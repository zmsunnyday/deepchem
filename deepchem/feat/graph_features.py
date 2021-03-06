import enum
import numpy as np
import deepchem as dc
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates
from deepchem.feat.mol_graphs import ConvMol, WeaveMol
from deepchem.data import DiskDataset
import multiprocessing
import logging


def _featurize_complex(featurizer, mol_pdb_file, protein_pdb_file, log_message):
  logging.info(log_message)
  return featurizer._featurize_complex(mol_pdb_file, protein_pdb_file)


def one_of_k_encoding(x, allowable_set):
  """Encodes elements of a provided set as integers.

  Parameters
  ----------
  x: object
    Must be present in `allowable_set`. 
  allowable_set: list
    List of allowable quantities.

  Example
  -------
  >>> import deepchem as dc
  >>> dc.feat.graph_features.one_of_k_encoding("a", ["a", "b", "c"])         
  [True, False, False]

  Raises
  ------
  `ValueError` if `x` is not in `allowable_set`.
  """
  if x not in allowable_set:
    raise ValueError("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element.

  Unlike `one_of_k_encoding`, if `x` is not in `allowable_set`, this method
  pretends that `x` is the last element of `allowable_set`.

  Parameters
  ----------
  x: object
    Must be present in `allowable_set`. 
  allowable_set: list
    List of allowable quantities.

  Examples
  --------
  >>> dc.feat.graph_features.one_of_k_encoding_unk("s", ["a", "b", "c"])    
  [False, False, True]
  """
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
  """For list of lists, gets the cumulative products of the lengths

  Note that we add 1 to the lengths of all lists (to avoid an empty list
  propagating a 0).

  Parameters
  ----------
  l: list of lists
    Returns the cumulative product of these lengths.

  Examples
  --------
  >>> dc.feat.graph_features.get_intervals([[1], [1, 2], [1, 2, 3]])        
  [1, 3, 12]

  >>> dc.feat.graph_features.get_intervals([[1], [], [1, 2], [1, 2, 3]])    
  [1, 1, 3, 12]
  """
  intervals = len(l) * [0]
  # Initalize with 1
  intervals[0] = 1
  for k in range(1, len(l)):
    intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

  return intervals


def safe_index(l, e):
  """Gets the index of e in l, providing an index of len(l) if not found

  Parameters
  ----------
  l: list
    List of values
  e: object
    Object to check whether `e` is in `l`

  Examples
  --------
  >>> dc.feat.graph_features.safe_index([1, 2, 3], 1)                       
  0
  >>> dc.feat.graph_features.safe_index([1, 2, 3], 7)                       
  3
  """
  try:
    return l.index(e)
  except:
    return len(l)


class GraphConvConstants(object):
  """This class defines a collection of constants which are useful for graph convolutions on molecules."""
  possible_atom_list = [
      'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
      'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
  ]
  """Allowed Numbers of Hydrogens"""
  possible_numH_list = [0, 1, 2, 3, 4]
  """Allowed Valences for Atoms"""
  possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
  """Allowed Formal Charges for Atoms"""
  possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
  """This is a placeholder for documentation. These will be replaced with corresponding values of the rdkit HybridizationType"""
  possible_hybridization_list = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
  """Allowed number of radical electrons."""
  possible_number_radical_e_list = [0, 1, 2]
  """Allowed types of Chirality"""
  possible_chirality_list = ['R', 'S']
  """The set of all values allowed."""
  reference_lists = [
      possible_atom_list, possible_numH_list, possible_valence_list,
      possible_formal_charge_list, possible_number_radical_e_list,
      possible_hybridization_list, possible_chirality_list
  ]
  """The number of different values that can be taken. See `get_intervals()`"""
  intervals = get_intervals(reference_lists)
  """Possible stereochemistry. We use E-Z notation for stereochemistry
     https://en.wikipedia.org/wiki/E%E2%80%93Z_notation"""
  possible_bond_stereo = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
  """Number of different bond types not counting stereochemistry."""
  bond_fdim_base = 6


def get_feature_list(atom):
  """Returns a list of possible features for this atom.

  Parameters
  ----------
  atom: RDKit.rdchem.Atom
    Atom to get features for

  Examples
  --------
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles("C")
  >>> atom = mol.GetAtoms()[0]
  >>> dc.feat.graph_features.get_feature_list(atom)
  [0, 4, 4, 3, 0, 2]

  Note
  ----
  This method requires RDKit to be installed.

  Returns
  -------
  features: list
    List of length 6. The i-th value in this list provides the index of the
    atom in the corresponding feature value list. The 6 feature values lists
    for this function are `[GraphConvConstants.possible_atom_list,
    GraphConvConstants.possible_numH_list,
    GraphConvConstants.possible_valence_list,
    GraphConvConstants.possible_formal_charge_list,
    GraphConvConstants.possible_num_radical_e_list]`.
  """
  possible_atom_list = GraphConvConstants.possible_atom_list
  possible_numH_list = GraphConvConstants.possible_numH_list
  possible_valence_list = GraphConvConstants.possible_valence_list
  possible_formal_charge_list = GraphConvConstants.possible_formal_charge_list
  possible_number_radical_e_list = GraphConvConstants.possible_number_radical_e_list
  possible_hybridization_list = GraphConvConstants.possible_hybridization_list
  # Replace the hybridization
  from rdkit import Chem
  #global possible_hybridization_list
  possible_hybridization_list = [
      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
      Chem.rdchem.HybridizationType.SP3D2
  ]
  features = 6 * [0]
  features[0] = safe_index(possible_atom_list, atom.GetSymbol())
  features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
  features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
  features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
  features[4] = safe_index(possible_number_radical_e_list,
                           atom.GetNumRadicalElectrons())

  features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
  return features


def features_to_id(features, intervals):
  """Convert list of features into index using spacings provided in intervals

  Parameters
  ----------
  features: list
    List of features as returned by `get_feature_list()`
  intervals: list
    List of intervals as returned by `get_intervals()`  

  Returns
  -------
  id: int 
    The index in a feature vector given by the given set of features.
  """
  id = 0
  for k in range(len(intervals)):
    id += features[k] * intervals[k]

  # Allow 0 index to correspond to null molecule 1
  id = id + 1
  return id


def id_to_features(id, intervals):
  """Given an index in a feature vector, return the original set of features.

  Parameters
  ----------
  id: int 
    The index in a feature vector given by the given set of features.
  intervals: list
    List of intervals as returned by `get_intervals()`  

  Returns
  -------
  features: list
    List of features as returned by `get_feature_list()`
  """
  features = 6 * [0]

  # Correct for null
  id -= 1

  for k in range(0, 6 - 1):
    # print(6-k-1, id)
    features[6 - k - 1] = id // intervals[6 - k - 1]
    id -= features[6 - k - 1] * intervals[6 - k - 1]
  # Correct for last one
  features[0] = id
  return features


def atom_to_id(atom):
  """Return a unique id corresponding to the atom type

  Parameters
  ----------
  atom: RDKit.rdchem.Atom
    Atom to convert to ids.

  Returns
  -------
  id: int 
    The index in a feature vector given by the given set of features.
  """
  features = get_feature_list(atom)
  return features_to_id(features, intervals)


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
  """Helper method used to compute per-atom feature vectors.

  Many different featurization methods compute per-atom features such as ConvMolFeaturizer, WeaveFeaturizer. This method computes such features.

  Parameters
  ----------
  bool_id_feat: bool, optional
    Return an array of unique identifiers corresponding to atom type.
  explicit_H: bool, optional
    If true, model hydrogens explicitly
  use_chirality: bool, optional
    If true, use chirality information.

  Returns
  -------
  np.ndarray of per-atom features.
  """
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    from rdkit import Chem
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(bond, use_chirality=False):
  """Helper method used to compute bond feature vectors.

  Many different featurization methods compute bond features
  such as WeaveFeaturizer. This method computes such features.

  Parameters
  ----------
  use_chirality: bool, optional
    If true, use chirality information.

  Note
  ----
  This method requires RDKit to be installed.

  Returns
  -------
  bond_feats: np.ndarray
    Array of bond features. This is a 1-D array of length 6 if `use_chirality`
    is `False` else of length 10 with chirality encoded.
  """
  try:
    from rdkit import Chem
  except ModuleNotFoundError:
    raise ValueError("This method requires RDKit to be installed.")
  bt = bond.GetBondType()
  bond_feats = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
  ]
  if use_chirality:
    bond_feats = bond_feats + one_of_k_encoding_unk(
        str(bond.GetStereo()), possible_bond_stereo)
  return np.array(bond_feats)


def pair_features(mol, edge_list, canon_adj_list, bt_len=6,
                  graph_distance=True):
  """Helper method used to compute atom pair feature vectors.

  Many different featurization methods compute atom pair features
  such as WeaveFeaturizer. Note that atom pair features could be
  for pairs of atoms which aren't necessarily bonded to one
  another.

  Parameters
  ----------
  mol: RDKit Mol
    Molecule to compute features on.
  edge_list: list
    List of edges to consider
  canon_adj_list: list of lists
    `canon_adj_list[i]` is a list of the atom indices that atom `i` shares a
    list. This list is symmetrical so if `j in canon_adj_list[i]` then `i in
    canon_adj_list[j]`.
  bt_len: int, optional (default 6)
    The number of different bond types to consider.
  graph_distance: bool, optional (default True)
    If true, use graph distance between molecules. Else use euclidean distance.

  Note
  ----
  This method requires RDKit to be installed.

  Returns
  -------
  features: np.ndarray
    Of shape `(N, N, bt_len + max_distance + 1)`. This is the array of pairwise
    features for all atom pairs.
  """
  if graph_distance:
    max_distance = 7
  else:
    max_distance = 1
  N = mol.GetNumAtoms()
  features = np.zeros((N, N, bt_len + max_distance + 1))
  num_atoms = mol.GetNumAtoms()
  rings = mol.GetRingInfo().AtomRings()
  for a1 in range(num_atoms):
    for a2 in canon_adj_list[a1]:
      # first `bt_len` features are bond features(if applicable)
      features[a1, a2, :bt_len] = np.asarray(
          edge_list[tuple(sorted((a1, a2)))], dtype=float)
    for ring in rings:
      if a1 in ring:
        # `bt_len`-th feature is if the pair of atoms are in the same ring
        features[a1, ring, bt_len] = 1
        features[a1, a1, bt_len] = 0.
    # graph distance between two atoms
    if graph_distance:
      distance = find_distance(
          a1, num_atoms, canon_adj_list, max_distance=max_distance)
      features[a1, :, bt_len + 1:] = distance
  # Euclidean distance between atoms
  if not graph_distance:
    coords = np.zeros((N, 3))
    for atom in range(N):
      pos = mol.GetConformer(0).GetAtomPosition(atom)
      coords[atom, :] = [pos.x, pos.y, pos.z]
    features[:, :, -1] = np.sqrt(np.sum(np.square(
      np.stack([coords] * N, axis=1) - \
      np.stack([coords] * N, axis=0)), axis=2))

  return features


def find_distance(a1, num_atoms, canon_adj_list, max_distance=7):
  """Computes distances from provided atom.

  Parameters
  ----------
  a1: RDKit atom
    The source atom to compute distances from.
  num_atoms: int
    The total number of atoms.
  canon_adj_list: list of lists
    `canon_adj_list[i]` is a list of the atom indices that atom `i` shares a
    list. This list is symmetrical so if `j in canon_adj_list[i]` then `i in
    canon_adj_list[j]`.
  max_distance: int, optional (default 7)
    The max distance to search.

  Returns
  -------
  distances: np.ndarray
    Of shape `(num_atoms, max_distance)`. Provides a one-hot encoding of the
    distances. That is, `distances[i]` is a one-hot encoding of the distance
    from `a1` to atom `i`.
  """
  distance = np.zeros((num_atoms, max_distance))
  radial = 0
  # atoms `radial` bonds away from `a1`
  adj_list = set(canon_adj_list[a1])
  # atoms less than `radial` bonds away
  all_list = set([a1])
  while radial < max_distance:
    distance[list(adj_list), radial] = 1
    all_list.update(adj_list)
    # find atoms `radial`+1 bonds away
    next_adj = set()
    for adj in adj_list:
      next_adj.update(canon_adj_list[adj])
    adj_list = next_adj - all_list
    radial = radial + 1
  return distance


class ConvMolFeaturizer(MolecularFeaturizer):
  """This class implements the featurization to implement Duvenaud graph convolutions.

  Duvenaud graph convolutions [1]_ construct a vector of descriptors for each
  atom in a molecule. The featurizer computes that vector of local descriptors.

  References
  ---------

  .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
         learning molecular fingerprints." Advances in neural information
         processing systems. 2015.

  Note
  ----
  This class requires RDKit to be installed.
  """
  name = ['conv_mol']

  def __init__(self, master_atom=False, use_chirality=False,
               atom_properties=[]):
    """
    Parameters
    ----------
    master_atom: Boolean
      if true create a fake atom with bonds to every other atom.
      the initialization is the mean of the other atom features in
      the molecule.  This technique is briefly discussed in
      Neural Message Passing for Quantum Chemistry
      https://arxiv.org/pdf/1704.01212.pdf
    use_chirality: Boolean
      if true then make the resulting atom features aware of the
      chirality of the molecules in question
    atom_properties: list of string or None
      properties in the RDKit Mol object to use as additional
      atom-level features in the larger molecular feature.  If None,
      then no atom-level properties are used.  Properties should be in the
      RDKit mol object should be in the form
      atom XXXXXXXX NAME
      where XXXXXXXX is a zero-padded 8 digit number coresponding to the
      zero-indexed atom index of each atom and NAME is the name of the property
      provided in atom_properties.  So "atom 00000000 sasa" would be the
      name of the molecule level property in mol where the solvent
      accessible surface area of atom 0 would be stored.

    Since ConvMol is an object and not a numpy array, need to set dtype to
    object.
    """
    self.dtype = object
    self.master_atom = master_atom
    self.use_chirality = use_chirality
    self.atom_properties = list(atom_properties)

  def _get_atom_properties(self, atom):
    """
    For a given input RDKit atom return the values of the properties
    requested when initializing the featurize.  See the __init__ of the
    class for a full description of the names of the properties

    Parameters
    ----------
    atom: RDKit.rdchem.Atom
      Atom to get the properties of
    returns a numpy lists of floats of the same size as self.atom_properties
    """
    values = []
    for prop in self.atom_properties:
      mol_prop_name = str("atom %08d %s" % (atom.GetIdx(), prop))
      try:
        values.append(float(atom.GetOwningMol().GetProp(mol_prop_name)))
      except KeyError:
        raise KeyError("No property %s found in %s in %s" %
                       (mol_prop_name, atom.GetOwningMol(), self))
    return np.array(values)

  def _featurize(self, mol):
    """Encodes mol as a ConvMol object."""
    # Get the node features
    idx_nodes = [(a.GetIdx(),
                  np.concatenate((atom_features(
                      a, use_chirality=self.use_chirality),
                                  self._get_atom_properties(a))))
                 for a in mol.GetAtoms()]

    idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
    idx, nodes = list(zip(*idx_nodes))

    # Stack nodes into an array
    nodes = np.vstack(nodes)
    if self.master_atom:
      master_atom_features = np.expand_dims(np.mean(nodes, axis=0), axis=0)
      nodes = np.concatenate([nodes, master_atom_features], axis=0)

    # Get bond lists with reverse edges included
    edge_list = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    ]

    # Get canonical adjacency list
    canon_adj_list = [[] for mol_id in range(len(nodes))]
    for edge in edge_list:
      canon_adj_list[edge[0]].append(edge[1])
      canon_adj_list[edge[1]].append(edge[0])

    if self.master_atom:
      fake_atom_index = len(nodes) - 1
      for index in range(len(nodes) - 1):
        canon_adj_list[index].append(fake_atom_index)

    return ConvMol(nodes, canon_adj_list)

  def feature_length(self):
    return 75 + len(self.atom_properties)

  def __hash__(self):
    atom_properties = tuple(self.atom_properties)
    return hash((self.master_atom, self.use_chirality, atom_properties))

  def __eq__(self, other):
    if not isinstance(self, other.__class__):
      return False
    return self.master_atom == other.master_atom and \
           self.use_chirality == other.use_chirality and \
           tuple(self.atom_properties) == tuple(other.atom_properties)


class WeaveFeaturizer(MolecularFeaturizer):
  """This class implements the featurization to implement Weave convolutions.
  
  Weave convolutions were introduced in [1]_. Unlike Duvenaud graph
  convolutions, weave convolutions require a quadratic matrix of interaction
  descriptors for each pair of atoms. These extra descriptors may provide for
  additional descriptive power but at the cost of a larger featurized dataset.

  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
         fingerprints." Journal of computer-aided molecular design 30.8 (2016):
         595-608.

  Note
  ----
  This class requires RDKit to be installed.
  """

  name = ['weave_mol']

  def __init__(self, graph_distance=True, explicit_H=False,
               use_chirality=False):
    """
    Parameters
    ----------
    graph_distance: bool, optional
      If true, use graph distance. Otherwise, use Euclidean
      distance.
    explicit_H: bool, optional
      If true, model hydrogens in the molecule.
    use_chirality: bool, optional
      If true, use chiral information in the featurization
    """
    # Distance is either graph distance(True) or Euclidean distance(False,
    # only support datasets providing Cartesian coordinates)
    self.graph_distance = graph_distance
    # Set dtype
    self.dtype = object
    # If includes explicit hydrogens
    self.explicit_H = explicit_H
    # If uses use_chirality
    self.use_chirality = use_chirality
    if self.use_chirality:
      self.bt_len = int(
          GraphConvConstants.bond_fdim_base) + len(possible_bond_stereo)
    else:
      self.bt_len = int(GraphConvConstants.bond_fdim_base)

  def _featurize(self, mol):
    """Encodes mol as a WeaveMol object."""
    # Atom features
    idx_nodes = [(a.GetIdx(),
                  atom_features(
                      a,
                      explicit_H=self.explicit_H,
                      use_chirality=self.use_chirality))
                 for a in mol.GetAtoms()]
    idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
    idx, nodes = list(zip(*idx_nodes))

    # Stack nodes into an array
    nodes = np.vstack(nodes)

    # Get bond lists
    edge_list = {}
    for b in mol.GetBonds():
      edge_list[tuple(sorted([b.GetBeginAtomIdx(),
                              b.GetEndAtomIdx()]))] = bond_features(
                                  b, use_chirality=self.use_chirality)

    # Get canonical adjacency list
    canon_adj_list = [[] for mol_id in range(len(nodes))]
    for edge in edge_list.keys():
      canon_adj_list[edge[0]].append(edge[1])
      canon_adj_list[edge[1]].append(edge[0])

    # Calculate pair features
    pairs = pair_features(
        mol,
        edge_list,
        canon_adj_list,
        bt_len=self.bt_len,
        graph_distance=self.graph_distance)

    return WeaveMol(nodes, pairs)


class AtomicConvFeaturizer(ComplexNeighborListFragmentAtomicCoordinates):
  """This class computes the Atomic Convolution features"""

  # TODO (VIGS25): Complete the description

  name = ['atomic_conv']

  def __init__(self,
               labels,
               neighbor_cutoff,
               frag1_num_atoms=70,
               frag2_num_atoms=634,
               complex_num_atoms=701,
               max_num_neighbors=12,
               batch_size=24,
               atom_types=[
                   6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35.,
                   53., -1.
               ],
               radial=[[
                   1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                   7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
               ], [0.0, 4.0, 8.0], [0.4]],
               layer_sizes=[32, 32, 16],
               strip_hydrogens=True,
               learning_rate=0.001,
               epochs=10):
    """
    Parameters

    labels: numpy.ndarray
      Labels which we want to predict using the model
    neighbor_cutoff: int
      TODO (VIGS25): Add description
    frag1_num_atoms: int
      Number of atoms in first fragment
    frag2_num_atoms: int
      Number of atoms in second fragment
    complex_num_atoms: int
      TODO (VIGS25) : Add description
    max_num_neighbors: int
      Maximum number of neighbors possible for an atom
    batch_size: int
      Batch size used for training and evaluation
    atom_types: list
      List of atoms recognized by model. Atoms are indicated by their
      nuclear numbers.
    radial: list
      TODO (VIGS25): Add description
    layer_sizes: list
      List of layer sizes for the AtomicConvolutional Network
    strip_hydrogens: bool
      Whether to remove hydrogens while computing neighbor features
    learning_rate: float
      Learning rate for training the model
    epochs: int
      Number of epochs to train the model for
    """

    self.atomic_conv_model = dc.models.atomic_conv.AtomicConvModel(
        frag1_num_atoms=frag1_num_atoms,
        frag2_num_atoms=frag2_num_atoms,
        complex_num_atoms=complex_num_atoms,
        max_num_neighbors=max_num_neighbors,
        batch_size=batch_size,
        atom_types=atom_types,
        radial=radial,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate)

    super(AtomicConvFeaturizer, self).__init__(
        frag1_num_atoms=frag1_num_atoms,
        frag2_num_atoms=frag2_num_atoms,
        complex_num_atoms=complex_num_atoms,
        max_num_neighbors=max_num_neighbors,
        neighbor_cutoff=neighbor_cutoff,
        strip_hydrogens=strip_hydrogens)

    self.epochs = epochs
    self.labels = labels

  def featurize_complexes(self, mol_files, protein_files):
    features = []
    failures = []
    for i, (mol_file, protein_pdb) in enumerate(zip(mol_files, protein_files)):
      logging.info("Featurizing %d / %d" % (i, len(mol_files)))
      new_features = self._featurize_complex(mol_file, protein_pdb)
      # Handle loading failures which return None
      if new_features is not None:
        features.append(new_features)
      else:
        failures.append(ind)

    features = np.asarray(features)
    labels = np.delete(self.labels, failures)
    dataset = DiskDataset.from_numpy(features, labels)

    # Fit atomic conv model
    self.atomic_conv_model.fit(dataset, nb_epoch=self.epochs)

    # Add the Atomic Convolution layers to fetches
    layers_to_fetch = [
        self.atomic_conv_model._frag1_conv, self.atomic_conv_model._frag2_conv,
        self.atomic_conv_model._complex_conv
    ]

    # Extract the atomic convolution features
    atomic_conv_features = list()
    batch_generator = self.atomic_conv_model.default_generator(
        dataset=dataset, epochs=1)

    for X, y, w in batch_generator:
      frag1_conv, frag2_conv, complex_conv = self.atomic_conv_model.predict_on_generator(
          [(X, y, w)], outputs=layers_to_fetch)
      concatenated = np.concatenate(
          [frag1_conv, frag2_conv, complex_conv], axis=1)
      atomic_conv_features.append(concatenated)

    batch_size = self.atomic_conv_model.batch_size

    if len(features) % batch_size != 0:
      num_batches = (len(features) // batch_size) + 1
      num_to_skip = num_batches * batch_size - len(features)
    else:
      num_to_skip = 0

    atomic_conv_features = np.asarray(atomic_conv_features)
    atomic_conv_features = atomic_conv_features[-num_to_skip:]
    atomic_conv_features = np.squeeze(atomic_conv_features)

    return atomic_conv_features, failures
