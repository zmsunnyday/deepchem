"""
Encodes smiles/smarts as RDKIT objects directly.
"""

import logging
from deepchem.feat import MolecularFeaturizer
from deepchem.feat import ReactionFeaturizer

logger = logging.getLogger(__name__)


class RawFeaturizer(MolecularFeaturizer):

  def __init__(self, smiles=False):
    self.smiles = smiles

  def _featurize(self, mol):
    from rdkit import Chem
    if self.smiles:
      return Chem.MolToSmiles(mol)
    else:
      return mol

class RawReactionFeaturizer(ReactionFeaturizer):
  """Featurize SMARTS as RDKit Reaction objects.

  This featurizer uses RDKit's
  `rdkit.Chem.rdChemReactions.ReactionFromSmarts` to parse in
  input SMARTS strings which encode reactions.
  """

  def __init__(self, smarts=True):
    """
    Parameters
    ----------
    smarts: bool, optional
      If True, process smarts into rdkit Reaction objects. Else don't process.
    """
    self.smarts = smarts 

  def _featurize(self, mol):
    """
    mol: string
      The SMARTS string to process.
    """
    from rdkit.Chem import rdChemReactions
    if self.smarts:
      smarts = mol 
      # Sometimes smarts have extraneous information at end of
      # form " |f:0" that causes parsing to fail. Not sure what
      # this information is, but just ignoring for now.
      smarts = smarts.split(" ")[0]
      rxn = rdChemReactions.ReactionFromSmarts(smarts)
      return rxn
    else:
      return mol
