
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from math import log10
from rdkit import DataStructs
from rdkit.Chem import MolStandardize
from neutral import NeutraliseCharges
from multiprocessing import Pool
from rdkit import RDLogger
import pandas as pd
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles):
    if len(smiles)==0:
        return ''
    mol = Chem.MolFromSmiles(smiles)
    lfc = MolStandardize.fragment.LargestFragmentChooser()
    
    if mol is not None:
        mol2 = lfc.choose(mol)
        smi2=Chem.MolToSmiles(mol2, isomericSmiles=True)
        smi,_=NeutraliseCharges(smi2)
        return smi
    else:
        return ''


def run(line):
    
    smi=line.smiles.values.tolist()
    p=Pool(30)
    smi=p.map(process_tautomer,smi) 
    if smi==''  is None:
        return None
    else:
        data={'smiles':smi
  }
        data=pd.DataFrame(data)
        return data

def process_tautomer(smi):

    smiles=canonicalize_smiles(smi)
    mol = Chem.MolFromSmiles(smiles)
    enumerator = rdMolStandardize.TautomerEnumerator()
    processed=enumerator.Canonicalize(mol)
   # processed = enumerator.Enumerate(mol)
    smi2=Chem.MolToSmiles(processed, isomericSmiles=True)

    return smi2
def enumerate_tautomers_smiles(smiles):
    """Return a set of tautomers as SMILES strings, given a SMILES string.

    :param smiles: A SMILES string.
    :returns: A set containing SMILES strings for every possible tautomer.
    :rtype: set of strings.
    """
    # Skip sanitize as standardize does this anyway
    smiles=canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    params=rdMolStandardize.CleanupParamers()
    params.maxTautomers = 5
    tautomers = rdMolStandardize.TautomerEnumerator().Enumerate(mol)
    return {Chem.MolToSmiles(m, isomericSmiles=True) for m in tautomers}

