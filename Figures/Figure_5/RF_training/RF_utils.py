from functools import partial
from rdkit import Chem

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from polyml import MultiModel

from mordred import MolecularDistanceEdge, get_descriptors_in_module, descriptors, Calculator

def mae(a1, a2):
    return np.mean(np.abs(a1-a2))

def rmse(a1, a2):
    return np.sqrt(np.mean((a1-a2)**2))

import time

def calc_mordred_df(df, descriptor_list = [i for i in get_descriptors_in_module(descriptors) if i not in [MolecularDistanceEdge.MolecularDistanceEdge]]):
    calc = Calculator(descriptor_list, ignore_3D=True)
    df["mol"] = df.smiles_polymer.map(Chem.MolFromSmiles)
    df_mord = calc.pandas(list(df.mol.values))
    
    return df_mord