from argparse import ArgumentParser
from pathlib import Path
from shutil import copy

import tensorflow as tf

from polyml.preprocessors import PolymerPreprocessor, WeightBinPreprocessor
import random
from typing import Dict

from polyml import MultiModel
import pandas as pd
import hashlib

import networkx as nx
import numpy as np

from nfp.preprocessing.features import atom_features_v2, bond_features_v1

pwd = Path(__file__).parent

def main(model_folder, save_folder, df_folder, kfolds):
    print(model_folder)
    mm = MultiModel().load_models(model_folder)
    df = pd.read_csv(df_folder)
    df.loc[:, "id"] = df.apply(lambda row: hashlib.sha224(
        f'{row.smiles_monomer}{row.pm}{row.distribution}'.encode('utf-8')).hexdigest(), axis=1
    )

    df.loc[:, "Mn"] = 1

    df_predictions = pd.DataFrame()
    print(kfolds)
    for kfold in kfolds:
        print(kfold, len(mm.models), "printing")
        df_kfold = mm.models[kfold].predict(df).copy()
        df_kfold["kfold"] = kfold

        df_kfold.to_csv(f"/projects/invpoly/kshebek/stereochemistry_hub/predictions/prediction/results/copo_pred_{kfold}.csv")
        df_predictions = pd.concat([df_predictions, df_kfold])

    # df_predictions.to_csv(save_folder, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Parameters
    parser.add_argument("--kfolds", nargs="+", default=[])    
    # Save Options
    parser.add_argument("--df_folder")
    parser.add_argument("--save_folder", default=None)
    parser.add_argument("--model_folder", default=None)

    values = parser.parse_args()
    
    save_folder = values.save_folder
    kfolds = values.kfolds
    df_folder = values.df_folder
    model_folder = values.model_folder

    main(
        model_folder = model_folder,
        kfolds=[int(i) for i in values.kfolds],
        save_folder=save_folder,
        df_folder=df_folder,
    )

