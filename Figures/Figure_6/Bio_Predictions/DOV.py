from rdkit.Chem import AllChem
from rdkit import DataStructs
from collections import defaultdict
import pandas as pd

# Code that generates a dictionary to calculate DOV
# hashed_morgan_element: set(pm1, pm2)

def get_DOV_dict(df, radius=2):
    DOV_dict = defaultdict(set)

    for i, row in df.iterrows():
        m = AllChem.MolFromSmiles(row.smiles_polymer)
        fp = AllChem.GetHashedMorganFingerprint(m, radius=radius,
        )

        for key in fp.GetNonzeroElements():
            DOV_dict[key].add(row.pm)
    
    return DOV_dict

# Function that takes in a molecule and a dpm value
# gets the morgan fingerprints, and sees if that feature
# has been trained for a similar pm

def get_num_missing_features(smi, pm, dpm, DOV_dict, radius=2):
    num_missing_features = 0
    
    m = AllChem.MolFromSmiles(smi)
    fp = AllChem.GetHashedMorganFingerprint(m, radius=radius)
    
    for i, key in enumerate(fp.GetNonzeroElements()):
        if key not in DOV_dict:
            num_missing_features += 1
        else:
            matching_pm = False
            for pm_loc in DOV_dict[key]:
                min_pm = pm_loc-dpm
                max_pm = pm_loc+dpm
                if min_pm <= pm and pm <= max_pm:
                    matching_pm = True
                    break
            
            if not matching_pm:
                num_missing_features += 1
    
    return num_missing_features

def get_kfold_DOV_error(kfold, radius=2, dpm=0.05):
    df_results = pd.DataFrame()
    kfold_DOV_dict = get_DOV_dict(kfold.df_train, radius=radius)
    
    for i, row in kfold.df_validate_results.iterrows():
        num_missing_features = get_num_missing_features(row.smiles_polymer, row.pm, dpm, kfold_DOV_dict)
        tg_ae = abs(row.Tg - row.Tg_pred) if row.Tg else None
        tm_ae = abs(row.Tm - row.Tm_pred) if row.Tg else None
        
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    [{"num_missing_features": num_missing_features,
                     "tg_ae": tg_ae,
                     "tm_ae": tm_ae
                    }],
                    columns=["num_missing_features", "tg_ae", "tm_ae"]
                )
            ]
        )
        
        
    df_results = pd.concat([kfold.df_validate_results.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)
    return df_results

def get_smiles_DOV(smi, pm, kfold_DOV_dict, radius=2, dpm=0.05):
    df_results = pd.DataFrame()
    
    
    num_missing_features = get_num_missing_features(smi, pm, dpm, kfold_DOV_dict, radius=radius)
        
    return num_missing_features
    
def get_DOV_errs(num_missing_features, kfolds_DOV):
    kfolds_DOV["Tg_err"] = abs(kfolds_DOV["Tg_err"])
    kfolds_DOV["Tm_err"] = abs(kfolds_DOV["Tm_err"])
    df = kfolds_DOV[kfolds_DOV.num_missing_features <= num_missing_features].groupby(by="kfold_i").agg("mean").reset_index()
    df["lte_missing_features"] = num_missing_features
    return df