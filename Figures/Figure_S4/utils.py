
import pickle
from tokenize import group
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import glob 
import shutil
import warnings 

from polyml import MultiModel, SingleModel
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

tqdm.pandas()

class Utils():

    def __init__(self) -> None:
        self.nrel_colors = ['#0B5E90',
                            '#0079C2',
                            '#00A4E4',
                            '#5DD2FF',
                            '#A16911',
                            '#F4A11A',
                            '#FFC423',
                            '#FFD200',
                            '#3D6321',
                            '#5D9732',
                            '#8CC63F',
                            '#C1EE86',
                            '#6F2D01',
                            '#933C06',
                            '#D9531E',
                            '#FE6523',
                            '#4B545A',
                            '#5E6A71',
                            '#D1D5D8',
                            '#DEE2E5',
                            '#000000',
                            '#212121',
                            '#282D30',
                            '#3A4W46']

    def pickle_read(self,file):
        """

        Args:
            file (string): location and filename with file extension 

        Returns:
            _type_: stored object
        """
        with open(file, 'rb') as f:
            return pickle.load(f)
    
    def pickle_write(self,save_object,file):
        """

        Args:
            save_object (object): any object to be pickled
            file (string): location and filename with file extension 
        """

        with open(file, 'wb') as f:
            pickle.dump(save_object, f)

    def log10_transform(self,df:pd.DataFrame,cols):
        """
        Performs log10 transformation of columns in dataframe

        Args:
            df (pd.DataFrame): dataframe for which to apply log10 transformations
            cols (list): cols in dataframe for which to apply log10 transormations
        """
        df[cols] = df[cols].apply(np.log10)
        df = df.rename({col:'log10_'+col for col in cols},axis=1)
        return df

    def log10_inversetransform(self,df,cols):
        """
        Performs inverse log10 transformation of columns in dataframe

        Args:
            df (pd.DataFrame): dataframe for which to apply log10 inverse transformations
            cols (list): cols in dataframe for which to apply log10 inverse transormations
        """
        df[cols] = df[cols].apply(lambda col: 10**col)
        df = df.rename({col:col.replace('log10_','') for col in cols},axis=1)
        return df

    def bigdata_predict(self,dfpoly: pd.DataFrame,models: List[SingleModel],loc='.',save_dfs=True,verbose=1) -> None:
        """function for making predictions and storing results in temporary csv file
        this breaks out the prediction task in polyml for big datasets so the results
        can be stored in a temporary location and so the process could be deployed across
        HPC.
        

        Args:
            dfpoly (pd.DataFrame): dataframe containing polymer smiles stroings
            models (List[SingleModel]): list of singlemodels, would be Multmodel.models if using multimodel object
            loc (str, optional): location where temp data folder will be made. Defaults to '.'. 
            verbose (int, optional): prints times if >0. Defaults to 0.
        """

        # make temp dir
        path_storepath = Path(loc).joinpath('temp_data')
        Path.mkdir(path_storepath,exist_ok=True,parents=True)
        
        for i,mm_i in enumerate(models):

            if verbose>0:
                start = datetime.datetime.now().replace(microsecond=0)
                print('Predicting for model: {}'.format(i))
                print('Start Time: {}'.format(start))
            
            # make predictions and store
            dfpred_i = mm_i.predict(dfpoly)
            dfpred_i['model_id'] = i
            pred_storepath_i = str(path_storepath)+'/pred_model_{}.csv.gz'.format(i)
            dfpred_i.to_csv(pred_storepath_i.format(i),compression='gzip')
            
            if verbose>0:
                end = datetime.datetime.now().replace(microsecond=0)
                print('Time to Finish: {}\n'.format(end-start))        
        
        # aggregrate final dfpred
        print('Aggregating Files')
        path_storepath_temp = Path(loc+'/temp_data')
        filename_preds = glob.glob(str(path_storepath_temp) + "/*.csv.gz")
        dfpreds = pd.DataFrame()
        for filename_pred  in filename_preds:
            dfpreds = pd.concat([dfpreds,
                        pd.read_csv(filename_pred,index_col=0)])
        
        if save_dfs:
            dfpreds.to_csv(Path(loc).joinpath('dfpreds.csv.gz'),compression='gzip')
        
        return dfpreds
    
    def bigdata_aggregate(self,loc='.',aggregate=True,keep_temp_folder=True,save_dfs=True,merge_replicate_structures=True) -> pd.DataFrame:
        """aggregates data generated from bigdata_predict

        Prediction and aggregation are done in two steps for HPC purposes.

        Args:
            loc (str, optional): location of temp_data folder from bigdata_predict. Defaults to '.'.
            aggregate (bool, optional): if true will calculate mean,std,count of dfpreds for each model. Defaults to True.
            keep_temp_folder (bool, optional): if false will remove the temp data folder. Defaults to True.
            save_dfs (bool, optional): will stor
        Returns:
            pd.DataFrame: dfpreds and dfpredagg
        """
        start = datetime.datetime.now().replace(microsecond=0)
        print('Aggregating Data, Start Time: {}'.format(start))
        dfpreds = pd.read_csv(Path(Path(loc).joinpath('dfpreds.csv.gz')),index_col=0)


        # get keys for aggregating
        pred_keys = dfpreds.keys()[dfpreds.keys().str.contains('_pred')]
        group_keys = dfpreds.keys()[~dfpreds.keys().str.contains('_pred')].drop('model_id')
        if merge_replicate_structures:
            group_keys = group_keys.drop(['replicate_structure','smiles_polymer'])

        # converts object cols to strs for merging
        dfpreds[group_keys] = dfpreds[group_keys].astype('str')

        # define aggregate function
        aggfunc = {col:[np.mean,np.std,np.count_nonzero] for col in pred_keys}
        if merge_replicate_structures:
            aggfunc['smiles_polymer'] = 'first'

        # aggregate
        dfpredagg = dfpreds.groupby(list(group_keys),as_index=False).agg(aggfunc)

        # flatten and clean column names
        colnames = ['_'.join(col).strip() for col in dfpredagg.columns.values]
        colnames = [c[:-1] if c[-1:]=='_' else c for c in colnames]
        colnames = [c.replace('_nonzero','').replace('_first','')  for c in colnames]
        dfpredagg.columns = colnames
            #assert dfpredagg.shape[0] ==int(dfpreds.shape[0]/dfpreds.model_id.nunique()/dfpreds.replicate_structure.nunique())
            
        end = datetime.datetime.now().replace(microsecond=0)
        print('Time to Finish: {}\n'.format(end-start))   
        if not keep_temp_folder:
            shutil.rmtree(Path(loc).joinpath('temp_data/'))

        if save_dfs:
            dfpredagg.to_csv(Path(loc).joinpath('dfpredagg.csv.gz'),compression='gzip')
        
        return dfpreds,dfpredagg

    def aggregate(self,dfpred,merge_keys,merge_replicate_structures=True,merge_modelid=True,stats = [np.mean,np.count_nonzero,np.std,min,max]):
        """_summary_

        Args:
            dfpred (_type_): _description_
            merge_keys (_type_): _description_
            merge_replicate_structures (bool, optional): _description_. Defaults to True.
            merge_modelid (bool, optional): _description_. Defaults to True.
            stats (list, optional): _description_. Defaults to [np.mean,np.count_nonzero,np.std,min,max].

        Returns:
            _type_: _description_
        """
        # getting all other columns other than ones which will be merged
        meta_keys = dfpred.keys()[~dfpred.keys().isin(merge_keys)]

        # removing items from meta_keys based on desired merging
        if merge_modelid == True:
            try:
                meta_keys = meta_keys.drop('model_id')
            except:
                pass
        if merge_replicate_structures == True:
            try:
                meta_keys = meta_keys.drop('replicate_structure')
            except:
                pass

        # meta_keys
        meta_keys = meta_keys.drop('smiles_polymer')
        aggfunc = {k:stats for k in merge_keys}
        aggfunc['smiles_polymer'] = 'first'
        dfpredagg = dfpred.groupby(list(meta_keys),as_index=False).agg(aggfunc)

        # flatten and clean column names
        colnames = ['_'.join(col).strip() for col in dfpredagg.columns.values]
        colnames = [c[:-1] if c[-1:]=='_' else c for c in colnames]
        colnames = [c.replace('_nonzero','').replace('_first','')  for c in colnames]
        dfpredagg.columns = colnames
        return dfpredagg

    def round_var(self,col):
        rounddic = {'Density':[np.round,2],
                    'Glass_Transition':[int,0],
                    'Melt_Temp':[int,0],
                    'YoungMod':['E',100],
                    'log10_Permeability_O2':[np.round,2],
                    'log10_Permeability_H2O':[np.round,2],
                    'log10_Permeability_CO2':[np.round,2],
                    'log10_Permeability_N2':[np.round,2],
                    'Tensile_Strength':['E',1],
                    'log10_ElongBreak':[np.round,2]
                    }

        varname = col.name.replace('_pred_mean','').replace('_pred_std','')


        if varname in rounddic.keys():

            roundfunc = rounddic[varname]
            if roundfunc[0]==int:
                try:col = col.astype(int)
                except:pass
            elif roundfunc[0]==np.round:
                try:col = roundfunc[0](col,decimals=roundfunc[1])
                except:pass
            elif roundfunc[0]=='E':
                try:col = (((col/roundfunc[1]).astype(int))*roundfunc[1]).astype(int)
                except:pass
            else:
                pass

        else:pass
        return col
    
class DoV():
    def __init__(self,fileloc_training_fingerprints=''):
        self.fingerprint_col = 'smiles_polymer'
        try: 
            self.dftrain_fps = pd.read_csv(fileloc_training_fingerprints,index_col=0)
        except:
            warnings.warn('No training fingperprints file found. Training fingerprints can be generated using get_fingerprints')

    def get_fp(self,smiles,radius=2):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, radius,useFeatures=False)
        return pd.Series(fp.GetNonzeroElements(),name=smiles)

    def get_fps(self,df,radius=2):

        return df.progress_apply(lambda row: self.get_fp(row[self.fingerprint_col],radius),axis=1).fillna(0)

    def __count_fps_overlap(self,row,dftrain_fps,alltraincount,aggmethod='count'):
        row.index = row.index.astype(int)
        testcount = row[row!=0]

        dfcount = pd.DataFrame({'testcount':testcount,'alltraincount':alltraincount}).fillna(0)
        dfcount = dfcount[dfcount.testcount>0]
        returnvalue = sum(dfcount.alltraincount==0)

        return pd.Series(returnvalue,name='min_occ')

    def get_fps_overlap(self,df,dftrain_fps=None,radius=2,aggmethod='count'):
        if type(dftrain_fps) != pd.DataFrame: 
            dftrain_fps = self.dftrain_fps
        alltraincount = dftrain_fps.sum(0)
        alltraincount.index = alltraincount.index.astype(int)

        df_fps  = self.get_fps(df,radius=radius)
        dfoccur = df_fps.apply(lambda row:self.__count_fps_overlap(row,dftrain_fps,alltraincount,aggmethod=aggmethod),axis=1)
        dfoccur.columns=['fps_notin_train']
        return pd.concat([df,dfoccur],axis=1)

class Analyze():

    def __init__(self):
        pass 

    def get_epochs(self,location,kfolds=10):
        """gets dataframe of loss, val loss by epoch

        Args:
            location (string): location of models
            kfolds (int or list): integer will get a list of kfolds from 0 to that integer. List will get those specific kfolds
        """
        if type(kfolds)==int:kfolds = range(0,kfolds)
        if type(kfolds)==list:pass

        dfepochs = pd.DataFrame()
        for kfold in kfolds:
            df = pd.read_csv(location+'model_{}/log.csv'.format(kfold))
            df['kfold'] = kfold
            dfepochs = pd.concat([dfepochs,df])
        return dfepochs

    def get_epochs_avg(self,dfepochs):
        aggfunc = {'loss':np.mean,'val_loss':np.mean,'kfold':np.count_nonzero}
        dfepochs_avg = (dfepochs.groupby(['epoch'])
                         .agg(aggfunc)
                         .reset_index()).rename({'kfold':'kfolds_complete'},axis=1)
        return dfepochs_avg

    def plot_epochs(self,ax1,ax2,dfepochs,kfolds = None,verbose=0):
        if kfolds == None:kfolds = dfepochs.kfold.unique().tolist()
        if type(kfolds)==int:kfolds = range(0,kfolds)
        if type(kfolds)==list:pass

        #max epoch
        df = pd.pivot_table(dfepochs,columns= 'kfold',index='epoch')
        epoch_max = df[~df.isnull().any(1)].reset_index().epoch.max()

        #max epochs_i
        #if verbose>0:
            #print(pd.DataFrame(dfepochs.groupby('kfold').epoch.max()).T)

        #plotting each loss and val loss
        for kfold in kfolds:

            dfepochs_i = dfepochs[dfepochs.kfold==kfold]

            #Loss
            x1 = dfepochs_i.epoch
            y1 = dfepochs_i.loss            
            ax1.plot(x1,y1,color='lightgrey')

            #Val loss
            y1 = dfepochs_i.val_loss 
            ax2.plot(x1,y1,color='lightgrey')

        #plotting mean loss & val loss
        dfepochs_avg = self.get_epochs_avg(dfepochs)

        x2 = dfepochs_avg.epoch
        y2 = dfepochs_avg.loss
        ax1.plot(x2,y2,color='red')
        ax1.set_title('{}\n'.format('Loss'))

        y2 = dfepochs_avg.val_loss
        ax2.plot(x2,y2,color='red')
        ax2.set_title('Validation\n{}'.format('Loss'))
        sns.despine(offset=5)

        #report val
        dfreturn = None
        if verbose==1:
            aggfunc = {'kfolds_complete':np.min,'loss':np.mean,'val_loss':np.mean}
            dfreturn = dfepochs_avg.tail(5).agg(aggfunc)
            print('\n',dfreturn.round(3))

        if verbose==2:
            dfreturn = (pd.DataFrame(pd.pivot_table(dfepochs,columns='kfold',index='epoch',values=['loss','val_loss'])
                               .tail().mean()).reset_index().rename({0:'value'},axis=1))
            dfreturn = pd.pivot_table(dfreturn,index='kfold',columns='level_0')
            dfreturn.columns = ['loss','val_loss']
            dfreturn = pd.merge(dfreturn,dfepochs.groupby('kfold').epoch.max(),left_on='kfold',right_on='kfold',how='inner').T
            print(dfreturn.round(3))

        return dfreturn