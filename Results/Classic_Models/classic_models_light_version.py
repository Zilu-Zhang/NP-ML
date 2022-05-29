## load packages

# suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# basic modules
import pandas as pd
import numpy as np
import math
from tqdm import trange
import argparse

# rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

# sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# add command line parser for improved flexibility
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-t', '--training', default=None, help='link of training dataset, weblink')
parser.add_argument('-tn', '--training_name', default=None, help='name of training dataset, string')
parser.add_argument('-e', '--external', default=None, help='link of external dataset, weblink')
parser.add_argument('-m', '--mode', nargs='+', default=None, help='cross validation mode, list')
parser.add_argument('-osum', '--output_summary', default=None, help='output AUC results (including file type), string')
parser.add_argument('-oraw', '--output_raw', default=None, help='output raw proba results (including file type), string')
args = vars(parser.parse_args())

# define chemical features for molecular descriptions 
descr = Descriptors._descList
calc = [x[1] for x in descr]
d_name = [x[0] for x in descr]

def describe_mol(mol, descriptor=True):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,4,nBits=2048) # radius=4, bits=2048
    fp_list = []
    fp_list.extend(fp.ToBitString())
    fp_expl = []
    fp_expl = [float(x) for x in fp_list]
    if not descriptor:
        return fp_expl
    else:
        ds_n = []
        for d in calc:
            v = d(mol)
            if v > np.finfo(np.float32).max: # postprocess descriptors for freak large values 
                ds_n.append(np.finfo(np.float32).max)
            elif math.isnan(v):
                ds_n.append(np.float32(0.0))
            else:
                ds_n.append(np.float32(v))
        return fp_expl + list(ds_n) # features vector of each input: 2048-bit MorganFP + RDKit descriptors

training_filepath = args['training']
external_filepath = args['external']

training_df = pd.read_csv(training_filepath)

## data generation
def molecule_featurization(df):
    # concatenation
    X_concat_w_o = [describe_mol(Chem.MolFromSmiles(df['drug_smile'][i]), descriptor=False) + 
                    describe_mol(Chem.MolFromSmiles(df['excp_smile'][i]), descriptor=False) 
                    for i in trange(len(df), desc='X_concat_w_o')]
    X_concat_w = [describe_mol(Chem.MolFromSmiles(df['drug_smile'][i]), descriptor=True) + 
                  describe_mol(Chem.MolFromSmiles(df['excp_smile'][i]), descriptor=True) 
                  for i in trange(len(df), desc='X_concat_w')]

    # codrug (aggregate SMILE)
    X_codrug_w_o = [describe_mol(Chem.MolFromSmiles(df['pair_smile'][i]), descriptor=False) 
                    for i in trange(len(df), desc='X_codrug_w_o')]
    X_codrug_w = [describe_mol(Chem.MolFromSmiles(df['pair_smile'][i]), descriptor=True) 
                    for i in trange(len(df), desc='X_codrug_w')]

    # addition (summation of feature vector)
    drug_features = [describe_mol(Chem.MolFromSmiles(df['drug_smile'][i]), descriptor=False)
                     for i in trange(len(df), desc='X_addition_w_o_drug_features')]

    excp_features = [describe_mol(Chem.MolFromSmiles(df['excp_smile'][i]), descriptor=False)
                     for i in trange(len(df), desc='X_addition_w_o_excp_features')]
    X_addition_w_o = []
    for i in trange(len(drug_features), desc='X_addition_w_o'):
        X_addition_w_o.append([sum(value) for value in zip(drug_features[i], excp_features[i])])


    drug_features = [describe_mol(Chem.MolFromSmiles(df['drug_smile'][i]), descriptor=True)
                     for i in trange(len(df), desc='X_addition_w_drug_features')]

    excp_features = [describe_mol(Chem.MolFromSmiles(df['excp_smile'][i]), descriptor=True)
                     for i in trange(len(df), desc='X_addition_w_excp_features')]
    X_addition_w = []
    for i in trange(len(drug_features), desc='X_addition_w'):
        X_addition_w.append([sum(value) for value in zip(drug_features[i], excp_features[i])])

    return [X_concat_w_o, X_concat_w, X_codrug_w_o, X_codrug_w, X_addition_w_o, X_addition_w]


def preprocessing(X_train, X_test, transformation):
    if transformation == 'original':
        X_train_working, X_test_working = X_train.copy(), X_test.copy()
    elif transformation == 'scaling':
        scaler = MinMaxScaler().fit(X_train)
        X_train_working, X_test_working = scaler.transform(X_train), scaler.transform(X_test)
    elif transformation == 'standardization':
        scaler = StandardScaler().fit(X_train)
        X_train_working, X_test_working = scaler.transform(X_train), scaler.transform(X_test)
    elif transformation == 'hybrid_standardization': # only apply z score transformaion on descriptors. but not fps
        fp_index = np.concatenate((np.arange(2048), np.arange(2048+208,2048*2+208)))
        des_index = np.concatenate((np.arange(2048, 2048+208), np.arange(2048*2+208, 2048*2+208*2)))
        scaler = StandardScaler().fit(X_train[des_index])
        X_train_working = pd.concat([X_train[fp_index].reset_index(drop=True), 
                                     pd.DataFrame(scaler.transform(X_train[des_index]))], 
                                     axis=1, join='inner')
        X_train_working.reset_index(drop=True, inplace=True)
        X_test_working = pd.concat([X_test[fp_index].reset_index(drop=True), 
                                    pd.DataFrame(scaler.transform(X_test[des_index]))], 
                                    axis=1, join='inner')
        X_test_working.reset_index(drop=True, inplace=True)
    return X_train_working, X_test_working    

def generate_list(count):
  return [[] for i in range(count)]

def consolidate_data(df_list):
    output_df = pd.concat(df_list)
    output_df.reset_index(drop=True, inplace=True)
    return output_df

def validation_pipeline(X, y, training_dataset, training_dataset_name, 
                        model, mode, transformation, identifier, external_X=None, external_y=None):
    repeats = 10
    cv_results, raw_summary_all_df = generate_list(2)
    header = ['Repeat_'+str(i+1) for i in range(repeats)]
    header.insert(0, 'Model')
    header.insert(1, 'Featurizer')
    header.insert(2, 'Transformation')
    header.insert(3, 'Training_dataset')
    header.insert(4, 'Mode')
    header.insert(5, 'AUC_avg')
    header.insert(6, 'AUC_std')
    X = pd.DataFrame(X)
    y = np.array(y)
    
    for m in model:
        model_name = type(m).__name__
        entry = []
        if mode == 'Standard':
            for _ in trange(repeats, desc='{}, {}, {}, {}'.format(model_name, identifier, transformation, mode)):
                y_index, y_true, y_pred_proba = generate_list(3)
                kf = KFold(n_splits=10, shuffle=True)
                for train_index, test_index in kf.split(X):
                    X_train, y_train = X.iloc[train_index], y[train_index]
                    X_test, y_test = X.iloc[test_index], y[test_index]
                    y_index.extend(test_index)
                    y_true.extend(y_test)
                    X_train_working, X_test_working = preprocessing(X_train, X_test, transformation)
                    if isinstance(m, LinearSVC):
                        y_pred_proba.extend(m.fit(X_train_working, y_train).decision_function(X_test_working))
                    else:
                        y_pred_proba.extend(m.fit(X_train_working, y_train).predict_proba(X_test_working)[:, 1])            
                entry.append(roc_auc_score(y_true, y_pred_proba))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [identifier]*len(y_index),
                                                                        [transformation]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, y_true, y_pred_proba]),
                                                 columns=['Model', 'Featurizer', 'Transformation', 'Training_dataset', 
                                                          'Mode', 'Index', 'Label', 'Pred_Proba_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=np.column_stack([y_index, y_pred_proba]),
                                          columns=['index', 'pred_proba'])
                    raw_df = raw_df.astype({'index': int})
                    raw_df.sort_values(['index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns),
                                          value=raw_df['pred_proba'].to_list(),
                                          column='Pred_Proba_{}'.format(_+1))

        elif mode == 'Stratified':
            for _ in trange(repeats, desc='{}, {}, {}, {}'.format(model_name, identifier, transformation, mode)):
                y_index, y_true, y_pred_proba = generate_list(3)
                skf = StratifiedKFold(n_splits=10, shuffle=True)
                for train_index, test_index in skf.split(X, y):
                    X_train, y_train = X.iloc[train_index], y[train_index]
                    X_test, y_test = X.iloc[test_index], y[test_index]
                    y_index.extend(test_index)
                    y_true.extend(y_test)
                    X_train_working, X_test_working = preprocessing(X_train, X_test, transformation)
                    if isinstance(m, LinearSVC):
                        y_pred_proba.extend(m.fit(X_train_working, y_train).decision_function(X_test_working))
                    else:
                        y_pred_proba.extend(m.fit(X_train_working, y_train).predict_proba(X_test_working)[:, 1])            
                entry.append(roc_auc_score(y_true, y_pred_proba))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [identifier]*len(y_index),
                                                                        [transformation]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, y_true, y_pred_proba]),
                                                 columns=['Model', 'Featurizer', 'Transformation', 'Training_dataset', 
                                                          'Mode', 'Index', 'Label', 'Pred_Proba_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=np.column_stack([y_index, y_pred_proba]),
                                          columns=['index', 'pred_proba'])
                    raw_df = raw_df.astype({'index': int})
                    raw_df.sort_values(['index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns),
                                          value=raw_df['pred_proba'].to_list(),
                                          column='Pred_Proba_{}'.format(_+1))

        elif mode == 'LODO': # Leave One Drug Out
            for _ in trange(repeats, desc='{}, {}, {}, {}'.format(model_name, identifier, transformation, mode)):
                y_index, y_true, y_pred_proba = generate_list(3)
                for drug in set(training_dataset['drug_name']):
                    train_index = training_dataset[training_dataset['drug_name'] != drug].index
                    test_index = training_dataset[training_dataset['drug_name'] == drug].index
                    X_train, y_train = X.iloc[train_index], y[train_index]
                    X_test, y_test = X.iloc[test_index], y[test_index]
                    y_index.extend(test_index)
                    y_true.extend(y_test)
                    X_train_working, X_test_working = preprocessing(X_train, X_test, transformation)
                    if isinstance(m, LinearSVC):
                        y_pred_proba.extend(m.fit(X_train_working, y_train).decision_function(X_test_working))
                    else:
                        y_pred_proba.extend(m.fit(X_train_working, y_train).predict_proba(X_test_working)[:, 1])
                    
                entry.append(roc_auc_score(y_true, y_pred_proba))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [identifier]*len(y_index),
                                                                        [transformation]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, y_true, y_pred_proba]),
                                                 columns=['Model', 'Featurizer', 'Transformation', 'Training_dataset', 
                                                          'Mode', 'Index', 'Label', 'Pred_Proba_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=np.column_stack([y_index, y_pred_proba]),
                                          columns=['index', 'pred_proba'])
                    raw_df = raw_df.astype({'index': int})
                    raw_df.sort_values(['index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns),
                                          value=raw_df['pred_proba'].to_list(),
                                          column='Pred_Proba_{}'.format(_+1))

        elif mode == 'LOEO': # Leave One Excipient Out
            for _ in trange(repeats, desc='{}, {}, {}, {}'.format(model_name, identifier, transformation, mode)):
                y_index, y_true, y_pred_proba = generate_list(3)
                for excp in set(training_dataset['excp_name']):
                    train_index = training_dataset[training_dataset['excp_name'] != excp].index
                    test_index = training_dataset[training_dataset['excp_name'] == excp].index
                    X_train, y_train = X.iloc[train_index], y[train_index]
                    X_test, y_test = X.iloc[test_index], y[test_index]
                    y_index.extend(test_index)
                    y_true.extend(y_test)
                    X_train_working, X_test_working = preprocessing(X_train, X_test, transformation)
                    if isinstance(m, LinearSVC):
                        y_pred_proba.extend(m.fit(X_train_working, y_train).decision_function(X_test_working))
                    else:
                        y_pred_proba.extend(m.fit(X_train_working, y_train).predict_proba(X_test_working)[:, 1])
                
                entry.append(roc_auc_score(y_true, y_pred_proba))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [identifier]*len(y_index),
                                                                        [transformation]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, y_true, y_pred_proba]),
                                                 columns=['Model', 'Featurizer', 'Transformation', 'Training_dataset', 
                                                          'Mode', 'Index', 'Label', 'Pred_Proba_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=np.column_stack([y_index, y_pred_proba]),
                                          columns=['index', 'pred_proba'])
                    raw_df = raw_df.astype({'index': int})
                    raw_df.sort_values(['index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns),
                                          value=raw_df['pred_proba'].to_list(),
                                          column='Pred_Proba_{}'.format(_+1))

        elif mode == 'External':
            external_X = pd.DataFrame(external_X)
            external_y = np.array(external_y)
            X_train_working, X_test_working = preprocessing(X, external_X, transformation)
            y_train, y_test = y, external_y
            y_index = external_X.index.to_list()
            for _ in trange(repeats, desc='{}, {}, {}, {}'.format(model_name, identifier, transformation, mode)):
                y_true, y_pred_proba = generate_list(2)
                if isinstance(m, LinearSVC):
                    y_pred_proba.extend(m.fit(X_train_working, y_train).decision_function(X_test_working))
                else:
                    y_pred_proba.extend(m.fit(X_train_working, y_train).predict_proba(X_test_working)[:, 1])
                y_true.extend(y_test)
                entry.append(roc_auc_score(y_true, y_pred_proba))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [identifier]*len(y_index),
                                                                        [transformation]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, y_true, y_pred_proba]),
                                                 columns=['Model', 'Featurizer', 'Transformation', 'Training_dataset', 
                                                          'Mode', 'Index', 'Label', 'Pred_Proba_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=np.column_stack([y_index, y_pred_proba]),
                                          columns=['index', 'pred_proba'])
                    raw_df = raw_df.astype({'index': int})
                    raw_df.sort_values(['index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns),
                                          value=raw_df['pred_proba'].to_list(),
                                          column='Pred_Proba_{}'.format(_+1))

        score = entry.copy()
        entry.insert(0, model_name) # model name
        entry.insert(1, identifier) # featurizer
        entry.insert(2, transformation) # transformation
        entry.insert(3, training_dataset_name) # training set
        entry.insert(4, mode) # cv mode
        entry.insert(5, np.mean(score)) # mean auc scores
        entry.insert(6, np.std(score)) # standard deviation
        cv_results.append(entry)
        raw_summary_all_df.append(raw_summary_df)

    
    cv_output_df = pd.DataFrame(columns=header, data=cv_results)
    raw_output_df = consolidate_data(raw_summary_all_df)
    
    return cv_output_df, raw_output_df

model = [GaussianNB(), KNeighborsClassifier(3), DecisionTreeClassifier(), 
         RandomForestClassifier(n_estimators=500), MLPClassifier(), LinearSVC()]

print('--------------------')
print('Training/validation set molecule featurization...')
training_X = molecule_featurization(training_df)
training_y = training_df['class']

mode = args['mode']
if 'External' in mode:
    external_df = pd.read_csv(external_filepath)
    print('\n--------------------')
    print('External test set molecule featurization...')
    external_X = molecule_featurization(external_df)
    external_y = external_df['class']
else:
    external_X, external_y = None, None


transformation = ['original', 'standardization']
identifier = ['concat_w_o', 'concat_w', 
              'codrug_w_o', 'codrug_w',
              'addition_w_o', 'addition_w']

training_dataset_name = args['training_name']

valid_output_all, raw_proba_all = generate_list(2)

for i in range(len(training_X)):
    for t in transformation:
        if t == 'hybrid_standardization':
            if identifier[i] == 'concat_w' or identifier[i] == 'codrug_w':
                if external_X == None:
                    results = [validation_pipeline(training_X[i], training_y, training_df, 
                                                   training_dataset_name=training_dataset_name,
                                                   model=model, mode=k, transformation=t, identifier=identifier[i]) for k in mode]        
                else:
                    results = [validation_pipeline(training_X[i], training_y, training_df, 
                                                   training_dataset_name=training_dataset_name,
                                                   model=model, mode=k, transformation=t, identifier=identifier[i],
                                                   external_X=external_X[i], external_y=external_y) for k in mode]

                valid_output_, raw_proba_ = [results[x][0] for x in range(len(results))], [results[x][1] for x in range(len(results))]
                
                valid_output = consolidate_data(valid_output_)
                valid_output_all.append(valid_output)

                raw_proba = consolidate_data(raw_proba_)
                raw_proba_all.append(raw_proba)

        else:
            if external_X == None:
                    results = [validation_pipeline(training_X[i], training_y, training_df, 
                                                   training_dataset_name=training_dataset_name,
                                                   model=model, mode=k, transformation=t, identifier=identifier[i]) for k in mode]        
            else:
                    results = [validation_pipeline(training_X[i], training_y, training_df, 
                                                   training_dataset_name=training_dataset_name,
                                                   model=model, mode=k, transformation=t, identifier=identifier[i],
                                                   external_X=external_X[i], external_y=external_y) for k in mode]

            valid_output_, raw_proba_ = [results[x][0] for x in range(len(results))], [results[x][1] for x in range(len(results))]
                
            valid_output = consolidate_data(valid_output_)
            valid_output_all.append(valid_output)

            raw_proba = consolidate_data(raw_proba_)
            raw_proba_all.append(raw_proba)


valid_output_df = consolidate_data(valid_output_all)
raw_proba_df = consolidate_data(raw_proba_all)

valid_output_df.to_csv(args['output_summary'], index=False)
raw_proba_df.to_csv(args['output_raw'], index=False)
print('\n--------------------')
print('Job finished successfully!')

                    
