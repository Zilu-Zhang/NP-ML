# For cross validation, use labeled triples to distinguish training and test split;
# For making predictions, first generate unified dataset containing both training and 
# prediction dataset, and then similarly use labeled triples to distungihsh the two

## load packages
import pandas as pd
import numpy as np
import json
import requests
from tqdm import trange
import collections.abc
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem

from chemicalx.data import DrugFeatureSet
from chemicalx.data import ContextFeatureSet
from chemicalx.data import LabeledTriples

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


## load models
import torch
from chemicalx.data import BatchGenerator
from chemicalx.models import DeepDDI
from chemicalx.models import DeepDDS
from chemicalx.models import DeepDrug
from chemicalx.models import DeepSynergy
from chemicalx.models import EPGCNDS
from chemicalx.models import GCNBMP
from chemicalx.models import MRGNN
from chemicalx.models import SSIDDI




def mol_featurization(df, col_list, test_index=None):
  # integrate all mols information
  [drug_name_col, drug_smile_col, excp_name_col, excp_smile_col, label_col] = col_list
  mol_name = df[drug_name_col].to_list() + df[excp_name_col].to_list()
  mol_smile = df[drug_smile_col].to_list() + df[excp_smile_col].to_list()
  mol_fp = [list(AllChem.GetMorganFingerprintAsBitVect(m, 4, nBits=2048)) 
            for m in [Chem.MolFromSmiles(s) for s in mol_smile]]
  # generate drug_feature_set
  drug_set_dic = {}
  for i in range(len(mol_name)):
    drug_set_dic[mol_name[i]] = {'smiles': mol_smile[i],
                                 'features': mol_fp[i]}
  # generate context_feature_set
  context_set_dic = {'Nanoparticle': df[label_col].to_list()}

  # generate labeled_triple_set
  triple_df = df[[drug_name_col, excp_name_col, label_col]].rename(columns={drug_name_col: 'drug_1',
                                                                            excp_name_col: 'drug_2',
                                                                            label_col: 'label'})
  triple_df.insert(loc=2, column='context', value=['Nanoparticle']*len(triple_df))
  if test_index:
    train_triple = triple_df.iloc[:test_index, :]
    test_triple = triple_df.iloc[test_index:, :]
    triple_df = [train_triple, test_triple]

  return drug_set_dic, context_set_dic, triple_df

def merge_drug(dic1, dic2):
  dic1.update(dic2)
  return dic1

def merge_context(dic1, dic2):
  dic1_context = dic1['Nanoparticle']
  dic2_context = dic2['Nanoparticle']
  return {'Nanoparticle': dic1_context + dic2_context}

def merge_triple(df1, df2):
    output_df = pd.concat([df1, df2])
    output_df.reset_index(drop=True, inplace=True)
    return output_df

# reset the parameters
def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def generate_list(count):
  return [[] for i in range(count)]

def consolidate_data(df_list):
    output_df = pd.concat(df_list)
    output_df.reset_index(drop=True, inplace=True)
    return output_df

## core function
def make_predictions(
    training_dataset, 
    test_dataset,
    model,
    drug_molecules,
    batch_size,
    epoch):
    # generate batches
    optimizer = torch.optim.Adam(model.parameters())
    generator = BatchGenerator(batch_size=batch_size,
                               context_features=True,
                               drug_features=True,
                               drug_molecules=drug_molecules,
                               drug_feature_set=training_dataset[0],
                               context_feature_set=training_dataset[1],
                               labeled_triples=training_dataset[2])
    # model training
    model.apply(reset_weights) # clear history or last trained model
    model.train()
    loss = torch.nn.BCELoss()
    epoch = epoch
    
    for _ in trange(epoch, desc=model._get_name()):
        for batch in generator:
            optimizer.zero_grad()
            prediction = model(*model.unpack(batch))
            if isinstance(prediction,collections.abc.Sequence):
                prediction = prediction[0]
            loss_value = loss(prediction, batch.labels)
            loss_value.backward()
            optimizer.step()
            
    
    # model testing
    model.eval()
    generator.drug_feature_set = test_dataset[0]
    generator.context_feature_set = test_dataset[1]
    generator.labeled_triples = test_dataset[2]
    
    predictions = []
    for batch in generator:
        prediction = model(*model.unpack(batch))
        if isinstance(prediction, collections.abc.Sequence):
            prediction = prediction[0]
        prediction = prediction.detach().cpu().numpy()
        identifiers = batch.identifiers
        identifiers["prediction"] = prediction
        predictions.append(identifiers)
    
    return predictions

def validation_pipeline(training_dataset, training_dataset_name, 
                        model, parameter, mode, batch_size=1024, epoch=200):
    repeats = 10
    cv_results, raw_summary_all = generate_list(2)
    header = ['Repeat'+str(i+1) for i in range(repeats)]
    header.insert(0, 'Model')
    header.insert(1, 'Parameter')
    header.insert(2, 'Dataset')
    header.insert(3, 'Mode')
    header.insert(4, 'AUC_avg')
    header.insert(5, 'AUC_std')
    
    n = 0
    for m in model:
        model_name = m._get_name()
        drug_molecules = False if isinstance(m, (DeepDDI, DeepSynergy, MatchMaker)) else True
        entry = []
        
        if mode == 'Standard': # perform 10 fold cross validation
            for _ in trange(repeats, desc='{}, {}, {}'.format(model_name, parameter[n], mode)):
                predictions = []
                kf = KFold(n_splits=10, shuffle=True) # 10-fold
                y_index = []
                for train_index, test_index in kf.split(training_dataset[2].data):
                    cv_train = LabeledTriples(training_dataset[2].data.iloc[train_index])
                    cv_test = LabeledTriples(training_dataset[2].data.iloc[test_index])
                    y_index.extend(test_index)
                    
                    predictions.append(make_predictions([training_dataset[0], training_dataset[1], cv_train], # cv_training dataset
                                                        [training_dataset[0], training_dataset[1], cv_test], # cv_test dataset
                                                        m, # model
                                                        drug_molecules, # drug_molecules
                                                        batch_size, # batch
                                                        epoch)[0])
                predictions_df = pd.concat(predictions)
                entry.append(roc_auc_score(predictions_df['label'], predictions_df['prediction']))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [parameter[n]]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, predictions_df['label'], 
                                                                        predictions_df['prediction']]), 
                                                  columns=['Model', 'Parameter', 'Dataset', 'Mode',
                                                           'Index', 'Label', 'Prediction_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=zip(y_index, predictions_df['prediction']),
                                          columns=['Index', 'Prediction'])
                    raw_df = raw_df.astype({'Index': int})
                    raw_df.sort_values(['Index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns), 
                                          value=raw_df['Prediction'].to_list(),
                                          column='Prediction_{}'.format(_+1))

        if mode == 'Stratified': # perform 10 fold stratified cross validation
            for _ in trange(repeats, desc='{}, {}, {}'.format(model_name, parameter[n], mode)):
                predictions = []
                skf = KFold(n_splits=10, shuffle=True) # 10-fold
                for train_index, test_index in skf.split(training_dataset[2].data, training_dataset[2].data['label']):
                    cv_train = LabeledTriples(training_dataset[2].data.iloc[train_index])
                    cv_test = LabeledTriples(training_dataset[2].data.iloc[test_index])
                    
                    predictions.append(make_predictions([training_dataset[0], training_dataset[1], cv_train], # cv_training dataset
                                                        [training_dataset[0], training_dataset[1], cv_test], # cv_test dataset
                                                        m, # model
                                                        drug_molecules, # drug_molecules
                                                        batch_size, # batch
                                                        epoch)[0])
                predictions_df = pd.concat(predictions)
                entry.append(roc_auc_score(predictions_df['label'], predictions_df['prediction']))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [parameter[n]]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, predictions_df['label'], 
                                                                        predictions_df['prediction']]), 
                                                  columns=['Model', 'Parameter', 'Dataset', 'Mode',
                                                           'Index', 'Label', 'Prediction_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=zip(y_index, predictions_df['prediction']),
                                          columns=['Index', 'Prediction'])
                    raw_df = raw_df.astype({'Index': int})
                    raw_df.sort_values(['Index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns), 
                                          value=raw_df['Prediction'].to_list(),
                                          column='Prediction_{}'.format(_+1))
    
        if mode == 'LODO': # leave-one-drug-out validation
            for _ in trange(repeats, desc='{}, {}, {}'.format(model_name, parameter[n], mode)):
                predictions = []
                for drug in set(training_dataset[2].data['drug_1']):
                    train_index = training_set[2].data[training_set[2].data['drug_1'] != drug].index
                    test_index = training_set[2].data[training_set[2].data['drug_1'] == drug].index
                    cv_train = LabeledTriples(training_dataset[2].data.iloc[train_index])
                    cv_test = LabeledTriples(training_dataset[2].data.iloc[test_index])                    
                    
                    predictions.append(make_predictions([training_dataset[0], training_dataset[1], cv_train], # cv_training dataset
                                                        [training_dataset[0], training_dataset[1], cv_test], # cv_test dataset
                                                        m, # model
                                                        drug_molecules, # drug_molecules
                                                        batch_size, # batch
                                                        epoch)[0])
                predictions_df = pd.concat(predictions)
                entry.append(roc_auc_score(predictions_df['label'], predictions_df['prediction']))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [parameter[n]]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, predictions_df['label'], 
                                                                        predictions_df['prediction']]), 
                                                  columns=['Model', 'Parameter', 'Dataset', 'Mode',
                                                           'Index', 'Label', 'Prediction_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=zip(y_index, predictions_df['prediction']),
                                          columns=['Index', 'Prediction'])
                    raw_df = raw_df.astype({'Index': int})
                    raw_df.sort_values(['Index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns), 
                                          value=raw_df['Prediction'].to_list(),
                                          column='Prediction_{}'.format(_+1))

       if mode == 'LOEO': # leave-one-excipient-out validation
            for _ in trange(repeats, desc='{}, {}, {}'.format(model_name, parameter[n], mode)):
                predictions = []
                for drug in set(training_dataset[2].data['drug_2']):
                    train_index = training_set[2].data[training_set[2].data['drug_2'] != drug].index
                    test_index = training_set[2].data[training_set[2].data['drug_2'] == drug].index
                    cv_train = LabeledTriples(training_dataset[2].data.iloc[train_index])
                    cv_test = LabeledTriples(training_dataset[2].data.iloc[test_index])                    
                    
                    predictions.append(make_predictions([training_dataset[0], training_dataset[1], cv_train], # cv_training dataset
                                                        [training_dataset[0], training_dataset[1], cv_test], # cv_test dataset
                                                        m, # model
                                                        drug_molecules, # drug_molecules
                                                        batch_size, # batch
                                                        epoch)[0])
                predictions_df = pd.concat(predictions)
                entry.append(roc_auc_score(predictions_df['label'], predictions_df['prediction']))
                if _ == 0:
                    raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                        [parameter[n]]*len(y_index),
                                                                        [training_dataset_name]*len(y_index),
                                                                        [mode]*len(y_index),
                                                                        y_index, predictions_df['label'], 
                                                                        predictions_df['prediction']]), 
                                                  columns=['Model', 'Parameter', 'Dataset', 'Mode',
                                                           'Index', 'Label', 'Prediction_1'])
                    raw_summary_df = raw_summary_df.astype({'Index': int})
                    raw_summary_df.sort_values(['Index'], inplace=True)
                else:
                    raw_df = pd.DataFrame(data=zip(y_index, predictions_df['prediction']),
                                          columns=['Index', 'Prediction'])
                    raw_df = raw_df.astype({'Index': int})
                    raw_df.sort_values(['Index'], inplace=True)
                    raw_summary_df.insert(loc=len(raw_summary_df.columns), 
                                          value=raw_df['Prediction'].to_list(),
                                          column='Prediction_{}'.format(_+1))
         
        score = entry.copy()
        entry.insert(0, model_name) # model name
        entry.insert(1, parameter[n]) # parameter
        entry.insert(2, training_dataset_name)
        entry.insert(3, mode) # validation mode
        entry.insert(4, np.mean(score)) # mean auc scores
        entry.insert(5, np.std(score)) # standard deviation
        cv_results.append(entry)
        raw_summary_all.append(raw_summary_df)
        n += 1
    
    cv_output_df = pd.DataFrame(columns=header, data=cv_results)
    raw_output_df = consolidate_data(raw_summary_all)
    return cv_output_df, raw_output_df

def external_validation_pipeline(training_dataset, test_dataset, training_dataset_name,
                                 model, parameter, mode, batch_size=1024, epoch=200):
    repeats = 10
    cv_results, raw_summary_all = generate_list(2)
    header = ['Repeat'+str(i+1) for i in range(repeats)]
    header.insert(0, 'Model')
    header.insert(1, 'Params')
    header.insert(2, 'Dataset')
    header.insert(3, 'Mode')
    header.insert(4, 'AUC_avg')
    header.insert(5, 'AUC_std')

    n = 0
    for m in model:
        model_name = m._get_name()
        drug_molecules = False if isinstance(m, (DeepDDI, DeepSynergy, MatchMaker)) else True
        entry = []
        for _ in trange(repeats, desc='{}, {}, {}'.format(model_name, parameter[n], mode)):
            predictions_df = make_predictions([training_dataset[0], training_dataset[1], cv_train], # cv_training dataset
                                                        [training_dataset[0], training_dataset[1], cv_test], # cv_test dataset
                                                        m, # model
                                                        drug_molecules, # drug_molecules
                                                        batch_size, # batch
                                                        epoch)[0]
            entry.append(roc_auc_score(predictions_df['label'], predictions_df['prediction']))
            if _ == 0:
                raw_summary_df = pd.DataFrame(data=np.column_stack([[model_name]*len(y_index),
                                                                    [parameter[n]]*len(y_index),
                                                                    [training_dataset_name]*len(y_index),
                                                                    [mode]*len(y_index),
                                                                    y_index, predictions_df['label'], 
                                                                    predictions_df['prediction']]), 
                                              columns=['Model', 'Parameter', 'Dataset', 'Mode',
                                                       'Index', 'Label', 'Prediction_1'])
                raw_summary_df = raw_summary_df.astype({'Index': int})
                raw_summary_df.sort_values(['Index'], inplace=True)
            else:
                raw_df = pd.DataFrame(data=zip(y_index, predictions_df['prediction']),
                                      columns=['Index', 'Prediction'])
                raw_df = raw_df.astype({'Index': int})
                raw_df.sort_values(['Index'], inplace=True)
                raw_summary_df.insert(loc=len(raw_summary_df.columns), 
                                      value=raw_df['Prediction'].to_list(),
                                      column='Prediction_{}'.format(_+1))
        score = entry.copy()
        entry.insert(0, model_name) # model name
        entry.insert(1, parameter[n]) # parameter
        entry.insert(2, training_dataset_name)
        entry.insert(3, mode) # validation mode
        entry.insert(4, np.mean(score)) # mean auc scores
        entry.insert(5, np.std(score)) # standard deviation
        cv_results.append(entry)
        raw_summary_all.append(raw_summary_df)
        n += 1

    cv_output_df = pd.DataFrame(columns=header, data=cv_results)
    raw_output_df = consolidate_data(raw_summary_all_df)

    return cv_output_df, raw_output_df

def prediction_pipeline(training_dataset, test_dataset, model, batch_size=1024, epoch=200):
    prediction_summary = []
    for m in model:
        model_name = m._get_name()
        drug_molecules = False if isinstance(m, (DeepDDI,DeepSynergy,MatchMaker)) else True
        entry = []
        predictions_df = pd.concat(make_predictions(training_dataset,
                                                    test_dataset,
                                                    m, # model 
                                                    drug_molecules,
                                                    batch_size=batch_size,
                                                    epoch=epoch)
        )
        predictions_df.insert(loc=0, value=[model_name]*len(predictions_df), column='model')
        prediction_summary.append(predictions_df)

    output_df = pd.concat(prediction_summary)
    return output_df

# add command line parser for improved flexibility
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-t', '--training', default=None, help='link of training dataset, weblink')
parser.add_argument('-tn', '--training_name', default=None, help='name of training dataset, string')
parser.add_argument('-e', '--external', default=None, help='link of external dataset, weblink')
parser.add_argument('-m', '--mode', nargs='+', default=None, help='cross validation mode, list')
parser.add_argument('-osum', '--output_summary', default=None, help='output AUC results (including file type), string')
parser.add_argument('-oraw', '--output_raw', default=None, help='output raw proba results (including file type), string')
args = vars(parser.parse_args())

# load training/validation dataset and generate all required inputs for deep models
training_filepath = args['training']
training_df = pd.read_csv(training_filepath)
training_drug_set, training_context_set, training_triple_df = mol_featurization(training_df, training_df.columns[:5])
training_dataset = [training_drug_set, training_context_set, training_triple_df]
training_dataset_name = args['training_name']

mode = args['mode']
if 'External' in mode:
    external_filepath = args['external']
    external_df = pd.read_csv(external_filepath)
    external_drug_set, external_context_set, external_triple_df = mol_featurization(exteral_df, external_df.columns[:5])

    # generate unified set used for making predictions
    unified_drug_set = merge_drug(training_drug_set, external_drug_set)
    unified_context_set = merge_context(training_context_set, external_context_set)
    unified_dataset_training = [unified_drug_set, unified_context_set, training_triple_df]
    unified_dataset_external = [unified_drug_set, unified_context_set, external_triple_df]

valid_output_all, raw_proba_all = generate_list(2)

parameter = [
             ['hidden_layers_num=9, default', 'hidden_layers_num=5', 'hidden_layers_num=12'],
             ['dropout_rate=0.5, default', 'dropout_rate=0.25', 'dropout_rate=0.75'],
             ['dropout_rate=0.5, default', 'dropout_rate=0.25', 'dropout_rate=0.75'],
             ['dropout_rate=0.5, default', 'dropout_rate=0.25', 'dropout_rate=0.75'],
             ['dropout_rate=0.1, default', 'dropout_rate=0.05', 'dropout_rate=0.3'],
             ['hidden_channels=32, default', 'hidden_channels=16', 'hidden_channels=64'],
             ['hidden_conv_layers=1, default', 'hidden_conv_layers=3', 'hidden_conv_layers=5'],
             ['hidden_channels=32, default', 'hidden_channels=16', 'hidden_channels=64'],
             ['head_number=(2,2), default', 'head_number=(1,1)', 'head_number=(4,4)']
             ]
## executation
drug_channels = 2048
for mode in modes:
    if mode != 'External':
        context_channels = len(training_context_set['Nanoparticle'])
        model = [
            ## feature/FP-based models
            DeepDDI(drug_channels=drug_channels, hidden_layers_num=9), # default
            DeepDDI(drop_channels=drug_channels, hidden_layers_num=5),
            DeepDDI(drop_channels=drug_channels, hidden_layers_num=12),

            DeepSynergy(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.5), # default
            DeepSynergy(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.25),
            DeepSynergy(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.75),

            MatchMaker(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.5), # default
            MatchMaker(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.25),
            MatchMaker(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.75),

            ## graph-based models
            DeepDDS(context_channels=context_channels, dropout_rate=0.5), # default
            DeepDDS(context_channels=context_channels, dropout_rate=0.25),
            DeepDDS(context_channels=context_channels, dropout_rate=0.75),

            DeepDrug(dropout_rate=0.1), # default
            DeepDrug(dropout_rate=0.05),
            DeepDrug(dropout_rate=0.3),

            EPGCNDS(hidden_channels=32), # default
            EPGCNDS(hidden_channels=16),
            EPGCNDS(hidden_channels=64),

            GCNBMP(hidden_conv_layers=1), # default
            GCNBMP(hidden_conv_layers=3),
            GCNBMP(hidden_conv_layers=5),

            MRGNN(hidden_channels=32), # dafault
            MRGNN(hidden_channels=16),
            MRGNN(hidden_channels=64),

            SSIDDI(head_number=(2,2)), # default
            SSIDDI(head_number=(1,1)),
            SSIDDI(head_number=(4,4))] 

        results = [validation_pipeline(training_dataset, training_dataset_name, model, 
                                       parameter, mode)]
        valid_output_, raw_proba_ = [results[x][0] for x in range(len(results))], [results[x][1] for x in range(len(results))]

        valid_output = consolidate_data(valid_output_)
        valid_output_all.append(valid_output)

        raw_proba = consolidate_data(raw_proba_)
        raw_proba_all.append(raw_proba)

    else:
        context_channels = len(unified_context_set['Nanoparticle'])
        model = [
            ## feature/FP-based models
            DeepDDI(drug_channels=drug_channels, hidden_layers_num=9), # default
            DeepDDI(drop_channels=drug_channels, hidden_layers_num=5),
            DeepDDI(drop_channels=drug_channels, hidden_layers_num=12),

            DeepSynergy(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.5), # default
            DeepSynergy(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.25),
            DeepSynergy(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.75),

            MatchMaker(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.5), # default
            MatchMaker(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.25),
            MatchMaker(context_channels=context_channels,drug_channels=drug_channels, dropout_rate=0.75),

            ## graph-based models
            DeepDDS(context_channels=context_channels, dropout_rate=0.5), # default
            DeepDDS(context_channels=context_channels, dropout_rate=0.25),
            DeepDDS(context_channels=context_channels, dropout_rate=0.75),

            DeepDrug(dropout_rate=0.1), # default
            DeepDrug(dropout_rate=0.05),
            DeepDrug(dropout_rate=0.3),

            EPGCNDS(hidden_channels=32), # default
            EPGCNDS(hidden_channels=16),
            EPGCNDS(hidden_channels=64),

            GCNBMP(hidden_conv_layers=1), # default
            GCNBMP(hidden_conv_layers=3),
            GCNBMP(hidden_conv_layers=5),

            MRGNN(hidden_channels=32), # dafault
            MRGNN(hidden_channels=16),
            MRGNN(hidden_channels=64),

            SSIDDI(head_number=(2,2)), # default
            SSIDDI(head_number=(1,1)),
            SSIDDI(head_number=(4,4))] 

        results = [external_validation_pipeline(unified_dataset_training, unified_dataset_external,
                                                training_dataset_name, model,
                                                parameter, mode)]
        valid_output_, raw_proba_ = [results[x][0] for x in range(len(results))], [results[x][1] for x in range(len(results))]

        valid_output = consolidate_data(valid_output_)
        valid_output_all.append(valid_output)

        raw_proba = consolidate_data(raw_proba_)
        raw_proba_all.append(raw_proba)


valid_output_df = consolidate_data(valid_output_all)
raw_proba_df = consolidate_data(raw_proba_all)

valid_output_df.to_csv(args['chemicalx_output_summary'], index=False)
raw_proba_df.to_csv(args['chemicalx_output_raw'], index=False)
print('\n--------------------')
print('Job finished successfully!')