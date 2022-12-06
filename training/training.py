#Pre-processing stuff
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emoji

#Modules required for training BERT model
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import set_seed
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from transformers import RobertaForSequenceClassification
from transformers import BertForPreTraining
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

#Modules required for Sklearn metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Modules required for the SVM Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import pickle

#Modules required for splitting data
import string
import pickle
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Modules required for scikit learn modesl
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

#Miscellaneous modules
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import re
import sys
import gc
import os

#Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

#K-fold library
from sklearn.model_selection import StratifiedKFold
#Convert number to ordinal numbers
from number_parser import parse_ordinal

import matplotlib
import transformers
matplotlib.use('Agg')
#How to use it:
#1st parameter: Number of folds
#2nd parameter: GPU using (0-5)
#3rd parameter: Random seed (first validation used 17)
#4th parameter: random state (first validation used 1)
#5th parameter: Training set path
#6th parameter: Testing set path
#7th parameter: Experiment Name
print('Transformers version: '+str(transformers.__version__))
print('Reading arguments... \n')
no_folds = int(sys.argv[1])
no_gpu = int(sys.argv[2])
seed_val = int(sys.argv[3])
training_dataframe = str(sys.argv[5])
testing_dataframe = str(sys.argv[6])
experiment_name = str(sys.argv[7])
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
set_seed(seed_val)

print('Reading datasets (Face Mask)... \n')
df_training = pd.read_csv(training_dataframe, delimiter="\t", names=["tweet_text", "tweet_id","tweet_class", "label"], index_col=None, header = 0)
df_validation = pd.read_csv(testing_dataframe, delimiter="\t", names=["tweet_text", "tweet_id","tweet_class", "label"], index_col=None, header = 0)

#Training dataset will only contain a specific percentage of the original dataset
percentage_text = ""
percentage_training = 1.0

if percentage_training < 1.0:
    percentage_text = '('+str(int(percentage_training*100))+'%)'

if percentage_training > 1.0:
    percentage_text = '(+'+str(int(percentage_training*100)-100)+'%)'

#Encoding the labels
possible_labels = df_training.tweet_class.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

#Inserting the labels with the encoded labels
df_training['label'] = df_training.tweet_class.replace(label_dict)
df_validation['label'] = df_validation.tweet_class.replace(label_dict)
#Making sure the labels are integers
df_training['label'] = df_training['label'].astype('int')
df_validation['label'] = df_validation['label'].astype('int')

#Example of dataset labeling
print('Example of how dataset was labeled: \n')
print(df_training.groupby('label').first())
label_dict_inverse = {v: k for k, v in label_dict.items()}
print('Values for each label \n')
print(label_dict_inverse)

#Creating a dataframe to store the results of each fine-tuned model
column_names_with = ["SVM", "LR", "RF", "NB", "DT", "BERT", "BERTL", "CTBERT","RoBERTa","RoBERTa_Large"]
model_results_with = pd.DataFrame(columns = column_names_with)


print('Training data size: '+str(len(df_training)))
print(df_training.head())
print('Evaluation data size: '+str(len(df_validation)))
print(df_validation.head())

#Creating the required folders to store the model's checkpoint
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
fold_names = [ordinal(n) for n in range(1,no_folds+1)]

#Creating the models folder
isExist = os.path.exists('models')
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs('models')

for n in range(1,int(sys.argv[4])+1):
    if not os.path.exists('models/'+str(n)+'_randomstate'):
        os.makedirs('models/'+str(n)+'_randomstate')
    for m in range(0,no_folds):
        if not os.path.exists('models/'+str(n)+'_randomstate/'+str(fold_names[m])+'_fold'):
            os.makedirs('models/'+str(n)+'_randomstate/'+str(fold_names[m])+'_fold')
        if not os.path.exists('models/'+str(n)+'_randomstate/'+str(fold_names[m])+'_fold/'+str(experiment_name)):
            os.makedirs('models/'+str(n)+'_randomstate/'+str(fold_names[m])+'_fold/'+str(experiment_name))

#Creating the metrics folder
isExist = os.path.exists('metrics')
if not isExist:
   # Create a new directory because it does not exist
    os.makedirs('metrics')

if not os.path.exists('models/'+str(n)+'_randomstate'):
    os.makedirs('metrics/'+str(sys.argv[4])+'_randomstate')
if not os.path.exists('metrics/'+str(sys.argv[4])+'_randomstate/best_fold'):
    os.makedirs('metrics/'+str(sys.argv[4])+'_randomstate/best_fold')
if not os.path.exists('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)):
    os.makedirs('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name))
if not os.path.exists('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports'):
    os.makedirs('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports')
if not os.path.exists('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions'):
    os.makedirs('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions')
if not os.path.exists('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix'):
    os.makedirs('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix')
  

#Final values for the training and validation (data and target values)
X = df_training.tweet_text.values
Y = df_training.label.values

#Performance metrics
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def accuracy_per_class(preds, labels):    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

#Training process loop

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def generate_dataloaders(X_train, X_test, y_train, y_test):
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=350, 
        return_tensors='pt',
        truncation=True
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        X_test, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=350, 
        return_tensors='pt',
        truncation=True
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(y_train)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(y_test)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    #Data Loader
    batch_size = 8
    dataloader_tr = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    dataloader_vl = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

    return dataloader_tr, dataloader_vl

def training_process(model, model_name, no_epochs):
    global best_f1_score, best_fold, current_fold
    for epoch in tqdm(range(1, no_epochs+1)):
        
        model.train()
        
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
        #We save only the last epoch
        if epoch == epochs:     
            torch.save(model.state_dict(),'models/'+str(sys.argv[4])+'_randomstate/'+fold_names[current_fold]+'_fold/'+str(experiment_name)+'/finetuned_'+model_name+percentage_text+'_epoch_'+str(epoch)+'.model')
            
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        val_accuracy = accuracy_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        tqdm.write(f'Accuracy Score: {val_accuracy}')

    if best_f1_score < val_f1:
        best_f1_score = val_f1
        best_fold = fold_names[current_fold]
    
    current_fold = current_fold+1

#This will delete the fine-tuned models from the non-best folds
#Keeping only the best fold for each model (from each randomstate)
def keep_best_fold(the_fold_names,the_best_fold,model_type,mod_name, perc_text):
    model_to_remove = ""
    for name_fold in the_fold_names:
        if name_fold != the_best_fold:
            if model_type == "transformers":
                model_to_remove = 'models/'+str(sys.argv[4])+'_randomstate/'+name_fold+'_fold/'+str(experiment_name)+'/finetuned_'+mod_name+perc_text+'_epoch_5.model'
            if model_type == "sklearn":
                model_to_remove = 'models/'+str(sys.argv[4])+'_randomstate/'+name_fold+'_fold/'+str(experiment_name)+'/trained_'+mod_name+perc_text+'.pickle'
            os.remove(model_to_remove)

def generate_reports(model_name):
    #Loading and evaluating the model (from the best fold) into the testing dataset
    encoded_data_val = tokenizer.batch_encode_plus(
        df_validation.tweet_text.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=350, 
        return_tensors='pt',
        truncation=True
    )

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_validation.label.values)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    #Data Loader
    batch_size = 8
    dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

    _, predictions, true_vals = evaluate(dataloader_validation)
    accuracy_per_class(predictions, true_vals)

    #Storing the classification report
    bert_preds_flat = np.argmax(predictions, axis=1).flatten()
    bert_labels_flat = true_vals.flatten()

    #For the ensemble model
    model_results_with[model_name] = bert_preds_flat.tolist()

    bert_model_classification_report = classification_report(bert_labels_flat.tolist(), bert_preds_flat.tolist(), target_names=label_dict_inverse.values(), output_dict=True)
    bert_model_classification_report = pd.DataFrame(bert_model_classification_report).transpose()
    #Exporting the results and model
    bert_model_classification_report.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports/'+model_name+percentage_text+'_metrics.csv', index= True)
    #Exporting the labels and predictions
    predictions_columns = ["True","Predicted"]
    df_predictions = pd.DataFrame(columns = predictions_columns)
    df_predictions["True"] = bert_labels_flat.tolist()
    df_predictions["Predicted"] = bert_preds_flat.tolist()
    df_predictions.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions/'+model_name+percentage_text+'_predictions.csv',index=False)

    #Exporting confusion matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confusion_matrix(bert_labels_flat.tolist(), bert_preds_flat.tolist(), normalize='all'),
    annot=True, fmt='.2f', xticklabels=label_dict_inverse.values(), yticklabels=label_dict_inverse.values())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix/'+model_name+percentage_text+'_matrix.png')

#===============================================BERT MODEL ======================================================================
#To avoid our CUDA's memory being full, we empty the cache first
gc.collect()
torch.cuda.empty_cache()
device = "cuda:"+str(no_gpu)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#BERT Pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model_name = "BERT"

#5-fold cross validation (stratified)
skf = StratifiedKFold(n_splits=5, random_state=int(sys.argv[4]), shuffle = True)
#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0

best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
                                          
    dataloader_train, dataloader_validation = generate_dataloaders(X_train, X_test, y_train, y_test) 

    #Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)           
    epochs = 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
    model = model.to(device)
    

    training_process(model, model_name, epochs)

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "transformers", model_name, percentage_text)

#Loading Best model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model.to(device)
model.load_state_dict(torch.load('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/finetuned_'+model_name+percentage_text+'_epoch_5.model', map_location=torch.device('cuda')))
generate_reports(model_name)

#================================ BERT-LARGE ====================================
#To avoid our CUDA's memory being full, we empty the cache first
gc.collect()
torch.cuda.empty_cache()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:"+str(no_gpu)

#BERT Pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model_name = "BERTL"

#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
                                          
    dataloader_train, dataloader_validation = generate_dataloaders(X_train, X_test, y_train, y_test) 

    #Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)           
    epochs = 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
    model = model.to(device)
    

    training_process(model,model_name,epochs)

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "transformers", model_name, percentage_text)

#Loading Best model
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model.to(device)
model.load_state_dict(torch.load('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/finetuned_'+model_name+percentage_text+'_epoch_5.model', map_location=torch.device('cuda')))
generate_reports(model_name)
#=============================== CT-BERT =======================================
#To avoid our CUDA's memory being full, we empty the cache first
gc.collect()
torch.cuda.empty_cache()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:"+str(no_gpu)

#BERT Pre-trained model
model = BertForSequenceClassification.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model_name = "CT-BERT"

#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
                                          
    dataloader_train, dataloader_validation = generate_dataloaders(X_train, X_test, y_train, y_test) 

    #Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)           
    epochs = 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
    model = model.to(device)
    

    training_process(model,model_name,epochs)

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "transformers", model_name, percentage_text)

#Loading Best model
model = BertForSequenceClassification.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model.to(device)
model.load_state_dict(torch.load('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/finetuned_'+model_name+percentage_text+'_epoch_5.model', map_location=torch.device('cuda')))
generate_reports(model_name)

#=============================== RoBERTa =======================================
#To avoid our CUDA's memory being full, we empty the cache first
gc.collect()
torch.cuda.empty_cache()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:"+str(no_gpu)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
#BERT Pre-trained model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model_name = "RoBERTa"

#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
                                          
    dataloader_train, dataloader_validation = generate_dataloaders(X_train, X_test, y_train, y_test) 

    #Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)           
    epochs = 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
    model = model.to(device)
    

    training_process(model,model_name,epochs)

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "transformers", model_name, percentage_text)

#Loading Best model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model.to(device)
model.load_state_dict(torch.load('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/finetuned_'+model_name+percentage_text+'_epoch_5.model', map_location=torch.device('cuda')))
generate_reports(model_name)

#========================= RoBERTa Large =============================
#To avoid our CUDA's memory being full, we empty the cache first
gc.collect()
torch.cuda.empty_cache()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:"+str(no_gpu)

#BERT Tokenizer and encoding Data
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)                                      
#BERT Pre-trained model
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model_name = "RoBERTa_Large"

#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
                                          
    dataloader_train, dataloader_validation = generate_dataloaders(X_train, X_test, y_train, y_test) 

    #Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)           
    epochs = 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
    model = model.to(device)
    

    training_process(model,model_name,epochs)

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "transformers", model_name, percentage_text)

#Loading Best model
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model.to(device)
model.load_state_dict(torch.load('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/finetuned_'+model_name+percentage_text+'_epoch_5.model', map_location=torch.device('cuda')))
generate_reports(model_name)

#====================== Support Vector Machines ==========================
#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    print("Evaluating and getting metrics from the SVC model..")
    model_name ="SVM"
    model_svc = make_pipeline(TfidfVectorizer(), SVC(class_weight='balanced', kernel='linear',probability=True))
    model_svc.fit(X_train, y_train)
    model_svc_labels = model_svc.predict(X_test)
    pickle.dump(model_svc, open('models/'+str(sys.argv[4])+'_randomstate/'+fold_names[current_fold]+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'wb'))

    val_f1 = f1_score(y_test, model_svc_labels, average='weighted')

    if best_f1_score < val_f1:
            best_f1_score = val_f1
            best_fold = fold_names[current_fold]
        
    current_fold = current_fold+1

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "sklearn", model_name, percentage_text)

#Storing the classification report
best_model_svc = pickle.load(open('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'rb'))
preds_flat = best_model_svc.predict(df_validation.tweet_text.values)
labels_flat = df_validation.label.values

#For the ensemble model
model_results_with[model_name] = preds_flat.tolist()

model_classification_report = classification_report(labels_flat.tolist(), preds_flat.tolist(), target_names=label_dict_inverse.values(), output_dict=True)
model_classification_report = pd.DataFrame(model_classification_report).transpose()
#Exporting the results and model
model_classification_report.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports/'+model_name+percentage_text+'_metrics.csv', index= True)
#Exporting the labels and predictions
predictions_columns = ["True","Predicted"]
df_predictions = pd.DataFrame(columns = predictions_columns)
df_predictions["True"] = labels_flat.tolist()
df_predictions["Predicted"] = preds_flat.tolist()
df_predictions.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions/'+model_name+percentage_text+'_predictions.csv',index=False)

#Exporting confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(labels_flat.tolist(), preds_flat.tolist(), normalize='all'),
annot=True, fmt='.2f', xticklabels=label_dict_inverse.values(), yticklabels=label_dict_inverse.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix/'+model_name+percentage_text+'_matrix.png')

#====================== Logistic Regression ==============================
#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    print("Evaluating and getting metrics from the Logistic Regression model model..")
    model_name ="LR"
    model_lr = make_pipeline(TfidfVectorizer(), LogisticRegression(solver='newton-cg', class_weight='balanced', n_jobs=-1))
    model_lr.fit(X_train, y_train)
    model_lr_labels = model_lr.predict(X_test)
    pickle.dump(model_lr, open('models/'+str(sys.argv[4])+'_randomstate/'+fold_names[current_fold]+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'wb'))

    val_f1 = f1_score(y_test, model_lr_labels, average='weighted')

    if best_f1_score < val_f1:
            best_f1_score = val_f1
            best_fold = fold_names[current_fold]
        
    current_fold = current_fold+1

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "sklearn", model_name, percentage_text)

#Storing the classification report
best_model_lr = pickle.load(open('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'rb'))
preds_flat = best_model_lr.predict(df_validation.tweet_text.values)
labels_flat = df_validation.label.values

#For the ensemble model
model_results_with[model_name] = preds_flat.tolist()

model_classification_report = classification_report(labels_flat.tolist(), preds_flat.tolist(), target_names=label_dict_inverse.values(), output_dict=True)
model_classification_report = pd.DataFrame(model_classification_report).transpose()
#Exporting the results and model
model_classification_report.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports/'+model_name+percentage_text+'_metrics.csv', index= True)
#Exporting the labels and predictions
predictions_columns = ["True","Predicted"]
df_predictions = pd.DataFrame(columns = predictions_columns)
df_predictions["True"] = labels_flat.tolist()
df_predictions["Predicted"] = preds_flat.tolist()
df_predictions.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions/'+model_name+percentage_text+'_predictions.csv',index=False)

#Exporting confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(labels_flat.tolist(), preds_flat.tolist(), normalize='all'),
annot=True, fmt='.2f', xticklabels=label_dict_inverse.values(), yticklabels=label_dict_inverse.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix/'+model_name+percentage_text+'_matrix.png')


#======================Random Forest =====================================
#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    print("Evaluating and getting metrics from the Logistic Random Forest model..")
    model_name ="RF"
    model_rf = make_pipeline(TfidfVectorizer(), RandomForestClassifier(class_weight='balanced', n_jobs=-1, max_depth=50))
    model_rf.fit(X_train, y_train)
    model_rf_labels = model_rf.predict(X_test)
    pickle.dump(model_rf, open('models/'+str(sys.argv[4])+'_randomstate/'+fold_names[current_fold]+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'wb'))

    val_f1 = f1_score(y_test, model_rf_labels, average='weighted')

    if best_f1_score < val_f1:
            best_f1_score = val_f1
            best_fold = fold_names[current_fold]
        
    current_fold = current_fold+1

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "sklearn", model_name, percentage_text)

#Storing the classification report
best_model_rf = pickle.load(open('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'rb'))
preds_flat = best_model_rf.predict(df_validation.tweet_text.values)
labels_flat = df_validation.label.values

#For the ensemble model
model_results_with[model_name] = preds_flat.tolist()

model_classification_report = classification_report(labels_flat.tolist(), preds_flat.tolist(), target_names=label_dict_inverse.values(), output_dict=True)
model_classification_report = pd.DataFrame(model_classification_report).transpose()
#Exporting the results and model
model_classification_report.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports/'+model_name+percentage_text+'_metrics.csv', index= True)
#Exporting the labels and predictions
predictions_columns = ["True","Predicted"]
df_predictions = pd.DataFrame(columns = predictions_columns)
df_predictions["True"] = labels_flat.tolist()
df_predictions["Predicted"] = preds_flat.tolist()
df_predictions.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions/'+model_name+percentage_text+'_predictions.csv',index=False)

#Exporting confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(labels_flat.tolist(), preds_flat.tolist(), normalize='all'),
annot=True, fmt='.2f', xticklabels=label_dict_inverse.values(), yticklabels=label_dict_inverse.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix/'+model_name+percentage_text+'_matrix.png')

#====================== Naive Bayes ======================================
#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    print("Evaluating and getting metrics from the Naive Bayes model model..")
    model_name ="NB"
    model_nb = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model_nb.fit(X_train, y_train)
    model_nb_labels = model_nb.predict(X_test)
    pickle.dump(model_nb, open('models/'+str(sys.argv[4])+'_randomstate/'+fold_names[current_fold]+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'wb'))

    val_f1 = f1_score(y_test, model_nb_labels, average='weighted')

    if best_f1_score < val_f1:
            best_f1_score = val_f1
            best_fold = fold_names[current_fold]
        
    current_fold = current_fold+1

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "sklearn", model_name, percentage_text)

#Storing the classification report
best_model_nb = pickle.load(open('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'rb'))
preds_flat = best_model_nb.predict(df_validation.tweet_text.values)
labels_flat = df_validation.label.values

#For the ensemble model
model_results_with[model_name] = preds_flat.tolist()

model_classification_report = classification_report(labels_flat.tolist(), preds_flat.tolist(), target_names=label_dict_inverse.values(), output_dict=True)
model_classification_report = pd.DataFrame(model_classification_report).transpose()
#Exporting the results and model
model_classification_report.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports/'+model_name+percentage_text+'_metrics.csv', index= True)
#Exporting the labels and predictions
predictions_columns = ["True","Predicted"]
df_predictions = pd.DataFrame(columns = predictions_columns)
df_predictions["True"] = labels_flat.tolist()
df_predictions["Predicted"] = preds_flat.tolist()
df_predictions.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions/'+model_name+percentage_text+'_predictions.csv',index=False)

#Exporting confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(labels_flat.tolist(), preds_flat.tolist(), normalize='all'),
annot=True, fmt='.2f', xticklabels=label_dict_inverse.values(), yticklabels=label_dict_inverse.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix/'+model_name+percentage_text+'_matrix.png')

#====================== Decision Tree ======================================
#This will determine which fold has the best F1-score and will be ised for the testing dataset
best_f1_score = 0
best_fold = fold_names[0]
current_fold = 0 #0 is the first one

for train_index, test_index in skf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    print("Evaluating and getting metrics from the Decision Tree model..")
    model_name ="DT"
    model_dt = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier(class_weight='balanced', max_depth=50))
    model_dt.fit(X_train, y_train)
    model_dt_labels = model_dt.predict(X_test)
    pickle.dump(model_dt, open('models/'+str(sys.argv[4])+'_randomstate/'+fold_names[current_fold]+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'wb'))

    val_f1 = f1_score(y_test, model_dt_labels, average='weighted')

    if best_f1_score < val_f1:
            best_f1_score = val_f1
            best_fold = fold_names[current_fold]
        
    current_fold = current_fold+1

#Keeping only the model from the best fold
keep_best_fold(fold_names,best_fold, "sklearn", model_name, percentage_text)

#Storing the classification report
best_model_dt = pickle.load(open('models/'+str(sys.argv[4])+'_randomstate/'+best_fold+'_fold/'+str(experiment_name)+'/trained_'+model_name+percentage_text+'.pickle', 'rb'))
preds_flat = best_model_dt.predict(df_validation.tweet_text.values)
labels_flat = df_validation.label.values

#For the ensemble model
model_results_with[model_name] = preds_flat.tolist()

model_classification_report = classification_report(labels_flat.tolist(), preds_flat.tolist(), target_names=label_dict_inverse.values(), output_dict=True)
model_classification_report = pd.DataFrame(model_classification_report).transpose()
#Exporting the results and model
model_classification_report.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports/'+model_name+percentage_text+'_metrics.csv', index= True)
#Exporting the labels and predictions
predictions_columns = ["True","Predicted"]
df_predictions = pd.DataFrame(columns = predictions_columns)
df_predictions["True"] = labels_flat.tolist()
df_predictions["Predicted"] = preds_flat.tolist()
df_predictions.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions/'+model_name+percentage_text+'_predictions.csv',index=False)

#Exporting confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(labels_flat.tolist(), preds_flat.tolist(), normalize='all'),
annot=True, fmt='.2f', xticklabels=label_dict_inverse.values(), yticklabels=label_dict_inverse.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix/'+model_name+percentage_text+'_matrix.png')


#======================= Exporting the final results from all the models (together) ==========================================
final_results= model_results_with.mode(axis=1)
final_results_int = [int(a) for a in final_results[0].tolist()] 
bert_model_classification_report = classification_report(df_validation.label.values, final_results_int, target_names=label_dict_inverse.values(), output_dict=True)
bert_model_classification_report = pd.DataFrame(bert_model_classification_report).transpose()
#Exporting the results and model
bert_model_classification_report.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/reports/ensemble'+percentage_text+'_metrics.csv', index= True)
#Exporting the labels and predictions
predictions_columns = ["True","Predicted"]
df_predictions = pd.DataFrame(columns = predictions_columns)
df_predictions["True"] = df_validation.label.values
df_predictions["Predicted"] = final_results_int
df_predictions.to_csv('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/predictions/ensemble'+percentage_text+'_predictions.csv',index=False)
#Exporting confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(df_validation.label.values, final_results_int, normalize='all'),
annot=True, fmt='.2f', xticklabels=label_dict_inverse.values(), yticklabels=label_dict_inverse.values())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('metrics/'+str(sys.argv[4])+'_randomstate/best_fold/'+str(experiment_name)+'/matrix/ensemble'+percentage_text+'_matrix.png')