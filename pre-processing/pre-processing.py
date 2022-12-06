#Pre-processing stuff
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emoji

#Modules required for training BERT model
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from transformers import set_seed
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
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
#1st parameter: split percentage
#2nd parameter: Random seed (first validation used 17)
#3rd paramtere: random state (first validation used 1)
print('Transformers version: '+str(transformers.__version__))
print('Reading arguments... \n')
percentage_training = float(sys.argv[1])
seed_val = int(sys.argv[2])
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
set_seed(seed_val)

print('Reading datasets (Face Mask)... \n')
df_training_fm_part_1 = pd.read_csv("hydrated_labeled_tweets/df_face_masks_train.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_fm_part_2 = pd.read_csv("hydrated_labeled_tweets/df_face_masks_val.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_fm_part_3 = pd.read_csv("hydrated_labeled_tweets/df_face_masks_train_noisy.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
frames = [df_training_fm_part_1, df_training_fm_part_2, df_training_fm_part_3]
df_training_fm = pd.concat(frames)
print(df_training_fm)

df_test_fm = pd.read_csv("hydrated_labeled_tweets/df_face_masks_test.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)

df_training_fm = df_training_fm.astype(str)
df_test_fm = df_test_fm.astype(str)

df_training_fm['tweet_class'] = df_training_fm['tweet_class'].str.lower()
df_test_fm['tweet_class'] = df_test_fm['tweet_class'].str.lower()

print('Reading datasets (Fauci)... \n')
df_training_fa_part_1 = pd.read_csv("hydrated_labeled_tweets/df_fauci_train.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_fa_part_2 = pd.read_csv("hydrated_labeled_tweets/df_fauci_val.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_fa_part_3 = pd.read_csv("hydrated_labeled_tweets/df_fauci_train_noisy.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
frames = [df_training_fa_part_1, df_training_fa_part_2, df_training_fa_part_3]
df_training_fa = pd.concat(frames)
df_test_fa = pd.read_csv("hydrated_labeled_tweets/df_fauci_test.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)

df_training_fa = df_training_fa.astype(str)
df_test_fa = df_test_fa.astype(str)

df_training_fa['tweet_class'] = df_training_fa['tweet_class'].str.lower()
df_test_fa['tweet_class'] = df_test_fa['tweet_class'].str.lower()

print('Reading datasets (School Closures)... \n')
df_training_sc_part_1 = pd.read_csv("hydrated_labeled_tweets/df_school_closures_train.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_sc_part_2 = pd.read_csv("hydrated_labeled_tweets/df_school_closures_val.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_sc_part_3 = pd.read_csv("hydrated_labeled_tweets/df_school_closures_train_noisy.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
frames = [df_training_sc_part_1, df_training_sc_part_2, df_training_sc_part_3]
df_training_sc = pd.concat(frames)
df_test_sc = pd.read_csv("hydrated_labeled_tweets/df_school_closures_test.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)

df_training_sc = df_training_sc.astype(str)
df_test_sc = df_test_sc.astype(str)

df_training_sc['tweet_class'] = df_training_sc['tweet_class'].str.lower()
df_test_sc['tweet_class'] = df_test_sc['tweet_class'].str.lower()

print('Reading datasets (Stay at home orders)... \n')
df_training_sh_part_1 = pd.read_csv("hydrated_labeled_tweets/df_stay_at_home_orders_train.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_sh_part_2 = pd.read_csv("hydrated_labeled_tweets/df_stay_at_home_orders_val.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
df_training_sh_part_3 = pd.read_csv("hydrated_labeled_tweets/df_stay_at_home_orders_train_noisy.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)
frames = [df_training_sh_part_1, df_training_sh_part_2, df_training_sh_part_3]
df_training_sh = pd.concat(frames)
df_test_sh = pd.read_csv("hydrated_labeled_tweets/df_stay_at_home_orders_test.tsv", delimiter="\t", names=["tweet_text", "tweet_id","tweet_class"], index_col=None, header = 0)

df_training_sh = df_training_sh.astype(str)
df_test_sh = df_test_sh.astype(str)

df_training_sh['tweet_class'] = df_training_sh['tweet_class'].str.lower()
df_test_sh['tweet_class'] = df_test_sh['tweet_class'].str.lower()

#Decoding to utf-8
df_training_fm['tweet_text'] = df_training_fm['tweet_text'].str.strip("b\'\"")
df_test_fm['tweet_text'] = df_test_fm['tweet_text'].str.strip("b\'\"")

df_training_fa['tweet_text'] = df_training_fa['tweet_text'].str.strip("b\'\"") 
df_test_fa['tweet_text'] = df_test_fa['tweet_text'].str.strip("b\'\"")

df_training_sc['tweet_text'] = df_training_sc['tweet_text'].str.strip("b\'\"")
df_test_sc['tweet_text'] = df_test_sc['tweet_text'].str.strip("b\'\"")

df_training_sh['tweet_text'] = df_training_sh['tweet_text'].str.strip("b\'\"")
df_test_sh['tweet_text'] = df_test_sh['tweet_text'].str.strip("b\'\"")

#Removing empty columns and rows (both training and dataset)
df_training_fm = df_training_fm.dropna(how='all', axis=1)
df_training_fm = df_training_fm.dropna(how='all')
df_test_fm = df_test_fm.dropna(how='all', axis=1)
df_test_fm = df_test_fm.dropna(how='all')

df_training_fa = df_training_fa.dropna(how='all', axis=1)
df_training_fa = df_training_fa.dropna(how='all')
df_test_fa = df_test_fa.dropna(how='all', axis=1)
df_test_fa = df_test_fa.dropna(how='all')

df_training_sc = df_training_sc.dropna(how='all', axis=1)
df_training_sc = df_training_sc.dropna(how='all')
df_test_sc = df_test_sc.dropna(how='all', axis=1)
df_test_sc = df_test_sc.dropna(how='all')

df_training_sh = df_training_sh.dropna(how='all', axis=1)
df_training_sh = df_training_sh.dropna(how='all')
df_test_sh = df_test_sh.dropna(how='all', axis=1)
df_test_sh = df_test_sh.dropna(how='all')


#Training dataset will only contain a specific percentage of the original dataset
percentage_text = ""

if percentage_training < 1.0:
    #Stratify sampling
    df_training_fm = df_training_fm.groupby('tweet_class', group_keys=False).apply(lambda x: x.sample(frac=percentage_training, replace=False, random_state=int(sys.argv[3])))
    df_training_fa = df_training_fa.groupby('tweet_class', group_keys=False).apply(lambda x: x.sample(frac=percentage_training, replace=False, random_state=int(sys.argv[3])))
    df_training_sc = df_training_sc.groupby('tweet_class', group_keys=False).apply(lambda x: x.sample(frac=percentage_training, replace=False, random_state=int(sys.argv[3])))
    df_training_sh = df_training_sh.groupby('tweet_class', group_keys=False).apply(lambda x: x.sample(frac=percentage_training, replace=False, random_state=int(sys.argv[3])))
    percentage_text = '('+str(int(percentage_training*100))+'%)'

if percentage_training > 1.0:
    percentage_text = '(+'+str(int(percentage_training*100)-100)+'%)'

#Encoding the labels
possible_labels_fm = df_training_fm.tweet_class.unique()
possible_labels_fa = df_training_fa.tweet_class.unique()
possible_labels_sc = df_training_sc.tweet_class.unique()
possible_labels_sh = df_training_sh.tweet_class.unique()

label_dict_fm = {}
for index, possible_label in enumerate(possible_labels_fm):
    label_dict_fm[possible_label] = index

label_dict_fa = {}
for index, possible_label in enumerate(possible_labels_fa):
    label_dict_fa[possible_label] = index

label_dict_sc = {}
for index, possible_label in enumerate(possible_labels_sc):
    label_dict_sc[possible_label] = index

label_dict_sh = {}
for index, possible_label in enumerate(possible_labels_sh):
    label_dict_sh[possible_label] = index

#Inserting the labels with the encoded labels
df_training_fm['label'] = df_training_fm.tweet_class.replace(label_dict_fm)
df_test_fm['label'] = df_test_fm.tweet_class.replace(label_dict_fm)

df_training_fa['label'] = df_training_fa.tweet_class.replace(label_dict_fa)
df_test_fa['label'] = df_test_fa.tweet_class.replace(label_dict_fa)

df_training_sc['label'] = df_training_sc.tweet_class.replace(label_dict_sc)
df_test_sc['label'] = df_test_sc.tweet_class.replace(label_dict_sc)

df_training_sh['label'] = df_training_sh.tweet_class.replace(label_dict_sh)
df_test_sh['label'] = df_test_sh.tweet_class.replace(label_dict_sh)

#Making sure the labels are integers
df_training_fm['label'] = df_training_fm['label'].astype('int')
df_test_fm['label'] = df_test_fm['label'].astype('int')

df_training_fa['label'] = df_training_fa['label'].astype('int')
df_test_fa['label'] = df_test_fa['label'].astype('int')

df_training_sc['label'] = df_training_sc['label'].astype('int')
df_test_sc['label'] = df_test_sc['label'].astype('int')

df_training_sh['label'] = df_training_sh['label'].astype('int')
df_test_sh['label'] = df_test_sh['label'].astype('int')

#Example of dataset labeling
print('Example of how dataset was labeled: \n')
print(df_training_fm.groupby('label').first())
label_dict_inverse = {v: k for k, v in label_dict_fm.items()}
print('Values for each label \n')
print(label_dict_inverse)

def remove_leading_usernames(tweet):
    regex_str = '^[\s.]*@[A-Za-z0-9_]+\s+'
    original = tweet
    change = re.sub(regex_str, '', original)
    while original != change:
        original = change
        change = re.sub(regex_str, '', original)
    return change

def process_tweet(tweet):
    #Remove www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))\s+','',tweet)
    tweet = re.sub('\s+((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    #Remove RTs
    tweet = re.sub('^RT @[A-Za-z0-9_]+: ', '', tweet)
    # Incorrect apostraphe
    tweet = re.sub(r"’", "'", tweet)
    #Remove @username
    tweet = remove_leading_usernames(tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace ampersands
    tweet = re.sub(r' &amp; ', ' and ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    #Remove emoticons
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    tweet = emoticon_pattern.sub(r'', tweet)
    #Remove emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    #Remove additional emojis
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    #trim
    tweet = tweet.strip('\'"')

    return tweet.lower().strip()

print(df_training_fm.dtypes)

df_training_fm['tweet_text'] = df_training_fm['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_training_fm['tweet_text'] = df_training_fm['tweet_text'].apply(lambda x : process_tweet(x))
df_test_fm['tweet_text'] = df_test_fm['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_test_fm['tweet_text'] = df_test_fm['tweet_text'].apply(lambda x : process_tweet(x))

print('Training data size (Face Mask): '+str(len(df_training_fm)))
print(df_training_fm.head())
print('Evaluation data size (Face Mask): '+str(len(df_test_fm)))
print(df_test_fm.head())

df_training_fa['tweet_text'] = df_training_fa['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_training_fa['tweet_text'] = df_training_fa['tweet_text'].apply(lambda x : process_tweet(x))
df_test_fa['tweet_text'] = df_test_fa['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_test_fa['tweet_text'] = df_test_fa['tweet_text'].apply(lambda x : process_tweet(x))

print('Training data size (Fauci): '+str(len(df_training_fa)))
print(df_training_fa.head())
print('Evaluation data size (Fauci): '+str(len(df_test_fa)))
print(df_test_fa.head())

df_training_sc['tweet_text'] = df_training_sc['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_training_sc['tweet_text'] = df_training_sc['tweet_text'].apply(lambda x : process_tweet(x))
df_test_sc['tweet_text'] = df_test_sc['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_test_sc['tweet_text'] = df_test_sc['tweet_text'].apply(lambda x : process_tweet(x))

print('Training data size (School closures): '+str(len(df_training_sc)))
print(df_training_sc.head())
print('Evaluation data size (School closures): '+str(len(df_test_sc)))
print(df_test_sc.head())

df_training_sh['tweet_text'] = df_training_sh['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_training_sh['tweet_text'] = df_training_sh['tweet_text'].apply(lambda x : process_tweet(x))
df_test_sh['tweet_text'] = df_test_sh['tweet_text'].apply(lambda x : remove_leading_usernames(x))
df_test_sh['tweet_text'] = df_test_sh['tweet_text'].apply(lambda x : process_tweet(x))

print('Training data size (Stay at home orders): '+str(len(df_training_sh)))
print(df_training_sh.head())
print('Evaluation data size (Stay at home orders): '+str(len(df_test_sh)))
print(df_test_sh.head())

#Exporting pre-processed datasets
df_training_fm.to_csv('df_face_masks_train_preprocessed.tsv', sep="\t", encoding = 'utf-8')
df_test_fm.to_csv('df_face_masks_test_preprocessed.tsv', sep="\t", encoding = 'utf-8')

df_training_fa.to_csv('df_fauci_train_preprocessed.tsv', sep="\t", encoding = 'utf-8')
df_test_fa.to_csv('df_fauci_test_preprocessed.tsv', sep="\t", encoding = 'utf-8')

df_training_sc.to_csv('df_school_closures_train_preprocessed.tsv', sep="\t", encoding = 'utf-8')
df_test_sc.to_csv('df_school_closures_test_preprocessed.tsv', sep="\t", encoding = 'utf-8')

df_training_sh.to_csv('df_stay_at_home_orders_train_preprocessed.tsv', sep="\t", encoding = 'utf-8')
df_test_sh.to_csv('df_stay_at_home_orders_test_preprocessed.tsv', sep="\t", encoding = 'utf-8')