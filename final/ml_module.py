# coding: utf-8
# import section

import os

import time as time
import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle

import random as rnd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

from collections import defaultdict
from collections import Counter

from compute_feature_data import *
from utils import *

print "ml_module.py"

print "Compiling the helper C++ post processors"
os.system("g++ -o fast_csv fast_csv.cpp")
os.system("g++ -o concatenate_csv concatenate_csv.cpp")


# load initial data
print "Load initial data"
train_X_file = "given_data/training_data.csv"
train_y_file = "given_data/training_ground_truth.csv"
test_X_file = "given_data/testing_data.csv"

# load processed data
print "Load processed data"
test_data = pd.read_pickle(test_X_file + "_post")
train_data = pd.read_pickle(train_X_file + "_post")

# debug

train_counts = Counter(train_data['category_name'])
test_counts = Counter(test_data['category_name'])
print train_counts
print test_counts


# Load training and testing regression pair data
# ==============================================

train_df = pd.read_pickle('ml_train_set')
print "train_df # : {}".format(len(train_df))
test_df = pd.read_pickle('ml_test_set')
print "test df # : {}".format(len(test_df))
features = list(train_df.columns[:-3])
print "Features :{}".format(features)

print Counter(train_df.category)
print Counter(test_df.category)

def enrich_initial_dataset():
    def construct_new_samples_fname_po_tags(data):
        fname_po_tags = set()
        block_key = defaultdict(list)
        for index, row in data.iterrows():
            name_tokens = row.norm_name.split()
            #if len(name_tokens):
            if len(name_tokens) > 1 and len(row.tag_tokens):
                hash_key = name_tokens[0] + name_tokens[1] + row.state_code + str(row.tag_tokens)
                #hash_key = name_tokens[0] + row.state_code + str(row.tag_tokens)
                for el in block_key[hash_key]:
                    fname_po_tags.add((el, index))
                block_key[hash_key].append(index)
        
        print len(fname_po_tags)
        return fname_po_tags

      
    positive_set = cPickle.load(open("blocking_positive_set_train"))        
    negative_set = cPickle.load(open("blocking_negative_set_train"))
    positive_not_considered = cPickle.load(open("blocking_positive_set_not_train"))
    truth_set = positive_set | positive_not_considered        
    total_test_set = cPickle.load(open("blocking_total_set_test")) 
    print len(positive_set), len(negative_set), len(positive_not_considered)

    assert len(negative_set & positive_set) == 0
    assert len(negative_set & positive_not_considered) == 0

    new_sample_train = construct_new_samples_fname_po_tags(train_data[train_data.category_name == 0])
    print len(new_sample_train - (positive_set | negative_set | positive_not_considered)), len(new_sample_train & positive_not_considered)
    new_sample_test = construct_new_samples_fname_po_tags(test_data[test_data.category_name == 0])
    print len(new_sample_test - total_test_set)

    new_items_positive = [(p, 1) for p in new_sample_train & positive_not_considered]
    new_items_negative = [(p, 0) for p in new_sample_train - (positive_set | negative_set) - (new_sample_train & positive_not_considered)]
    print "new positive: {} new negative: {}".format(len(new_items_positive), len(new_items_negative))
    new_train_data = bulk_process (train_data, new_items_negative, "ipython_train")
    new_items_list = [(p, 2) for p in new_sample_test - total_test_set]
    new_test_data = bulk_process (test_data, new_items_list, "ipython_test")
    print "new_items: {}".format(len(new_sample_test - total_test_set))

    new_train_data = pd.DataFrame(cPickle.load(open("bulk_process_ipython_train_tmp")), columns =features + ["pair_1", "pair_2", "label"])
    new_test_data = pd.DataFrame(cPickle.load(open("bulk_process_ipython_test_tmp")), columns = features + ["pair_1", "pair_2", "label"])
    return new_train_data, new_test_data

new_train_data, new_test_data = enrich_initial_dataset()
print "Additional data - train: {} - test: {}".format(len(new_train_data), len(new_test_data))

train_df = pd.concat([train_df, new_train_data])
test_df = pd.concat([test_df, new_test_data])

print "Split data into train and test"
train, test = train_test_split(train_df, test_size = 0.2, random_state = 0)


# create training features and matrix for xgboost
features = [el for el in train.columns if el not in ["pair_1", "pair_2", "label"]]
print "Train ({}) Test ({}) Features: {}".format(len(train.index), len(test.index), features)
dtrain = xgb.DMatrix(train.as_matrix(features), train["label"])
dtest = xgb.DMatrix(test.as_matrix(features), test["label"])

dtrain_set = set(zip(train.pair_1, train.pair_2))
dtest_set = set(zip(test.pair_1, test.pair_2))


# Train model
# ===========

param = {'max_depth': 4, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 30
bst = xgb.train(param, dtrain, num_round, watchlist)


# predict labels

preds_prob = bst.predict(dtest)
preds  = np.array([round(el) for el in preds_prob])
labels = dtest.get_label()

print classification_report (labels, preds)
print "MSE error: {}".format(((preds_prob - labels) ** 2).sum())
print "Errors by name category: {}".format(Counter(test[preds != labels].category))
print "Errors by gender: {}".format(Counter(test[preds != labels].gender_match))
print "Errors by match_mi: {}".format(Counter(test[preds != labels].mi_match))

print "Confusion matrix all: {} / {}".format(len(test[test.category == 0]), len(test[test.category == 1]))
print confusion_matrix(labels, preds)

print "Person:  FN : {} FP : {}".format(len(test[(preds == 0) & (labels == 1) & (test.category == 0)]),
                                        len(test[(preds == 1) & (labels == 0) & (test.category == 0)]))
print "Company: FN : {} FP : {}".format(len(test[(preds == 0) & (labels == 1) & (test.category == 1)]),
                                        len(test[(preds == 1) & (labels == 0) & (test.category == 1)]))
# Test bst model on test_df
# ==========================

print "test on {} records".format(len(test_df.index))
dpred = xgb.DMatrix(test_df.as_matrix(features), test_df["label"])
#bst = xgb.Booster(model_file='xgb.model')
real_preds = bst.predict(dpred)
at_row = 0
for index, item in test_df.category.iteritems():
    if item == 3:
        real_preds[at_row] = 0
    at_row += 1
    
real_preds_int = np.array([el >= 0.5 for el in real_preds])

positive_pred = sum(1 for i in range(len(real_preds)) if int(real_preds[i] > 0.5))
negative_pred = len(real_preds) - positive_pred

print "Positive {} / Negative {}".format(positive_pred, negative_pred)

# write prediction to file

dtest_csv = compute_df_csv(test_df, real_preds)
write_df_csv(dtest_csv)


# load initial positive and negative test sets

train_truth_df = train_df[train_df.label == 1]


def construct_transitive_closure(df, pred_label, pred_prob):
    graph = defaultdict(list)

    pred_set = set()
    pred_dict = defaultdict(float)
    real_dict = defaultdict(float)

    pred_at = 0
    for index, row in df.iterrows():
        pred_dict[(row.pair_1, row.pair_2)] = pred_prob[pred_at]
        pred_set.add((row.pair_1, row.pair_2))
        if pred_label[pred_at]:
            #pred_set.add((test.pair_1, test_pair_2))
            graph[row.pair_1].append(row.pair_2)
        pred_at += 1

    for x in graph:
        graph[x].sort()
        
    added_set = set()
    new_set = set()
    
    new_edges = 0
    
    for x in graph:
        adj_x = graph[x]
        if len(adj_x) > 1:
            for j in range(len(adj_x)):
                for i in xrange(j):
                    if (adj_x[i], adj_x[j]) not in pred_set:
                        new_edges += 1
                        new_set.add((adj_x[i], adj_x[j]))
                    if (adj_x[i], adj_x[j]) in pred_set and pred_dict[(adj_x[i], adj_x[j])] <= 0.5:
                        #if pred_dict[(adj_x[i], adj_x[j])] > 0.1:
                        added_set.add((adj_x[i], adj_x[j]))
                        pred_dict[(adj_x[i], adj_x[j])] = min (pred_dict[(x, adj_x[i])], pred_dict[(x, adj_x[j])])
                        #print "transitive closure {} - {} / {} : {}".format(i, j, len(adj_x), pred_dict[(adj_x[i], adj_x[j])])

    print "New edges :{}".format(new_edges)
                    
    return added_set, new_set, pred_dict

print "Computing transitive closure"
added_set, new_set, pred_dict = construct_transitive_closure(test_df, real_preds_int, real_preds)
print "Changed edges: {} / new edges : {}".format(len(added_set), len(new_set))

transitive_new_test = [(p, 2) for p in new_set]
new_transitive_data = bulk_process (test_data, transitive_new_test, "ipython_transitive_test")

def test_on_df(df, features, bst):
    print "testing on {} records".format(len(df.index))
    dtmp = xgb.DMatrix(df.as_matrix(features), df["label"])
    real_preds = bst.predict(dtmp)

    positive_pred = sum(1 for i in range(len(real_preds)) if int(real_preds[i] > 0.5))
    negative_pred = len(real_preds) - positive_pred

    print "Positive {} Negative {}".format(positive_pred, negative_pred)
    
    df_csv = df[['pair_1', 'pair_2']]
    df_csv['pred'] = list(real_preds)
    
    return df_csv

new_transitive_data =  pd.DataFrame(cPickle.load(open("bulk_process_ipython_transitive_test_tmp")), columns = features + ["pair_1", "pair_2", "label"])

#print features
#rule1_csv = test_on_df(new_test_data, features, bst)
#print len(rule1_csv)
#rule1_csv.pred = rule1_csv.pred.map(lambda el : 0.5)

transitive_sure_csv = test_on_df(new_transitive_data, features, bst)
#rule1_set = set()
#for index, row in additional_csv.iterrows():
#    additional_csv
#print len(additional_csv)


#write_df_csv(pd.concat([dtest_csv, additional_csv]))
write_df_csv(transitive_sure_csv, "transitive_sure.csv")
#write_df_csv(rule1_csv, "rule1.csv")

# Build the final output csv

submission_id = "final_output.csv"
os.system("cat submission.csv transitive_sure.csv > {}".format(submission_id + "_tmp"))
os.system("./concatenate_csv < {} > {}".format(submission_id + "_tmp", submission_id))

