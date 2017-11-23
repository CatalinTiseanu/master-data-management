import pandas as pd
import networkx as nx
import numpy as np
import os
import xgboost as xgb
from collections import defaultdict

def contains_digits(s):
    return any(char.isdigit() for char in s)


def compute_min_form(name):
    return "".join(sorted(name))


def compute_token_min_form(name_tokens):
    return "".join([compute_min_form(t) for t in name_tokens])


def compute_df_csv(df, real_preds):
    final_csv = df[['pair_1', 'pair_2']]
    final_csv['pred'] = list(real_preds)
    return final_csv


def write_df_csv(df, filename = "submission.csv"):
    print "Writing {}".format(filename)

    df.to_csv("final.csv", header = False, index = False)

    print "Fast CSV cpp"
    os.system("./fast_csv < {} > {}".format("final.csv", filename))