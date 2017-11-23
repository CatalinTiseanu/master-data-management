"""
This module extras useful information form the nip data file
(1) firstname -> gender mapping
(2) set of possible medical suffixes
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import cPickle

nip_filename = "nip_data/npidata_20050523-20150809.csv"

def load_nip(filename):
  df = pd.read_csv(big_data_filename, usecols = ["NPI","Entity Type Code","Replacement NPI",
                                               "Provider Organization Name (Legal Business Name)",
                                               "Provider Last Name (Legal Name)",
                                               "Provider First Name",
                                               "Provider Middle Name",
                                               "Provider Credential Text",
                                               "Provider Gender Code"])

def compute_firstname_gender_dict(df):
  male_view = df[df['Provider Gender Code'] == 'M']["Provider First Name"].str.lower()
  female_view = df[df['Provider Gender Code'] == 'F']["Provider First Name"].str.lower()

  male_first_names = set(male_view)
  female_first_names = set(female_view)

  male_counter = Counter(male_view)
  female_counter = Counter(female_view)

  intersection_set = male_first_names & female_first_names

  name_gender_dict = {}

  for key in male_first_names - intersection_set:
    name_gender_dict[key] = 'M'
  for key in female_first_names - intersection_set:
    name_gender_dict[key] = 'F'

  for key in male_first_names & female_first_names:
    if max(male_counter[key], female_counter[key]) >= 10 * min(male_counter[key], female_counter[key]): 
      if male_counter[key] > female_counter[key]:
        name_gender_dict[key] = 'M'
      else:
        name_gender_dict[key] = 'F'
    else:
      name_gender_dict[key] = 'U'
    
    print name_gender_dict[key], key, male_counter[key], female_counter[key]
    
  name_cnt = Counter(df["Provider First Name"].str.lower())

  cPickle.dump(name_gender_dict, open("processed_data/first_names_by_gender.dict", "w"))
  cPickle.dump(name_cnt, open("processed_data/first_names_by_cnt.dict", "w"))

def compute_suffix_set(df)
  credentials_view = df[df['Provider Credential Text'].notnull()]['Provider Credential Text'].str.lower()
  total_list = []
  for el in credentials_view:
    tokens = el.replace(".", "").split(",")
    for t in tokens:
        total_list.append(t.strip())
  suffix_cnt = Counter(total_list)
  print suffix_cnt
  cPickle.dump(suffix_cnt, open("processsed_data/suffix_token_cnt.dict", "w"))

def process_nip():
  df = load_nip(nip_filename)
  compute_suffix_set(df)
  compute_firstname_gender_dict(df)
