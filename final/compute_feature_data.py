import numpy as np
import pandas as pd
import jellyfish as jf

from termcolor import colored
import sys
import gc

from collections import defaultdict, Counter
from ast import literal_eval

from math import *

from multiprocessing import Process, Queue, cpu_count, Pool

import itertools as it

import random as rnd
import time
import itertools as it
import cPickle

import copy

from os import system

from utils import *

from ml_module import train_regression
from name_recognizer_module import NameCategorizer

import threading
from tag_module import TagLinker, process_raw_taxonomy


given_data = "given_data/"

train_X_file = given_data + "training_data.csv"
train_y_file = given_data + "training_ground_truth.csv"
test_X_file = given_data + "testing_data.csv"

def load_truth_set():
    print "Loading truth data - output dict + set"
    train_y = pd.read_csv(train_y_file)

    id1 = train_y.id1
    id2 = train_y.id2

    truth_dict = {}
    truth_set = set()

    for el in id1:
        truth_dict[int(el)] = set()

    for i in range(len(id1)):
        truth_dict[int(id1[i])].add(int(id2[i]))
        truth_set.add((int(id1[i]), int(id2[i])))

    print "Loaded {} pairs".format(len(id1))

    return truth_dict, truth_set

def column_features():
    return ["name_match_jw", "street_match_jw", "address_match_jw",
            #"name_match_l", "street_match_l", "address_match_l",
            "last_name_match", "gender_match", "first_name_match", "match_name_jaccard", "min_name_tokens", "max_name_tokens",
            "prefix_match", "suffix_match", "mi_match",
            "match_state_code", "match_po_code", "match_city",
            "category", "same_category", "same_dot",
            "match_tags", "mean_match_tags", "tags_best_dist",
            "pair_1", "pair_2", "label"]

def category_pair(c1, c2):
    if c1 == c2:
        return c1
    
    rc1 = min (c1, c2)
    rc2 = max (c1, c2)

    if rc1 == 0:
        return 3 + rc2 - 1
    if rc1 == 1:
        return 5


# given two rows, compute the similartity features between them
def compute_features(row1, row2, pair, label, TL):
    #if index % 1000 == 0:
    #    print index

    name_match_jw = jf.jaro_winkler(row1.norm_name, row2.norm_name)
    street_match_jw = jf.jaro_winkler(row1.street_name, row2.street_name)
    address_match_jw = jf.jaro_winkler(row1.address, row2.address)

    same_category = row1.category_name == row2.category_name
    
    is_person = row1.category_name == 0 and row2.category_name == 0

    match_name_jaccard = 0

    name_set_1 = set(row1.norm_name.split())
    name_set_2 = set(row2.norm_name.split())

    same_dot = 1

    if is_person:
        nid1 = row1.name_id.split(",")
        nid2 = row2.name_id.split(",")
        if len(nid1) > 1 and len(nid2) > 1 and any("." in t for t in nid1[1:]) != any("." in t for t in nid2[1:]):
            same_dot = 0

    # rarity_score = 0
    # if len(name_set_1) and len(name_set_2):
    #     intersect_set = name_set_1 & name_set_2
    #     match_name_jaccard = 1.0 * len(intersect_set) / len(name_set_1 | name_set_2)

    #     for t in intersect_set:
    #         if t in rarity_counter:
    #             cur_score = sqrt(1.0 / rarity_counter[t])
    #             if cur_score > rarity_score:
    #                 rarity_score = cur_score

    min_name_tokens = min(len(name_set_1), len(name_set_2))
    max_name_tokens = max(len(name_set_1), len(name_set_2))

    # state features
    match_state_code = False
    if row1.state_code and row2.state_code and row1.state_code == row2.state_code:
        match_state_code = True
    match_po_code = 2.0
    if row1.po_code > 0 and row2.po_code > 0:
        match_po_code = 1.0 * abs(row1.po_code - row2.po_code) / max(row1.po_code, row2.po_code)

    match_city = False
    if row1.city_name and row2.city_name and compute_token_min_form(row1.city_name.split()) == compute_token_min_form(row2.city_name.split()):
        match_city = True

    prefix_match = 1.0
    if len(row1.prefix | row2.prefix):
        prefix_match = 1.0 * len(row1.prefix & row2.prefix) / len(row1.prefix | row2.prefix)
    
    suffix_match = 1.0
    if len(row1.suffix | row2.suffix):
        suffix_match = 1.0 * len(row1.suffix & row2.suffix) / len(row1.suffix | row2.suffix)

    last_name_match = -1
    if (row1.last_name or row2.last_name):
        if row1.last_name == row2.last_name:
            last_name_match = 1
        elif row1.last_name_2 != "" and row1.last_name_2 == row2.last_name:
            last_name_match = 1
        elif row2.last_name_2 != "" and row1.last_name == row2.last_name_2:
            last_name_match = 1
        elif row1.last_name_2 != "" and row1.last_name_2 == row2.last_name_2:
            last_name_match = 1
        else:
            last_name_match = 0

    gender_match = 4

    if row1.gender == row2.gender:
        gender_match = row1.gender
    elif row1.gender <= 1 and row2.gender <= 1:
        gender_match = 3

    first_name_match = -1

    # addendum
    if is_person:
        first_name_match = 0

    if is_person and len(row1.norm_name.split()) and len(row2.norm_name.split()):
        fname1 = row1.norm_name.split()[0]
        fname2 = row2.norm_name.split()[0]

        if (fname1 == fname2) or (len(fname2) == 1 and fname2 in fname1) or (len(fname1) == 1 and fname1 in fname2):
            first_name_match = 1

        if row1.gender != row2.gender and max(row1.gender, row2.gender) == 2 and (row1.norm_name.split()[0] == row2.norm_name.split()[0]):
            gender_match = min(row1.gender, row2.gender)


    mi_match = 2
    if row1.middle and row2.middle:
        if row1.middle in row2.middle or row2.middle in row1.middle:
            mi_match = 3
        else:
            mi_match = 0
    elif row1.middle or row2.middle:
        mi_match = 1

    mean_match_tags = 0.0
    match_tags = 0

    tags_best_dist = 100

    if row1.tag_tokens == row2.tag_tokens:
        match_tags = 4
        mean_match_tags = 4
    else:
        for t1 in row1.tag_tokens:
            for t2 in row2.tag_tokens:
                cur = TL.match_tags(t1, t2)
                match_tags = max(cur, match_tags)
                mean_match_tags += cur

                cur_dist = TL.shortest_path_length(t1, t2)
                if cur_dist < tags_best_dist:
                    tags_best_dist = cur_dist

        mean_match_tags /= (len(row1.tag_tokens) * len(row2.tag_tokens))

    return (name_match_jw, street_match_jw, address_match_jw,
            #name_match_l, street_match_l, address_match_l,
            last_name_match, gender_match, first_name_match, match_name_jaccard, min_name_tokens, max_name_tokens,
            prefix_match, suffix_match, mi_match,
            match_state_code, match_po_code, match_city,
            category_pair(row1.category_name, row2.category_name), same_category, same_dot,
            match_tags, mean_match_tags, tags_best_dist,
            pair[0], pair[1], label)

def bulk_process(df, positive_negative, name):
    TL = TagLinker()
    TL.load("taxonomy/taxonomy_graph.in")

    print "{} - computing feature for {} pairs".format(name, len(positive_negative))
    nr_proc = 0
    data = []
    for index, entry in enumerate(positive_negative):
        pair, label = entry
        
        features = compute_features(df.loc[int(pair[0])], df.loc[int(pair[1])], pair, label, TL)
        
        if nr_proc and nr_proc % 1000 == 0:
            print "{} / {} {}".format(nr_proc, len(positive_negative), features)
            print df.loc[int(pair[0])].tag_tokens, df.loc[int(pair[1])].tag_tokens, features[-3]

        data.append(features)
        nr_proc += 1
    
    print "Job {} finished, writing to: {}".format(name, "bulk_process_{}_tmp".format(name))
    cPickle.dump(data, open("bulk_process_{}_tmp".format(name), "wb"))
    del data
    time.sleep(1)

def compute_ml_data(filename, mode, previous = None):
    print "Compute regression data: {}, {}, {}".format(filename, mode, previous)
    df = pd.read_pickle(filename + "_post")
    print "Loaded df {}".format(df.shape)

    # ml_data = cPickle.load(open("ml_dataset"))
    #truth_dict, truth_set = load_truth_set()
    #positive_negative = cPickle.load(open("positive_negative.out"))

    if mode == "train":
        truth_dict, truth_set = load_truth_set()
    
        positive_set = cPickle.load(open("blocking_positive_set_{}".format(mode)))
        
        # Add all available training sets !
        positive_set |= truth_set

        positive_list = [((int(p[0]), int(p[1])), 1) for p in positive_set]

        negative_set = cPickle.load(open("blocking_negative_set_{}".format(mode)))
        negative_list = [((int(p[0]), int(p[1])), 0) for p in negative_set]
        
        new_negatives = min(len(negative_list), 20 * len(positive_list))

        print "Undersampling negatives: {} -> {}".format(len(negative_list), new_negatives)
        negative_list = rnd.sample(negative_list, new_negatives)

        assert not (negative_set & positive_set)

        print "Train: {} {}".format(len(positive_list), len(negative_list))

        positive_negative = positive_list + negative_list

        del positive_set
        del negative_set
        del positive_list
        del negative_list
        del truth_set
        del truth_dict
    else:
        total_set = cPickle.load(open("blocking_total_set_test"))

        if previous:
            prev_set = cPickle.load(open(previous[0]))
            print "Previous blocking_set defined : {}".format(prev_set)
            total_set = total_set - prev_set

        total_list = [((int(p[0]), int(p[1])), 2) for p in total_set]
        positive_negative = total_list

        print "Test: {}".format(len(total_list))

    #total_token_list = [t for el in df.norm_name for t in el.split() if len(t) >= 4]
    #rarity_counter = Counter(total_token_list)

    print "Garbage collection: {}".format(gc.collect())

    rnd.shuffle(positive_negative)

    #print positive_negative
    #n_cores = cpu_count()
    n_cores = 3
    n_list = len(positive_negative)
    n_block = n_list / n_cores + 1

    jobs = []
    data = []
    
    for i in range(n_cores):
        start_id = i * n_block
        end_id = min (n_list, start_id + n_block)
        jobs.append(Process(target = bulk_process, args = (df, positive_negative[start_id:end_id], "job_{}".format(i))))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()    
    
    print "Sleeping for 2 seconds"
    time.sleep(2)    

    for i in range(n_cores):
        print "Processing file {}".format(i)
        tmp_filename = "bulk_process_{}_tmp".format("job_{}".format(i))
        data_tmp = cPickle.load(open(tmp_filename, 'rb'))
        system("rm -f {}".format(tmp_filename))
        data.extend(data_tmp)

    print "data length: {}".format(len(data))

    ml_df = pd.DataFrame(data, columns = column_features())
    if previous:
        print "Previous df defined : {}".format(previous[1])
        prev_df = pd.read_pickle(previous[1])
        ml_df = pd.concat([ml_df, prev_df])


    #print "Computing feature for {} pairs".format(len(positive_negative))

    #data = Parallel(n_jobs=10)(delayed(compute_features)(df.loc[int(pair[0])], df.loc[int(pair[1])], index, pair, label) for index, (pair, label) in enumerate(positive_negative))

    ml_df.to_pickle("ml_{}_set".format(mode))
    # combine into data set

def process_data(filename):
    NC = NameCategorizer()

    data = pd.read_csv(filename)
    data = data.set_index('id')

    N = len(data.index)
    print ("{} records in {}".format(N, filename))
    
    norm_name = []
    category_name = []

    street_name = []
    tag_tokens = []

    #name_dict = defaultdict(list)
    #street_dict = defaultdict(list)

    state_name = []
    state_code = []
    po_code = []
    city_name_v = []
    
    start_time = time.time()
    nr_proc = 0

    #soundex_dict = defaultdict(list)
    #soundex_name = []

    middle_initial_v = []
    prefix_v = []
    suffix_v = []

    last_name_v = []
    last_name_2_v = []
    gender_v = []

    min_name = []
    category_2_counter = 0

    for index, row in data.iterrows():
        nr_proc += 1

        # Tags
        row.tags = unicode(row.tags.decode("utf-8"))
        tag = row.tags.lower().replace(r"""','""", "<#>")
        tag = tag[1:-1]
        tokens = set(tag.split("<#>"))

        tag_tokens.append(tokens)

        assert type(tokens) == set

        # Name
        if type(row.name_id) != str:
            row.name_id = ""

        row.name_id = unicode(row.name_id.decode("utf-8"))
        category, name, prefix, suffix, middle_initial, last_name, gender, last_name_2 = NC.categorize_name(row.name_id, tokens, index)
     
        name = unicode(name)

        assert type(prefix) == set
        assert type(suffix) == set

        #soundex_enc = jf.soundex(name)
        #soundex_name.append(soundex_enc)
        #soundex_dict[soundex_enc].append(index)

        min_name_str = "".join(sorted(name))
        min_name.append(min_name_str)

        norm_name.append(name)
        category_name.append(category)

        prefix_v.append(prefix)
        suffix_v.append(suffix)
        middle_initial_v.append(middle_initial)

        last_name_v.append(last_name)
        last_name_2_v.append(compute_token_min_form(last_name_2))
        gender_v.append(gender)
        #name_dict[name].append(index)

        # Address
        row.address = unicode(row.address.decode("utf-8"))
        address = row.address.strip().lower()
        fields = map(lambda el : el.strip(), address.split(","))

        if len(fields) >= 3:
            street_name.append(fields[0])
            #street_dict[fields[0]].append(index)  
        else:
            street_name.append("")
            print "no street"            

        if len(fields) >= 2:
            state_name.append(fields[-2])
            #street_dict[fields[-2]].append(index) 
            state_term = fields[-2].replace ("-", " ")
            state_tokens = state_term.split()

            if state_tokens[0].isalpha() and state_tokens[1].isdigit():
                state_code.append(state_tokens[0])
                po_code.append(int(state_tokens[1]))
            else:
                #print "bad state_token:{}".format(state_tokens)
                state_code.append("")
                po_code.append(-1)
        else:
            state_name.append("")
            #print "no street"      

        if len(fields) >= 4:
            city_name_v.append(compute_token_min_form(fields[-3]))
        else:
            city_name_v.append("")

        if category == 2:
            category_2_counter += 1
        #    print name, tokens, category_2_counter

        #if last_name_2 != "":
        #    print colored(row.name_id + " | " + last_name_2 + " | " + city_name_v[-1], "green")

        if nr_proc and nr_proc % 1000 == 0:
            cur_time = time.time()
            print "Processed {} records ({} / sec)".format(nr_proc, int(1000 / (cur_time - start_time)))
            print row.name_id, name, "({})".format(gender), category, fields[0], tokens, index, min_name_str, prefix, suffix, middle_initial
            start_time = cur_time

    print "Category 2: {}".format(category_2_counter)

    data['norm_name'] = norm_name
    data['category_name'] = category_name
    data['street_name'] = street_name
    data['state_name'] = state_name
    data['city_name'] = city_name_v

    data['state_code'] = state_code
    data['po_code'] = po_code

    data['tag_tokens'] = tag_tokens
    data['min_name'] = min_name

    data['prefix'] = prefix_v
    data['suffix'] = suffix_v
    data['middle'] = middle_initial_v

    data['last_name'] = last_name_v
    data['last_name_2'] = last_name_2_v
    data['gender'] = gender_v
    #data['soundex_name'] = soundex_name

    data.to_pickle(filename + "_post")


def compute_block(df, column_name, truth_set):
    # Street
    column_dict = defaultdict(list)
    column_set = set()

    for index, key in df[column_name].iteritems():
        #if df.loc[index].category_name != 0:
        #    continue
        if not key: continue
        new_key = key.replace(" ", "")
        new_key = key.replace("'", "")
        new_key = new_key.replace(".", "")
        column_dict["".join(sorted(new_key))].append(index)

    print "Unique {}: {}".format(column_name, len(column_dict))

    considered = 0

    for key in column_dict:
        if len(column_dict[key]) <= 1:
            continue
        considered += 1
        if considered % 1000 == 0:
            print considered, key, len(column_dict[key])
        l = column_dict[key]
        for i in xrange(len(l)):
            for j in xrange(i):
                column_set.add((min(l[i], l[j]), max(l[i], l[j])))

    print "Block val {}: {} {}".format(column_name, len(column_set), len(column_set & truth_set))

    return column_set

def compute_block_min_name_zip(df, truth_set):
    # Street
    column_dict = defaultdict(list)
    column_set = set()

    for index, row in df.iterrows():
        key = row.min_name

        if not key: continue
        
        new_key = key.replace(" ", "")
        new_key = key.replace("'", "")
        new_key = new_key.replace(".", "")
        
        if row.category_name == 1:
            new_key += "$$" + row.state_code

        column_dict["".join(sorted(new_key))].append(index)

    print "Unique {}: {}".format("min_name", len(column_dict))

    considered = 0

    for key in column_dict:
        if len(column_dict[key]) <= 1:
            continue
        considered += 1
        if considered % 1000 == 0:
            print considered, key, len(column_dict[key])
        l = column_dict[key]
        for i in xrange(len(l)):
            for j in xrange(i):
                column_set.add((min(l[i], l[j]), max(l[i], l[j])))

    print "Block val {}: {} {}".format("min_name", len(column_set), len(column_set & truth_set))

    return column_set

def compute_zip_block(df, truth_set):
    column_name = 'state_name'
    column_dict = defaultdict(list)
    column_set = set()

    for index, key in df[column_name].iteritems():
        tokens = key.replace("-", " ")
        tokens = key.split()

        if tokens[0].isalpha() and tokens[1].isdigit():
            column_dict["".join(sorted(tokens[0] + tokens[1]))].append(index)
            #column_dict[tokens[0] + tokens[1]].append(index)
    print "Unique {}: {}".format(column_name, len(column_dict))

    considered = 0

    for key in column_dict:
        if len(column_dict[key]) <= 1:
            continue
        considered += 1
        if considered % 1000 == 0:
            print considered, key, len(column_dict[key])
        l = column_dict[key]
        for i in xrange(len(l)):
            for j in xrange(i):
                column_set.add((min(l[i], l[j]), max(l[i], l[j])))

    print "Block val {}: {} {}".format(column_name, len(column_set), len(column_set & truth_set))

    return column_set

def compute_term_block(df, column_name, truth_set):
    # Street
    column_dict = defaultdict(list)
    column_set = set()

    for index, key in df[column_name].iteritems():
        tokens = key.strip().split()
        for t in tokens:
            column_dict["".join(sorted(t))].append(index)
            #column_dict[t].append(index)

    sorted_list = []
    for key, value in column_dict.items():
        if len(column_dict[key]) > 1 and len(column_dict[key]) <= 20:
            sorted_list.append((len(column_dict[key]), key))

    sorted_list.sort()

    tmp_dict = {}
    for key in column_dict:
        if len(column_dict[key]) > 1 and len(column_dict[key]) <= 20:
            tmp_dict[key] = column_dict[key]
    column_dict = tmp_dict

    print sorted_list

    print "Unique {}: {}".format(column_name, len(column_dict))

    considered = 0

    for key in column_dict:
        if len(column_dict[key]) <= 1:
            continue
        considered += 1
        if considered % 1000 == 0:
            print considered, key, len(column_dict[key])
        l = column_dict[key]
        for i in xrange(len(l)):
            for j in xrange(i):
                column_set.add((min(l[i], l[j]), max(l[i], l[j])))

    print "Block val {}: {} {}".format(column_name, len(column_set), len(column_set & truth_set))

    return column_set


def compute_blocking(filename, mode):
    df = pd.read_pickle(filename)

    total_set = set()

    if mode == "train":
        truth_dict, truth_set = load_truth_set()

        min_name_set = compute_block(df, 'min_name', truth_set)
        total_set |= min_name_set

        street_name_set = compute_block(df, 'street_name', truth_set)
        total_set |= street_name_set

        positive_set = total_set & truth_set
        negative_set = total_set - positive_set
        
        positive_not_considered = truth_set - positive_set

        print "positive {} - negative {} - not considered {}".format(len(positive_set), len(negative_set), len(positive_not_considered))

        cPickle.dump(positive_set, open("blocking_positive_set_{}".format(mode), "w"))
        cPickle.dump(negative_set, open("blocking_negative_set_{}".format(mode), "w"))
    
        cPickle.dump(positive_not_considered, open("blocking_positive_set_not_{}".format(mode), "w"))
    else:
        truth_set = set()

        #min_name_set = compute_block (df, 'min_name', truth_set)
        min_name_set = compute_block_min_name_zip (df, truth_set)
        total_set |= min_name_set

        street_name_set = compute_block(df, 'street_name', truth_set) 
        total_set |= street_name_set

        print "total set: {}".format(len(total_set))

        cPickle.dump(total_set, open("blocking_total_set_{}".format(mode), "w"))

    print "Final: {} - {}".format(len(total_set), len(total_set & truth_set))

def compute_name_block(df, truth_set, positive_not_considered):
    # women:
    #   (1) first 2 names if not initials
    #   (2) first and last_name(2)
    # men:
    #   (1) first + last_name(2)
    #   (2) middle + last_name (2)
    #   (3) initials + last_name (2)

    final_set = set()

    women_set = defaultdict(list)
    men_set = defaultdict(list)
    min_name_dict = defaultdict(list)

    for index, row in df.iterrows():
        if row.category_name != 0:
            continue

        key = row.min_name
        if not key: continue
        
        new_key = key.replace(" ", "")
        new_key = key.replace("'", "")
        new_key = new_key.replace(".", "")
        
        min_name_dict["".join(sorted(new_key))].append(index)

        last_names = [compute_min_form(el) for el in [row.last_name, row.last_name_2] if el]
        first_names = []

        name_tokens = row.norm_name.split()

        initials = ""

        initial_name_tokens = [el.strip() for el in row.name_id.lower().split(",")[0].split()]
        initial_name_tokens = [el.replace(".", "") for el in initial_name_tokens]

        my_pref = ""
        if len(row.prefix):
            initial_name_tokens = initial_name_tokens[1:]

        initial_name_tokens = [el for el in initial_name_tokens if el]
        last_name_tokens = [el for el in initial_name_tokens if len(el) > 1]

        for t in initial_name_tokens:
            if t:
                my_pref += t[0]

        for t in name_tokens:
            if len(t) > 1 and t not in last_names:
                first_names.append(t)
                initials += t[0]
            #elif len(t) == 1


        #if index in set([377801, 141250]):
        #    print colored(row, 'green')
        #    print colored(str(first_names) + " | " + str(last_names) + " | " + str(initial_name_tokens), 'green')
        #    print colored(str(row.middle) + " | " + str(last_name_tokens) + " | " + compute_token_min_form(row.city_name.split()), 'green')

        # LAST_NAME, CITY
        #if len(last_names) >= 1:
        #    men_set[last_names[0] + "@" + compute_token_min_form(row.city_name.split())].append(index)
        #    women_set[last_names[0] + "@" + compute_token_min_form(row.city_name.split())].append(index)

        # BOTH LAST NAMES
        if len(last_names) >= 2:
            men_set[last_names[0] + "$$$" + last_names[1]].append(index)
            women_set[last_names[0] + "$$$" + last_names[1]].append(index)

        # IF MEN THEN (FIRST_NAME, LAST_NAME), (SECOND_NAME, LAST_NAME)
        if row.gender == 0 or row.gender == 2:
            if len(first_names):
                for last_name in last_names:
                    men_set[first_names[0] + "$" + last_name].append(index)
                    men_set[first_names[-1] + "$" + last_name].append(index)
        
        # IF WOMAN
        if row.gender == 1 or row.gender == 2:
            #if len(first_names) >= 2:
            #    women_set[first_names[0] + "$" + first_names[1]].append(index)            
            
            # FIRST NAME, LAST NAME
            if len (first_names) >= 1:
                for last_name in last_names:
                    women_set[first_names[0] + "$" + last_name].append(index)        
                    women_set[first_names[-1] + "$" + last_name].append(index)

            #if len(row.middle) and len(first_names) and row.state_code:
            #    if index in set([377801, 141250]):
            #        print colored(first_names[0] + "#" + row.middle + "#" + compute_token_min_form(row.city_name.split()), 'red')
            #    women_set[first_names[0] + "$" + row.middle + "$" + compute_token_min_form(row.state_code.split())].append(index)
            #    women_set[first_names[0] + "#" + row.middle + "#" + compute_token_min_form(row.city_name.split())].append(index)
                #print colored(row.name_id + " | " + first_names[0] + "$" + row.middle + "$" + compute_token_min_form(row.state_name.split()), 'green')
            #elif len(first_names) >= 2 and len(initial_name_tokens) >= 2 and row.state_code:
            #    women_set[first_names[0] + "$" + initial_name_tokens[1][0] + "$" + compute_token_min_form(row.state_code.split())].append(index)
            #elif len(first_names) >= 2 and row.city_name:
            #    women_set[first_names[0] + "#" + compute_token_min_form(row.city_name.split())]

            #if len(first_names) >= 1 and len(last_name_tokens) >= 2:
                #if index in set([377801, 141250]):
                #    print colored(first_names[0] + "#" + last_name_tokens[-1][0] + "#" + compute_token_min_form(row.city_name.split()), 'red')
                #women_set[first_names[0] + "#" + last_name_tokens[-1][0] + "#" + compute_token_min_form(row.city_name.split())].append(index)

            #if len(row.tag_tokens) == 1 and len(first_names) >= 1:
            #    women_set[first_names[0] + "&" + str(row.tag_tokens) + "&" + compute_token_min_form(row.state_code.split())].append(index)

        if row.gender != 1 and len(last_names) == 1:
            if len(my_pref) >= 2:
                #print colored(row.name_id + " | " + my_pref + " | " + str(last_names), "green")
                men_set[my_pref + "$$" + last_names[0]].append(index)
                women_set[my_pref + "$$" + last_names[0]].append(index)

    print "Unique men: {} / women: {}".format(len(men_set), len(women_set))

    considered = 0

    for key in men_set:
        if len(men_set[key]) <= 1:
            continue
        considered += 1
        if considered % 1000 == 0:
            print considered, key, len(men_set[key])
        l = men_set[key]
        for i in xrange(len(l)):
            for j in xrange(i):
                final_set.add((min(l[i], l[j]), max(l[i], l[j])))


    for key in women_set:
        if len(women_set[key]) <= 1:
            continue
        considered += 1
        if considered % 1000 == 0:
            print considered, key, len(women_set[key])
        l = women_set[key]
        for i in xrange(len(l)):
            for j in xrange(i):
                final_set.add((min(l[i], l[j]), max(l[i], l[j])))

    #for key in min_name_dict:
    #    if len(min_name_dict[key]) <= 1:
    #        continue
    #    considered += 1
        #if considered % 1000 == 0:
        #    print considered, key, len(women_set[key])
    #    l = min_name_dict[key]
    #    for i in xrange(len(l)):
    #        for j in xrange(i):
    #            final_set.add((min(l[i], l[j]), max(l[i], l[j])))

    print "Block val {} {} - {}".format(len(final_set), len(final_set & truth_set), len(final_set & positive_not_considered))

    #assert (134587, 237910) in final_set
    #assert (3051, 7339) in final_set
    # flog = open("not_seen", "w")
    # for p in truth_set:
    #     pair1 = df.loc[p[0]]
    #     pair2 = df.loc[p[1]]

    #     if (pair1.name, pair2.name) in final_set:
    #         continue

    #     if pair1.category_name == 0 and pair2.category_name == 0:
    #         # not considered
    #         flog.write(str(pair1) + "\n" + str(pair2) + "\n")
    # flog.close()

    return final_set


def train_pipeline():
    process_data(train_X_file)
    compute_blocking(train_X_file + "_post", "train")
    compute_ml_data(train_X_file, "train")
    train_regression("ml_train_set", train_X_file + "_post", "train");

def test_pipeline():
    process_data(test_X_file)
    compute_blocking(test_X_file + "_post", "test")
    compute_ml_data(test_X_file, "test")
    #train_regression("ml_test_set", test_X_file + "_post", "test");

def test_FRIL():
    df = pd.read_pickle(train_X_file + "_post")
    truth_dict, truth_set = load_truth_set()
    fril_csv = pd.read_csv("FRIL-v2.1.5/duplicates.csv")

    positive_set = cPickle.load(open("blocking_positive_set_{}".format("train")))
    min_name_set = compute_block(df, 'min_name', truth_set)
    #street_name_set = compute_block(df, 'street_name', truth_set)

    match_id = fril_csv["id"]
    fril_set = set()

    for i in range(len(match_id) / 2):
        match1 = int(match_id.iloc[2 * i])
        match2 = int(match_id.iloc[2 * i + 1])

        if match1 > match2:
            match2, match1 = match1, match2
        fril_set.add((match1, match2))

    valid_fril_set = fril_set & truth_set

    print "{} {} - {} {} {}".format(len(truth_set), len(fril_set), len(truth_set & fril_set), len(valid_fril_set | positive_set), len(positive_set))

def redo_feature_data():
    print "Processing initial data"
    process_data(train_X_file)
    process_data(test_X_file)
    
    print "Redoing blocking sets"
    compute_blocking(train_X_file + "_post", "train")
    compute_blocking(test_X_file + "_post", "test")

    print "Redoing regression data"
    compute_ml_data(train_X_file, "train")
    compute_ml_data(test_X_file, "test")

if __name__ == '__main__':
    print "Deduplicator v1.0"
    print "Num cores: {}".format(cpu_count())

    redo_feature_data()
