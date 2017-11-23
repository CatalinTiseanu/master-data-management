"""
This module in charge of taking a name field from the initial csv and:
(1) Decide wheter it is a company or a person
(2) If it's a person, also compute first_name, gender, prefix title (such as Dr) and suffix titles (such as M.D)
"""

from utils import *

from collections import defaultdict, Counter
from termcolor import colored

import cPickle

class NameCategorizer:
    # takes care of misspellings
    base_titles_table = [["dr", "rd"],
                   ["mr", "rm"],
                   ["miss", "ms", "sm", 'mssi', 'smsi', 'imss', 'issm', 'msis', 'sism', 'ssim', 'smis', 'isms', 'ssmi', 'sims'],
                   ["mrs", 'msr', 'rms', 'rsm', 'smr', 'srm'],
                   ["prof", 'oprf', 'rofp', 'pfro', 'orpf', 'forp', 'orfp', 'frop', 'fopr', 'opfr', 'rfop', 'rfpo', 'fpor', 'pfor', 'rpfo', 'porf', 'ofpr', 'pofr', 'prfo', 'ofrp', 'ropf', 'rpof', 'fpro', 'frpo']]

    name_suffixes = ["jr", "sr", "rs", "rj", "i", "ii", "iii", "iv", "v", "vi", "vii", "vii", "ix", "x"]

    # company tokens ended up not being used in the end
    company_tokens = set(['of', 'the', 'and', 'for',
                          'inc', 'incorporated', 'co', 'company', 'ltd', 'limited', 'associates', 'asc', 'practice', 'partnership', 'group',
                          'center', 'centre', 'corporation', 'corp', 'clinic', 'clinics', 'gynecology',
                          'home', 'health', 'healthcare', 'medical', 'memorial',
                          'care', 'group', 'university', 'college', 'family', 'practice', 'llc', 'llp',
                          'store', 'stores', 'associates', 'ltd', 'state', 'county', 'lp',
                          'corporation','services', 'service', 'clinic', 'drugstore', 'drugs', 'drug', 'therapy',
                          'village', 'psychologists', 'physicians', 'consultants', 'dept', 'physical', 'regional',
                          'radiology', 'oncology', 'advanced', 'imaging', 'unit', 'specialists', 'vascular', 'surgery', 'cardiovascular',
                          'cardiology', 'renal', 'family', 'nutrition', 'pulmonary',
                          'company', 'treatment', 'dialysis', 'home', 'centers', 'therapies', 'emergency', 'ambulance', 'rehab',
                          'prostethics', 'general', 'institute', 'vascular', 'diagnostic', 'rehabilitation', 'womens',
                          'pediatric', 'diseases', 'disease', 'infectious', 'resources', 'department', 'neurology', 'diagnostics',
                          'division', 'mri', 'skill', 'anesthesia', 'professional', 'centers', 'pathology',
                          'vision', 'partners', 'building', 'community', 'homecare', 'hospice', 'system', 'mobile', 'transport', 'prescription',
                          'physician', 'eyesight', 'network', 'provider', 'evaluation', 'surgeons', 'association', 'orthopaedics'
                          'vamc'])    

    # not used
    buildings = set(["hospital", "hospitals", "pharmacy", "college", "university"])

    titles = set()
    medical_titles = set()
    base_titles_form = {}
    medical_suffix_form = {}               

    """ Load firstname -> gender mapping""" 
    def process_names(self):
        d = cPickle.load(open("first_names_by_gender.dict"))
        firstnames = set(d.keys())

        self.gender = dict(map(lambda (k,v): (k, {'M':0, 'F':1, 'U':2}[v]), d.iteritems()))

        return firstnames

    def __init__(self):
        print "Initialiazing NameCategorizer"

        self.firstnames = self.process_names()

        self.medical_suffix = cPickle.load(open("suffix_token_cnt.dict"))
        self.medical_suffix = filter(lambda el : self.medical_suffix[el] > 50, self.medical_suffix)

        for term in self.medical_suffix:
            self.medical_titles.add(term)
            self.medical_suffix_form[term] = term

        self.medical_titles.add('dm')
        self.medical_suffix_form['dm'] = 'md'

        self.medical_titles.add('pdh')
        self.medical_suffix_form['pdh'] = 'phd'

        self.medical_titles -= set(["llc", "co", "inc", "icn", "asc", "ltd", "lp", "none"])

        self.sorted_buildings = set()
        for t in self.buildings:
            self.sorted_buildings.add("".join(sorted(t)))

        for title_row in self.base_titles_table:
            for i in range(0, len(title_row)):
                self.titles.add(title_row[i])
                self.base_titles_form[title_row[i]] = title_row[0]

        

        print "Titles: {}".format(self.titles)
        print "Medical titles: {}".format(self.medical_titles)

    # category 0 means person name, 1 means organization name
    def construct_human_form(self, name, hyphenated_name, index):
        prefix = set()
        suffix = set()
        middle_initial = ""

        last_name = ""
        name_suffixes = ""

        last_name_2 = ""

        if len(hyphenated_name.split(",")) > 0 and len(hyphenated_name.split(",")[0].split()) > 1:
            if "-" in hyphenated_name.split(",")[0].split()[-1]:
                last_name_2 = compute_min_form(hyphenated_name.split(",")[0].split()[-1].split("-")[0].strip())

        if not name:
            return (0, "", prefix, suffix, "", "", 2, last_name_2)

        old_name_parts = name.split(",")
        first_part = old_name_parts[0].split()
        name_tokens = []

        if first_part[0] in self.titles:
            prefix.add(self.base_titles_form[first_part[0]])

        for t in first_part:
            if t in self.name_suffixes:
                name_suffixes += t

        for sep in old_name_parts[1:]:
            tokens = sep.split()
            for t in tokens:
                if t in self.medical_titles:
                    suffix.add(self.medical_suffix_form[t])
                else:
                    suffix.add(t)

        real_name_tokens = []
        for t in first_part:
            if len(t) > 1:
                if t not in self.titles and t not in self.name_suffixes:
                    if t not in self.medical_titles:
                        name_tokens.append(compute_min_form(t))
                        real_name_tokens.append(t)
                    else:
                        suffix.add(self.medical_suffix_form[t])
            else:
                # middle initial
                middle_initial += t

        if len(name_tokens) > 0:
            last_name = name_tokens[-1] + name_suffixes
        else:
            print colored("No initial first_name, recursing: {} > {} - {}".format(index, name, " ".join(prefix) + ",".join(old_name_parts[1:])), "red")
            new_name = " ".join(prefix) + ",".join(old_name_parts[1:])

            if new_name != name:
                return self.construct_human_form(" ".join(prefix) + ",".join(old_name_parts[1:]), hyphenated_name, index)
            else:
                return (0, "", prefix, suffix, "", "", 2, last_name_2)

        name = " ".join(name_tokens)

        first_name = ""
        if len(name) > 0 and len(real_name_tokens[0]) > 1:
            first_name = real_name_tokens[0]

        gender = 2

        if "mister" in prefix:
            gender = 0
        elif "miss" in prefix or "ms" in prefix or "mrs" in prefix:
            gender = 1

        if gender == 2 and first_name in self.firstnames:
            gender = self.gender[first_name]

        return (0, name, prefix, suffix, middle_initial, last_name, gender, last_name_2)

    # does the suffix (i.e, , M.D., D.O., etc) contain a dot 
    def has_dot(self, real_name):
        fname_tokens = real_name.split(",")
        if len(fname_tokens) > 1 and any("." in t for t in fname_tokens[1:]):
            return True
        return False

    def categorize_name(self, name, tag_tokens, index):
        name = name.lower()

        if len(name) < 1:
            print colored("Empty name: {}".format(index), "red")
            return (1, name, set(), set(), "", "", 2, "")

        initial_char = name[0]
        
        name = name.strip()

        # remove punctuation marks
        name = name.replace(".", "")
        hyphenated_name = name
        name = name.replace("-", " ")
        
        name = name.strip()

        old_name = name

        if not name:
            print colored("Empty name: {}".format(index), "red")
            return (1, " ".join([compute_min_form(t) for t in name.split()]), set(), set(), "", "", 2, "")

        if initial_char == ' ':
            return self.construct_human_form(old_name, hyphenated_name, index)
        else:
            if len(name.split()) < 1:
                print colored("Error for : {} - {}".format(name, index), "red")
            first_token = name.split()[0]
            if first_token in self.titles:
                return self.construct_human_form(old_name, hyphenated_name, index)
            else:
                name = name.replace (",", "")
                return (1, " ".join([compute_min_form(t) for t in name.split()]), set(), set(), "", "", 2, "")
