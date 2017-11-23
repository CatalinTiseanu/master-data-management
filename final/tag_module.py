"""
Described in detail in the documentations
"""

import pandas as pd
import networkx as nx


from collections import defaultdict
import cPickle

class TagLinker:
    def __init__(self):
        self.tags_count = defaultdict(int)
        self.tags_set = set()
        self.tags_list = defaultdict(list)
        self.connected_set = set()
        self.G = None

    def create(self, tag_tokens):
        for index, tokens in tag_tokens.iteritems():
            for t in tokens:
                self.tags_set.add(t)
                self.tags_count[t] += 1
                self.tags_list[t].append(index)


    def get_list(self, t):
        if t in self.tags_set:
            return self.tags_list[t]
        return None

    def get_count(self, t):
        if t in self.tags_set:
            return self.tags_count[t]
        return None

    def shortest_path_length(self, t1, t2):
        if t1 in self.tags_list and t2 in self.tags_list:
            return nx.shortest_path_length(self.graph, t1, t2)
        else:
            return 100

    def save(self, filename):
        cPickle.dump(self.tags_list, filename)

    def load(self, filename):
        self.G = cPickle.load(open(filename))
        self.graph = nx.Graph()

        #edges = [(key.lower, val) for key in self.G for child in self.G[key]]

        self.tags_list = set([child.lower() for key in self.G for child in self.G[key]])
        self.child_of = set([(key.lower(), child.lower()) for key in self.G for child in self.G[key]])

        self.graph.add_edges_from(list(self.child_of))
        assert ("internal medicine", "cardiovascular disease") in self.child_of
        assert not ("internal medicine", "legal medicine") in self.child_of

    def match_tags(self, tag1, tag2):
        if tag1 == tag2:
            return 3

        if tag1.endswith(tag2) or tag2.endswith(tag1):
            return 2

        if tag1.startswith(tag2) or tag2.startswith(tag1):
            return 2

        if (tag1, tag2) in self.child_of or (tag2, tag1) in self.child_of:
            return 1

        return 0

    def process_raw_taxonomy(self, filename):
        lines = [line.rstrip('\n') for line in open(filename)]

        prev_level = 0

        final_lines = ["ROOT"]

        for line in lines:
            if line.startswith("#"):
                cur_level = 1
                tag = line[1:].strip()
            elif " - " not in line:
                cur_level = 2
                tag = line.split("[definition]")[0].strip()
            else:
                tokens = [el.strip() for el in line.split(" - ")]
                if len(tokens) == 1 or tokens[-1].split(" ")[0].endswith("00000X"):
                    tag = tokens[0]
                    cur_level = 3
                else:
                    tag = tokens[0]
                    cur_level = 4

            final_lines.append((" " * cur_level * 10) + " " + tag + "\n")

        open("taxonomy/processed_taxonomy", "w").writelines(final_lines)

    def get_indent(self, lines):
        return len(lines) - len(lines.lstrip())

    def process_tree_taxonomy(self, filename):
        lines = [line.rstrip('\n') for line in open(filename)]

        g = defaultdict(list)

        for i in range(1, len(lines)):
            for j in range(i - 1, -1, -1):
                if (self.get_indent(lines[j]) < self.get_indent(lines[i])):
                    g[lines[j].lstrip()].append(lines[i].lstrip())
                    break


        cPickle.dump(g, open("taxonomy/taxonomy_graph.in", "w"))


def process_raw_taxonomy(filename):
    TL = TagLinker()
    TL.process_raw_taxonomy(filename)
    TL.process_tree_taxonomy("taxonomy/processed_taxonomy")
