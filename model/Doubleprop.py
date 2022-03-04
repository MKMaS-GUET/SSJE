# coding: utf-8

# In[1]:

import os
from nltk.tokenize import sent_tokenize
from nltk.parse.stanford import StanfordDependencyParser

# path_to_jar = "./stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar"
# path_to_models_jar = "./stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar"
java_path = r"C:\Program Files\Java\jre1.8.0_221\bin\java.exe"
os.environ['JAVAHOME'] = java_path
os.environ['STANFORD_PARSER'] = 'F:\stanford_nlp\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'F:\stanford_nlp\stanford-parser-3.9.2-models.jar'
# dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
dependency_parser = StanfordDependencyParser()
from copy import deepcopy
import pickle

# # Global Variable
# In[2]:

MR = set()  ## MR sentiment set
MR.add('amod')
MR.add('nmod')
MR.add('pnmod')
MR.add('nsubj')
MR.add('s')
MR.add('dobj')
MR.add('iobj')
MR.add('desc')
MR.add('xcomp')  ## test
NN = set()  ## NN pos set
NN.add('NN')
NN.add('NNS')
NN.add('NNP')
JJ = set()  ## JJ pos set
JJ.add('JJ')
JJ.add('JJS')
JJ.add('JJR')
CONJ = set()  ## CONJ sentiment set
CONJ.add('conj')
modES = set()  ## modify sentiment equality
modES.add('amod')
modES.add('pnmod')
modES.add('nmod')
gjES = set()  ##
gjES.add('dobj')
gjES.add('iobj')
gjES.add('nsubj')

# In[3]:
class Opinion:  ## Opinion word class
    def __init__(self, t):
        self.token = t
        self.is_seed = False
# In[4]:
class Target:  ## target word class
    def __init__(self, t):
        self.token = t

# In[5]:

class Label:
    def __init__(self, t, p, a):
        self.token = t
        self.polarity = p
        self.add_info = a
# # Function

# In[8]:
def dep_senti_equality(a, b):
    if a in modES and b in modES:
        return True
    elif a in gjES and b in gjES:
        return True
    else:
        return False

# In[9]:

def get_opinion_set(Opinion_class_set):
    O_s = set()
    for opinion in Opinion_class_set:
        O_s.add(opinion.token)
    return O_s

def get_target_set(Target_class_set):
    T_s = set()
    for target in Target_class_set:
        T_s.add(target.token)
    return T_s

def check_neg_rel(opinion_word, s):
    sent = s.split()
    index = 0
    tok_index = -1
    for token in sent:
        if token == opinion_word:
            tok_index = index
        index += 1
    start_index = 0
    end_index = len(sent) - 1
    if (tok_index - 3) > 0:
        start_index = tok_index - 3
    if (tok_index + 3) < (len(sent) - 1):
        end_index = tok_index + 3
    ##print(tok_index, start_index, end_index)
    for i in range(start_index, end_index, 1):
        if sent[i] == 'not' or sent[i] == 'no' or '''n't''' in sent[i]:
            return -1
    return 1


# # Rule

# ## Rule 1 Opinion to Target

# In[14]:

def Rule_1_O_to_T_1(review_sent, opinion_set,sent,dep):
    # sent = review_sent.sentence ## newly add
    opinion_word_set = get_opinion_set(opinion_set)
    new_T_set = set()
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    for t in dep.triples():
        root_t, dep_rel, leaf_t = t
        ##print(t)
        if leaf_t[0] in opinion_word_set and dep_rel in MR and root_t[1] in NN:
            target = Target(root_t[0])
            new_T_set.add(target)
            pair =[str(target.token),str(leaf_t[0])]
            # is_new_flag = True
            # for t in review_sent.pred_pair.keys():
            #     if t== target.token:
            #         is_new_flag = False
            #         break
            # if is_new_flag:
            # print("1Rule_1_O_to_T_1:", t)
            if target.token == leaf_t[0]:
                print("1Rule_1_O_to_T_1:", sent, target.token, leaf_t[0])
            else:
                if pair not in review_sent.pred_pair:
                    review_sent.pred_pair.append(pair)
        elif root_t[0] in opinion_word_set and dep_rel in MR and leaf_t[1] in NN:
            target = Target(leaf_t[0])
            new_T_set.add(target)
            # pair = {str(target.token):str(root_t[0])}
            pair = [str(target.token), str(root_t[0])]
            is_new_flag = True
            # for t in review_sent.pred_pair.keys():
            #     if t == target.token:
            #         is_new_flag = False
            #         break
            # if is_new_flag:
            #     review_sent.pred_pair.add(pair)
            #     # print("2Rule_1_O_to_T_1:", pair)
            if str(target.token) == str(root_t[0]):
                print("2Rule_1_O_to_T_1:", sent, target.token, root_t[0])
            else:
                if pair not in review_sent.pred_pair:
                    review_sent.pred_pair.append(pair)
    return new_T_set

def Rule_1_O_to_T_2(review_sent, opinion_set,sent,dep):
    # sent = review_sent.sentence  ## newly add
    opinion_word_set = get_opinion_set(opinion_set)
    new_T_set = set()
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    H = set()  ## head of dependency sentiment
    opinion = ""
    for t in dep.triples():
        root_t, dep_rel, leaf_t = t
        if leaf_t[0] in opinion_word_set and dep_rel in MR:
            l = root_t[0] + "#" + leaf_t[0]
            opinion = leaf_t[0]
            H.add(l)
    for h in H:
        head, leaf = h.split("#")
        for t in dep.triples():
            root_t, dep_rel, leaf_t = t
            if root_t[0] == head and dep_rel in MR and leaf_t[1] in NN:
                target = Target(leaf_t[0])
                new_T_set.add(target)
                # pair = {str(target.token): str(opinion)}
                pair = [str(target.token),opinion]
                is_new_flag = True
                # for t in review_sent.pred_pair.keys():
                #     if t == target.token:
                #         is_new_flag = False
                #         break
                # if is_new_flag:
                #     print("1Rule_1_O_to_T_2:", pair)
                if str(target.token) == str(opinion):
                    print("1Rule_1_O_to_T_2:", sent,pair)
                    print(t)
                    print(H)
                else:
                    if pair not in review_sent.pred_pair:
                        review_sent.pred_pair.append(pair)
    return new_T_set

# ## Rule 2 Target to Opinion
def Rule_2_T_to_O_1(review_sent, target_set,sent,dep):
    # sent = review_sent.sentence  ## newly add
    target_word_set = get_opinion_set(target_set)
    new_O_set = set()
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    for t in dep.triples():
        root_t, dep_rel, leaf_t = t
        ##print(t)
        if root_t[0] in target_word_set and dep_rel in MR and leaf_t[1] in JJ:
            opinion = Opinion(leaf_t[0])
            new_O_set.add(opinion)
            ## update target in review_sent
            # is_new_flag = True
            # pair = {str(root_t[0]):str(opinion.token)}
            pair = [str(root_t[0]),opinion.token]
            # for t in review_sent.pred_target_set:
            #     if root_t[0] == t.token:
            #         is_new_flag = False
            # if is_new_flag:
                # print("1Rule_2_T_to_O_1:", pair)
            if root_t[0] == opinion.token:
                print("1Rule_2_T_to_O_1:", sent, root_t[0], str(opinion.token))
            else:
                if pair not in review_sent.pred_pair:
                    review_sent.pred_pair.append(pair)
        elif leaf_t[0] in target_word_set and dep_rel in MR and root_t[1] in JJ:
            opinion = Opinion(root_t[0])
            new_O_set.add(opinion)
            # pair = {leaf_t[0]:opinion.token}
            pair = [str(leaf_t[0]),opinion.token]
            ## update target in review_sent
            is_new_flag = True
            # for t in review_sent.pred_target_set:
            #     if leaf_t[0] == t.token:
            #         is_new_flag = False
            # if is_new_flag:

                # print("2Rule_2_T_to_O_1:", pair)
            if leaf_t[0] == opinion.token:
                print("2Rule_2_T_to_O_1:", sent, leaf_t[0], str(opinion.token))
            else:
                if pair not in review_sent.pred_pair:
                    review_sent.pred_pair.append(pair)
    return new_O_set

def Rule_2_T_to_O_2(review_sent, target_set,sent,dep):
    # sent = review_sent.sentence  ## newly add
    target_word_set = get_opinion_set(target_set)
    new_O_set = set()
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    ## result_triples = my_dependency_parser(sent)
    H = set()  ## head of dependency sentiment
    for t in dep.triples():
        root_t, dep_rel, leaf_t = t
        if leaf_t[0] in target_word_set and dep_rel in MR:
            H.add(root_t[0] + "#" + leaf_t[0])
    for h in H:
        head, leaf = h.split("#")
        for t in dep.triples():
            root_t, dep_rel, leaf_t = t
            if root_t[0] == head and dep_rel in MR and leaf_t[1] in JJ:
                opinion = Opinion(leaf_t[0])
                new_O_set.add(opinion)
                # pair = {leaf: opinion.token}
                pair = [leaf,opinion.token]
                ## update target in review_sent
                # is_new_flag = True
                # for t in review_sent.pred_target_set:
                #     if leaf == t.token:
                #         is_new_flag = False
                # if is_new_flag:

                    # print("2Rule_2_T_to_O_2:", pair)
                if leaf == opinion.token:
                    print("2Rule_2_T_to_O_2:", sent, leaf, str(opinion.token))
                else:
                    if pair not in review_sent.pred_pair:
                        review_sent.pred_pair.append(pair)
    return new_O_set


def Rule_3_T_to_T_1(review_sent, target_set,sent,dep):
    # sent = review_sent.sentence
    target_word_set = get_target_set(target_set)
    ##print(target_word_set)
    new_T_set = set()
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    for t in list(dep.triples()):
        root_t, dep_rel, leaf_t = t
        if leaf_t[0] in target_word_set and root_t[1] in NN and dep_rel in CONJ:
            if root_t[0] not in target_word_set:
                target = Target(root_t[0])
                review_sent.pred_target_set.add(deepcopy(target))
                new_T_set.add(target)
            else:
                is_root_new_flag = True
                is_leaf_new_flag = True
                for r_sent in review_sent.pred_target_set:
                    if r_sent.token == root_t[0]:
                        is_root_new_flag = False
                    if r_sent.token == leaf_t[0]:
                        is_leaf_new_flag = False
                if is_root_new_flag:
                    t = Target(root_t[0])
                    review_sent.pred_target_set.add(deepcopy(t))
                if is_leaf_new_flag:
                    t = Target(leaf_t[0])
                    review_sent.pred_target_set.add(deepcopy(t))

        elif root_t[0] in target_word_set and leaf_t[1] in NN and dep_rel in CONJ:
            target = Target(leaf_t[0])
            review_sent.pred_target_set.add(deepcopy(target))
            new_T_set.add(target)
    return new_T_set

def Rule_3_T_to_T_2(review_sent, target_set,sent,dep):
    # sent = review_sent.sentence
    target_word_set = get_target_set(target_set)
    new_T_set = set()
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    for i, val in enumerate(list(dep.triples())):
        root_t, dep_rel, leaf_t = val
        ##print(val)
        for j, val2 in enumerate(list(dep.triples()), i):
            root_t2, dep_rel2, leaf_t2 = val2
            ##print(val2)
            if root_t[0] == root_t2[0] and dep_senti_equality(dep_rel, dep_rel2) and leaf_t[1] in NN and leaf_t2[
                0] in target_word_set and leaf_t[0] != leaf_t2[0]:
                ##print("here")
                if leaf_t[0] not in target_word_set:
                    target = Target(leaf_t[0])
                    review_sent.pred_target_set.add(deepcopy(target))
                    new_T_set.add(target)
                else:
                    is_new_flag = True
                    for r_sent in review_sent.pred_target_set:
                        if r_sent.token == leaf_t[0]:
                            is_new_flag = False
                            break
                    if is_new_flag:
                        t = Target(leaf_t[0])
                        review_sent.pred_target_set.add(deepcopy(t))
                    is_new_flag = True
                    for r_sent in review_sent.pred_target_set:
                        if r_sent.token == leaf_t2[0]:
                            is_new_flag = False
                            break
                    if is_new_flag:
                        t = Target(leaf_t2[0])
                        review_sent.pred_target_set.add(deepcopy(t))
    return new_T_set

# If find new opinion word, then add it to opinion expand set and assign its parent's polarity to it.
# If the finding opinion is not new, and not a opinion seed, then update its polarity with (its polarity + parent polarity)/ 2
# If the finding opinion is not new, and it is a opinion seed, then do nothing
def Rule_4_O_to_O_1(review_sent, opinion_set,sent,dep):
    new_op_set = set()
    # sent = review_sent.sentence
    opinion_word_set = get_opinion_set(opinion_set)
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    for t in dep.triples():
        root_t, dep_rel, leaf_t = t
        # The moive is A and B, given B(leaf) -> A(root)
        if leaf_t[0] in opinion_word_set and root_t[1] in JJ and dep_rel in CONJ:
            # child opinion is new opinion, initialize an new opinion instance and assign parent polarity to it.
            # and add it to opinion expand set
            if root_t[0] not in opinion_word_set:
                opinion = Opinion(root_t[0])
                opinion_set.add(opinion)
                new_op_set.add(opinion)
                opinion_word_set = get_opinion_set(opinion_set)

        ## The moive is A and B, given A(root) -> B(leaf)
        if root_t[0] in opinion_word_set and leaf_t[1] in JJ and dep_rel in CONJ:
            # child opinion is new opinion, initialize an new opinion instance and assign parent polarity to it.
            # and add it to opinion expand set
            if leaf_t[0] not in opinion_word_set:
                opinion = Opinion(leaf_t[0])
                opinion_set.add(opinion)
                new_op_set.add(opinion)
                opinion_word_set = get_opinion_set(opinion_set)
    return new_op_set

# no return value
# If find new opinion word, then add it to opinion expand set and assign its parent's polarity to it.
# If the finding opinion is not new, and not a opinion seed, then update its polarity with (its polarity + parent polarity)/ 2
# If the finding opinion is not new, and it is a opinion seed, then do nothing
def Rule_4_O_to_O_2(review_sent, opinion_set,sent,dep):
    new_op_set = set()
    # sent = review_sent.sentence
    opinion_word_set = get_opinion_set(opinion_set)
    # result = dependency_parser.raw_parse(sent)
    # dep = result.__next__()
    for t in dep.triples():
        root_t, dep_senti_t, leaf_t = t
        if leaf_t[0] in opinion_word_set:
            HEAD = root_t[0]
            REL = dep_senti_t
            for u in dep.triples():
                root_u, dep_senti_u, leaf_u = u
                if leaf_u[0] != leaf_t[0] and dep_senti_equality(dep_senti_u, REL) and root_u[0] == HEAD and leaf_u[
                    1] in JJ:
                    # child opinion is new opinion, initialize an new opinion instance and assign parent polarity to it.
                    # and add it to opinion expand set
                    if leaf_u[0] not in opinion_word_set:
                        opinion = Opinion(leaf_u[0])
                        opinion_set.add(opinion)
                        new_op_set.add(opinion)
                        opinion_word_set = get_opinion_set(opinion_set)
    return new_op_set

def DoubleProp(opinion_seed, product_review_list):
    # output set initialization

    Opinion_Expand = opinion_seed
    Product_Reviews_list = product_review_list
    ##print(Opinion_Expand)
    All_Feature = set()
    itr = 0
    while True:
        print("iteraton " + str(itr) + ":")
        ## print(Opinion_Expand)
        new_feature_i_set = set()
        new_opinion_i_set = set()
        new_feature_j_set = set()
        new_opinion_j_set = set()
        for one_product_review in Product_Reviews_list:
            sent = one_product_review.sentence  ## newly add
            result = dependency_parser.raw_parse(sent)
            dep = result.__next__()
            temp_new_feature_R11_set = Rule_1_O_to_T_1(one_product_review, Opinion_Expand,sent,dep)
            for new_feature in temp_new_feature_R11_set:
                is_new_flag = True
                for feature in All_Feature:
                    if new_feature.token == feature.token:
                        is_new_flag = False
                        break
                if is_new_flag:
                    new_feature_i_set.add(new_feature)
                    ##if feature not in All_Feature:#no use "if" since add will handle the repeated feature
                    ##   new_feature_i_set.add(feature)
            temp_new_feature_R12_set = Rule_1_O_to_T_2(one_product_review, Opinion_Expand,sent,dep)
            for new_feature in temp_new_feature_R12_set:
                is_new_flag = True
                for feature in All_Feature:
                    if new_feature.token == feature.token:
                        is_new_flag = False
                        break
                if is_new_flag:
                    new_feature_i_set.add(new_feature)
                    ##if feature not in All_Feature:#no use "if" since add will handle the repeated feature
                    ##   new_feature_i_set.add(feature)

                # Rule 4
                # print("implementing rule4")
            temp_new_opinion_R41_set = Rule_4_O_to_O_1(one_product_review, Opinion_Expand,sent,dep)
            temp_new_opinion_R42_set = Rule_4_O_to_O_2(one_product_review, Opinion_Expand,sent,dep)
            for opinion in temp_new_opinion_R41_set:
                new_opinion_i_set.add(opinion)
            for opinion in temp_new_opinion_R42_set:
                new_opinion_i_set.add(opinion)

        All_Feature = All_Feature | new_feature_i_set
        Opinion_Expand = Opinion_Expand | new_opinion_i_set

            ##print("2")
            ## print(Opinion_Expand)

            # for st in one_product_review.review_sentences_list:
                # Rule 3
                # print("implementing rule3")
        for one_product_review in Product_Reviews_list:
            sent = one_product_review.sentence  ## newly add
            result = dependency_parser.raw_parse(sent)
            dep = result.__next__()
            temp_new_feature_R31_set = Rule_3_T_to_T_1(one_product_review, new_feature_i_set,sent,dep)
            temp_new_feature_R32_set = Rule_3_T_to_T_2(one_product_review, new_feature_i_set,sent,dep)
            for feature in temp_new_feature_R31_set:
                new_feature_j_set.add(feature)
            for feature in temp_new_feature_R32_set:
                new_feature_j_set.add(feature)
            # print("implementing rule2")
            # Rule 2
            temp_new_opinion_R21_set = Rule_2_T_to_O_1(one_product_review, new_feature_i_set,sent,dep)
            for new_opinion in temp_new_opinion_R21_set:
                is_new_flag = True
                for opinion in Opinion_Expand:
                    if new_opinion.token == opinion.token:
                        is_new_flag = False
                        break
                if is_new_flag:
                    new_opinion_j_set.add(new_opinion)
                ##if opinion not in Opinion_Expand:#no use "if" since add will handle the repeated opinion
                ##    new_opinion_j_set.add(opinion)
            temp_new_opinion_R22_set = Rule_2_T_to_O_2(one_product_review, new_feature_i_set,sent,dep)
            for new_opinion in temp_new_opinion_R22_set:
                is_new_flag = True
                for opinion in Opinion_Expand:
                    if new_opinion.token == opinion.token:
                        is_new_flag = False
                        break
                if is_new_flag:
                    new_opinion_j_set.add(new_opinion)
                ##if opinion not in Opinion_Expand:#no use "if" since add will handle the repeated opinion
                ##    new_opinion_j_set.add(opinion)

        new_feature_i_set = new_feature_i_set | new_feature_j_set
        new_opinion_i_set = new_opinion_i_set | new_opinion_j_set
        ## reset feature polarity to zero for each review
        for feature in new_feature_i_set:
            feature.polarity = 0

        print("new feature i set: \n" + str(len(new_feature_i_set)) + " new features")
        for feature in new_feature_i_set:
            print(feature.token, feature.polarity)
        print("new opinion i set: \n" + str(len(new_opinion_i_set)) + " new opinions")
        for opinion in new_opinion_i_set:
            print(opinion.token)

        All_Feature = All_Feature | new_feature_j_set
        Opinion_Expand = Opinion_Expand | new_opinion_j_set
        itr += 1
        ##print("3")
        ##print(Opinion_Expand)
        if len(new_feature_i_set) == 0 and len(new_opinion_i_set) == 0:
            break
    return All_Feature, Opinion_Expand