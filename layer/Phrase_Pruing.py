import copy
opinion_seed_file_name = './data/Opinion_Seed.txt'
class Opinion:  ## Opinion word class
    def __init__(self, t):
        self.token = t
        self.is_seed = False
def get_opinion_set(Opinion_class_set):
    O_s = set()
    for opinion in Opinion_class_set:
        O_s.add(opinion.token)
    return O_s
O_S = set()  ## opinion seed set
file = open(opinion_seed_file_name)
lines = file.readlines()
for line in lines:
    token, polarity = line.split(", ")
    o = Opinion(token)
    o.is_seed = True
    O_S.add(o)
def get_phrase(root):
    def helper(root, np, vp):
        if type(root) == str:
            return
        l = len(root)
        for i in range(l):
            if root.height() < 3:
                break
            if len(root) >= i:
                if root[i]:
                    helper(root[i], np, vp)
                    if (root.height() < 5) and 1 < len(root):
                        # print(root.leaves())
                        if (str(root.label()) == 'NP') and root.height() == 3:
                            np_index = root.leaves()
                            if 'the' in root.leaves():
                                np_index.remove('the')
                            if np_index not in np and len(np_index)>1:
                                np.append(np_index)
                        if (str(root.label()) == 'VP') and len(root) <= 3:
                            if root.leaves() not in vp:
                                is_append = True
                                for word in root.leaves():
                                    if word in ['was','is']:
                                        is_append = False
                                        break
                                if is_append:
                                    vp.append(root.leaves())
    if root is None:
        return []
    np = []
    vp = []
    helper(root, np, vp)
    return np, vp
def phrase_pruning(tag, np):
    # print(tag)
    np_new = copy.deepcopy(np)
    for i in range(len(np)):
        jj_count = 0
        r = True
        for j in range(len(np[i])):
            word = np[i][j]
            opinion_word_set = get_opinion_set(O_S)
            if tag[word] in ['CC']:
                np_new[i] = []
            if word in opinion_word_set or word in ['that','operating','anything','everything','new','newer','thing','more','problems','other','old','.','older','all']:
                np_new[i].remove(word)
            if tag[word] in ['DT','PRP$','FW','PRP']:
                np_new[i].remove(word)
            if tag[word] in ['JJ']:
                jj_count += 1
                if r:
                    pre_jj_word = word
                    r = False
                if jj_count > 1:
                    np_new[i].remove(pre_jj_word)
                    pre_jj_word = word
    np_new1 = copy.deepcopy(np_new)
    for i in range(len(np_new)):
        if len(np_new[i]) <= 1:
            np_new1.remove(np_new[i])
    return np_new1
