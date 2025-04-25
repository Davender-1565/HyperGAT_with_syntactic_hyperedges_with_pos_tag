import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from scipy.sparse.linalg import eigsh
import sys
import re
import collections
from collections import Counter
import numpy as np
from multiprocessing import Process, Queue
import pandas as pd
import os
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.utils import *
from nltk import pos_tag


def clean_str(string):
    string = re.sub(r'([a-z])([A-Z])', r'\1 \2', string)
    string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_simple_version(string, dataset):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def show_statisctic(clean_docs):
    min_len = 10000
    aver_len = 0
    max_len = 0 
    num_sentence = sum([len(i) for i in clean_docs])
    ave_num_sentence = num_sentence * 1.0 / len(clean_docs)

    for doc in clean_docs:
        for sentence in doc:
            temp = sentence
            aver_len = aver_len + len(temp)

            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)

    aver_len = 1.0 * aver_len / num_sentence

    print('min_len_of_sentence : ' + str(min_len))
    print('max_len_of_sentence : ' + str(max_len))
    print('min_num_of_sentence : ' + str(min([len(i) for i in clean_docs])))
    print('max_num_of_sentence : ' + str(max([len(i) for i in clean_docs])))
    print('average_len_of_sentence: ' + str(aver_len))
    print('average_num_of_sentence: ' + str(ave_num_sentence))
    print('Total_num_of_sentence : ' + str(num_sentence))

    return max([len(i) for i in clean_docs])


def clean_document(doc_sentence_list, dataset):
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    stemmer = WordNetLemmatizer()

    word_freq = Counter()

    for doc_sentences in doc_sentence_list:
        for sentence in doc_sentences:
            temp = word_tokenize(clean_str(sentence))
            temp = ' '.join([stemmer.lemmatize(word) for word in temp])

            words = temp.split()
            for word in words:
                word_freq[word] += 1

    highbar = word_freq.most_common(10)[-1][1]
    clean_docs = []
    for doc_sentences in doc_sentence_list:
        clean_doc = []
        count_num = 0
        for sentence in doc_sentences:
            temp = word_tokenize(clean_str(sentence))
            temp = ' '.join([stemmer.lemmatize(word) for word in temp])

            words = temp.split()
            doc_words = []
            for word in words:
                if dataset in ['mr', 'Tweet']:
                    if word not in stop_words:
                        doc_words.append(word)
                elif (word not in stop_words) and (word_freq[word] >= 5) and (word_freq[word] < highbar):
                    doc_words.append(word)

            clean_doc.append(doc_words)
            count_num += len(doc_words)

            if dataset == '20ng' and count_num > 2000:
                break
            
        clean_docs.append(clean_doc)

    return clean_docs


def split_validation(train_set, valid_portion, SEED):
    np.random.seed(SEED)

    train_set_x = [i for i, j in train_set]
    train_set_y = [j for i, j in train_set]

    if valid_portion == 0.0:
        return (train_set_x, train_set_y)

    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, max_num_sentence, keywords_dic, num_categories, LDA=False, num_pos_hyperedges=0):
        inputs = data[0]
        
        # Pad or truncate sequences at all levels
        padded_inputs = []
        max_len_outer = max(len(seq) for seq in inputs)  # Maximum number of sentences per document
        max_len_inner = max(len(s) for seq in inputs for s in seq)  # Maximum number of tokens per sentence

        for seq in inputs:
            padded_seq = []
            for s in seq:
                # Pad or truncate each sentence
                if len(s) > max_len_inner:
                    padded_seq.append(s[:max_len_inner])  # Truncate longer sentences
                else:
                    padded_seq.append(s + [0] * (max_len_inner - len(s)))  # Pad shorter sentences
            # Pad or truncate the document
            if len(padded_seq) > max_len_outer:
                padded_inputs.append(padded_seq[:max_len_outer])  # Truncate longer documents
            else:
                padded_inputs.append(padded_seq + [[0] * max_len_inner] * (max_len_outer - len(padded_seq)))  # Pad shorter documents
        
        self.inputs = np.asarray(padded_inputs)
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.keywords = keywords_dic
        self.LDA = LDA
        self.num_categories = num_categories
        self.num_pos_hyperedges = num_pos_hyperedges

    def generate_batch(self, batch_size, shuffle=False):
        if shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, iList):
        inputs, targets = self.inputs[iList], self.targets[iList]
        items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []

        # Calculate maximum dimensions
        max_n_node = 0
        for doc in inputs:
            unique_words = set()
            for sentence in doc:
                sentence_list = sentence.tolist() if isinstance(sentence, np.ndarray) else sentence
                unique_words.update(w for w in sentence_list if w != 0)
            max_n_node = max(max_n_node, len(unique_words))
        max_n_node = max(max_n_node, 1)

        # Calculate total hyperedges - FIXED VERSION
        if len(inputs) == 0:
            max_n_edge = 1
        else:
            max_n_edge = max([len(doc) for doc in inputs])  # Sequential
        
        if self.LDA:
            max_n_edge += self.num_categories
        if self.num_pos_hyperedges > 0:
            max_n_edge += self.num_pos_hyperedges

        for idx, u_input in enumerate(inputs):
            temp_s = []
            for s in u_input:
                if isinstance(s, np.ndarray):
                    s = s.tolist()
                if len(s) > 0:
                    temp_s.extend(w for w in s if w != 0)
            
            temp_l = list(set(temp_s))    
            temp_dic = {temp_l[i]: i for i in range(len(temp_l))}        
            n_node.append(temp_l)
            alias_inputs.append([temp_dic[i] for i in temp_s])
            node_dic.append(temp_dic)

        max_se_len = max([len(i) for i in alias_inputs]) if alias_inputs else 0

        for idx in range(len(inputs)):
            u_input = inputs[idx]
            node = n_node[idx]
            items.append(node + (max_n_node - len(node)) * [0])

            rows = []
            cols = []
            vals = []

            for s in range(len(u_input)):
                for i in np.arange(len(u_input[s])):
                    if u_input[s][i] == 0:
                        continue

                    rows.append(node_dic[idx][u_input[s][i]])
                    cols.append(s)
                    vals.append(1.0)

            if len(cols) == 0:
                s = 0
            else:
                s = max(cols) + 1

            if self.LDA or self.num_pos_hyperedges > 0:
                # Add LDA hyperedges if enabled
                if self.LDA:
                    for i in node:
                        if i in self.keywords:
                            temp = [h for h in self.keywords[i] if h < self.num_categories]
                            rows += [node_dic[idx][i]] * len(temp)
                            cols += [topic + s for topic in temp]
                            vals += [1.0] * len(temp)
                    s += self.num_categories

                # Add POS hyperedges
                if self.num_pos_hyperedges > 0:
                    for i in node:
                        if i in self.keywords:
                            pos_edges = [h for h in self.keywords[i] 
                                        if h >= (self.num_categories if self.LDA else 0) 
                                        and h < (self.num_categories + self.num_pos_hyperedges)]
                            if pos_edges:
                                rows += [node_dic[idx][i]] * len(pos_edges)
                                cols += [h + s for h in pos_edges]
                                vals += [1.0] * len(pos_edges)

            # Validate dimensions before creating matrix
            if cols:  # Only check if cols is not empty
                max_col = max(cols)
                if max_col >= max_n_edge:
                    # Adjust invalid indices
                    cols = [min(c, max_n_edge-1) for c in cols]
                    print(f"Warning: Adjusted column indices (max was {max_col}, matrix width {max_n_edge})")

            u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))
            
            alias_inputs[idx] = [j for j in range(max_n_node)]
            node_masks.append([1 for j in node] + (max_n_node - len(node)) * [0])

        return alias_inputs, HT, items, targets, node_masks