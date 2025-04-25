import pickle
from collections import defaultdict

# Function to build vocab_dic
def build_vocab(corpus_file):
    vocab_dic = defaultdict(int)
    with open(corpus_file, 'r', encoding='ISO-8859-1') as f:  # Use the correct encoding
        for line in f:
            words = line.strip().split()
            for word in words:
                vocab_dic[word] += 1
    # Assign unique indices to words
    vocab_dic = {word: idx for idx, word in enumerate(vocab_dic.keys(), start=1)}
    return vocab_dic

# Function to build labels_dic
def build_labels(labels_file):
    labels_dic = {}
    with open(labels_file, 'r', encoding='ISO-8859-1') as f:  # Use the correct encoding
        for idx, line in enumerate(f):
            label = line.strip()
            labels_dic[label] = idx
    return labels_dic

# Paths to your preprocessed data files
corpus_file = 'R52_corpus.txt'  # Replace with the actual filename
labels_file = 'R52_labels.txt'  # Replace with the actual filename

# Build vocab_dic and labels_dic
vocab_dic = build_vocab(corpus_file)
labels_dic = build_labels(labels_file)

# Save vocab_dic and labels_dic
with open('vocab_dic.pkl', 'wb') as f:
    pickle.dump(vocab_dic, f)
with open('labels_dic.pkl', 'wb') as f:
    pickle.dump(labels_dic, f)

print("vocab_dic and labels_dic saved successfully!")