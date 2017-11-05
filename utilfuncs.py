import re
import codecs
import sys
import operator
import math
import numpy as np
from collections import defaultdict

def update_vocab(symbol, idxvocab, vocabxid):
    idxvocab.append(symbol)
    vocabxid[symbol] = len(idxvocab) - 1 

def gen_vocab(dummy_symbols, corpus, stopwords, vocab_minfreq, vocab_maxfreq, verbose):
    idxvocab = []
    vocabxid = defaultdict(int)
    vocab_freq = defaultdict(int)
    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for word in line.strip().split():
            vocab_freq[word] += 1
        if line_id % 1000 == 0 and verbose:
            sys.stdout.write(str(line_id) + " processed\r")
            sys.stdout.flush()

    #add in dummy symbols into vocab
    for s in dummy_symbols:
        update_vocab(s, idxvocab, vocabxid)

    #remove low fequency words
    for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
        if f < vocab_minfreq:
            break
        else:
            update_vocab(w, idxvocab, vocabxid)

    #ignore stopwords, frequent words and symbols for the document input for topic model
    stopwords = set([item.strip().lower() for item in open(stopwords)])
    freqwords = set([item[0] for item in sorted(vocab_freq.items(), key=operator.itemgetter(1), \
        reverse=True)[:int(float(len(vocab_freq))*vocab_maxfreq)]]) #ignore top N% most frequent words for topic model
    alpha_check = re.compile("[a-zA-Z]")
    symbols = set([ w for w in vocabxid.keys() if ((alpha_check.search(w) == None) or w.startswith("'")) ])
    ignore = stopwords | freqwords | symbols | set(dummy_symbols) | set(["n't"])
    ignore = set([vocabxid[w] for w in ignore if w in vocabxid])

    return idxvocab, vocabxid, ignore