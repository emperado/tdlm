import sys, random, os, time, math
import numpy as np
import gensim.models as g
# import codecs
import constants as cf
from utilfuncs import *

random.seed(1)
np.random.seed(1)
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
vocabxid = {}
idxvocab = []

wordvec = g.Word2Vec.load('./word2vec/imdb.bin')
word_embd_size = wordvec.vector_size

idxvocab, vocabxid, tm_ignore = gen_vocab(cf.dummy_symbols, cf.train_corpus, cf.stopwords, cf.vocab_minfreq, cf.vocab_maxfreq)
train_sents, train_docs, train_docids, train_stats = gen_data(vocabxid, cf.dummy_symbols, tm_ignore, cf.train_corpus)
valid_sents, valid_docs, valid_docids, valid_stats = gen_data(vocabxid, cf.dummy_symbols, tm_ignore, cf.valid_corpus)

