import re
import codecs
import sys
import operator
import math
import numpy as np
from collections import defaultdict
import constants as cf

def pad(lst, max_len, pad_symbol):
    return lst + [pad_symbol] * (max_len - len(lst))

def get_batch(sents, docs, tags, idx, doc_len, sent_len, tag_len, batch_size, pad_id, remove_curr):
    x, y, m, d, t = [], [], [], [], []

    for docid, seqid, seq in sents[(idx*batch_size):((idx+1)*batch_size)]:
        if remove_curr:
            dw = docs[docid][:seqid] + docs[docid][(seqid+1):]
        else:
            dw = docs[docid]
        dw = [item for sublst in dw for item in sublst][:doc_len]
        d.append(pad(dw, doc_len, pad_id))
        x.append(pad(seq[:-1], sent_len, pad_id))
        y.append(pad(seq[1:], sent_len, pad_id))
        m.append([1.0]*(len(seq)-1) + [0.0]*(sent_len-len(seq)+1))
        if tags != None:
            t.append(pad(tags[docid][:tag_len], tag_len, pad_id))
        else:
            t.append([])

    for _ in xrange(batch_size - len(d)):
        d.append([pad_id]*doc_len)
        x.append([pad_id]*sent_len)
        y.append([pad_id]*sent_len)
        m.append([0.0]*sent_len)
        t.append([pad_id]*tag_len)

	return x, y, m, d, t

def update_vocab(symbol, idxvocab, vocabxid):
    idxvocab.append(symbol)
    vocabxid[symbol] = len(idxvocab) - 1 

def gen_data(vocabxid, dummy_symbols, ignore, corpus):
	sents = ([], []) #tuple of (tm_sents, lm_sents); each element is [(doc_id, seq_id, seq)]
	docs = ([], []) 
	sent_lens = [] #original sentence lengths
	doc_lens = [] #original document lengths
	docids = [] #original document IDs

	for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
		#every tm_sents starts with start symbol and end only after all the tokens
		tm_sents=[]
		tm_sents.append(vocabxid[cf.start_symbol])
		lm_sents=[]

		#**************improvisation can be here with tokenisations
		for token in line.strip().split("\t"):
			sent=[vocabxid[cf.start_symbol]]
			for word in token.strip().split():
				if word in vocabxid:
					sent.append(vocabxid[word])
					if (vocabxid[word] not in ignore):
                        			tm_sents.append(vocabxid[word])
				else:
                    			sent.append(vocabxid[cf.unk_symbol])
				sent.append(vocabxid[cf.end_symbol])
			lm_sents.append(sent)

		if len(tm_sents)>1:
			docids.append(line_id)
			sent_lens.extend([len(item)-1 for item in lm_sents])
			doc_lens.append(len(tm_sents))

			#truncating tm_sents into sizes of m1=3
			m1=3
			seq_id = 0
			doc_seqs = []
			for si in range(int(math.ceil(len(tm_sents) * 1.0 / m1))):
				seq = tm_sents[si*m1:((si+1)*m1+1)]
				if len(seq) > 1:
					sents[0].append((len(docs[0]), seq_id, seq))
					doc_seqs.append(seq[1:])
					seq_id += 1
			docs[0].append(doc_seqs)

			m2=30
			seq_id = 0
			doc_seqs = []
			for s in lm_sents:
				for si in range(int(math.ceil(len(s) * 1.0 / m2))):
					seq = s[si*m2:((si+1)*m2+1)]
					if len(seq) > 1:
						sents[1].append((len(docs[1]), seq_id, seq))
						doc_seqs.append([w for w in seq[1:] if w not in ignore]) #output sentence = seq[1:]
						seq_id += 1
			docs[1].append(doc_seqs)

	return sents, docs, docids, (np.mean(sent_lens), max(sent_lens), min(sent_lens), np.mean(doc_lens), max(doc_lens), min(doc_lens))

def gen_vocab(dummy_symbols, corpus, stopwords, vocab_minfreq, vocab_maxfreq):
    idxvocab = []
    vocabxid = defaultdict(int)
    vocab_freq = defaultdict(int)
    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for word in line.strip().split():
            vocab_freq[word] += 1

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
