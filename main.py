import sys, random, os, time, math
import numpy as np
import gensim.models as g
# import codecs
import constants as cf
from utilfuncs import *
from model import TopicModel as TM
from model import LanguageModel as LM
import torch

def init_embedding(model, idxvocab):
	word_emb = []
	for vi, v in enumerate(idxvocab):
		if v in model:
			word_emb.append(model[v])
		else:
			word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
	return np.array(word_emb)

def fetch_batch_and_train(sents, docs, tags, model, seq_len, i, (tm_costs, tm_words, lm_costs, lm_words), (m_tm_cost, m_tm_train, m_lm_cost,m_lm_train)):
	x, y, m, d, t = get_batch(sents, docs, tags, i, cf.doc_len, seq_len, cf.tag_len, cf.batch_size, 0,(True if isinstance(model, LM) else False))

	global tm_train

	if isinstance(model, LM):
		if cf.topic_number > 0:
			tm_cost=m_tm_cost
			lm_cost=m_lm_cost
			# tm_cost, _, lm_cost, _ = sess.run([m_tm_cost, m_tm_train, m_lm_cost, m_lm_train], \
		        # {model.x: x, model.y: y, model.lm_mask: m, model.doc: d, model.tag: t})
		else:
		    #pure lstm
			tm_cost=m_tm_cost
			lm_cost=m_lm_cost
			# tm_cost, _, lm_cost, _ = sess.run([m_tm_cost, m_tm_train, m_lm_cost, m_lm_train], \
		        # {model.x: x, model.y: y, model.lm_mask: m})
	else:
		tm_cost=m_tm_cost
		lm_cost=m_lm_cost
	    # tm_cost, _, lm_cost, _ = sess.run([m_tm_cost, m_tm_train, m_lm_cost, m_lm_train], \
	        # {model.y: y, model.tm_mask: m, model.doc: d, model.tag: t})
	print tm_cost
	print "hi555"
	if tm_cost != None:
		tm_costs += tm_cost * cf.batch_size #keep track of full batch loss (not per example batch loss)
		tm_words += np.sum(m)
	if lm_cost != None:
		lm_costs += lm_cost * cf.batch_size
		lm_words += np.sum(m)

	tm_train.varFunc(y,m,d,t)

	return tm_costs, tm_words, lm_costs, lm_words


def run_epoch(sents, docs, labels, tags, models, is_training):

    ####unsupervised topic and language model training####

    #generate the batches
	tm_num_batches, lm_num_batches = int(math.ceil(float(len(sents[0]))/cf.batch_size)), int(math.ceil(float(len(sents[1]))/cf.batch_size))
	batch_ids = [ (item, 0) for item in range(tm_num_batches) ] + [ (item, 1) for item in range(lm_num_batches) ]
	seq_lens = (cf.tm_sent_len, cf.lm_sent_len)
	#shuffle batches and sentences
	random.shuffle(batch_ids)
	random.shuffle(sents[0])
	random.shuffle(sents[1])

	#set training and cost ops for topic and language model training
	tm_cost_ops = (None, None, None, None)
	lm_cost_ops = (None, None, None, None)
	if models[0] != None:
		tm_cost_ops = (models[0].tm_cost, (models[0].tm_train_op if is_training else None), None, None)
	if models[1] != None:
		lm_cost_ops = (None, None, models[1].lm_cost, (models[1].lm_train_op if is_training else None))
	cost_ops = (tm_cost_ops, lm_cost_ops)

	start_time = time.time()
	lm_costs, tm_costs, lm_words, tm_words = 0.0, 0.0, 0.0, 0.0
	for bi, (b, model_id) in enumerate(batch_ids):
		if model_id==0:					#if language included comment this line
			tm_costs, tm_words, lm_costs, lm_words = fetch_batch_and_train(sents[model_id], docs[model_id], tags,models[model_id], seq_lens[model_id], b, (tm_costs, tm_words, lm_costs, lm_words), cost_ops[model_id])



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

num_classes = 0



print "hello i am here1"
print wordvec["the"]

flag=0

tm_train = TM(is_training=True,  vocab_size=len(idxvocab), batch_size=cf.batch_size, num_steps=3, num_classes=num_classes, cf=cf)
tm_valid = TM(is_training=False, vocab_size=len(idxvocab), batch_size=cf.batch_size, num_steps=3, num_classes=num_classes, cf=cf)

tm_train.conv_word_embedding = torch.from_numpy(init_embedding(wordvec, idxvocab))


for i in range(cf.epoch_size):
	print "hello i am here2"
	run_epoch(train_sents, train_docs, None, None, (tm_train, None), True)
	print "hello i am here3"
	curr_ppl = run_epoch(valid_sents, valid_docs, None, None, (tm_valid, None), False)


if cf.topic_number > 0:
	print "\nTopics\n======"
	# topics, entropy = tm_train.get_topics(sess, topn=20)
	for ti, t in enumerate(topics):
		print "Topic", ti, "[", ("%.2f" % entropy[ti]), "] :", " ".join([ idxvocab[item] for item in t ])
