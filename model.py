import torch

class TopicModel:
	def __init__(self, is_training, vocab_size, batch_size, num_steps, num_classes, cf):
		self.conv_size = len(cf.filter_sizes)*cf.filter_number
		# self.doc = torch.IntTensor(cf.doc_len)
		self.conv_word_embedding = torch.FloatTensor(vocab_size, 50).zero_()
		self.topic_output_embedding = torch.FloatTensor(cf.k,cf.topic_embedding_size).zero_()
		self.topic_input_embedding = torch.FloatTensor(cf.k,self.conv_size).zero_()
		self.tm_softmax_w = torch.FloatTensor(cf.topic_embedding_size,vocab_size).zero_()
		if is_training and config.num_samples > 0:
			self.tm_softmax_w_t = torch.transpose(self.tm_softmax_w,0,1)
		self.tm_softmax_b = torch.FloatTensor(vocab_size).zero_()
		self.eye = torch.eye(cf.k)

		doc_inputs = torch.nn.LookupTable(self.conv_word_embedding, self.doc)
		if is_training and cf.tm_keep_prob < 1.0:
			# doc_inputs = torch.nn.Dropout(doc_inputs, cf.tm_keep_prob, seed=1)
		doc_inputs = doc_inputs.unsqueeze(-1)

		pooled_outputs=[]
		for i, filter_size in enumerate(cf.filter_sizes):
			filter_w = torch.rand(filter_size, 50, 1, cf.filter_number)
			filter_b = torch.FloatTensor(cf.filter_number).zero_()

			model1 = torch.nn.Conv2d(1,1,filter_w.size(), stride=(1,1,1,1), padding=1)
			model1.weight.data=filter_w
			model1.bias.data=filter_b
			inputs = torch.autograd.Variable(torch.FloatTensor(doc_inputs))
			conv = model1(inputs)

			model2 = torch.nn.MaxPool2D((1,(cf.doc_len-filter_size+1),1,1), stride=(1,1,1,1), padding=1)
			h = model2(conv)
			pooled_outputs.append(h)
			
		conv_pooled = torch.cat(pooled_outputs,3)
		conv_pooled = conv_pooled.view(-1, self.conv_size)

		self.attention = torch.nn.SoftMax(torch.sum(self.topic_input_embedding.unsqueeze(0).mul(conv_pooled.unsqueeze(1)), 2))
		self.mean_topic = torch.sum(self.attention.unsqueeze(2).mul(self.topic_output_embedding.unsqueeze(0)),1)

		if is_training and cf.tm_keep_prob < 1.0:
			self.mean_topic_dropped = torch.nn.Dropout(self.mean_topic, cf.tm_keep_prob, seed=cf.seed)
		else:
			self.mean_topic_dropped = self.mean_topic

		self.conv_hidden = self.mean_topic_dropped.repeat(1, num_steps).view(batch_size*num_steps, cf.topic_embedding_size)

		if not (is_training and cf.num_samples > 0):
			self.tm_logits = self.conv_hidden.mm(self.tm_softmax_w) + self.tm_softmax_b
			# tm_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(self.tm_logits, self.y.view(-1))
			# tm_crossent = torch.nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100)

		tm_crossent_m = tm_crossent * self.tm_mask.view(-1)
		self.tm_cost = torch.sum(tm_crossent_m) / batch_size

		if not is_training:
			return

		topicnorm = self.topic_output_embedding / torch.sqrt(torch.sum(torch.mul(self.topic_output_embedding,self.topic_output_embedding),1, keep_dims=True))
		uniqueness = torch.max(torch.mul(topicnorm.mm(torch.transpose(topicnorm)) - self.eye,topicnorm.mm(torch.transpose(topicnorm)) - self.eye))
		self.tm_cost += cf.alpha * uniqueness
		self.tm_train_op = torch.optim.Adam(self.tm_cost, lr=cf.learning_rate)

###########################################################
	def get_topics(self, sess, topn):
		topics = []
		entropy = []
		tw_dist = sess.run(tf.nn.softmax(tf.matmul(self.topic_output_embedding, self.tm_softmax_w) + self.tm_softmax_b))
		for ti in range(self.cf.topic_number):
			best = matutils.argsort(tw_dist[ti], topn=topn, reverse=True)
			topics.append(best)
			entropy.append(scipy.stats.entropy(tw_dist[ti]))

		return topics, entropy