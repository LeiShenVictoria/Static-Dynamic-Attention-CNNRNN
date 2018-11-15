import sys
import math
import time
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

class Encoder(nn.Module):
	def __init__(self, emb_size, hidden_size):
		super(Encoder,self).__init__()
		self.emb_size = emb_size
		self.hidden_size = hidden_size

		self.gru = nn.GRU(self.emb_size,self.hidden_size,batch_first=True)

	def forward(self, input, hidden):
		output,hidden = self.gru(input,hidden)

		return hidden

class Attention(nn.Module):
	def __init__(self, hidden_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size

		self.W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
		self.U = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
		self.v = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, inputs, hidden, pad_matrix):
		attn_hidden = torch.mm(hidden.squeeze(0),self.W)
		list_weight = []
		for input in inputs:
			weight = torch.mm(self.tanh(attn_hidden+torch.mm(input.squeeze(0),self.U)),self.v)
			list_weight.append(weight)
		attn_weights = torch.cat(tuple(list_weight),1).masked_fill_(Variable(pad_matrix.cuda()),-100000)
		attn_weights = self.softmax(attn_weights)

		return attn_weights

class Decoder(nn.Module):
	def __init__(self, decoder_dict_size, hidden_size, emb_size):
		super(Decoder, self).__init__()
		self.decoder_dict_size = decoder_dict_size
		self.hidden_size = hidden_size
		self.emb_size = emb_size

		self.gru = nn.GRU(self.hidden_size+self.emb_size,self.hidden_size,batch_first=True)
		self.out = nn.Linear(self.hidden_size,self.decoder_dict_size)
		self.log_softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden, context_attn):
		input_combine = torch.cat((input, context_attn), 2)
		output, hidden = self.gru(input_combine, hidden)

		output = self.log_softmax(self.out(output.squeeze(1)))
		return output, hidden

class StaticModel(nn.Module):
	def __init__(self, dict_size, emb_size, hidden_size, batch_size, dropout, max_senten_len, teach_forcing):
		super(StaticModel, self).__init__()
		self.dict_size = dict_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.dropout = dropout
		self.max_senten_len = max_senten_len
		self.teach_forcing = teach_forcing

		self.encoder = Encoder(self.emb_size, self.hidden_size)
		self.decoder = Decoder(self.dict_size, self.hidden_size, self.emb_size)
		self.attention = Attention(self.hidden_size)

		self.embedding = nn.Embedding(self.dict_size,self.emb_size)
		
		self.dropout = nn.Dropout(self.dropout)

	def init_parameters(self,emb_matrix):
		n = 0
		for name, weight in self.named_parameters():
			if weight.requires_grad:
				if  "embedding.weight" in name:
					weight = nn.Parameter(emb_matrix)
				elif weight.ndimension() < 2:
					weight.data.fill_(0)
				else:
					weight.data = init.xavier_uniform(weight.data)
				n += 1
		print ("{} weights requires grad.".format(n))

	def forward(self, reply_tensor_batch, contexts_tensor_batch, pad_matrix_batch, ini_idx):
		encoder_hidden = Variable(torch.zeros(1,self.batch_size,self.hidden_size)).cuda()
		list_context_hidden = []
		list_decoder_output = []

		for context_tensor in contexts_tensor_batch:
			context_embedded = self.embedding(Variable(context_tensor).cuda())
			context_hidden = self.encoder(self.dropout(context_embedded),encoder_hidden)
			list_context_hidden.append(context_hidden)

		attn_weights = self.attention(list_context_hidden,list_context_hidden[-1],pad_matrix_batch)
		context_hiddens = torch.cat(tuple(list_context_hidden),0).transpose(0,1)
		context_attn = torch.bmm(attn_weights.unsqueeze(1), context_hiddens)

		decoder_input = torch.cuda.LongTensor([ini_idx]*self.batch_size)
		decoder_hidden = self.dropout(list_context_hidden[-1])

		for reply_tensor in reply_tensor_batch:
			decoder_embedded = self.embedding(Variable(decoder_input)).unsqueeze(1)
			decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, context_attn)
			list_decoder_output.append(decoder_output)
			if self.teach_forcing:
				decoder_input = reply_tensor.cuda()
			else:
				top_values,top_indices = decoder_output.data.topk(1)
				decoder_input = top_indices.squeeze(1)

		return list_decoder_output

	def predict(self, contexts_tensor_batch, reply_index2word, pad_matrix_batch, ini_idx, sep_char=''):
		encoder_hidden = Variable(torch.zeros(1,self.batch_size,self.hidden_size),volatile=True).cuda()

		list_context_hidden = []
		predict_words_array = [['' for i in range(self.max_senten_len)] for i in range(self.batch_size)]
		predict_sentences = ["" for i in range(self.batch_size)]

		for context_tensor in contexts_tensor_batch:
			context_embedded = self.embedding(Variable(context_tensor,volatile=True).cuda())
			context_hidden = self.encoder(context_embedded,encoder_hidden)
			list_context_hidden.append(context_hidden)

		attn_weights = self.attention(list_context_hidden,list_context_hidden[-1],pad_matrix_batch)
		context_hiddens = torch.cat(tuple(list_context_hidden),0).transpose(0,1)
		context_attn = torch.bmm(attn_weights.unsqueeze(1), context_hiddens)

		decoder_input = torch.cuda.LongTensor([ini_idx]*self.batch_size)
		decoder_hidden = list_context_hidden[-1]

		for di in range(self.max_senten_len):
			decoder_embedded = self.embedding(Variable(decoder_input,volatile=True)).unsqueeze(1)
			decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, context_attn)
			
			decoder_output[:,-2:] = float("-inf")
			top_values,top_indices = decoder_output.data.topk(1)
			batch_topi = [top_indices[i][0] for i in range(self.batch_size)]
			for i in range(self.batch_size):
				predict_words_array[i][di] = reply_index2word[batch_topi[i]]
			decoder_input = top_indices.squeeze(1)

		for i in range(self.batch_size):
			predict_sentences[i] = sep_char.join(predict_words_array[i])
		return predict_sentences
