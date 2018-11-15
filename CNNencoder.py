import sys
import math
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes, dropout):
        super(CNNEncoder, self).__init__()
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.dropoutrate = dropout

        '''
        ModuleList: can be indexed like a regular Python list
        Conv2d: input size(N, C_in, H, W) and output size(N, C_out, H_out, W_out)
        para: in_channels, out_channels, kernel_sizes, stride=1)
        here, we build different CNN modules with kernels of different sizes, 345
        in total, we have 3(len(kernel_sizes)) * kernel_num filters
        kernel_num: the number of filters per filter size
         '''
        self.convs1 = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (K, self.embed_dim)) for K in self.Ks])
        self.dropout = nn.Dropout(self.dropoutrate)

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        '''
        return the representation of a sentence from the penultimate layer
        '''
        return x

class RNNEncoder(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(RNNEncoder,self).__init__()
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

        # Parameter is the subclass of Variable, when it's used with Module, it will be added to the parameters() iteration
        # of Module, and requires_grad is Ture, while requires_grad is False for narmal Variable
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
        self.gru = nn.GRU(self.hidden_size+self.emb_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.decoder_dict_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, context_attn):
        input_combine = torch.cat((input, context_attn), 2)
        output, hidden = self.gru(input_combine, hidden)

        output = self.log_softmax(self.out(output.squeeze(1)))
        return output, hidden

class StaticModel(nn.Module):
    def __init__(self, dict_size, emb_size, hidden_size, batch_size, dropout, max_senten_len, teach_forcing, kernel_num, kernel_sizes):
        super(StaticModel, self).__init__()
        self.dict_size = dict_size
        self.emb_size = emb_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.max_senten_len = max_senten_len
        self.teach_forcing = teach_forcing

        self.cnnencoder = CNNEncoder(self.dict_size, self.emb_size, self.kernel_num, self.kernel_sizes, self.dropout)
        self.rnnencoder = RNNEncoder(self.emb_size, self.hidden_size)
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
            #print("context_tensor")
            #print(context_tensor.size())
            context_embedded = self.embedding(Variable(context_tensor).cuda())
            #print("context_embedded")
            #print(context_embedded.size())
            context_hidden_cnn = self.cnnencoder(self.dropout(context_embedded)).unsqueeze(0)
            context_hidden_rnn = self.rnnencoder(self.dropout(context_embedded),encoder_hidden)
            #print(context_hidden.size()) #80*1024
            context_hidden = torch.cat((context_hidden_cnn, context_hidden_rnn), 2)
            list_context_hidden.append(context_hidden)
        #print("list_context_hidden")
        #print(len(list_context_hidden))

        attn_weights = self.attention(list_context_hidden,list_context_hidden[-1],pad_matrix_batch)
        #print("attn_weights")
        #print(attn_weights.size())
        context_hiddens = torch.cat(tuple(list_context_hidden),0).transpose(0,1)
        #print("context_hiddens")
        #print(context_hiddens.size())
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
            context_hidden_cnn = self.cnnencoder(context_embedded).unsqueeze(0)
            context_hidden_rnn = self.rnnencoder(self.dropout(context_embedded),encoder_hidden)
            context_hidden = torch.cat((context_hidden_cnn, context_hidden_rnn), 2)
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
