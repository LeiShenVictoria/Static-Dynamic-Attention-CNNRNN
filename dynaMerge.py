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
        super(RNNEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)

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

    def forward(self, hidden, list_context_hidden, context_hiddens, pad_matrix):
        attn_hidden = torch.mm(hidden.squeeze(0), self.W)
        list_weight = []
        for context_hidden in list_context_hidden:
            weight = torch.mm(self.tanh(attn_hidden + torch.mm(context_hidden.squeeze(0), self.U)), self.v)
            list_weight.append(weight)
        attn_weights = torch.cat(tuple(list_weight), 1).masked_fill_(Variable(pad_matrix.cuda()), -100000)
        attn_weights = self.softmax(attn_weights)
        print("********attn_weights**********")
        print(attn_weights)
        print("\n\n\n")
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), context_hiddens)
        return attn_applied
        # return c_t

class MergeGate(nn.Module):
    def __init__(self, hidden_size):
        super(MergeGate, self).__init__()
        self.hidden_size = hidden_size
        self.WSh = nn.Linear(hidden_size, hidden_size)
        self.WSc = nn.Linear(hidden_size, hidden_size)
        self.WSr = nn.Linear(hidden_size, hidden_size)
        self.wS = nn.Linear(hidden_size, 1)

    def forward(self, attn_applied_c, attn_applied_r, hidden):
        content_c = self.WSc(attn_applied_c) + self.WSh(hidden.transpose(0,1))
        # print("hidden:")
        #print(hidden.size())->(1,80,1024)
        #print("content_c:")
        #print(content_c.size())->(80,1,1024)
        #print("WSc,attn_applied_c:")
        #print(self.WSc(attn_applied_c).size())->(80,1,1024)
        #print("attn_applied_c:")
        #print(attn_applied_c.size())->(80,1,1024)
        score_c = self.wS(F.tanh(content_c))
        #print("score_c")->(80,1,1)
        #print(score_c.size())

        content_r = self.WSr(attn_applied_r) + self.WSh(hidden.transpose(0,1))
        score_r = self.wS(F.tanh(content_r))
        #print("score_r")->(80,1,1)
        #print(score_r.size())

        gama_t = F.sigmoid(score_c - score_r)
        c_t = gama_t * attn_applied_c + (1 - gama_t) * attn_applied_r
        #print("c_t:")
        #print(c_t.size()) -> (80,1,1024)       
        return c_t


# class Decoder(nn.Module):
#     def __init__(self, decoder_dict_size, hidden_size, emb_size):
#         super(Decoder, self).__init__()
#         self.decoder_dict_size = decoder_dict_size
#         self.hidden_size = hidden_size
#         self.emb_size = emb_size
#
#         self.W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
#         self.U = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
#         self.v = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
#
#         self.gru = nn.GRU(self.hidden_size + self.emb_size, self.hidden_size, batch_first=True)
#         self.out = nn.Linear(self.hidden_size, self.decoder_dict_size)
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax(dim=1)
#         self.log_softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden, list_context_hidden, context_hiddens, pad_matrix):
#         attn_hidden = torch.mm(hidden.squeeze(0), self.W)
#         list_weight = []
#         for context_hidden in list_context_hidden:
#             weight = torch.mm(self.tanh(attn_hidden + torch.mm(context_hidden.squeeze(0), self.U)), self.v)
#             list_weight.append(weight)
#         attn_weights = torch.cat(tuple(list_weight), 1).masked_fill_(Variable(pad_matrix.cuda()), -100000)
#         attn_weights = self.softmax(attn_weights)
#
#         attn_applied = torch.bmm(attn_weights.unsqueeze(1), context_hiddens)
#
#         input_combine = torch.cat((input, attn_applied), 2)
#         output, hidden = self.gru(input_combine, hidden)
#
#         output = self.log_softmax(self.out(output.squeeze(1)))
#         return output, hidden


class Decoder(nn.Module):
    def __init__(self, decoder_dict_size, hidden_size, emb_size):
        super(Decoder, self).__init__()
        self.decoder_dict_size = decoder_dict_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        self.attention = Attention(self.hidden_size)
        self.merge = MergeGate(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.emb_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.decoder_dict_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, list_context_hidden_cnn, list_context_hidden_rnn, context_hiddens_cnn, context_hiddens_rnn, pad_matrix):
        attn_applied_cnn = self.attention(hidden, list_context_hidden_cnn, context_hiddens_cnn, pad_matrix)
        print("************cnn*********************")

        attn_applied_rnn = self.attention(hidden, list_context_hidden_rnn, context_hiddens_rnn, pad_matrix)
        print("**************rnn*******************")
        #print("***********attn_applied_cnn****************")
        #print(attn_applied_cnn)
        #print("***********attn_applied_rnn****************")
        #print(attn_applied_rnn)
        #print("attn_applied_cnn:")->[80,1,1024]
        #print(attn_applied_cnn.size())
        c_t = self.merge(attn_applied_cnn, attn_applied_rnn, hidden)
        input_combine = torch.cat((input, c_t), 2)
        output, hidden = self.gru(input_combine, hidden)
        output = self.log_softmax(self.out(output.squeeze(1)))
        return output, hidden

class DynamicModel(nn.Module):
    def __init__(self, dict_size, emb_size, hidden_size, batch_size, dropout, max_senten_len, teach_forcing, kernel_num, kernel_sizes):
        super(DynamicModel, self).__init__()
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
        self.attention = Attention(self.hidden_size)
        self.merge = MergeGate(self.hidden_size)
        self.decoder = Decoder(self.dict_size, self.hidden_size, self.emb_size)

        self.embedding = nn.Embedding(self.dict_size, self.emb_size)

        self.dropout = nn.Dropout(self.dropout)


    def init_parameters(self, emb_matrix):
        n = 0
        for name, weight in self.named_parameters():
            if weight.requires_grad:
                if "embedding.weight" in name:
                    weight = nn.Parameter(emb_matrix)
                elif weight.ndimension() < 2:
                    weight.data.fill_(0)
                else:
                    weight.data = init.xavier_uniform(weight.data)
                n += 1
        print ("{} weights requires grad.".format(n))

    def forward(self, reply_tensor_batch, contexts_tensor_batch, pad_matrix_batch, ini_idx):
        encoder_hidden = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda()
        list_context_hidden_cnn = []
        list_context_hidden_rnn = []
        list_decoder_output = []


        for context_tensor in contexts_tensor_batch:
            context_embedded = self.embedding(Variable(context_tensor).cuda())
            context_hidden_cnn = self.cnnencoder(self.dropout(context_embedded)).unsqueeze(0)
            context_hidden_rnn = self.rnnencoder(self.dropout(context_embedded), encoder_hidden)

            list_context_hidden_cnn.append(context_hidden_cnn)
            list_context_hidden_rnn.append(context_hidden_rnn)

        context_hiddens_cnn = torch.cat(tuple(list_context_hidden_cnn), 0).transpose(0, 1)
        context_hiddens_rnn = torch.cat(tuple(list_context_hidden_rnn), 0).transpose(0, 1)

        decoder_input = torch.LongTensor([ini_idx] * self.batch_size).cuda()
        decoder_hidden = self.dropout(list_context_hidden_rnn[-1])

        for reply_tensor in reply_tensor_batch:
            decoder_embedded = self.embedding(Variable(decoder_input)).unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, list_context_hidden_cnn, list_context_hidden_rnn,
                                                          context_hiddens_cnn, context_hiddens_rnn, pad_matrix_batch)
            list_decoder_output.append(decoder_output)
            decoder_input = reply_tensor.cuda()

        return list_decoder_output

    def predict(self, contexts_tensor_batch, reply_index2word, pad_matrix_batch, ini_idx, sep_char=''):
        encoder_hidden = Variable(torch.zeros(1, self.batch_size, self.hidden_size), volatile=True).cuda()

        list_context_hidden_cnn = []
        list_context_hidden_rnn = []
        predict_words_array = [['' for i in range(self.max_senten_len)] for i in range(self.batch_size)]
        predict_sentences = ["" for i in range(self.batch_size)]

        for context_tensor in contexts_tensor_batch:
            context_embedded = self.embedding(Variable(context_tensor, volatile=True).cuda())
            context_hidden_cnn = self.cnnencoder(context_embedded).unsqueeze(0)
            context_hidden_rnn = self.rnnencoder(context_embedded, encoder_hidden)
            list_context_hidden_cnn.append(context_hidden_cnn)
            list_context_hidden_rnn.append(context_hidden_rnn)

        context_hiddens_cnn = torch.cat(tuple(list_context_hidden_cnn), 0).transpose(0, 1) # seq_len, batch_size, hidden_size
        context_hiddens_rnn = torch.cat(tuple(list_context_hidden_rnn), 0).transpose(0, 1)

        decoder_input = torch.LongTensor([ini_idx] * self.batch_size).cuda()
        decoder_hidden = list_context_hidden_rnn[-1]

        for di in range(self.max_senten_len):
            decoder_embedded = self.embedding(Variable(decoder_input, volatile=True)).unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, list_context_hidden_cnn, list_context_hidden_rnn,
                                                          context_hiddens_cnn, context_hiddens_rnn, pad_matrix_batch)

            decoder_output[:, -2:] = float("-inf")
            top_values, top_indices = decoder_output.data.topk(1)
            batch_topi = [top_indices[i][0] for i in range(self.batch_size)]
            for i in range(self.batch_size):
                predict_words_array[i][di] = reply_index2word[batch_topi[i]]
            decoder_input = top_indices.squeeze(1)

        for i in range(self.batch_size):
            predict_sentences[i] = sep_char.join(predict_words_array[i])
        return predict_sentences
