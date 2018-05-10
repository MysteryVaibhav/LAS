import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import to_variable, to_tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from char_rnn import CharRNN
from beam_search import beamsearch_v2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, dropout=0.65):
        if not self.training:
            return inp
        tensor_mask = inp.data.new(1, inp.size(1), inp.size(2)).bernoulli_(1 - dropout)
        var_mask = torch.autograd.Variable(tensor_mask, requires_grad=False) / (1 - dropout)
        var_mask = var_mask.expand_as(inp)
        return var_mask * inp


class LAS(nn.Module):
    def __init__(self, params, output_size, max_seq_len):
        super(LAS, self).__init__()
        self.hidden_size = params.hidden_dimension
        self.embedding_dim = params.embedding_dimension
        self.n_layers = params.n_layers
        self.max_seq_len = max_seq_len
        self.output_size = output_size
        self.drop_out = nn.Dropout(0.4)
        self.cnn_encoder = CNN_Encoder(params.embedding_dimension)
        self.encoder = Encoder(params)      #pBilstm
        self.decoder = Decoder(params, output_size)
        if params.use_lm == 1:
            self.lm = CharRNN(params.hidden_dimension, 128, output_size + 1, n_layers=3)
            self.lm.load_state_dict(torch.load('models/bestModel_4.37.t7'))
            self.decoder.lm = self.lm

    def forward(self, input, input_len, label=None, label_len=None, tf_rate=None):
        #input = self.cnn_encoder(input)
        input = input.permute(1, 0, 2)
        keys, values = self.encoder(input, input_len)
        if label is None:
            # During decoding of test data
            return self.decoder.decode(keys, values)
        else:
            # During training
            return self.decoder(keys, values, label, label_len, input_len, tf_rate)


class CNN_Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=40, out_channels=128, kernel_size=3, padding=1),  # Mini-batch * 128 * len
            nn.Hardtanh(0, 20, inplace=True),
            #nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # Mini-batch * 256 * len
            #nn.BatchNorm1d(256),
            nn.Hardtanh(0, 20, inplace=True),
            #nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=256, out_channels=embedding_dim, kernel_size=3, padding=1),  # Mini-batch * 256 * len
        ])

    def forward(self, input):
        h = input.permute(0, 2, 1)  # batch_size * C * length
        for layer in self.layers:
            h = layer(h)
        return h.permute(0, 2, 1)   # reshaping it back


class Encoder(nn.Module):
    """
    pBILSTM model which encodes the sequence and returns key and value encodings
    """

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.hidden_size = params.hidden_dimension
        self.embedding_dim = params.embedding_dimension
        #self.locked_dropout = LockedDropout()
        self.n_layers = params.n_layers
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=params.embedding_dimension, hidden_size=params.hidden_dimension, bidirectional=True),
            nn.LSTM(input_size=params.hidden_dimension, hidden_size=params.hidden_dimension, bidirectional=True),
            nn.LSTM(input_size=params.hidden_dimension, hidden_size=params.hidden_dimension, bidirectional=True),
            nn.LSTM(input_size=params.hidden_dimension, hidden_size=params.hidden_dimension, bidirectional=True)])
        self.linear_key = nn.Linear(in_features=params.hidden_dimension, out_features=params.hidden_dimension)
        self.linear_values = nn.Linear(in_features=params.hidden_dimension, out_features=params.hidden_dimension)

    def forward(self, input, input_len):
        # TODO:Add some CNN layers later
        h = input
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                # After first lstm layer, pBiLSTM
                seq_len = h.size(0)
                if seq_len % 2 == 0:
                    h = h.permute(1, 0, 2).contiguous()
                    # Average pooling
                    h = h.view(h.size(0), h.size(1) // 2, 2, h.size(2)).sum(2) / 2
                    # Concat pooling
                    #h = h.view(h.size(0), h.size(1) // 2, h.size(2) * 2)
                    h = h.permute(1, 0, 2).contiguous()
                    input_len /= 2
                else:
                    print("Odd seq len should not occur!!")
                    exit()
            # First BiLSTM
            packed_h = pack_padded_sequence(h, input_len.cpu().numpy())
            h, _ = lstm(packed_h)
            h, _ = pad_packed_sequence(h)      # seq_len * bs * (2 * hidden_dim)
            #h = self.locked_dropout(h, 0.3)
            # Summing forward and backward representation
            h = h.view(h.size(0), h.size(1), 2, -1).sum(2) / 2       # h = ( h_forward + h_backward ) / 2
        keys = self.linear_key(h)           # bs * seq_len/8 * 256
        values = self.linear_values(h)      # bs * seq_len/8 * 256
        return keys, values

    
class MyLSTM(nn.LSTMCell):
    def __init__(self, input_size, hidden_size):
        super(MyLSTM, self).__init__(input_size, hidden_size)
        self.h0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)

    def forward(self, h, hx, cx):
        return super(MyLSTM, self).forward(h, (hx, cx))
    
    
class Decoder(nn.Module):

    def __init__(self, params, output_size):
        super(Decoder, self).__init__()
        self.vocab = output_size
        self.use_tf = params.use_tf
        self.use_lm = params.use_lm
        self.use_multi_head_attn = params.use_multi_head_attn
        self.num_tries = params.num_tries
        self.hidden_size = params.hidden_dimension
        self.embedding_dim = params.embedding_dimension
        self.is_stochastic = params.is_stochastic
        self.use_beam_decode = params.use_beam_decode
        self.plot = params.plot
        #self.locked_dropout = LockedDropout()
        self.collect_image = True
        self.max_decoding_length = params.max_decoding_length
        self.embed = nn.Embedding(num_embeddings=output_size, embedding_dim=self.hidden_size)
        self.lstm_cells = nn.ModuleList([
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=self.hidden_size)
        ])

        # For attention
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        if self.use_multi_head_attn == 1:
            self.v_k_q_1 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.v_k_q_2 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.v_k_q_3 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.v_k_q_4 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                 nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.linears = nn.ModuleList([self.v_k_q_1,
                                         self.v_k_q_2,
                                         self.v_k_q_3,
                                         self.v_k_q_4])
            self.multi_head_linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        # For character projection

        # For character projection
        self.projection_layer1 = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.non_linear = nn.LeakyReLU()
        self.projection_layer2 = nn.Linear(in_features=self.hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # Tying weights of last layer and embedding layer
        self.embed.weight = self.projection_layer2.weight# = self.embed.weight

    def forward(self, keys, values, label, label_len, input_len, teacher_force_rate):
        # Number of characters in the transcript
        embed = self.embed(label)          # bs * label_len * 256

        # Lm
        if self.use_lm == 1:
            hidden_lm = self.lm.init_hidden(label.size()[0])
            prediction_lm, _ = self.lm(label, hidden_lm)
            
        #embed = self.locked_dropout(embed, 0.1)
        output = None
        hidden_states = []

        # Create mask
        a = torch.arange(input_len[0]).unsqueeze(0).expand(len(input_len), -1)
        b = input_len.unsqueeze(1).float()
        mask = a < b
        if torch.cuda.is_available():
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor)).cuda()
        else:
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor))
        if self.plot == 1:
            context, attn_arr = self.get_context(self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous(), keys, values, mask, True)
        else:
            context = self.get_context(self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous(), keys, values, mask)
        if self.collect_image and self.plot == 1:
                atten_array = np.zeros((4, label_len.max(), keys.size()[0]))
                atten_array[0, 0, :] = attn_arr[0][0, 0, :]
                atten_array[1, 0, :] = attn_arr[1][0, 0, :]
                atten_array[2, 0, :] = attn_arr[2][0, 0, :]
                atten_array[3, 0, :] = attn_arr[3][0, 0, :]
            
        for i in range(label_len.max() - 1):
            if self.use_tf == 1:
                teacher_force = True
            else:
                teacher_force = True if np.random.random_sample() < teacher_force_rate else False
            if i==0 or teacher_force :
                h = embed[:, i, :]                                  # bs * 256
            else:
                h = self.embed(torch.max(h, dim=1)[1])
            h = torch.cat((h, context), dim=1)                  # bs * 512
            for j, lstm in enumerate(self.lstm_cells):
                if i == 0:
                    h_x_0, c_x_0 = lstm(h, lstm.h0.expand(embed.size(0), -1).contiguous(),
                                        lstm.c0.expand(embed.size(0), -1).contiguous())       # bs * 512
                    hidden_states.append((h_x_0, c_x_0))
                else:
                    h_x_0, c_x_0 = hidden_states[j]
                    hidden_states[j] = lstm(h, h_x_0, c_x_0)
                h = hidden_states[j][0]
            
            if self.plot == 1:
                context, attn_arr = self.get_context(self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous(), keys, values, mask, True)
            else:
                context = self.get_context(self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous(), keys, values, mask)
            if self.collect_image and self.plot:
                atten_array[0, i+1, :] = attn_arr[0][0, 0, :]
                atten_array[1, i+1, :] = attn_arr[1][0, 0, :]
                atten_array[2, i+1, :] = attn_arr[2][0, 0, :]
                atten_array[3, i+1, :] = attn_arr[3][0, 0, :]
            h = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
            h = self.projection_layer1(h)
            h = self.non_linear(h)
            h = self.projection_layer2(h)
            
            # if use LM
            if self.use_lm == 1:
                h = h*3 + prediction_lm[:, i, :-1]
            
            h = self.softmax(h)
            # Accumulating the output at each timestep
            if output is None:
                output = h.unsqueeze(1)
            else:
                output = torch.cat((output, h.unsqueeze(1)), dim=1)

        if self.collect_image and self.plot == 1:
            mat = np.matrix(atten_array[0])
            fig = plt.figure()
            plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
            plt.xlabel('Utterance length')
            plt.ylabel('Label')
            fig.savefig('sample_atten_1.png')
            
            mat = np.matrix(atten_array[1])
            fig = plt.figure()
            plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
            plt.xlabel('Utterance length')
            plt.ylabel('Label length')
            fig.savefig('sample_atten_2.png')
            
            mat = np.matrix(atten_array[2])
            fig = plt.figure()
            plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
            plt.xlabel('Utterance length')
            plt.ylabel('Label length')
            fig.savefig('sample_atten_3.png')
            
            mat = np.matrix(atten_array[3])
            fig = plt.figure()
            plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
            plt.xlabel('Utterance length')
            plt.ylabel('Label length')
            fig.savefig('sample_atten_4.png')
            self.collect = False
        return output                         # bs * max_label_seq_len * 33
    
    def get_context(self, h, keys, values, mask=None, get_attn=False):
        query = self.linear(h)  # bs * 2048, This is the query
        if get_attn:
            attn_array = []
        if self.use_multi_head_attn == 1:
            head = None
            for linear in self.linears:
                n_keys = linear[0](keys)
                n_query = linear[1](query)
                n_values = linear[2](values)
                attn = torch.bmm(n_query.unsqueeze(1), n_keys.permute(1, 2, 0)) * (1.0 / np.sqrt(self.hidden_size // 4))
                attn = F.softmax(attn, dim=2)
                if mask is not None:
                    attn = attn * mask
                    attn = attn / attn.sum(2).unsqueeze(2)
                if get_attn:
                    attn_array.append(attn)
                if head is None:
                    head = torch.bmm(attn, n_values.permute(1, 0, 2)).squeeze(1)
                else:
                    head = torch.cat((head, torch.bmm(attn, n_values.permute(1, 0, 2)).squeeze(1)), dim=1)
            context = self.multi_head_linear(head)
        else:
            attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8
            attn = F.softmax(attn, dim=2)
            if mask is not None:
                attn = attn * mask
                attn = attn / attn.sum(2).unsqueeze(2)
            context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)
        if get_attn:
            return context, attn_array
        return context
    
    
    def sample_gumbel(self, shape, eps=1e-10, out=None):
        """
        Sample from Gumbel(0, 1) based on (MIT license)
        """
        U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        return - torch.log(eps - torch.log(U + eps))

    def decode(self, keys, values):
        """
        :param keys:
        :param values:
        :return: Returns the best decoded sentence
        """
        bs = 1  # batch_size for decoding
        output = []
        raw_preds = []

        for _ in range(self.num_tries):
            hidden_states = []
            raw_pred = None
            raw_out = []
            # Initial context
            context = self.get_context(self.lstm_cells[2].h0, keys, values)
            h = self.embed(to_variable(torch.zeros(bs).long()))  # Start token provided for generating the sentence
            
            if self.use_lm == 1:
                hidden_lm = self.lm.init_hidden(bs)
                prediction_lm, _ = self.lm(to_variable((torch.ones(bs) * 66).long().unsqueeze(0)), hidden_lm)
                prediction_lm = prediction_lm.squeeze(0)
                
            for i in range(self.max_decoding_length):
                h = torch.cat((h, context), dim=1)
                for j, lstm in enumerate(self.lstm_cells):
                    if i == 0:
                        h_x_0, c_x_0 = lstm(h, lstm.h0,
                                            lstm.c0)  # bs * 512
                        hidden_states.append((h_x_0, c_x_0))
                    else:
                        h_x_0, c_x_0 = hidden_states[j]
                        hidden_states[j] = lstm(h, h_x_0, c_x_0)
                    h = hidden_states[j][0]

                context = self.get_context(h, keys, values)
                h = torch.cat((h, context), dim=1)

                # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
                h = self.projection_layer1(h)
                h = self.non_linear(h)
                h = self.projection_layer2(h)
                
                # if use lm
                if self.use_lm == 1:
                    h = h*3 + prediction_lm[:, :-1]
                    
                lsm = self.softmax(h)
                if self.is_stochastic > 0:
                    gumbel = torch.autograd.Variable(self.sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel
                # TODO: Do beam search later

                h_argmax = torch.max(h, dim=1)[1]
                raw_out.append(h_argmax.data.cpu().numpy()[0])
                if raw_pred is None:
                    raw_pred = lsm
                else:
                    raw_pred = torch.cat((raw_pred, lsm), dim=0)

                if h_argmax.data.cpu().numpy() == 0:
                    break

                # Primer for next character generation
                h = self.embed(h_argmax)
                if self.use_lm == 1:
                    prediction_lm, _ = self.lm(h_argmax.unsqueeze(0), hidden_lm)
                    prediction_lm = prediction_lm.squeeze(0)
            if self.use_beam_decode:
                #print(raw_pred.size())
                beam_search = beamsearch_v2(raw_pred)
                decoded = beam_search.decode()
                #print(decoded)
                #print(raw_out)
                raw_out = decoded
            output.append(raw_out)
            raw_preds.append(raw_pred)
        return output, raw_preds
    
