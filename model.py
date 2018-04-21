import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import to_variable, to_tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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

    def forward(self, input, input_len, label=None, label_len=None):
        input = input.permute(1, 0, 2)
        #input = self.cnn_encoder(input)
        keys, values = self.encoder(input, input_len)
        if label is None:
            # During decoding of test data
            return self.decoder.decode(keys, values)
        else:
            # During training
            return self.decoder(keys, values, label, label_len, input_len)


class CNN_Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=40, out_channels=128, kernel_size=3, padding=1),  # Mini-batch * 128 * len
            nn.Hardtanh(0, 20, inplace=True),
            #nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # Mini-batch * 256 * len
            #nn.BatchNorm1d(256),
            nn.Hardtanh(0, 20, inplace=True),
            #nn.LeakyReLU(),
            nn.Dropout(0.3),
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
                    h = h.view(h.size(0), h.size(1) // 2, 2, h.size(2)).sum(2) / 2
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
        self.hidden_size = params.hidden_dimension
        self.embedding_dim = params.embedding_dimension
        self.is_stochastic = params.is_stochastic
        self.max_decoding_length = params.max_decoding_length
        self.embed = nn.Embedding(num_embeddings=output_size, embedding_dim=self.hidden_size)
        self.lstm_cells = nn.ModuleList([
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=self.hidden_size)
        ])

        # For attention
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        # For character projection
        self.projection_layer1 = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.non_linear = nn.LeakyReLU()
        self.projection_layer2 = nn.Linear(in_features=self.hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # Tying weights of last layer and embedding layer
        #self.projection_layer2.weight = self.embed.weight

    def forward(self, keys, values, label, label_len, input_len):
        # Number of characters in the transcript
        embed = self.embed(label)          # bs * label_len * 256
        output = None
        hidden_states = []
        # Initial context
        query = self.linear(self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous())  # bs * 256, This is the query
        attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8

        # Create mask
        a = torch.arange(input_len[0]).unsqueeze(0).expand(len(input_len), -1)
        b = input_len.unsqueeze(1).float()
        mask = a < b
        if torch.cuda.is_available():
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor)).cuda()
        else:
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor))
        #attn.data.masked_fill_((1 - mask).unsqueeze(1), -float('inf'))
        attn = F.softmax(attn, dim=2)
        attn = attn * mask
        attn = attn / attn.sum(2).unsqueeze(2)
        context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)

        for i in range(label_len.max() - 1):
            h = embed[:, i, :]                                  # bs * 256
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
            
            query = self.linear(h)              # bs * 2048, This is the query
            attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))         # bs * 1 * seq_len/8
            #attn.data.masked_fill_((1 - mask).unsqueeze(1), -float('inf'))
            attn = F.softmax(attn, dim=2)
            attn = attn * mask
            attn = attn / attn.sum(2).unsqueeze(2)
            context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)       # bs * 256
            h = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
            h = self.projection_layer1(h)
            h = self.non_linear(h)
            h = self.softmax(self.projection_layer2(h))
            # Accumulating the output at each timestep
            if output is None:
                output = h.unsqueeze(1)
            else:
                output = torch.cat((output, h.unsqueeze(1)), dim=1)
        return output                         # bs * max_label_seq_len * 33
    
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

        for _ in range(100):
            hidden_states = []
            raw_pred = None
            raw_out = []
            # Initial context
            query = self.linear(self.lstm_cells[2].h0)  # bs * 256, This is the query
            attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8
            attn = F.softmax(attn, dim=2)
            context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)

            h = self.embed(to_variable(torch.zeros(bs).long()))  # Start token provided for generating the sentence
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

                query = self.linear(h)  # bs * 2048, This is the query
                attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8
                # attn.data.masked_fill_((1 - mask).unsqueeze(1), -float('inf'))
                attn = F.softmax(attn, dim=2)
                context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)  # bs * 256
                h = torch.cat((h, context), dim=1)

                # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
                h = self.projection_layer1(h)
                h = self.non_linear(h)
                h = self.projection_layer2(h)
                lsm = self.softmax(h)
                if self.is_stochastic > 0:
                    gumbel = torch.autograd.Variable(self.sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel
                # TODO: Do beam search later

                h = torch.max(h, dim=1)[1]
                raw_out.append(h.data.cpu().numpy()[0])
                if raw_pred is None:
                    raw_pred = lsm
                else:
                    raw_pred = torch.cat((raw_pred, lsm), dim=0)

                if h.data.cpu().numpy() == 0:
                    break

                # Primer for next character generation
                h = self.embed(h)
            output.append(raw_out)
            raw_preds.append(raw_pred)
        return output, raw_preds


