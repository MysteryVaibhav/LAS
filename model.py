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
        self.encoder = Encoder(params)      #pBilstm
        self.decoder = Decoder(params, output_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        # TODO: Complete this after done implemeting the model
        #self.encoder.weight.data.uniform_(-init_range, init_range)
        #self.decoder.bias.data.fill_(0)
        #self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, input_len, label=None, label_len=None):
        keys, values = self.encoder(input, input_len)
        if label is None:
            # During decoding of test data
            return self.decoder.decode(keys, values)
        else:
            # During training
            return self.decoder(keys, values, label, label_len)


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
            nn.LSTM(input_size=2 * params.hidden_dimension, hidden_size=2 * params.hidden_dimension, bidirectional=True),
            nn.LSTM(input_size=4 * params.hidden_dimension, hidden_size=4 * params.hidden_dimension, bidirectional=True),
            nn.LSTM(input_size=8 * params.hidden_dimension, hidden_size=8 * params.hidden_dimension, bidirectional=True)])
        self.linear_key = nn.Linear(in_features=8 * params.hidden_dimension, out_features=params.hidden_dimension)
        self.linear_values = nn.Linear(in_features=8 * params.hidden_dimension, out_features=params.hidden_dimension)

    def forward(self, input, input_len):
        # TODO:Add some CNN layers later
        h = input.transpose(0, 1)    # seq_len * bs * 40
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                # After first lstm layer, pBiLSTM
                seq_len = h.size(0)
                if seq_len % 2 == 0:
                    even_seq = to_variable(to_tensor(np.arange(0, seq_len, 2)).long())
                    odd_seq = to_variable(to_tensor(np.arange(1, seq_len + 1, 2)).long())
                    h_even = torch.index_select(h, dim=0, index=even_seq)
                    h_odd = torch.index_select(h, dim=0, index=odd_seq)
                    h = torch.cat((h_even, h_odd), dim=2)       # seq_len/2 * bs * (2^n * hidden_dim)
                    input_len /= 2
                else:
                    print("Odd seq len should not occur!!")
                    exit()
            # First BiLSTM
            packed_h = pack_padded_sequence(h, input_len.cpu().numpy())
            h, _ = lstm(packed_h)
            h, _ = pad_packed_sequence(h)      # seq_len * bs * (2 * hidden_dim)
            # Summing forward and backward representation
            h = h.view(h.size(0), h.size(1), 2, -1).sum(2).view(h.size(0), h.size(1), -1) / 2       # h = ( h_forward + h_backward ) / 2

        h = h.transpose(0, 1)               # bs * seq_len/8 * 2048
        keys = self.linear_key(h)           # bs * seq_len/8 * 256
        values = self.linear_values(h)      # bs * seq_len/8 * 256
        return keys, values


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
            nn.LSTMCell(input_size=2 * self.hidden_size, hidden_size=self.hidden_size),
            nn.LSTMCell(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            nn.LSTMCell(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size)
        ])

        # For attention
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        # For character projection
        self.projection_layer1 = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.projection_layer2 = nn.Linear(in_features=self.hidden_size, out_features=output_size)

        # Tying weights of last layer and embedding layer
        self.projection_layer2.weight = self.embed.weight

    def init_hidden(self, batch_size):
        return [(to_variable(torch.zeros(batch_size, self.hidden_size)),
                 to_variable(torch.zeros(batch_size, self.hidden_size))),
                (to_variable(torch.zeros(batch_size, 2 * self.hidden_size)),
                 to_variable(torch.zeros(batch_size, 2 * self.hidden_size))),
                (to_variable(torch.zeros(batch_size, 2 * self.hidden_size)),
                 to_variable(torch.zeros(batch_size, 2 * self.hidden_size)))]

    def forward(self, keys, values, label, label_len):
        # Number of characters in the transcript
        embed = self.embed(label)          # bs * label_len * 256
        output = to_variable(torch.zeros(label_len.max() - 1, embed.size(0), self.vocab))
        hidden = self.init_hidden(embed.size(0))
        context = to_variable(torch.zeros(embed.size(0), self.hidden_size), requires_grad=True)             # Initial context
        for i in range(label_len.max() - 1):
            h = embed[:, i, :]                 # bs * 256
            h = torch.cat((h, context), dim=1)  # bs * 512
            for j, lstm in enumerate(self.lstm_cells):

                # First LSTMCell
                if j == 0:
                    hidden[j] = lstm(h, hidden[j])       # bs * 256
                    h = hidden[j][0]
                    # At this point, we get the decoded values at each step :  bs * 256
                    query = self.linear(h)              # bs * 2048, This is the query
                    attn = torch.bmm(query.unsqueeze(1), keys.permute(0, 2, 1))         # bs * 1 * seq_len/8
                    attn = F.softmax(attn, dim=2)
                    context = torch.bmm(attn, values).squeeze(1)       # bs * 256
                    h = torch.cat((h, context), dim=1)

                # Remaining 2 LSTMCells
                else:
                    hidden[j] = lstm(h, hidden[j])       # bs * 512
                    h = hidden[j][0]

            # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
            h = self.projection_layer1(h)
            h = self.projection_layer2(h)

            # Accumulating the output at each timestep
            output[i] = h
        return output.permute(1, 0, 2)                  # bs * max_label_seq_len * 33

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
        bs = 1          # batch_size for decoding
        hidden = self.init_hidden(bs)
        output = []
        context = to_variable(torch.zeros(bs, self.hidden_size))
        h = self.embed(to_variable(torch.LongTensor([0])))      # Start token provided for generating the sentence
        for i in range(self.max_decoding_length):
            h = torch.cat((h, context), dim=1)
            for j, lstm in enumerate(self.lstm_cells):

                # First LSTMCell
                if j == 0:
                    hidden[j] = lstm(h, hidden[j])       # bs * 256
                    h = hidden[j][0]
                    # At this point, we get the decoded values at each step :  bs * 256
                    query = self.linear(h)              # bs * 2048, This is the query
                    attn = torch.bmm(query.unsqueeze(1), keys.permute(0, 2, 1))         # bs * 1 * seq_len/8
                    attn = F.softmax(attn, dim=2)
                    context = torch.bmm(attn, values).squeeze(1)       # bs * 256
                    h = torch.cat((h, context), dim=1)

                # Remaining 2 LSTMCells
                else:
                    hidden[j] = lstm(h, hidden[j])       # bs * 512
                    h = hidden[j][0]

            # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
            h = self.projection_layer1(h)
            h = self.projection_layer2(h)

            if self.is_stochastic > 0:
                gumbel = torch.autograd.Variable(self.sample_gumbel(shape=h.size(), out=h.data.new()))
                h += gumbel
            # TODO: Do beam search later
            h = torch.max(h, dim=1)[1]
            if h.data.cpu().numpy() == 0:
                break
            output.append(h.data.cpu().numpy()[0])

            # Primer for next character generation
            h = self.embed(h)
        return output

