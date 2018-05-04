import torch
import torch.nn as nn
from utils import to_variable


class CharRNN(nn.Module):
    def __init__(self, hidden_size, embedding_dim, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(output_size, embedding_dim)
        nn.init.xavier_uniform(self.encoder.weight)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        return to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)), \
               to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

    def forward(self, input, hidden):
        self.batch_size = input.size()[0]
        encoded = self.encoder(input)
        encoded = encoded.permute(1, 0, 2).contiguous()
        output, hidden = self.rnn(encoded, hidden)
        output = output.permute(1, 0, 2).contiguous()
        # no attention added, picking the last lstm output
        output = self.decoder(output)
        return output, hidden
