import torch
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

print_flag = 0
teacher_forcing = 1


def CreateOnehotVariable(input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data
    input_type = torch.torch.cuda.FloatTensor  # type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1, input_x, 1)).type(
        input_type)
    return onehot_x


def TestCreateOnehotVariable(input_x, encoding_dim=63):
    begin = np.zeros((1, 1, encoding_dim), dtype='float32')
    onehot_x = Variable(torch.from_numpy(begin)).cuda()
    return onehot_x


# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, rnn_unit='LSTM', dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim * 2, hidden_dim, 1, bidirectional=True,
                                   dropout=dropout_rate, batch_first=True)

    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        # print("THe number o time steps is", timestep)
        input_x = input_x.contiguous().view(batch_size, int(timestep / 2), feature_dim * 2)
        # Bidirectional RNN
        output, hidden = self.BLSTM(input_x)
        return output, hidden


class speller(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, max_label_length=100):
        super(speller, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.target_size = target_size
        self.float_type = torch.torch.cuda.FloatTensor
        self.max_label_length = max_label_length
        self.rnn_layer = nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim * 2, num_layers=3)
        self.attention = Attention(mlp_preprocess_input=True, preprocess_mlp_dim=128,
                                   activate='relu', input_feature_dim=hidden_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.character_distribution = nn.Linear(hidden_dim * 4, target_size)
        self.embedding_layer = nn.Embedding(target_size, embedding_dim, sparse=False)

    def fwd_step(self, input_word, last_hidden_state, listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word, last_hidden_state)
        attention_score, context = self.attention(rnn_output, listener_feature)
        # print("Returned from Attention")
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        # print("Shape of concat feature", concat_feature.shape)
        raw_pred = self.softmax(self.character_distribution(concat_feature))
        return raw_pred, hidden_state, context, attention_score

    def test(self, context_tensor):

        seed_word = TestCreateOnehotVariable(torch.torch.cuda.FloatTensor, self.target_size)
        rnn_input = torch.cat([seed_word[:, 0:1, :], context_tensor[:, 0:1, :]], dim=-1)
        hidden_state = None
        attention_tensor = []
        prediction_tensor = []
        while True:
            probs, hidden_state, context, attention = self.fwd_step(rnn_input, hidden_state, context_tensor)
            attention_tensor.append(attention)
            # print("The prediction was ",probs.shape)
            prediction = probs.unsqueeze(1)
            # values, idx = torch.max(prediction) 
            # print("The prediction was ",prediction.shape)
            # if int(prediction_numpy)  == 1:
            #     break  
            prediction_tensor.append(probs)
            rnn_input = torch.cat([prediction, context_tensor[:, 0:1, :]], dim=-1)
        prediction_tensor = torch.stack(prediction_tensor)
        return prediction_tensor

    def forward(self, context_tensor, ground_truth, tf_rate):

        teacher_forcing_value = tf_rate
        gt = ground_truth
        teacher_forcing = 1 if np.random.random_sample() < teacher_forcing_value else 0

        ground_truth = self.embedding_layer(ground_truth.long())
        # print("Shape of embedded ground truth is", ground_truth.shape)
        seed_word = CreateOnehotVariable(self.float_type(gt.data.float()), self.target_size)
        rnn_input = torch.cat([seed_word[:, 0:1, :], context_tensor[:, 0:1, :]], dim=-1)

        if print_flag == 1:
            print("  Shape of context matrix is ", context_tensor.shape)
            print("  Shape of seed word is ", seed_word.shape)
            print("  Shape of RNN Input is ", rnn_input.shape)

        max_len = ground_truth.shape[1]
        # print("I will predict ", max_len, "timesteps for each batch")
        hidden_state = None
        attention_tensor = []
        prediction_tensor = []
        for i in range(max_len - 1):
            probs, hidden_state, context, attention = self.fwd_step(rnn_input, hidden_state, context_tensor)
            if print_flag == 1:
                print("  Currently in timestep", i + 1)
                print("  Shape of attention tensor: ", attention.shape)
                print("  Shape of probs is ", probs.shape)
                print("  Shape of context: ", context.shape)
            attention_tensor.append(attention)
            prediction = probs.unsqueeze(1)
            prediction_tensor.append(probs)
            if print_flag == 1:
                print("  Shape of prediction: ", prediction.shape)
                print("  Shape of context tensor: ", context_tensor.shape)

            if teacher_forcing == 1:
                rnn_input = torch.cat([ground_truth[:, i + 1:i + 2, :].type(self.float_type), context.unsqueeze(1)],
                                      dim=-1)
            else:
                rnn_input = torch.cat([prediction, context.unsqueeze(1)], dim=-1)
            if print_flag == 1:
                print("  Time step done", i + 1)
                print('\n')

        prediction_tensor = torch.stack(prediction_tensor)
        prediction_tensor = prediction_tensor.transpose(0, 1)
        # print("Returing this shape from speller: ", prediction_tensor.shape)
        return prediction_tensor


class Attention(nn.Module):
    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, input_feature_dim=512):
        super(Attention, self).__init__()
        self.mlp_preprocess_input = False  # mlp_preprocess_input
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim, preprocess_mlp_dim)
            self.psi = nn.Linear(input_feature_dim, preprocess_mlp_dim)
            self.activate = getattr(F, activate)

    def forward(self, decoder_state, listener_feature):
        comp_decoder_state = decoder_state
        comp_listener_feature = listener_feature.transpose(1, 2)
        energy = torch.bmm(comp_decoder_state, comp_listener_feature).squeeze(dim=1)
        attention_score = self.softmax(energy)
        context = torch.sum(listener_feature * attention_score.unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                            dim=1)

        return attention_score, context


class listener(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(listener, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, int(hidden_dim / 2), 1, batch_first=True, bidirectional=True)
        self.pLSTM_layer1 = pBLSTMLayer(hidden_dim, hidden_dim, rnn_unit='lstm', dropout_rate=0.2)
        self.pLSTM_layer2 = pBLSTMLayer(hidden_dim * 2, hidden_dim, rnn_unit='lstm', dropout_rate=0.2)
        self.pLSTM_layer3 = pBLSTMLayer(hidden_dim * 2, hidden_dim, rnn_unit='lstm', dropout_rate=0.2)

    def forward(self, utterance_batch):
        batch_size = utterance_batch.size()[0]
        time_steps = utterance_batch.size()[1]
        feature_dim = utterance_batch.size()[2]
        x_input = utterance_batch.contiguous()
        lstm_out, self.hidden = self.lstm(x_input, None)

        # First pyramidal LSTM
        output, _ = self.pLSTM_layer1(lstm_out)
        timesteps = output.size()[1]
        if timesteps % 2 == 1:
            output = output[:, :timesteps - 1:]

        # Second Pyramidal LSTM
        output, _ = self.pLSTM_layer2(output)
        timesteps = output.size()[1]
        if timesteps % 2 == 1:
            output = output[:, :timesteps - 1:]

        # Third Pyramidal LSTM
        output, _ = self.pLSTM_layer3(output)
        # print("Output from pyramidal LSTM", output.shape)
        return output


'''    
i2w = {i:w for w,i in words_dict.items()}
embedding_dim = len(words_dict)
hidden_dim = 128
vocab_size = len(words_dict)
target_size = vocab_size
input_dim = 40
print("The target size is ", target_size)
baseline_listener = listener(input_dim, hidden_dim)
baseline_speller = speller(embedding_dim, hidden_dim, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(baseline_listener.parameters()) + list( baseline_speller.parameters()), lr=0.01)
baseline_listener.cuda()
baseline_speller.cuda()
objective = nn.CrossEntropyLoss(ignore_index=0)
objective.cuda()
'''
