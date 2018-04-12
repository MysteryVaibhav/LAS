import numpy as np
import torch.utils.data
from utils import *


class DataLoader:
    def __init__(self, params):
        self.params = params

        # Loading data
        self.train = self.expand(np.load(params.data_dir + 'dev.npy'))
        self.train_transcript = np.load(params.data_dir + 'dev_transcripts.npy')
        self.val = self.expand(np.load(params.data_dir + 'dev.npy'))
        self.val_transcript = np.load(params.data_dir + 'dev_transcripts.npy')
        self.test = self.expand(np.load(params.data_dir + 'test.npy'))
        self.max_seq_len = np.max([x.shape[0] for x in self.train] +
                                  [x.shape[0] for x in self.val] +
                                  [x.shape[0] for x in self.test])

        # Constructing vocab and charset
        self.vocab, self.charset = self.get_vocab()

        # Converting transcripts to chars
        self.train_label = self.char_to_int(self.train_transcript)
        self.val_label = self.char_to_int(self.val_transcript)

        # Setting pin memory and number of workers
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        # Creating data loaders
        dataset_train = CustomDataSet(self.train, self.train_label, False)
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=params.batch_size,
                                                             collate_fn=dataset_train.collate, shuffle=True, **kwargs)

        dataset_val = CustomDataSet(self.val, self.val_label, False)
        self.val_data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=params.batch_size,
                                                           collate_fn=dataset_val.collate, shuffle=False, **kwargs)

        dataset_test = CustomDataSet(self.test, [], True)
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                            collate_fn=dataset_test.collate, shuffle=False, **kwargs)

    def get_vocab(self):
        vocab = []
        charset = {}
        i = 1
        # Adding start/stop symbol
        vocab.append('<s>')
        charset['<s>'] = 0
        for each_utterance in self.train_transcript:
            for c in each_utterance:
                if c not in charset:
                    charset[c] = i
                    vocab.append(c)
                    i += 1
        return vocab, charset

    def char_to_int(self, transcripts):
        char_to_int = []
        for transcript in transcripts:
            # Appending start and stop
            char_to_int.append([0] + [self.charset[c] for c in transcript] + [0])
        return char_to_int

    @staticmethod
    def expand(data):
        for i, utterance in enumerate(data):
            # Repeating last frame
            while len(data[i]) % 8 != 0:
                data[i] = np.concatenate((data[i], [utterance[-1]]), axis=0)
        return data


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data, labels, is_test):
        self.data = data
        self.labels = labels
        self.num_of_samples = len(self.data)
        self.data = self.data
        self.is_test = is_test
        self.max_seq_len = np.max([x.shape[0] for x in self.data])

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        if self.is_test:
            return self.data[idx], len(self.data[idx])
        return self.data[idx], len(self.data[idx]), self.labels[idx], len(self.labels[idx])

    def collate(self, batch):
        inputs = np.array([x[0] for x in batch])
        input_seq_lens = [x.shape[0] for x in inputs]
        sorted_input_seq_len = np.flipud(np.argsort(input_seq_lens))
        input_lens = np.array([x[1] for x in batch])[sorted_input_seq_len]
        inputs = inputs[sorted_input_seq_len]
        utterance_max_len = np.max(input_lens)
        padded_input = np.zeros((len(batch), utterance_max_len, 40))

        i = 0
        for input in inputs:
            padded_input[i][:len(input), :] = input
            i += 1

        if self.is_test:
            return to_tensor(padded_input), to_tensor(input_lens).int()

        labels = np.array([x[2] for x in batch])[sorted_input_seq_len]
        label_lens = np.array([x[3] for x in batch])[sorted_input_seq_len]
        max_label_len = np.max(label_lens)
        padded_label = np.zeros((len(batch), max_label_len))
        label_mask = np.zeros((len(batch), max_label_len))
        i = 0
        for input in labels:
            padded_label[i][:len(input)] = input
            label_mask[i][:len(input)] = 1
            i += 1

        return to_tensor(padded_input), to_tensor(input_lens).int(), \
               to_tensor(padded_label).long(), to_tensor(label_lens).int(), to_tensor(label_mask).long()