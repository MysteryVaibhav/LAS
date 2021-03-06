import numpy as np
from utils import *


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_val_loss(self, my_net, loss_fn):
        losses = []
        my_net.eval()
        for (input_val, input_len, label, label_len, label_mask) in self.data_loader.val_data_loader:
            label = to_variable(label)
            prediction = my_net(to_variable(input_val), input_len, label, label_len)
            # Use prediction and compute the loss carefully
            var_label_mask = to_variable(label_mask[:, 1:].contiguous().view(-1).nonzero().squeeze())
            prediction = torch.index_select(prediction.contiguous().view(-1, len(self.data_loader.vocab)), dim=0,
                                            index=var_label_mask)
            label = torch.index_select(label[:, 1:].contiguous().view(-1), dim=0, index=var_label_mask)
            loss = loss_fn(prediction, label)
            losses.append(loss.data.cpu().numpy())
        return np.asscalar(np.mean(losses))

    def decode(self, my_net):
        """
        :param my_net:
        :return: Writes the decoded result of test set in submission.csv
        """
        my_net.eval()
        i = 0
        file = open('submission.csv', 'w')
        file.write("Id,Predicted\n")
        for (input_val, input_len) in self.data_loader.test_data_loader:
            output, raw_preds = my_net(to_variable(input_val), input_len)
            output = self.get_best_out(output, raw_preds)
            pred = [self.data_loader.vocab[idx] for idx in output]
            pred = ''.join(pred)
            file.write("{},{}\n".format(i, pred))
            print("{},{}\n".format(i, pred))
            i += 1
        file.close()

    def get_best_out(self, output, raw_preds):
        loss_fn = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
        best_loss = 1000
        best_out = None
        for i, each in enumerate(output):
            loss = loss_fn(raw_preds[i], to_variable(to_tensor(np.array(each)).long())).data.cpu().numpy()[0]
            if loss < best_loss:
                best_loss = loss
                best_out = each[:-1]
        return best_out


class Node:
    def __init__(self, char, prob, children=[]):
        self.char = char
        self.prob = prob
        self.children = children
