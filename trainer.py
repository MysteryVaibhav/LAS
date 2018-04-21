import numpy as np
import sys
from model import LAS
from utils import *
from timeit import default_timer as timer


class Trainer:
    def __init__(self, params, data_loader, evaluator):
        self.params = params
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.model = LAS(params, len(data_loader.vocab), data_loader.max_seq_len)

    @staticmethod
    def init_xavier(m):
        if type(m) == torch.nn.Linear:
            fan_in = m.weight.size()[1]
            fan_out = m.weight.size()[0]
            std = np.sqrt(6.0 / (fan_in + fan_out))
            m.weight.data.normal_(0, std)
            m.bias.data.zero_()

    def train(self):
        my_net = self.model
        my_net.apply(self.init_xavier)
        my_net.load_state_dict(torch.load('models/bestModelWeights_0.15.t7'))
        loss_fn = torch.nn.CrossEntropyLoss()
        optim = torch.optim.ASGD(my_net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.wdecay)
        if torch.cuda.is_available():
            my_net = my_net.cuda()
            loss_fn = loss_fn.cuda()
        
        try:
            prev_best = 100000
            for epoch in range(self.params.num_epochs):
                losses = []
                start_time = timer()
                minibatch = 1
                for (input_val, input_len, label, label_len, label_mask) in self.data_loader.train_data_loader:
                    optim.zero_grad()  # Reset the gradients
                    my_net.train()
                    label = to_variable(label)
                    prediction = my_net(to_variable(input_val), input_len, label, label_len)  # Feed forward
                    
                    # Use prediction and compute the loss carefully
                    var_label_mask = to_variable(label_mask[:, 1:].contiguous().view(-1).nonzero().squeeze())
                    prediction = torch.index_select(prediction.contiguous().view(-1, len(self.data_loader.vocab)),
                                                    dim=0, index=var_label_mask)
                    label = torch.index_select(label[:, 1:].contiguous().view(-1), dim=0, index=var_label_mask)

                    loss = loss_fn(prediction, label)
                    loss.backward()  # Back propagate the gradients

                    if self.params.clip_value > 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm(my_net.parameters(), self.params.clip_value)
                    optim.step()  # Update the network

                    losses.append(loss.data.cpu().numpy())
                    sys.stdout.write("[%d/%d] :: Training Loss: %f   \r" % (
                        minibatch, len(self.data_loader.train_label) // self.params.batch_size,
                        np.asscalar(np.mean(losses))))
                    sys.stdout.flush()
                    minibatch += 1

                # anneal lr
                optim_state = optim.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0][
                                                           'lr'] * self.params.learning_anneal
                optim.load_state_dict(optim_state)
                val_loss = self.evaluator.get_val_loss(my_net, loss_fn)
                print("Epoch {} : Training Loss: {:.5f}, Validation Loss: {:.5f}, Time elapsed {:.2f} mins".
                      format(epoch, np.asscalar(np.mean(losses)), val_loss, (timer() - start_time) / 60))

                if val_loss < prev_best:
                    prev_best = val_loss
                    print("Validation loss decreased... saving weights !")
                    torch.save(my_net.state_dict(), self.params.model_dir+'/bestModelWeights_{:.2f}.t7'.format(val_loss))
                else:
                    print("Validation loss didn't decrease... not saving !")

        except KeyboardInterrupt:
            print("Interrupted...saving model !!!")
            torch.save(my_net.state_dict(), 'model_interrupt.t7')

        return my_net
