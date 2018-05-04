import numpy as np
import sys
from model import LAS
from utils import *
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from logger import *


class Trainer:
    def __init__(self, params, data_loader, evaluator):
        self.params = params
        #self.logger = Logger('./logs')
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
        #my_net.load_state_dict(torch.load('models/bestModelWeights_19.35.t7'))
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
        optim = torch.optim.Adam(my_net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.wdecay)
        if torch.cuda.is_available():
            my_net = my_net.cuda()
            loss_fn = loss_fn.cuda()
        plot_losses = []
        val_losses = []
        #val_losses_1 = []
        try:
            prev_best = 100000
            for epoch in range(self.params.num_epochs):
                losses = []
                start_time = timer()
                minibatch = 1
                tf_rate = 0.9 - (0.9 - 0)*(epoch/10)
                for (input_val, input_len, label, label_len, label_mask) in self.data_loader.train_data_loader:
                    optim.zero_grad()  # Reset the gradients
                    my_net.train()
                    label = to_variable(label)
                    prediction = my_net(to_variable(input_val), input_len, label, label_len, tf_rate)  # Feed forward
                    
                    # Use prediction and compute the loss carefully
                    #var_label_mask = to_variable(label_mask[:, 1:].contiguous().view(-1).nonzero().squeeze())
                    #prediction = torch.index_select(prediction.contiguous().view(-1, len(self.data_loader.vocab)),
                    #                                dim=0, index=var_label_mask)
                    #label = torch.index_select(label[:, 1:].contiguous().view(-1), dim=0, index=var_label_mask)

                    #loss = loss_fn(prediction, label)
                    
                    loss = loss_fn(prediction.contiguous().view(-1, len(self.data_loader.vocab)), label[:, 1:].contiguous().view(-1))
                    var_label_mask = to_variable(label_mask[:, 1:].contiguous())
                    loss = (loss.view(var_label_mask.size()) * var_label_mask.float()).sum(1).mean()
                    loss.backward()  # Back propagate the gradients

                    if self.params.clip_value > 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm(my_net.parameters(), self.params.clip_value)
                    optim.step()  # Update the network

                    losses.append(loss.data.cpu().numpy())
                    if (minibatch - 1) % 70 == 0:
                        plot_losses.append(loss.data.cpu().numpy())
                        #logger.scalar_summary('Training Loss', loss.data.cpu().numpy(), len(plot_losses))
                    
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
                val_loss = self.evaluator.get_val_loss(my_net, loss_fn, self.data_loader.val_data_loader, tf_rate)
                #val_loss_1 = self.evaluator.get_val_loss(my_net, loss_fn, self.data_loader.val_data_loader_1, tf_rate)
                val_losses.append(val_loss)
                #val_losses_1.append(val_loss_1)
                
                print("Epoch {} : Training Loss: {:.5f}, Validation Loss: {:.5f}, Time elapsed {:.2f} mins".
                      format(epoch, np.asscalar(np.mean(losses)), val_loss, (timer() - start_time) / 60))

                if val_loss < prev_best:
                    prev_best = val_loss
                    print("Validation loss decreased... saving weights !")
                    torch.save(my_net.state_dict(), self.params.model_dir+'/bestModelWeights_{:.2f}.t7'.format(val_loss))
                else:
                    print("Validation loss didn't decrease... not saving !")
           
            # Plot 1
            fig = plt.figure()
            plt.plot(range(0, len(plot_losses)), plot_losses, color='b', label='train loss')
            plt.ylabel('loss')
            plt.xlabel('per 70 mini batch')
            plt.legend()
            fig.savefig('train_loss.png')
            
            # Plot 2
            fig = plt.figure()
            plt.plot(range(0, self.params.num_epochs), val_losses, color='b', label='train_split_val_set')
            #plt.plot(range(0, self.params.num_epochs), val_losses_1, color='r', label='given_val_set')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig('validation_loss.png')
            
            

        except KeyboardInterrupt:
            print("Interrupted...saving model !!!")
            torch.save(my_net.state_dict(), 'model_interrupt.t7')
            fig = plt.figure()
            plt.plot(range(0, len(plot_losses)), plot_losses, color='b', label='train loss')
            plt.ylabel('loss')
            plt.xlabel('per 70 mini batch')
            plt.legend()
            fig.savefig('d_g.png')

        return my_net
