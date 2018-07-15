import torch as t


class RNN(t.nn.Module):
    #TODO: complete ori RNN
    def __init__(self):
        super(RNN, self).__init__()
        self.input_dim = 300
        self.hidden_size = 150
        self.rnn_cell = t.nn.RNNCell(input_size=self.input_dim,hidden_size=self.hidden_size)

    def forward(self, inputs):
        pass
