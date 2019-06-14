# encoding: utf-8

"""
@author: huayuxiu

RCLSTM block
"""

import torch
from torch import nn
# from torch.autograd import Variable
from torch.nn import functional, init
import numpy as np
import random
import math

# generate mask matrix based on uniform distribution
def generate_mask_matrix(shape, connection=1.):
    s = np.random.uniform(size=shape)
    s_flat = s.flatten()
    s_flat.sort()
    threshold = s_flat[int(shape[0]*shape[1]*(1-connection))]
    super_threshold_indices = s>=threshold
    lower_threshold_indices = s<threshold
    s[super_threshold_indices] = 1.
    s[lower_threshold_indices] = 0.
    return s

def generate_weight_mask(shape, connection=1.):
    sub_shape = (shape[0], shape[1])
    w = []
    for _ in range(4):
        w.append(generate_mask_matrix(sub_shape, connection))
    return np.concatenate(w, axis=1).astype('float32')

class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.xavier_uniform(self.weight_ih.data, gain=init.calculate_gain('sigmoid'))
        init.xavier_uniform(self.weight_hh.data, gain=init.calculate_gain('sigmoid'))
        init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class RCLSTMCell(nn.Module):
    
    """RCLSTM cell"""
    
    def __init__(self, input_size, hidden_size, connectivity, device):
        
        super(RCLSTMCell, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.connectivity = connectivity
        self.mask_wih = torch.FloatTensor(input_size, 4 * hidden_size)
        self.mask_whh = torch.FloatTensor(hidden_size, 4 * hidden_size)
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        self.mask_wih = torch.from_numpy(
            generate_weight_mask((self.input_size, self.hidden_size), self.connectivity)).to(self.device)
            
        self.mask_whh = torch.from_numpy(
            generate_w((self.hidden_size, self.hidden_size), self.connectivity)).to(self.device)
        
        weight_ih_data = init.orthogonal(self.weight_ih.data)
        weight_ih_data = weight_ih_data * self.mask_wih.cpu().data
        self.weight_ih.data.set_(weight_ih_data)
        
        weight_hh_data = init.orthogonal(self.weight_hh.data)
        weight_hh_data = weight_hh_data * self.mask_whh.cpu().data
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is set to zero.
        init.constant(self.bias.data, val=0)
            
    def print_weight(self):
        print(self.weight_ih.data.nmupy())
            
    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh * self.mask_whh)
        wi = torch.mm(input_, self.weight_ih * self.mask_wih)
        f, i, o, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class RNN(nn.Module):

    """A module that runs multiple steps of LSTM or RCLSTM."""

    def __init__(self, device, cell_class, input_size, hidden_size, connectivity,
                 num_layers=1, batch_first=True, dropout=0):
        super(RNN, self).__init__()
        self.device = device
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.connectivity = connectivity
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            if cell_class == 'lstm':
                cell = LSTMCell(input_size=layer_input_size, hidden_size=hidden_size)
            else:
                cell = RCLSTMCell(input_size=layer_input_size, hidden_size=hidden_size, connectivity=connectivity, device=device)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

    @staticmethod
    def _forward_rnn(cell, input_, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[max_time-1-time], hx=hx)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if hx is None:
            hx = input_.data.new(batch_size, self.hidden_size).zero_()
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = RNN._forward_rnn(
                cell=cell, input_=input_, hx=hx)
            input_ = layer_output
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)
