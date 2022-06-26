import torch
import torch.nn as nn

## This implementation follows the implementation available here: https://github.com/ndrplz/ConvLSTM_pytorch

class ConvLSTM(nn.Module):
    '''
    Implementation of the model ConvLSTM. Paper link: https://dl.acm.org/doi/10.5555/2969239.2969329

    Parameters
    ..........
    input_dim (Int) - Number of input features or channels
    hidden_dim (Int or List of Int, Optional) - Default: [128, 64, 64]. Indicates the number of nodes or
                                            filters in all layers. If not a list, same value will be used for each layer.
    kernel_size (Tuple or List of Tuple, Optional) - Default: (3, 3). Indicates the filter size. If not list,
                                                     same kernel size will be used in all layers.
    num_layers (Int, Optional) - Default: 3. Indicates number of layers.
    bias (Boolean, Optional) - Default: True. Denotes whether bias parameter is True or False.
    '''

    def __init__(self, input_dim, hidden_dim = [128, 64, 64], kernel_size = (3, 3), num_layers = 3, bias=True):
        super(ConvLSTM, self).__init__()

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers

        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * num_layers

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Lengths of parameters hidden_dim, kernel_size, and num_layers are wrong')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        self.device = None

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(_ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        '''
        Parameters
        ..........
        input_tensor (Tensor) - History sequence part of the input sample
        hidden_state (Tuple, Optional) - A tuple of pair denoting the hidden state: (h, c). Default: None
        '''

        if self.device is None:
            self.device = input_tensor.device

        b, seq_len, channels, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w)

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        return layer_output, (h, c)

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width, self.device))
        return init_states



class _ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding="same",
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device), torch.zeros(batch_size, self.hidden_dim, height, width, device=device))




