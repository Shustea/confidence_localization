from numpy import arange
from torch import nn
import torch
from torch.multiprocessing import Process, Queue
import torch.nn.functional as F

import hydra

class SSMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_processes=0):
        super(SSMLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_processes = num_processes

        # Parameters for state-space model
        self.A = nn.Parameter(torch.normal(torch.zeros(hidden_dim, hidden_dim),
                               torch.ones(hidden_dim, hidden_dim)).exp())
        self.B = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        self.C = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        # self.delta = nn.Parameter(torch.zeros(hidden_dim))

        self.initialization()
        

    def initialization(self):
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        # nn.init.xavier_uniform_(self.delta)

    def worker(x, kernel, queue):
        queue.put(F.conv1d(x.permute(0, 1, -1, -2), kernel))

    def forward(self, x):
        """
        Forward pass for SSM Layer.
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, channels, seq_len, input_dim)
        """
        batch_size, channels, seq_len, _ = x.size()

        # discrete_A = torch.linalg.matrix_exp(self.delta @ self.A)
        # discrete_B = (self.delta @ self.A).inverse() @ torch.linalg.matrix_exp(discrete_A - torch.eye(self.hidden_dim).to(x.device)) @ (self.delta @ self.B)

        if self.num_processes > 0:

            self.share_memory()  # Share the model across processes

            processes = []
            queue = Queue()

            K = [self.C @ torch.pow(self.A, p_idx) @ self.B for p_idx in range(seq_len)]

            for rank in range(self.num_processes):
                p = Process(target=self.worker, args=(x, K[rank], queue))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            outputs = p

        else:
            outputs = []
            state = torch.zeros(batch_size, channels, self.hidden_dim, device=x.device) + 1e-6
            for t in range(seq_len):
                input_t = x[:, :, t, :]

                state = F.tanh(torch.matmul(state,
                                            self.A.to(x.device))
                                            + torch.matmul(input_t ,self.B.to(x.device)))

                output_t = state @ self.C.to(x.device)
                outputs.append(output_t.unsqueeze(-1))

        return torch.cat(outputs, dim=-1).permute([0, 1, -1, -2])


class Mamba(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, recivers_num):
        super(Mamba, self).__init__()
        self.layers = nn.ModuleList([
            SSMLayer(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.encode_to_hidden_state = nn.Linear(input_dim, hidden_dim)
        self.conv_layer = nn.Conv2d(2*(recivers_num - 1), 2*(recivers_num - 1), 2*(recivers_num - 1)-1, padding=2)

        self.selective_gate = nn.Linear(input_dim, hidden_dim)
         #maybe should return the length of input?

    def forward(self, x):
        """
        Forward pass for Mamba model.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        x_permute = x.permute([0, 1, -1, -2])
        x = self.conv_layer(self.encode_to_hidden_state(x_permute)).squeeze()
        s = F.selu(self.selective_gate(x_permute))
        for layer in self.layers:
            x = layer(x)
        return (x * s)