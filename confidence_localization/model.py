import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Mamba(nn.Module):
    def __init__(self, input_dim, hidden_dim, receivers_num, selective_scan_flag):
        super(Mamba, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.selective_scan_flag = selective_scan_flag

        self.in_projection = nn.Linear(input_dim, 2 * hidden_dim, bias=False)

        self.channel_conv = nn.Conv2d(2*(receivers_num - 1), 1, 2*(receivers_num - 1)-1, padding=2)

        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim,
                                 kernel_size=2*(receivers_num - 1)-1, 
                                 groups=hidden_dim,
                                 padding=2*(receivers_num - 1)-2 #casual
        )

        self.x_projection = nn.Linear(
            hidden_dim,
            hidden_dim * 3,
            bias=False
        )

        self.delta_t_projection = nn.Linear(
            hidden_dim, 
            input_dim, 
            bias=True
        )

        # State-space model parameters

        self.A_log = nn.Parameter(
            torch.log(repeat(
            torch.arange(1, input_dim + 1, dtype=torch.float32),
            'n -> n d', d=hidden_dim
        )), 
            requires_grad=True
        )

        self.D = nn.Parameter(
            torch.ones(hidden_dim, dtype=torch.float32),
            requires_grad=True
        )

        self.out_projection = nn.Linear(
            hidden_dim, input_dim
        )

    def ssm(self, x):
        _, n = self.A_log.shape

        # Compute state space parameters
        A = -torch.exp(self.A_log)  # shape -> (d_in, n)
        D = self.D


        x_dbl = self.x_projection(x)  # shape -> (batch, input, seq_len)

        delta, B, C = torch.split(
            x_dbl, 
            [n, n, n], 
            dim=-1
        )

        delta = F.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_internal_dim)

        if self.selective_scan_flag:
            return self.selective_scan(x, delta, A, B, C, D)
        else:
            return self.selective_scan_time_serial(x, delta, A, B, C, D)

    def forward(self, x):
        if len(x.shape) > 3:
            seq_len= x.shape[-1]
            x = self.channel_conv(x.permute(0,1,-1,-2)).squeeze(1)
        else:
            seq_len = x.shape[-2]

        x_and_res = self.in_projection(x)
        x, res = torch.split(x_and_res, [self.hidden_dim, self.hidden_dim], dim=-1)

        x = rearrange(x, 'b l n -> b n l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b n l -> b l n')

        x = F.silu(x)  # Equivalent to tf.nn.swish
        y = self.ssm(x)
        y = y * F.silu(res)

        return self.out_projection(y)

    def selective_scan(self, u, delta, A, B, C, D):
        dA = torch.einsum('bld,dn->bldn', delta, A)
        dB_u = torch.einsum('bld,bln,bln->bldn', delta, u, B)

        dA_cumsum = F.pad(dA[:, 1:], (0, 0, 0, 0, 1, 1, 0, 0))[:, 1:]
        
        dA_cumsum = torch.flip(dA_cumsum, dims=[1])  # Flip along axis 1
        
        # Cumulative sum along all the input tokens, parallel prefix sum, 
        # calculates dA for all the input tokens parallely
        dA_cumsum = torch.cumsum(dA_cumsum, dim=1) 

        # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
        dA_cumsum = torch.exp(dA_cumsum)  
        dA_cumsum = torch.flip(dA_cumsum, dims=[1])  # Flip back along axis 1

        x = dB_u * dA_cumsum
        # 1e-12 to avoid division by 0
        x = torch.cumsum(x, dim=1) / (dA_cumsum + 1e-12) 

        y = torch.einsum('bldn,bln->bln', x, C)
    
        return y + u * D.to(u.device)
    

    def selective_scan_time_serial(self, u, delta, A, B, C, D):
        batch_size, L, hidden = u.shape
        N = A.shape[0]  # Assuming A, B, C have the same last dimension N

        # Initialize output tensor
        y = torch.zeros((batch_size, L, hidden), device=u.device)

        # Initialize recurrent variables
        A_bar = torch.ones((batch_size, N, hidden), device=u.device)  # Accumulating exp(ΔA)
        x_accum = torch.zeros((batch_size, N, hidden), device=u.device)  # Accumulating x
        
        for t in range(L):
            
            dA = torch.einsum('bd,dn->bdn', delta[:, t], A)
            dB_u = torch.einsum('bd,bn,bn->bdn', delta[:, t], u[:, t], B[:,t,:])

            # Compute A_bar recursively
            A_bar.mul_(torch.exp(dA))

            # Compute x accumulation
            x_accum.add_(dB_u * A_bar)

            # Compute output
            y[:, t] = torch.einsum('bdn,bn->bn', x_accum / (A_bar + 1e-12), C[:, t, :])

        return y + u * D.to(u.device)