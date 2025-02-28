from numpy import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

class ChannelCNN(nn.Module):
    def __init__(self, receivers_num=4):
        super(ChannelCNN, self).__init__()
        self.channel_conv = nn.Conv2d(2*(receivers_num - 1), 1, 2*(receivers_num - 1)-1, padding=2)

    def forward(self, x):
        return self.channel_conv(x.permute(0,1,-1,-2)).squeeze(1)

class MambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, receivers_num):
        super(MambaBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.dt_rank = int(ceil(self.hidden_dim / 8))
        # self.dt_rank = self.hidden_dim

        #Define NN

        self.in_projection = nn.Linear(input_dim, 2 * self.hidden_dim, bias=False)

        self.conv1d = nn.Conv1d(self.hidden_dim, self.hidden_dim,
                                 kernel_size=2*(receivers_num - 1)-1, 
                                 groups=self.hidden_dim,
                                 padding=2*(receivers_num - 1)-2 #casual
        )

        self.x_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim * 2 + self.dt_rank,
            bias=False
        )

        self.delta_t_projection = nn.Linear(
            self.dt_rank, 
            self.input_dim, 
            bias=True
        )

        # State-space model parameters

        self.log_A = nn.Parameter(
                   torch.zeros(input_dim, hidden_dim)
        , 
            requires_grad=True
        )

        self.D = nn.Parameter(
            torch.ones(self.hidden_dim, dtype=torch.float32),
            requires_grad=True
        )

        self.out_projection = nn.Linear(
            self.hidden_dim, input_dim
        )

        # self.initialization()

    # def initialization(self):
    #     nn.init.xavier_uniform_(self.A)

    def ssm(self, x):
        d, n = self.log_A.shape

        # Compute state space parameters
        A = -torch.exp(self.log_A).float() # shape -> (d_in, n)
        D = self.D.float()

        x_dbl = self.x_projection(x)  # shape -> (batch, input, seq_len)

        delta, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.hidden_dim, self.hidden_dim], 
            dim=-1
        )

        delta = F.silu(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_internal_dim)

        return self.selective_scan(x, delta, A, B, C, D)
        
    def forward(self, x):
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
        
        # "Prefix-sum" of dA along the sequence dimension
        dA_cumsum = torch.cumsum(dA_cumsum, dim=1)
        dA_cumsum = torch.exp(dA_cumsum)  

        dA_cumsum = torch.flip(dA_cumsum, dims=[1])  # Flip back along axis 1

        x = dB_u * dA_cumsum
        x = torch.cumsum(x, dim=1) / (dA_cumsum + 1e-12) 

        y = torch.einsum('bldn,bln->bln', x, C)
    
        return y + u * D.to(u.device)
    

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    

class ResidualBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = MambaBlock(cfg.input_dim, cfg.hidden_dim, cfg.recivers_num)
        self.norm = RMSNorm(cfg.input_dim)

    def forward(self, x):
        return self.layer(self.norm(x)) + x
    

class DOAMAMBA(nn.Module):
    def __init__(self, cfg):
        super(DOAMAMBA, self).__init__()
        self.cfg = cfg
        self.channel_conv = ChannelCNN(cfg.recivers_num)
        self.mamba_layers = nn.ModuleList([
            ResidualBlock(cfg)
            for _ in range(cfg.num_layers)
        ])

        self.hidden = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.doa = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.logvar = nn.Linear(cfg.input_dim, cfg.input_dim)

    def forward(self, x):
        x = self.channel_conv(x)
        for layer in self.mamba_layers:
            x = F.tanh(layer(x))
        hidden = F.relu(self.hidden(x))
        return (self.unwrap_angle(self.doa(hidden)), self.logvar(hidden))
    
    def unwrap_angle(self, angle):
        return angle % (2 * torch.pi)

    def accuracy(self, est, gt, var):
        return torch.sum(torch.abs(est - gt) < var.sqrt()) / torch.numel(est)
        