from numpy import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
from einops import rearrange, repeat, einsum

class ChannelCNN(nn.Module):
    def __init__(self, receivers_num=4, out_channels=1):
        super(ChannelCNN, self).__init__()
        self.channel_conv = nn.Sequential(
        nn.Conv2d(receivers_num - 1, 32, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, out_channels, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )

    def forward(self, x):
        return self.channel_conv(x).squeeze(1).permute(0, 2, 1)

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

        self.A_log = nn.Parameter(
            torch.log(repeat(
            torch.arange(1, input_dim + 1, dtype=torch.float32),
            'n -> n d', d=hidden_dim
        )), 
            requires_grad=True
        )

        self.D = nn.Parameter(
            torch.ones(self.hidden_dim, dtype=torch.float32),
            requires_grad=True
        )

        self.out_projection = nn.Linear(
            self.hidden_dim, input_dim, bias=False
        )

        # self.initialization()

    # def initialization(self):
    #     nn.init.xavier_uniform_(self.A)

    def ssm(self, x):
        d, n = self.A_log.shape

        # Compute state space parameters
        A = -torch.exp(self.A_log) # shape -> (d_in, n)
        D = self.D.float()

        x_dbl = self.x_projection(x)  # shape -> (batch, input, seq_len)

        delta, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.hidden_dim, self.hidden_dim], 
            dim=-1
        )

        delta = F.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_internal_dim)

        return selective_scan(x, delta, A, B, C, D)
        
    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
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

def complex_log(input, eps=1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * torch.pi
    return torch.complex(real, imag)

def selective_scan(u, dt, A, B, C, D, mode='cumsum'):
    dA = torch.einsum('bld,dn->bldn', dt, A)
    dB_u = torch.einsum('bld,bln,bln->bldn', dt, u, B)
    # dA = dA.clamp(min=-20)
    
    padding =  (0, 0, 0, 0, 1, 0)

    if mode=='cumsum':            
        dA_cumsum = F.pad(dA[:, 1:], padding).cumsum(1).exp()
        x = dB_u / (dA_cumsum + 1e-12)
        x = x.cumsum(1) * dA_cumsum
        y = torch.einsum('bldn,bln->bln', x, C)
    
    elif mode=='logcumsumexp':  # more numerically stable (Heisen sequence)
        dB_u_log = complex_log(dB_u)
        dA_star = F.pad(dA[:, 1:].cumsum(1), padding)
        x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star
        y = torch.einsum('bldn,bln->bln', x_log.real.exp() * torch.cos(x_log.imag), C)
            
    return y + u * D
    

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
        return self.norm(self.layer(x)) + x
    
    
class UNet1D(nn.Module):
    def __init__(self, in_channels=257, out_channels=2, base_features=64):
        """
        1D U-Net.
        
        Args:
            in_channels  (int): Number of input channels.
            out_channels (int): Number of output channels (e.g., 2 for doa/logvar).
            base_features(int): Number of feature maps in the first encoder layer.
        """
        super(UNet1D, self).__init__()

        self.enc1 = self.double_conv(in_channels, base_features)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc2 = self.double_conv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # we can extend the U-Net depth by adding more encoders/pools.
        # For brevity, let's keep it two-level here.

        self.bottleneck = self.double_conv(base_features * 2, base_features * 4)

        self.up2 = nn.ConvTranspose1d(
            base_features * 4, base_features * 2, kernel_size=2, stride=2, output_padding=1
        )
        self.dec2 = self.double_conv(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose1d(
            base_features * 2, base_features, kernel_size=2, stride=2
        )
        self.dec1 = self.double_conv(base_features * 2, base_features)

        self.out_conv = nn.Conv1d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the 1D U-Net.
        x shape: (batch_size, in_channels, seq_len)
        """
        e1 = self.enc1(x)         
        p1 = self.pool1(e1)       

        e2 = self.enc2(p1)        
        p2 = self.pool2(e2)       

        b = self.bottleneck(p2)   

        u2 = self.up2(b)          
        c2 = torch.cat([u2, e2], dim=1)  
        d2 = self.dec2(c2)       

        u1 = self.up1(d2)         
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(c1)        

        out = self.out_conv(d1)
        return out
    
    def double_conv(self, in_ch, out_ch):
        """
        for Unet
        """
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
class UNet2D(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, base_features=64):
        """
        2D U-Net.
        
        Args:
            in_channels  (int): Number of input channels.
            out_channels (int): Number of output channels (e.g., 2 for doa/logvar).
            base_features(int): Number of feature maps in the first encoder layer.
        """
        super(UNet2D, self).__init__()

        self.enc1 = self.double_conv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = self.double_conv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # we can extend the U-Net depth by adding more encoders/pools.
        # For brevity, let's keep it two-level here.

        self.bottleneck = self.double_conv(base_features * 2, base_features * 4)

        self.up2 = nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2, 
            kernel_size=2, 
            stride=2, 
            output_padding=(1, 0)
        )
        self.dec2 = self.double_conv(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(
            base_features * 2, 
            base_features, 
            kernel_size=2, 
            stride=2, 
            output_padding=(0, 1)
        )
        self.dec1 = self.double_conv(base_features * 2, base_features)

        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the 2D U-Net.
        x shape: (batch_size, in_channels, height, width)
        """
        e1 = self.enc1(x)         
        p1 = self.pool1(e1)       

        e2 = self.enc2(p1)        
        p2 = self.pool2(e2)       

        b = self.bottleneck(p2)   

        u2 = self.up2(b)          
        c2 = torch.cat([u2, e2], dim=1)  
        d2 = self.dec2(c2)       

        u1 = self.up1(d2)         
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(c1)        

        out = self.out_conv(d1)
        return out
    
    def double_conv(self, in_ch, out_ch):
        """
        for Unet
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
class FeatureEncoder(nn.Module):
    def __init__(self, receivers_num=4, dilation=2):
        super(FeatureEncoder, self).__init__()
        
        self.pre_conv = nn.Sequential(
            nn.Conv2d(2*(receivers_num-1), 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        densenet = densenet121(pretrained=False)
        
        densenet.features[0] = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=dilation, dilation=dilation, bias=False)

        for module in densenet.features:
            if isinstance(module, nn.Conv2d):
                module.dilation = (dilation, dilation)
                module.padding = (dilation, dilation)

        self.dense_core = densenet.features

        self.post_conv = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pre_conv(x)
        return x