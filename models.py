import torch
from torch import nn



class ResidualConvBlock(nn.Module):
    """Convolutional Residual Block"""
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualConvBlock, self).__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.handle_channel = None
        if self.is_res and not self.same_channels:
            self.handle_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
    def forward(self, x):
        x_out = self.conv2(self.conv1(x))
        if not self.is_res:
            return x_out
        else:
            if self.handle_channel:
                x_out = x_out + self.handle_channel(x)
            else:
                x_out = x_out + x
            return x_out / 1.414 # normalize


class UNetDown(nn.Module):
    """UNet downward block (i.e., contracting block)"""
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels, True), # use skip-connection
                  ResidualConvBlock(out_channels, out_channels),
                  nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """UNet upward block (i.e., expanding block)"""
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2), # double h & w sizes
            ResidualConvBlock(out_channels, out_channels, True), # use residual-connection
            ResidualConvBlock(out_channels, out_channels)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_skip):
        return self.model(torch.cat([x, x_skip], dim=1))


class EmbedFC(nn.Module):
    """Fully Connected Embedding Layer"""
    def __init__(self, input_dim, hidden_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)[:, :, None, None]

class AdaIN(nn.Module):
    def __init__(self, mid_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(num_features, mid_dim*2)

    def forward(self, _x, ada_emb):
        B, C, H, W = _x.shape
        # _x = _x.reshape(B, -1)
        ada_emb = ada_emb.reshape(B, -1)
        x = self.norm(_x)
        ada_params = self.fc(ada_emb)[:, :, None, None]
        gamma, beta = ada_params.chunk(2, dim=1)
        return ((1.0 + gamma) * x + beta).reshape(B, C, H, W)


class HyperAda(nn.Module):
    def __init__(self, mid_dim, num_features, rank=8):
        super().__init__()
        self.rank = rank
        # self.up_weights = nn.Sequential(
        #     nn.Linear(num_features,mid_dim*rank),
        # )
        self.generate_weights = nn.Sequential(
            nn.Linear(num_features,64),
            nn.SiLU(),
            nn.Linear(64, mid_dim*rank*2)
        )

    def forward(self, x, ada_emb):
        x_in = x
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, -1, C)
        ada_emb = ada_emb.reshape(B, -1)
        x_weights = self.generate_weights(ada_emb)
        x_down, x_up = x_weights.view(-1, C, self.rank, 2).chunk(2, dim=-1)
        x_down, x_up = x_down.squeeze(-1), x_up.squeeze(-1)
        x = torch.einsum('bic,bco->bio', x, x_down)
        # x = torch.nn.functional.silu(x)
        x = torch.einsum('bic,bco->bio', x, x_up.permute(0, 2, 1))
        
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x + x_in

class ContextUnet(nn.Module):
    """Context UNet model
    Args:
        in_channels (int): Number of channels in the input image
        height (int): Height of the input image
        width (int): Width of the input image
        n_feat (int): Number of initial features i.e., hidden channels to which
            the input image be transformed
        n_cfeat (int): Number of context features i.e., class categories
        n_downs (int): Number of down (and up) blocks of UNet. Default: 2
    """
    def __init__(self, in_channels, height, width, n_feat, n_cfeat, n_downs=2, emb_type="normal"):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.n_downs = n_downs

        # Define initial convolution
        self.init_conv = ResidualConvBlock(in_channels, n_feat, True)
        
        # Define downward unet blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(2**i*n_feat, 2**(i+1)*n_feat))
        
        # Define at the center layers
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((height//2**len(self.down_blocks), width//2**len(self.down_blocks))), 
            nn.GELU())
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                2**n_downs*n_feat, 
                2**n_downs*n_feat, 
                (height//2**len(self.down_blocks), width//2**len(self.down_blocks))),
            nn.GroupNorm(8, 2**n_downs*n_feat),
            nn.GELU()
        )
        
        # Define upward unet blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downs, 0, -1):
            self.up_blocks.append(UNetUp(2**(i+1)*n_feat, 2**(i-1)*n_feat))

        # Define final convolutional layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, in_channels, 1, 1)
        )

        # Define time & context embedding blocks 
        self.timeembs = nn.ModuleList([EmbedFC(1, 2**i*n_feat) for i in range(n_downs, 0, -1)])
        self.emb_type = emb_type
        if emb_type == "adain":
            self.adas = nn.ModuleList([AdaIN(2**i*n_feat, (2**i*n_feat)*2) for i in range(n_downs, 0, -1)])
        elif emb_type == "hyperada":
            self.adas = nn.ModuleList([HyperAda(2**i*n_feat, (2**i*n_feat)*2) for i in range(n_downs, 0, -1)])
        self.contextembs = nn.ModuleList([EmbedFC(n_cfeat, 2**i*n_feat) for i in range(n_downs, 0, -1)])

    def forward(self, x, t, c):
        x = self.init_conv(x)
        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0: downs.append(down_block(x))
            else: downs.append(down_block(downs[-1]))
        up = self.up0(self.to_vec(downs[-1]))

        if (self.emb_type == "normal"):
            for up_block, down, contextemb, timeemb in zip(self.up_blocks, downs[::-1], self.contextembs, self.timeembs):
                up = up_block(up*contextemb(c) + timeemb(t), down)
        else:
            for up_block, down, contextemb, timeemb, emb_ada in zip(self.up_blocks, downs[::-1], self.contextembs, self.timeembs, self.adas):
                merged_feats= torch.cat([timeemb(t), contextemb(c)], dim=-1)
                up = up_block(emb_ada(up, merged_feats), down)

        
        return self.final_conv(torch.cat([up, x], axis=1))
