import torch
from torch import nn
import timm
import math


'''
def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
'''

class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        return attn_output



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)  
        self.adapter_up = nn.Linear(dim, 768)  
        self.adapter_mid = nn.Linear(dim, dim)
        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        # print('B, N, C',B, N, C)
        #torch.Size([32, 320, 8])
        x_down = self.adapter_down(x)  
        # print('x_down',x_down.size()) 
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up

class Bi_direct_adapter_w(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)  
        self.adapter_up = nn.Linear(dim, 768)  
        self.adapter_mid = nn.Linear(dim, dim)

        # self.adapter_weight_token = nn.Linear(dim, 1)
        # self.adapter_SE = SELayer(768)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        # nn.init.zeros_(self.adapter_weight_token.weight)
        # nn.init.zeros_(self.adapter_weight_token.bias)
        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        # print('x_S',x_S.size())
        # print('B, N, C',B, N, C)
        #torch.Size([32, 320, 8])
        x_down = self.adapter_down(x)  
        # print('x_down',x_down.size()) 
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        #x_down = self.act(x_down)
        # self.w_token = self.adapter_weight_token(x_down)
        # self.w = self.w_token.squeeze(-1).unsqueeze(-1)
        # x_down = x_down * self.w
        x_up = self.adapter_up(x_down) 
        #B,768,16,16
        # x_S = x_up[:,64:]
        # x_Z = x_up[:,:64]
        # x_S = x_S.permute(0,2,1).view(B,C,16,16)
        # x_Z = x_Z.permute(0,2,1).view(B,C,8,8)
        # x_S = self.adapter_SE(x_S)
        # x_Z = self.adapter_SE(x_Z)
        # x_S = x_S.view(B,C,256).permute(0,2,1)
        # x_Z = x_Z.view(B,C,64).permute(0,2,1)

        # x_up = torch.cat((x_Z,x_S),dim=1)
        #print("return adap x", x_up.size())
        return x_up

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class Bi_appearence_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768,dim)
        self.adapter_up = nn.Linear(2*dim, 768)  
        # self.adapter_mid = nn.Linear(384, dim)
        self.adapter_mid_2 = nn.Linear(2*dim, 2*dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.adapter_mid_2.weight)
        nn.init.zeros_(self.adapter_mid_2.bias)


        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x,xi):
        B, N, C = x.shape
        # print('B, N, C',B, N, C)   
        x_down = self.adapter_down(x)   
        # x_down = self.adapter_mid(x_down)
        xi_down = self.adapter_down(x)   
        # xi_down = self.adapter_mid(xi_down)
        x_cat = torch.cat((x_down,xi_down),-1)
        x_cat = self.adapter_mid_2(x_cat)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_cat)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up


class Bi_cross_adapter(nn.Module):
    def __init__(self, dim=8,type=None, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768,dim)
        self.adapter_up = nn.Linear(2*dim, 768)  
        # self.adapter_mid = nn.Linear(384, dim)
        self.adapter_mid_2 = nn.Linear(dim, dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.adapter_mid_2.weight)
        nn.init.zeros_(self.adapter_mid_2.bias)


        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x,xi):
        B, N, C = x.shape
        # print('B, N, C',B, N, C)   
        x_down = self.adapter_down(x)   
        # x_down = self.adapter_mid(x_down)
        xi_down = self.adapter_down(x)   
        # xi_down = self.adapter_mid(xi_down)
        # x_cat = torch.cat((x_down,xi_down),-1)
        x_sum = x_down + xi_down
        x_sum = self.adapter_mid_2(x_sum)
        #RGB branch need TIR info
        if type == 'RGB':
            x_sum = torch.cat((x_sum,xi_down),-1)
        #TIR branch need RGB info
        else:
            x_sum = torch.cat((x_sum,x_down),-1)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_sum)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up


"""


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        #print(x.shape)
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        #print(x_down.shape)

        x_patch = x_down[:, 64:].reshape(B, 16, 16, self.dim).permute(0, 3, 1, 2)   ############
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 16 * 16, self.dim)


        #x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up
"""