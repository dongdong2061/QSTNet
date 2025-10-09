import math
import logging
import pdb
from functools import partial, reduce
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_adapter

from ..layers.attn_adapt_blocks import CEABlock,CEABlock_Enhancement, CEABlock_Spatio_Temporal   ##BAT 
from ..layers.dualstream_attn_blocks import DSBlock ## Dual Stream without adapter


from lib.models.layers.attn import Attention
from lib.models.layers.adapter import Bi_direct_adapter


_logger = logging.getLogger(__name__)


class SK_Fusion(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SK_Fusion,self).__init__()
        d=max(in_channels//r,L)   # 计算从向量C降维到 向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                            #    nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
        


        self.conv_down =nn.Conv2d(
                    in_channels=in_channels,  # 输入通道数（假设为单通道）
                    out_channels=out_channels,  # 输出通道数
                    kernel_size=3,  # 3x3 卷积核
                    stride=1,       # 步长=1
                    padding=0       # 无填充
                )

    def forward(self, input):
        batch_size=input[0].size(0)
        output=input

        output[0] = self.conv1(input[0].reshape(batch_size,768,8,8))
        output[1] = self.conv2(input[1].reshape(batch_size,768,8,8))

        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        # print(U.size())            
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        # print(s.size())
        z=self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        # print(z.size())
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        # print(a_b.size())
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]  
        # print(a_b.size())
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]  
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        # print(a_b[0].size())
        # print(a_b[1].size())
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        # print("V",V.size())

        #修改  prompt tokens 数量
        V = self.conv_down(V)  
        # print(V.size())

        V = V.reshape(batch_size,36,self.out_channels)  # [batch_size,out_channels,H,W] -> [batch_size,out_channels,H*W]

        return V    # [batch_size,out_channels,H,W]


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, adapter_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #self.patch_embed_adapter = embed_layer(
        #    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """     #
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:  #ce_loc [3,6,9]
                ce_keep_ratio_i = ce_keep_ratio[ce_index]  #[1,1,1]
                ce_index += 1
            #20240825  for ablation study two blocks
            if i==3 or i == 6 or i == 9:
                blocks.append(
                CEABlock_Enhancement(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
            #2024.7.19 修改为只利用一个TDMF的encoder
            elif i ==11:
                blocks.append(
                CEABlock_Spatio_Temporal(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
            elif i<20:
                blocks.append(
                CEABlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
            else:
                blocks.append(
                    DSBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
        

        self.blocks = nn.Sequential(*blocks)       
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)
        self.adap_fusion_SK = SK_Fusion(embed_dim,embed_dim)


    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,dynamic_template=None,dynamic_last=None,Test=None):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb
        xi, zi = x_dte, z_dte


        z = self.patch_embed(z)
        x = self.patch_embed(x)


        xi = self.patch_embed(xi)
        zi = self.patch_embed(zi)

        if dynamic_template == None:
            pass
        else:
            self.dynamic_template = dynamic_template

###################################################################===========
        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        zi += self.pos_embed_z
        xi += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
            xi += self.search_segment_pos_embed      #//////////////////////////////////////////////////////////////////
            zi += self.template_segment_pos_embed
        #print(x.shape) #[Batch size, 256, 768]
        #z [bs,64,768]
        x = combine_tokens(z, x, mode=self.cat_mode)  ##[Batch size, 320, 768]
        #print("after cat",x.shape)

        xi = combine_tokens(zi, xi, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            xi = torch.cat([cls_tokens, xi], dim=1)

        x = self.pos_drop(x)
        xi = self.pos_drop(xi)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        global_index_ti = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_ti = global_index_ti.repeat(B, 1)

        global_index_si = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_si = global_index_si.repeat(B, 1)

        removed_indexes_s = []
        removed_indexes_si = []
        #用于统计两个模态间ce的差值
        # diff_sum = 0
        x_list = []
        test_tokens = True
        removed_flag = False
        for i, blk in enumerate(self.blocks):
            if i != 11:
                # print(i)
                x, global_index_t, global_index_s, removed_index_s, attn, \
                xi,dynamic_template, global_index_ti, global_index_si, removed_index_si, attn_i,vis_result = \
                    blk(x, xi, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x, ce_template_mask,
                        ce_keep_rate,dynamic_template)
                if self.ce_loc is not None and i in self.ce_loc:
                    removed_indexes_s.append(removed_index_s)
                    removed_indexes_si.append(removed_index_si)
            if i == 9:
                final_attn = attn
                vis_results = vis_result
            if i == 8:
                eight_attn = attn
            if i == 10:
                ten_attn = attn
            if i==3:
                vis_results_3 = vis_result
            if i == 6:
                vis_results_6 = vis_result

        # #for average  weights of tokens       
        # rgb_tokens_all = (vis_results[1] + vis_results_3[1] + vis_results_6[1])/3
        # tir_tokens_all = (vis_results[3] + vis_results_3[3] + vis_results_6[3])/3
        # x_temp ,xi_temp= x[:,64:],xi[:,64:]
        # rgb_tokens_all = rgb_tokens_all.unsqueeze(-1)
        # tir_tokens_all =  tir_tokens_all.unsqueeze(-1)   
        # x_temp = x_temp* rgb_tokens_all
        # xi_temp = xi_temp* tir_tokens_all
        # x = x[:,64:] + x_temp
        # xi = xi[:,64:] + xi_temp

        
        ###---2024.7.14 将最后一个block改为MSTE---###
        x_ori = x
        xi_ori = xi
        self.z_t,self.zi_t = x[0].unsqueeze(0)[:,:64],xi[0].unsqueeze(0)[:,:64]
        if Test:
            if  dynamic_template is not None:
                # self.MSTT = torch.cat([z,zi],dim=1)
                z_t ,zi_t = torch.chunk(dynamic_template,2,dim=1)
                z_last,zi_last = x[1].unsqueeze(0)[:,:64],xi[1].unsqueeze(0)[:,:64]
                # z_last,zi_last =  torch.chunk(dynamic_last,2,dim=1)
                # z_last_ = self.adap_fusion_pooling(z_last.reshape(1 ,768, 8,8)).reshape(1,9,768)
                # zi_last_ = self.adap_fusion_pooling(zi_last.reshape(1 ,768, 8,8)).reshape(1,9,768)
                # z_fusion = self.adap_fusion_conv(torch.cat([z_last_,zi_last_],dim=-1))
                z_fusion = self.adap_fusion_SK([z_last,zi_last])
            else:
                z_t ,zi_t = x[0].unsqueeze(0)[:,:64],xi[0].unsqueeze(0)[:,:64] 
                z_last,zi_last = x[1].unsqueeze(0)[:,:64],xi[1].unsqueeze(0)[:,:64]
                
                # z_last_ = self.adap_fusion_pooling(z_last.reshape(1 ,768, 8,8)).reshape(1,9,768)
                # zi_last_ = self.adap_fusion_pooling(zi_last.reshape(1 ,768, 8,8)).reshape(1,9,768)
                # z_fusion = self.adap_fusion_conv(torch.cat([z_last_,zi_last_],dim=-1)) 
                z_fusion = self.adap_fusion_SK([z_last,zi_last])

            # z_t = z
            # zi_t = zi
        else:
            # new_z = z.clone()
            # new_zi = zi.clone()
            # for i in range(0,new_z.shape[0]-1):
            #     new_z[i] = z[i+1]
            #     new_zi[i] = zi[i+1]
            # new_z[-1] = z[0]
            # new_zi[-1] = zi[0]
            z_t = x[0].unsqueeze(0)[:,:64]
            zi_t = xi[0].unsqueeze(0)[:,:64]
        xf,xif = [],[]
        # print('z_t',z_t.size())
        while B:
            if Test is not True:
                #consider more different tempaltes
                # z_last,zi_last = x_ori[-B+1].unsqueeze(0)[:,:64],xi_ori[-B+1].unsqueeze(0)[:,:64] 
                z_last,zi_last = x_ori[-B+1].unsqueeze(0)[:,:64],xi_ori[-B+1].unsqueeze(0)[:,:64]
                # z_last_ = self.adap_fusion_pooling(z_last.reshape(1 ,768, 8,8)).reshape(1,9,768)
                # zi_last_ = self.adap_fusion_pooling(zi_last.reshape(1 ,768, 8,8)).reshape(1,9,768)
                # z_fusion = self.adap_fusion_conv(torch.cat([z_last_,zi_last_],dim=-1))
                z_fusion = z_fusion = self.adap_fusion_SK([z_last,zi_last])
                # print('z_last',z_last.size())      
            #取出搜索区域tokens
            x = x_ori[-B].unsqueeze(0)[:,64:]
            xi = xi_ori[-B].unsqueeze(0)[:,64:]

            # print(z_t.size(),z_fusion.size(),x.size())

            x_t = torch.cat([z_t,z_fusion,x],dim=1)
            xi_t = torch.cat([zi_t,z_fusion,xi],dim=1)


            x, global_index_t, global_index_s, removed_index_s, attn, \
            xi,dynamic_template, global_index_ti, global_index_si, removed_index_si, attn_i,vis_result = \
                blk(x_t, xi_t, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x, ce_template_mask,
                    ce_keep_rate,dynamic_template,Test)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
                removed_indexes_si.append(removed_index_si)
            xf.append(x)
            xif.append(xi)
            z_t = x[:,:64]
            zi_t = xi[:,:64]
            # 0723 动态模态改为z_last
            z_last = x[:,64:100]
            zi_last = xi[:,64:100]            
            x = x[:,100:]
            xi = xi[:,100:]
            B -= 1
            dynamic_template = torch.cat([z_t,zi_t],dim=1)

        # out_att = attn

        x = torch.cat(xf,dim=0)
        xi = torch.cat(xif,dim=0)
        z_t = x[:,:64]
        zi_t = xi[:,:64]
        dynamic_last_zt = x[0][64:128].unsqueeze(0)
        dynamic_last_zti = xi[0][64:128].unsqueeze(0)
        dynamic_last = torch.cat([dynamic_last_zt,dynamic_last_zti],dim=1)
        #保留融合后的prompts
        x = x[:,64:]
        xi = xi[:,64:] 
        # print('x.size()',x.size()) 
        x = torch.cat([z_t,x],dim=1)
        xi = torch.cat([zi_t,xi],dim=1)
        # print('x',x.size())

        # x = x+x_
        # xi = xi+xi_
        x = self.norm(x)
        xi = self.norm(xi)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        lens_xi_new = global_index_si.shape[1]
        lens_zi_new = global_index_ti.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]
        zi = xi[:, :lens_zi_new]
        xi = xi[:, lens_zi_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        
        if removed_indexes_si and removed_indexes_si[0] is not None:
            removed_indexes_cat_i = torch.cat(removed_indexes_si, dim=1)

            pruned_lens_xi = lens_x - lens_xi_new                                ########################
            pad_xi = torch.zeros([B, pruned_lens_xi, xi.shape[2]], device=xi.device)
            xi = torch.cat([xi, pad_xi], dim=1)
            index_all = torch.cat([global_index_si, removed_indexes_cat_i], dim=1)
            # recover original token order
            C = xi.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            xi = torch.zeros_like(xi).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=xi)
        
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        xi = recover_tokens(xi, lens_zi_new, lens_x, mode=self.cat_mode)


    
        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)
        xi = torch.cat([zi, xi], dim=1)
        #x = torch.cat([x, xi], dim=0)
        #print("===========final out: ",x.size())
        # x = self.adap_fusion(x,xi)
        x = x + xi
        # x_fusion = self.adap_crossstage(x,x10)
        # x = x+x_fusion
        # x = torch.cat([x, xi], dim=1)
        #x = torch.cat([x, xi], dim=2)
        #x = self.adap_headcat(x)
        # x = self.adap_conv(x)

        #print("-------",x.shape)
        #print("attn",attn.size())
        aux_dict = {
            "attn": attn,
            "i_attn": attn_i,
            "final_attn":final_attn,
            "eight_attn":eight_attn,
            "ten_attn":ten_attn,
            "vis_results":vis_results,
            "vis_results_3":vis_results_3,
            "vis_results_6":vis_results_6,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "x":x,
            "xi":xi,
        }
        # print('diff_sum',diff_sum)
        return x, aux_dict,dynamic_template,dynamic_last

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,dynamic_template=None,dynamic_last = None,Test=None):
        dynamic_template_ = dynamic_template
        dynamic_last = dynamic_last
        # dynamic_template_list = []
        # if Test == False:
        #     for i in range(len(z)):
        #         z_ = z[i].repeat(8, 1, 1, 1)
        #         x_ = x[i].repeat(8, 1, 1, 1)
        #         x_, aux_dict_,dynamic_template_ = self.forward_features(z_, x_, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,dynamic_template=dynamic_template_)
        #         dynamic_template_list.append(dynamic_template_[0])
        #         dynamic_template_temp = torch.cat(dynamic_template_list,dim=0)
        #         dynamic_template_ = dynamic_template_temp
        # else:
        #     dynamic_template_ = dynamic_template
  
        x, aux_dict,dynamic_template,dynamic_last = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,dynamic_template=dynamic_template_,dynamic_last =dynamic_last,Test=Test)
        dynamic_template_save = dynamic_template.detach()
        dynamic_last_save = dynamic_last.detach()

        # dynamic_template_save = None
        return x, aux_dict,dynamic_template_save,dynamic_last_save


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
