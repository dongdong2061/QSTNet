import torch
import torch.nn as nn
import numpy as np


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim ** -0.5
       
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, T, S):
        q = self.linear_q(T)  # query
        k = self.linear_k(S)  # key
        v = self.linear_v(S)  # value
        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale
        attn_weights = self.softmax(attn_weights)
        
        # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, v)
        
        # 输出结果
        output = self.linear_out(attn_output)
        return output
class CrossAttention_For_Fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention_For_Fusion, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim ** -0.5
        self.proj_drop = nn.Dropout(0.1)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        #For visualize
        self.ReLU = nn.ReLU()
    
    def forward(self, T, x,xi):
        B,_,_ = T.shape
        x_ori = x
        xi_ori = xi
        x = x[:,128:]
        xi = xi[:,128:]
        q = self.linear_q(T)  # query
        # q_r = self.linear_q(x)  # query
        # q_t = self.linear_q(xi)  # query
        k = self.linear_k(x)  # key
        ki = self.linear_k(xi)  # key
        v = self.linear_v(x)
        vi = self.linear_v(xi)  # value
        # print(self.linear_q.weight.grad)
        #q torch.Size([32, 64, 768])
        #k torch.Size([32, 256, 768])
        # print('q',q.size())
        # print('k',k.size())
        # print('ki',ki.size())
        # print('v',v.size())
        #增加两个模态搜索区域的交叉注意力，利用另一个模态对当前模态增强
        # attn_weights_R = torch.matmul(q_r, ki.transpose(-2, -1))* self.scale
        # attn_weights_R = self.softmax(attn_weights_R)
        # x_r = attn_weights_R @ v
        # attn_weights_T = torch.matmul(q_t, k.transpose(-2, -1))* self.scale
        # attn_weights_T = self.softmax(attn_weights_T)
        # x_t = attn_weights_R @ vi

        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale

        #attn_weightsi torch.Size([32, 64, 192])
        attn_weightsi = torch.matmul(q, ki.transpose(-2, -1))* self.scale



        # weights = torch.cat([attn_weights,attn_weightsi],dim=-1)
        # weights = self.softmax(weights)
        # attn_weights,attn_weightsi = torch.chunk(weights,2,dim=-1)
        
        #attn_weights torch.Size([32, 64, 256])
        attn_weights = self.softmax(attn_weights)
        attn_weightsi = self.softmax(attn_weightsi)
        # vs_weight = attn_weights - attn_weightsi
        # 进行矩阵相减 A - B
        # print('attn_weights',attn_weights)
        # print('attn_weightsi',attn_weightsi)
        C = torch.sub(attn_weights, attn_weightsi)
        # zeros_count = torch.sum(C==0,dim=2)
        # print('C',zeros_count)
        # 如果结果为0，则将其置为0.5
        result = torch.where(C == 0, 0.5 * torch.ones_like(C), torch.where(C > 0, torch.ones_like(C), torch.zeros_like(C)))

        # 如果结果为负数，则将其置为0
        result = torch.where(C < 0, torch.zeros_like(C), result)

        #计算tensor中1的和
        count_ones_rgb = torch.sum(result == 1).item()
        count_zeroes_rgb = torch.sum(result == 0).item()
        
        #计算每列中1的数量
        column_sum = torch.sum(result==1,dim=1)
        tir_colum_sum = 128 - column_sum
        # one_count = torch.sum(result==1,dim=1)
        half_count = torch.sum(result==0.5,dim=1)
        # print('one_count',one_count)
        # print('half_count',half_count)
        #colum_sum torch.Size([32,1, 256])
        # print('colum_sum',column_sum.size())
        # 计算行数
        num_rows = result.size(1)
        # print('result.size(1)',result.size(1))
        # 计算每列中1的数量占行数的比例
        rgb_column_ratio = column_sum.float() / num_rows  # Convert column_sum to float for division
        # for weight visualization
        tir_ratio = tir_colum_sum.float() / num_rows
        # ratio = torch.cat((rgb_column_ratio.unsqueeze(-1),tir_ratio.unsqueeze(-1)),axis=-1)
        # ratio = self.softmax(ratio)
        # r,t = torch.chunk(ratio,2,dim=-1)
        # r,t = r.squeeze(dim=-1),t.squeeze(dim=-1)
        sub_rgb = rgb_column_ratio - tir_ratio
        sub_tir =  tir_ratio - rgb_column_ratio

        # vis_result = self.relu(sub_rgb)
        # vis_result[vis_result == 1] = 0.5
        # vis_result_i = self.relu(sub_tir)
        # vis_result_i[vis_result_i==1]=0
        # rgb_column_ratio[rgb_column_ratio == 1] = 0.5
        # tir_ratio[tir_ratio == 1] = 0.5
        # rgb_column_ratio[rgb_column_ratio == 0] = 0.5
        # tir_ratio[tir_ratio == 0] = 0.5
        # vis_result = rgb_column_ratio
        # vis_result_i = tir_ratio
        vis_result = (rgb_column_ratio - rgb_column_ratio.mean()) / (rgb_column_ratio.max() - rgb_column_ratio.min())
        vis_result_i = (tir_ratio - tir_ratio.mean()) / (tir_ratio.max() - tir_ratio.min())
        # vis_result = self.softmax(rgb_column_ratio)
        # vis_result_i = self.softmax(tir_ratio)
        # vis_result = r
        # vis_result_i = t
        # vis_result = rgb_column_ratio
        # vis_result_i = tir_ratio

        equal_ratio = half_count.float() / num_rows
        rgb_column_ratio = rgb_column_ratio.unsqueeze(-1)
        equal_ratio = equal_ratio.unsqueeze(-1)
        # rgb_column_ratio = rgb_column_ratio+1
        #rgb_column_ratio torch.Size([32, 256,1])
        # print('rgb_column_ratio',rgb_column_ratio.size(),rgb_column_ratio)
        
        tir_ratio = 1 - rgb_column_ratio- equal_ratio
        rgb_x = v*rgb_column_ratio + v*equal_ratio
        tir_xi = vi*tir_ratio + vi*equal_ratio
        x_s = rgb_x + tir_xi
        # print('x_s',x_s.size())
        # print('T',T.size())
        # # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, v)
        x_output = self.linear_out(attn_output)
        x_output = self.proj_drop(x_output)
        attn_outputi = torch.matmul(attn_weightsi, vi)
        xi_output = self.linear_out(attn_outputi)
        xi_output = self.proj_drop(xi_output)
        # print('attn_output',attn_output.size())
        # print('attn_outputi',attn_outputi.size())
        x_s = x_s 
        T = T + x_output + xi_output
        T = self.norm(T)


        # # 输出结果
        # output = self.linear_out(attn_output)
        return x_s,T,vis_result,vis_result_i,attn_weights,attn_weightsi

#for validation
class CrossModal_Fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModal_Fusion,self).__init__()
        self.adap_cross = CrossAttention(hidden_dim=768)
        # self.adap_down = nn.Linear(1536, 768)
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.relu = nn.ReLU()

    # def forward(self,x,x_z,xi,xi_z):
    def forward(self,x,xi):
        B,_,_ = x.shape
        z = x[:, :128]
        x_ = x[:, 128:]
        zi = xi[:, :128]
        xi_ = xi[:, 128:]
        # print(x.size())
        z = torch.cat((z,zi),dim=-1)
        # print('z1',z.size())
        # z = self.adap_down(z)
        # print('z2',z.size())
        output = self.adap_cross(x,xi)    
        return output



#for TQRT
class CrossModal_ST_Fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModal_ST_Fusion,self).__init__()
        self.adap_cross = CrossAttention_For_Fusion(hidden_dim=768)
        self.adap_down = nn.Linear(1536, 768)
        # self.adap_attn = CrossAttention(hidden_dim=768)
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.relu = nn.ReLU()

    # def forward(self,x,x_z,xi,xi_z):
    def forward(self,x,xi):
        B,_,_ = x.shape
        z = x[:, :128]
        x_ = x[:, 128:]
        zi = xi[:, :128]
        xi_ = xi[:, 128:]
        # print(x.size())
        #----for TQRT----#
        z = torch.cat((z,zi),dim=-1)
        # print('z1',z.size())
        z = self.adap_down(z)
        
        #for valid
        # z = self.adap_attn(zi,z)
        # zi = self.adap_attn(z,zi)
        # z = z + zi   
        
        
        # print('z2',z.size())
        output,T,vis_result,vis_result_i,attn_weights,attn_weightsi = self.adap_cross(z,x,xi)
        # output = self.norm1(output)
        # outputi = self.norm2(outputi)
        # x = torch.cat([output,x_],dim=1)
        # xi = torch.cat([outputi,xi_],dim=1)
        x_out = torch.cat((T,output),dim=1)
        return x_out,vis_result,vis_result_i,attn_weights,attn_weightsi

#for spatio-temporal
class CrossAttention_For_Fusion_Temporal(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention_For_Fusion_Temporal, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim ** -0.5
        self.proj_drop = nn.Dropout(0.1)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, T, x,xi):
        B,_,_ = T.shape
        x_ori = x
        xi_ori = xi
        x = x[:,128:]
        xi = xi[:,128:]
        q = self.linear_q(T)  # query
        q_r = self.linear_q(x)  # query
        q_t = self.linear_q(xi)  # query
        k = self.linear_k(x)  # key
        ki = self.linear_k(xi)  # key
        v = self.linear_v(x)
        vi = self.linear_v(xi)  # value
        # print(self.linear_q.weight.grad)
        #q torch.Size([32, 64, 768])
        #k torch.Size([32, 256, 768])
        # print('q',q.size())
        # print('k',k.size())
        # print('ki',ki.size())
        # print('v',v.size())
        #增加两个模态搜索区域的交叉注意力，利用另一个模态对当前模态增强
        # attn_weights_R = torch.matmul(q_r, ki.transpose(-2, -1))* self.scale
        # attn_weights_R = self.softmax(attn_weights_R)
        # x_r = attn_weights_R @ v
        # attn_weights_T = torch.matmul(q_t, k.transpose(-2, -1))* self.scale
        # attn_weights_T = self.softmax(attn_weights_T)
        # x_t = attn_weights_R @ vi

        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale
        #attn_weights torch.Size([32, 64, 256])
        attn_weights = self.softmax(attn_weights)
        #attn_weightsi torch.Size([32, 64, 192])
        attn_weightsi = torch.matmul(q, ki.transpose(-2, -1))* self.scale
        attn_weightsi = self.softmax(attn_weightsi)
        # vs_weight = attn_weights - attn_weightsi
        # 进行矩阵相减 A - B
        # print('attn_weights',attn_weights)
        # print('attn_weightsi',attn_weightsi)
        C = torch.sub(attn_weights, attn_weightsi)
        # zeros_count = torch.sum(C==0,dim=2)
        # print('C',zeros_count)
        # 如果结果为0，则将其置为0.5
        result = torch.where(C == 0, 0.5 * torch.ones_like(C), torch.where(C > 0, torch.ones_like(C), torch.zeros_like(C)))

        # 如果结果为负数，则将其置为0
        result = torch.where(C < 0, torch.zeros_like(C), result)

        vis_result = result

        #计算每列中1的数量
        column_sum = torch.sum(result==1,dim=1)

        
        # one_count = torch.sum(result==1,dim=1)
        half_count = torch.sum(result==0.5,dim=1)
        # print('one_count',one_count)
        # print('half_count',half_count)
        #colum_sum torch.Size([32,1, 256])
        # print('colum_sum',column_sum.size())
        # 计算行数
        num_rows = result.size(1)
        # print('result.size(1)',result.size(1))
        # 计算每列中1的数量占行数的比例
        rgb_column_ratio = column_sum.float() / num_rows  # Convert column_sum to float for division
        equal_ratio = half_count.float() / num_rows
        rgb_column_ratio = rgb_column_ratio.unsqueeze(-1)
        equal_ratio = equal_ratio.unsqueeze(-1)
        # rgb_column_ratio = rgb_column_ratio+1
        #rgb_column_ratio torch.Size([32, 256,1])
        # print('rgb_column_ratio',rgb_column_ratio.size(),rgb_column_ratio)
        
        rgb_x = v*rgb_column_ratio + v*equal_ratio
        tir_xi = vi*tir_ratio + vi*equal_ratio
        x_s = rgb_x + tir_xi
        # print('x_s',x_s.size())
        # print('T',T.size())
        # # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, v)
        x_output = self.linear_out(attn_output)
        x_output = self.proj_drop(x_output)
        attn_outputi = torch.matmul(attn_weightsi, vi)
        xi_output = self.linear_out(attn_outputi)
        xi_output = self.proj_drop(xi_output)
        # print('attn_output',attn_output.size())
        # print('attn_outputi',attn_outputi.size())
        x_s = x_s 
        T = T + x_output + xi_output
        T = self.norm(T)


        # # 输出结果
        # output = self.linear_out(attn_output)
        return x_s,T,vis_result

class CrossModal_ST_Fusion_Temporal(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModal_ST_Fusion_Temporal,self).__init__()
        self.adap_cross = CrossAttention_For_Fusion_Temporal(hidden_dim=768)
        self.adap_down = nn.Linear(1536, 768)
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.relu = nn.ReLU()

    # def forward(self,x,x_z,xi,xi_z):
    def forward(self,x,xi):
        B,_,_ = x.shape
        z = x[:, :128]
        x_ = x[:, 128:]
        zi = xi[:, :128]
        xi_ = xi[:, 128:]
        # print(x.size())
        z = torch.cat((z,zi),dim=-1)
        # print('z1',z.size())
        z = self.adap_down(z)
        # print('z2',z.size())
        output,T,vis_result = self.adap_cross(z,x,xi)
        # output = self.norm1(output)
        # outputi = self.norm2(outputi)
        # x = torch.cat([output,x_],dim=1)
        # xi = torch.cat([outputi,xi_],dim=1)
        x_out = torch.cat((T,output),dim=1)
        return x_out,T,vis_result



class CrossAttention_TS(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention_TS, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim ** -0.5
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, T, S,Ti,Si):
        q = self.linear_q(T)  # query
        k = self.linear_k(S)  # key
        v = self.linear_v(S)  # value
        # qi = self.linear_q(Ti)  # query
        ki = self.linear_k(Si)  # key
        vi = self.linear_v(Si)  # value
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))* self.scale
        attn_weights = self.softmax(attn_weights)
        # 计算注意力权重
        attn_weightsi = torch.matmul(q, ki.transpose(-2, -1))* self.scale
        attn_weightsi = self.softmax(attn_weightsi)
        
        # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, v)
        attn_outputi = torch.matmul(attn_weights, vi)
        # attn_output_cross = torch.matmul(attn_weightsi, v)
        # attn_outputi_cross = torch.matmul(attn_weights, vi)
        
        # 输出结果
        output = self.linear_out(attn_output) + T
        outputi = self.linear_out(attn_outputi) + Ti
        return output,outputi,attn_weights,attn_weightsi
class CrossModal_Templates_Enhancement(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModal_Templates_Update,self).__init__()
        self.adap_cross = CrossAttention_TS(hidden_dim=768)
        # self.adap_down = nn.Linear(768, 64) 
        # self.adap_up = nn.Linear(64, 768)  
        # self.adap_conv5x5 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        # self.adap_conv3x3 = nn.Conv2d(64, 64,kernel_size=3, stride=2, padding=1) 
        # self.adap_BN3 = nn.BatchNorm2d(64)
        # self.adap_BN5 = nn.BatchNorm2d(64)
        # self.adap_BN3i = nn.BatchNorm2d(64)
        # self.adap_BN5i = nn.BatchNorm2d(64)
        self.adap_max1 = torch.nn.MaxPool2d(3,stride=2,padding=1)
        self.adap_max2 = torch.nn.MaxPool2d(3,stride=2,padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    # def forward(self,x,x_z,xi,xi_z):
    def forward(self,x,xi):
        B,_,_ = x.shape
        z = x[:, :64]
        x_ = x[:, 64:]
        zi = xi[:, :64]
        xi_ = xi[:, 64:]
        x_z = x_
        # x_z = self.adap_down(x_z)
        x_z = x_z.permute(0,2,1).view(B,768,16,16)
        # x3 = self.adap_conv3x3(x_z)
        # x5 = self.adap_conv5x5(x_z)
        # x3 = self.adap_BN3(x3)
        # x5 = self.adap_BN5(x5)
        # print('x3',x3.size())
        # print('x5',x5.size())
        # x_z = x3+x5
        # x_z = self.relu(x_z)
        x_z = self.adap_max1(x_z)
        x_z = x_z.view(B,768,64).permute(0,2,1)
        # x_z = self.adap_up(x_z)
        x_zi_ = xi_
        # x_zi = self.adap_down(x_zi_)
        x_zi = x_zi.permute(0,2,1).view(B,64,16,16)
        # xi3 = self.adap_conv3x3(x_zi)
        # xi3 = self.adap_BN3i(xi3)
        # xi5 = self.adap_conv5x5(x_zi)
        # xi5 = self.adap_BN5i(xi5)
        # print('xi5',xi5.size())
        # x_zi = xi3+xi5
        # x_zi = self.relu(x_zi)
        x_zi = self.adap_max2(x_zi)
        x_zi = x_zi.view(B,768,64).permute(0,2,1)
        # x_zi = self.adap_up(x_zi)
        # z = self.adap_cross(z,x_z)
        # zi = self.adap_cross(zi,x_zi)
        output,outputi,attn_weights,attn_weightsi = self.adap_cross(z,x_z,zi,x_zi)
        output = self.norm1(output)
        outputi = self.norm2(outputi)
        x = torch.cat([output,x_],dim=1)
        xi = torch.cat([outputi,xi_],dim=1)
        return x,xi,attn_weights,attn_weightsi


class CrossModal_Templates_Update(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModal_Templates_Update,self).__init__()
        self.adap_cross = CrossAttention_TS(hidden_dim=768)
        self.adap_down = nn.Linear(768, 64) 
        self.adap_up = nn.Linear(64, 768)  
        self.adap_conv5x5 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.adap_conv3x3 = nn.Conv2d(64, 64,kernel_size=3, stride=2, padding=1) 
        self.adap_BN3 = nn.BatchNorm2d(64)
        self.adap_BN5 = nn.BatchNorm2d(64)
        self.adap_BN3i = nn.BatchNorm2d(64)
        self.adap_BN5i = nn.BatchNorm2d(64)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    # def forward(self,x,x_z,xi,xi_z):
    def forward(self,x,xi):
        B,_,_ = x.shape
        z = x[:, :64]
        x_ = x[:, 64:]
        zi = xi[:, :64]
        xi_ = xi[:, 64:]
        x_z = x_
        x_z = self.adap_down(x_z)
        x_z = x_z.permute(0,2,1).view(B,64,16,16)
        x3 = self.adap_conv3x3(x_z)
        x5 = self.adap_conv5x5(x_z)
        x3 = self.adap_BN3(x3)
        x5 = self.adap_BN5(x5)
        # print('x3',x3.size())
        # print('x5',x5.size())
        x_z = x3+x5
        x_z = self.relu(x_z)
        x_z = x_z.view(B,64,64).permute(0,2,1)
        x_z = self.adap_up(x_z)
        x_zi_ = xi_
        x_zi = self.adap_down(x_zi_)
        x_zi = x_zi.permute(0,2,1).view(B,64,16,16)
        xi3 = self.adap_conv3x3(x_zi)
        xi3 = self.adap_BN3i(xi3)
        xi5 = self.adap_conv5x5(x_zi)
        xi5 = self.adap_BN5i(xi5)
        # print('xi5',xi5.size())
        x_zi = xi3+xi5
        x_zi = self.relu(x_zi)
        x_zi = x_zi.view(B,64,64).permute(0,2,1)
        x_zi = self.adap_up(x_zi)
        # z = self.adap_cross(z,x_z)
        # zi = self.adap_cross(zi,x_zi)
        output,outputi,attn_weights,attn_weightsi = self.adap_cross(z,x_z,zi,x_zi)
        output = self.norm1(output)
        outputi = self.norm2(outputi)
        x = torch.cat([output,x_],dim=1)
        xi = torch.cat([outputi,xi_],dim=1)
        return x,xi,attn_weights,attn_weightsi

