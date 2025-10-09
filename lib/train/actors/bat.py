import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu

from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np

class BATActor(BaseActor):
    """ Actor for training BAT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # print('data',data)
        # forward pass
        out_dict = self.forward_pass(data)
        # print("out_dict",out_dict)
        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1
        # print('seq_name',data['seq_name'])
        # print('data',data['search_anno'])
        seq_name = data['seq_name'][0]
        # print('seq_name',seq_name)
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]
        # print('ce_keep_rate',ce_keep_rate) 
        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            seq_name =seq_name,
                            )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
        
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        
        # emd_loss = self.emd_loss(gt_boxes_vec,pred_boxes_vec)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss+0.2*pred_dict['ce_loss']
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()
                      }
            return loss, status
        else:
            return loss

    # def compute_track_loss(self,gt_track,pred_track):
    #     gt_centroids = (gt_track[:, :2] + gt_track[:, 2:]) / 2
    #     pred_centeroids = (pred_track[:, :2] + pred_track[:, 2:]) / 2
    #     sequence = torch.arange(1, 33, 1)   #构建时间序列
    #     # 合并 x, y 坐标和时序，构建三维坐标
    #     true_3d_coords = torch.cat((gt_centroids, sequence.unsqueeze(1)), dim=1)
    #     predicted_3d_coords = torch.cat((pred_centeroids, sequence.unsqueeze(1)), dim=1)
    #     # 计算两个点集之间的成对距离
    #     distances = F.pairwise_distance(true_3d_coords.unsqueeze(1), predicted_3d_coords.unsqueeze(0))
    
    # 定义EMD损失函数
    def emd_loss(self,gt_track,pred_track):
        device = torch.device('cuda:0')
        gt_centroids = (gt_track[:, :2] + gt_track[:, 2:]) / 2
        pred_centeroids = (pred_track[:, :2] + pred_track[:, 2:]) / 2
        sequence = torch.arange(1, 33, 1).to(device) #构建时间序列
        # 合并 x, y 坐标和时序，构建三维坐标
        true_3d_coords = torch.cat((gt_centroids, sequence.unsqueeze(1)), dim=1)
        predicted_3d_coords = torch.cat((pred_centeroids, sequence.unsqueeze(1)), dim=1)

        # 确保输入是NumPy数组
        predicted_points_np = predicted_3d_coords.cpu().detach().numpy()
        true_points_np = true_3d_coords.cpu().detach().numpy()

        x,y,z = predicted_points_np[:, 0],predicted_points_np[:, 1],predicted_points_np[:, 2]
        x2,y2,z2 = true_points_np[:, 0],true_points_np[:, 1],true_points_np[:, 2]

        distance_matrix = np.sqrt(np.power(x-x2,2)+np.power(y-y2,2)+np.power(z-z2,2))


        # # 计算两个点集之间的成对距离矩阵
        # predicted_distance_matrix = cdist(predicted_points_np, predicted_points_np)
        # true_distance_matrix = cdist(true_points_np, true_points_np)

        # # 初始化一个用于累积最小成本的变量
        # min_cost = 0

        # # 计算所有可能的匹配的成本
        # for i in range(len(predicted_points_np)):
        #     for j in range(len(true_points_np)):
        #         min_cost += np.minimum(predicted_distance_matrix[i, :], true_distance_matrix[j, :])

        # # EMD是最小成本的平均值
        # emd = min_cost / (len(predicted_points_np) * len(true_points_np))
        # # print('emd',emd)
        emd = torch.from_numpy(distance_matrix).to(device).mean() * 0.1
        # print('emd_torch',emd)
        return emd












        # # 计算两个点集之间的成对距离
        # distances = F.pairwise_distance(true_3d_coords.unsqueeze(1), predicted_3d_coords.unsqueeze(0))
        # # 将点集转换为NumPy数组
        # predicted_np = predicted_3d_coords.cpu().detach().numpy()
        # true_np = true_3d_coords.cpu().detach().numpy()
        
        # # 计算成对距离矩阵
        # distance_matrix = squareform(pdist(predicted_np))
        # true_distance_matrix = squareform(pdist(true_np))
        
        # # 计算EMD
        # emd = wasserstein_distance(distance_matrix, true_distance_matrix)
        
        # # 将EMD转换为PyTorch张量
        # emd_tensor = torch.tensor([emd], dtype=torch.float32)
        # emd_loss = emd_tensor.mean()
        
        # return emd_loss


        
