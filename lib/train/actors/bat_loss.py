import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

class BATActor(BaseActor):
    """ Actor for training BAT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.gt_bbox_past = None
        self.pred_bbox_past=None

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
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status,[pred_boxes_vec,gt_boxes_vec] = self.compute_losses(out_dict, data)

        return loss, status ,[pred_boxes_vec,gt_boxes_vec]

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        # template_image = data['template_images']
        # search_images = data['search_images']

        # print('template_image',template_image.size())
        # print('search_iamges',search_images.size())

        # # loader使用torchvision中自带的transforms函数
        # loader = transforms.Compose([
        #     transforms.ToTensor()])  

        # unloader = transforms.ToPILImage()

        # np_array = search_images.detach().cpu().numpy()[0]
        # # 使用 OpenCV 保存为图像
        # image = cv2.imwrite('output.jpg', np_array)
        # print("成功保存图像！")

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)
        
        # for i in range(search_images.shape[0]):
        #     np_array = search_images.detach().cpu().numpy()[i]
        #     img_name = '/DATA/dingzhaodong/project/BAT/test_images/output_{}.jpg'.format(i)
        #     cv2.imwrite(img_name, np_array)
        #     print("成功保存图像！",img_name)

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

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # print('gt_dict',gt_dict)
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        # print('gt_bbox',gt_bbox)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        # print('pred_boxes',pred_boxes)
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        #[32,4]
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        
        # print('pred_boxes_vec',pred_boxes_vec)
        # print('gt_boxes_vec',gt_boxes_vec)
        
        if gt_dict['i'] == 1:
            pass
        else:
            bbox_past = gt_dict['bbox_past']
            gt_bbox_past,pred_bbox_past = bbox_past
            # print('pred_bbox_past',pred_bbox_past)
            # print('gt_bbox_past',gt_bbox_past)
            # direction_loss = self.compute_direct_loss(gt_boxes_vec, pred_boxes_vec,gt_bbox_past,pred_bbox_past)
            direction_loss = self.compute_loss(gt_boxes_vec, pred_boxes_vec)

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
        if gt_dict['i'] == 1:
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                status = {"Loss/total": loss.item(),
                        "Loss/giou": giou_loss.item(),
                        "Loss/l1": l1_loss.item(),
                        "Loss/location": location_loss.item(),
                        "IoU": mean_iou.item()
                        }
                return loss, status,[pred_boxes_vec,gt_boxes_vec]
            else:
                return loss
        else:
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss+direction_loss
            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                status = {"Loss/total": loss.item(),
                        "Loss/giou": giou_loss.item(),
                        "Loss/l1": l1_loss.item(),
                        "Loss/location": location_loss.item(),
                        "IoU": mean_iou.item(),
                        "Direction_loss":direction_loss}
                return loss, status,[pred_boxes_vec,gt_boxes_vec]
            else:
                return loss
        

    def compute_direct_loss(self,gt_bbox_current,pred_bbox_current,gt_bbox_past,pred_bbox_past):
        def bbox_center(bbox):
            # 取左上角和右下角坐标的平均值
            x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return torch.stack((center_x, center_y), dim=1)
        # 计算两个边界框的中心点
        center_gc = bbox_center(gt_bbox_current)
        center_pc = bbox_center(pred_bbox_current)
        center_gpc = bbox_center(gt_bbox_past)
        center_ppc = bbox_center(pred_bbox_past)
        vec_pred = center_pc - center_ppc
        vec_gt = center_gc - center_gpc
        # 归一化向量
        gt_vectors_norm = F.normalize(vec_gt, p=2, dim=1)
        pred_vectors_norm = F.normalize(vec_pred, p=2, dim=1)
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(gt_vectors_norm, pred_vectors_norm, dim=1)
        # 计算方向损失（1 - 余弦相似度）
        direction_loss = 1 - cosine_similarity
        average_direction_loss = torch.mean(direction_loss)
        # print("average_direction_loss:", average_direction_loss)
        return average_direction_loss
    
    def compute_loss(self,gt_bbox_current,pred_bbox_current):
        def bbox_center(bbox):
            # 取左上角和右下角坐标的平均值
            x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return torch.stack((center_x, center_y), dim=1)
        # 计算每一对连续帧之间的移动向量
        # 计算两个边界框的中心点
        center_gc = bbox_center(gt_bbox_current)
        center_pc = bbox_center(pred_bbox_current)
        # 对于真实值
        gt_movement_vectors = center_gc[1:] - center_gc[:-1]
        # 对于预测值
        pred_movement_vectors = center_pc[1:] - center_pc[:-1]

        # 计算向量长度（归一化向量）
        gt_vector_norms = torch.norm(gt_movement_vectors, p=2, dim=1)
        pred_vector_norms = torch.norm(pred_movement_vectors, p=2, dim=1)

        # 归一化移动向量
        gt_normalized_movement_vectors = gt_movement_vectors / gt_vector_norms.unsqueeze(-1)
        pred_normalized_movement_vectors = pred_movement_vectors / pred_vector_norms.unsqueeze(-1)

        # 计算余弦相似度
        cosine_similarity = torch.sum(gt_normalized_movement_vectors * pred_normalized_movement_vectors, dim=1)

        # 计算方向损失（1 - 余弦相似度）
        direction_losses = 1 - cosine_similarity

        # 计算方向损失的均值
        average_direction_loss = direction_losses.mean()

        # print("Average Direction Loss:", average_direction_loss.item())
        return average_direction_loss



