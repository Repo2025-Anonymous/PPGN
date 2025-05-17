import model.resnet as resnet
import torch
import math
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import Dinov2Model

class similarity_func(nn.Module):
    def __init__(self):
        super(similarity_func, self).__init__()
    def forward(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

class masked_average_pooling(nn.Module):
    def __init__(self):
        super(masked_average_pooling, self).__init__()
    def forward(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
class proto_generation(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_average_pooling = masked_average_pooling()
    def forward(self, fea_list, mask_list):
        feature_fg_protype_list = []
        feature_bg_protype_list = []

        for k in range(len(fea_list)):
            feature_fg_protype = self.masked_average_pooling(fea_list[k], (mask_list[k] == 1).float())[None, :]
            feature_bg_protype = self.masked_average_pooling(fea_list[k], (mask_list[k] == 0).float())[None, :]
            feature_fg_protype_list.append(feature_fg_protype)
            feature_bg_protype_list.append(feature_bg_protype)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_protype_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_protype_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        return FP, BP

class DynamicPrototypeGenerator(nn.Module):
    def __init__(self, channal):
        super(DynamicPrototypeGenerator, self).__init__()
        self.similarity_func = similarity_func()
        self.channal = channal
        self.fg_thres = nn.Parameter(torch.tensor(0.5))
        self.bg_thres = nn.Parameter(torch.tensor(0.5))

    def forward(self, res_fea, dinov2_fea, res_out, dinov2_out):
        bs = res_fea.shape[0]
        res_out = res_out.softmax(1).view(bs, 2, -1)
        dinov2_out = dinov2_out.softmax(1).view(bs, 2, -1)

        res_out_fg, res_out_bg = res_out[:, 1], res_out[:, 0]
        dinov2_out_fg, dinov2_out_bg = dinov2_out[:, 1], dinov2_out[:, 0]

        res_fg_ls = []
        res_bg_ls = []
        dinov2_fg_ls = []
        dinov2_bg_ls = []

        for epi in range(bs):
            fg_thres = self.fg_thres
            bg_thres = self.bg_thres

            res_cur_feat = res_fea[epi].view(self.channal, -1)
            dinov2_cur_feat = dinov2_fea[epi].view(self.channal, -1)

            if (((res_out_fg[epi] > fg_thres).bool() & (dinov2_out_fg[epi] > fg_thres).bool())).sum() > 0:
                res_fg_feat = res_cur_feat[:, ((res_out_fg[epi] > fg_thres).bool() & (dinov2_out_fg[epi] > fg_thres).bool())]    # .mean(-1)
                dinov2_fg_feat = dinov2_cur_feat[:, ((res_out_fg[epi] > fg_thres).bool() & (dinov2_out_fg[epi] > fg_thres).bool())]                # .mean(-1)

            else:
                res_fg_feat = res_cur_feat[:, torch.topk(res_out_fg[epi], 12).indices]  # .mean(-1)
                dinov2_fg_feat = dinov2_cur_feat[:, torch.topk(dinov2_out_fg[epi], 12).indices]  # .mean(-1)

            if (((res_out_bg[epi] > bg_thres).bool() & (dinov2_out_bg[epi] > bg_thres).bool())).sum() > 0:
                res_bg_feat = res_cur_feat[:, ((res_out_bg[epi] > bg_thres).bool() & (dinov2_out_bg[epi] > bg_thres).bool())]  # .mean(-1)
                dinov2_bg_feat = dinov2_cur_feat[:, ((res_out_bg[epi] > bg_thres).bool() & (dinov2_out_bg[epi] > bg_thres).bool())]  # .mean(-1)

            else:
                res_bg_feat = res_cur_feat[:, torch.topk(res_out_bg[epi], 12).indices]  # .mean(-1)
                dinov2_bg_feat = dinov2_cur_feat[:, torch.topk(dinov2_out_bg[epi], 12).indices]  # .mean(-1)

            # global proto
            res_fg_proto = res_fg_feat.mean(-1, keepdim=True).transpose(-2, -1)
            res_bg_proto = res_bg_feat.mean(-1, keepdim=True).transpose(-2, -1)
            dinov2_fg_proto = dinov2_fg_feat.mean(-1, keepdim=True).transpose(-2, -1)
            dinov2_bg_proto = dinov2_bg_feat.mean(-1, keepdim=True).transpose(-2, -1)

            res_fg_ls.append(res_fg_proto)
            res_bg_ls.append(res_bg_proto)
            dinov2_fg_ls.append(dinov2_fg_proto)
            dinov2_bg_ls.append(dinov2_bg_proto)

        # global proto
        res_fg_proto = torch.cat(res_fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        res_bg_proto = torch.cat(res_bg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        dinov2_fg_proto = torch.cat(dinov2_fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        dinov2_bg_proto = torch.cat(dinov2_bg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        return res_fg_proto, res_bg_proto, dinov2_fg_proto, dinov2_bg_proto


class GaussianModel(nn.Module):
    def __init__(self, dim):
        super(GaussianModel, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.fc_mean = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc_logvar = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)

    def reparameterize(self, mu, logvar, k):
        std = logvar.mul(0.5).exp()
        eps = torch.randn(k, *mu.shape, device=mu.device)
        sample_z = mu + eps * std
        return sample_z

    def forward(self, res_prototype, dinov2_prototype):

        fused_proto = self.alpha * res_prototype + self.beta * dinov2_prototype

        mean = self.fc_mean(fused_proto)
        log_var = self.fc_logvar(fused_proto)

        samples = self.reparameterize(mean, log_var, k=50)

        uncertainty = samples.var(dim=0)
        uncertainty = 10 * (uncertainty / uncertainty.norm(dim=1, keepdim=True))

        fused_proto = (1 - uncertainty) * fused_proto

        fused_proto = fused_proto + res_prototype + dinov2_prototype
        return fused_proto, mean, log_var

class PseudoGenerator(nn.Module):
    def __init__(self, dim):
        super(PseudoGenerator, self).__init__()
        self.similarity_func = similarity_func()

        self.fused_gs_fp = GaussianModel(dim)
        self.fused_gs_bp = GaussianModel(dim)

        self.dpg = DynamicPrototypeGenerator(dim)

    def forward(self, res_supp_fp, res_supp_bp, res_query_fea, dinov2_supp_fp, dinov2_supp_bp, dinov2_query_fea):

        # step1: 用support的原型得到query的第一次分割结果
        res_query_out = self.similarity_func(res_query_fea, res_supp_fp, res_supp_bp)
        dinov2_query_out = self.similarity_func(dinov2_query_fea, dinov2_supp_fp, dinov2_supp_bp)

        # step2: 把query_out当做伪标签，利用ssp得到query的原型
        res_query_fp, res_query_bp = self.res_dpg(res_query_fea, res_query_out)
        dinov2_query_fp, dinov2_query_bp = self.dinov2_dpg(dinov2_query_fea, dinov2_query_out)

        fused_query_fp, mean_query_fp, log_var_query_fp = self.fused_gs_fp(res_query_fp, dinov2_query_fp)
        fused_query_bp, mean_query_bp, log_var_query_bp = self.fused_gs_bp(res_query_bp, dinov2_query_bp)

        return res_query_out, dinov2_query_out, fused_query_fp, fused_query_bp, mean_query_fp, log_var_query_fp, mean_query_bp, log_var_query_bp


class IFA_MatchingNet(nn.Module):
    def __init__(self, backbone, shot=1):
        super(IFA_MatchingNet, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3

        self.dinov2_vit_base = Dinov2Model.from_pretrained("./pretrained/dinov2-base")

        self.shot = shot
        self.similarity_func = similarity_func()
        self.proto_generation = proto_generation()

        self.DynamicPrototypeGenerator = DynamicPrototypeGenerator(channal=768)
        self.fused_uncertainty_fp = GaussianModel(dim=768)
        self.fused_uncertainty_bp = GaussianModel(dim=768)

        self.conv = nn.Conv2d(1024, 768, 1)

        # 初始化可学习的参数alpha和beta,用于将ResNet50和DINOv2的分割结果进行融合
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gama = nn.Parameter(torch.tensor(0.5))
        self.lamta = nn.Parameter(torch.tensor(0.5))

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):

        b, c, h, w = img_q.shape

        # feature maps of support images
        res_supp_feat_list = []

        #   使用ResNet50提取support的特征
        for k in range(len(img_s_list)):
            res_s_0 = self.layer0(img_s_list[k])                               #  s_0: [128, 100, 100]
            res_s_1 = self.layer1(res_s_0)                                     #  s_1: [256, 100, 100]
            res_s_2 = self.layer2(res_s_1)                                     #  s_2: [512,  50,  50]
            res_s_3 = self.layer3(res_s_2)                                     #  s_3: [1024, 50,  50]
            res_s_3 = self.conv(res_s_3)
            res_s_3 = F.normalize(res_s_3, p=2, dim=1)
            res_supp_feat_list.append(res_s_3)

        #   计算ResNet50分支support的原型
        res_supp_fp, res_supp_bp = self.proto_generation(res_supp_feat_list, mask_s_list)

        #   使用ResNet50提取query的特征
        res_q_0 = self.layer0(img_q)
        res_q_1 = self.layer1(res_q_0)
        res_q_2 = self.layer2(res_q_1)
        res_q_3 = self.layer3(res_q_2)
        res_q_3 = self.conv(res_q_3)
        res_q_3 = F.normalize(res_q_3, p=2, dim=1)

        # feature maps of support images
        dinov2_supp_feat_list = []

        #   使用DINNOv2提取support的特征
        for k in range(len(img_s_list)):
            dinov2_supp_features = self.dinov2_vit_base(F.interpolate(img_s_list[k], size=(700, 700), mode="bilinear", align_corners=True))
            dinov2_supp_features = dinov2_supp_features.last_hidden_state[:, 1:, :]                       # [B, 784, 768]
            dinov2_supp_features = dinov2_supp_features.reshape(b, 50, 50, 768).permute(0, 3, 1, 2)       # [B, 768, 28, 28]
            dinov2_supp_features = F.normalize(dinov2_supp_features, p=2, dim=1)
            dinov2_supp_feat_list.append(dinov2_supp_features)

        #   计算DINOv2分支support的原型
        dinov2_supp_fp, dinov2_supp_bp = self.proto_generation(dinov2_supp_feat_list, mask_s_list)

        #   使用DINOv2提取query的特征
        dinov2_query_features = self.dinov2_vit_base(F.interpolate(img_q, size=(700, 700), mode="bilinear", align_corners=True))
        dinov2_query_features = dinov2_query_features.last_hidden_state[:, 1:, :]                                       # [B, 784, 768]
        dinov2_query_features = dinov2_query_features.reshape(b, 50, 50, 768).permute(0, 3, 1, 2)                       # [B, 768, 28, 28]
        dinov2_query_features = F.normalize(dinov2_query_features, p=2, dim=1)

        #  利用从ResNet50分支和DINOv2分支得到的support的原型，对query第一次进行预测
        pred_res_query_1 = self.similarity_func(res_q_3, res_supp_fp, res_supp_bp)
        pred_dinov2_query_1 = self.similarity_func(dinov2_query_features, dinov2_supp_fp, dinov2_supp_bp)

        #  第一次使用DynamicPrototypeGenerator模块，得到resNet50和DINOv2分支的query的原型
        res_dpg_fg_proto_1, res_dpg_bg_proto_1, dinov2_dpg_fg_proto_1, dinov2_dpg_bg_proto_1 = (
            self.DynamicPrototypeGenerator(res_fea=res_q_3, dinov2_fea=dinov2_query_features, res_out=pred_res_query_1, dinov2_out=pred_dinov2_query_1))

        #  使用非确定性建模对获得的query的原型进行混合建模
        fused_query_fp, mean_query_fp, log_var_query_fp = self.fused_uncertainty_fp(res_dpg_fg_proto_1, dinov2_dpg_fg_proto_1)
        fused_query_bp, mean_query_bp, log_var_query_bp = self.fused_uncertainty_bp(res_dpg_bg_proto_1, dinov2_dpg_bg_proto_1)

        #  使用query的混合原型对query进行第二次预测
        pred_res_query_2 = self.similarity_func(res_q_3, fused_query_fp, fused_query_bp)
        pred_dinov2_query_2 = self.similarity_func(dinov2_query_features, fused_query_fp, fused_query_bp)

        #  第二次使用DynamicPrototypeGenerator模块，得到resNet50和DINOv2分支的query的原型
        res_dpg_fg_proto_2, res_dpg_bg_proto_2, dinov2_dpg_fg_proto_2, dinov2_dpg_bg_proto_2 = (
            self.DynamicPrototypeGenerator(res_fea=res_q_3, dinov2_fea=dinov2_query_features, res_out=pred_res_query_2, dinov2_out=pred_dinov2_query_2))

        #  使用query的原型对query进行第三次预测
        pred_res_query_3 = self.similarity_func(res_q_3, res_dpg_fg_proto_2, res_dpg_bg_proto_2)
        pred_dinov2_query_3 = self.similarity_func(dinov2_query_features, dinov2_dpg_fg_proto_2, dinov2_dpg_bg_proto_2)

        #   将ResNet50分支和DINOv2分支query的预测分割值进行融合
        query_out = self.alpha * pred_res_query_2 + self.beta * pred_res_query_3 + self.gama * pred_dinov2_query_2 + self.lamta * pred_dinov2_query_3

        #   对分割结果进行上采样
        pred_res_query_1 = F.interpolate(pred_res_query_1, size=(h, w), mode="bilinear", align_corners=True)                # q_out:[B, 2, 400, 400]
        pred_res_query_2 = F.interpolate(pred_res_query_2, size=(h, w), mode="bilinear", align_corners=True)                # q_out:[B, 2, 400, 400]
        pred_res_query_3 = F.interpolate(pred_res_query_3, size=(h, w), mode="bilinear", align_corners=True)                # q_out:[B, 2, 400, 400]
        pred_dinov2_query_1 = F.interpolate(pred_dinov2_query_1, size=(h, w), mode="bilinear", align_corners=True)          # q_out:[B, 2, 400, 400]
        pred_dinov2_query_2 = F.interpolate(pred_dinov2_query_2, size=(h, w), mode="bilinear", align_corners=True)          # q_out:[B, 2, 400, 400]
        pred_dinov2_query_3 = F.interpolate(pred_dinov2_query_3, size=(h, w), mode="bilinear", align_corners=True)          # q_out:[B, 2, 400, 400]
        query_out = F.interpolate(query_out, size=(h, w), mode="bilinear", align_corners=True)                              # q_out:[B, 2, 400, 400]

        return (query_out, pred_res_query_3, pred_res_query_2, pred_res_query_1, pred_dinov2_query_3, pred_dinov2_query_2,
                pred_dinov2_query_1, mean_query_fp, log_var_query_fp, mean_query_bp, log_var_query_bp)