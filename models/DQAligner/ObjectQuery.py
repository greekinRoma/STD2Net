import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectQueryModule(nn.Module):
    def __init__(self, channels=[16, 32, 64], cls_objects=1):
        super().__init__()
        self.cls_objects = cls_objects
        self.channels = channels
        self.kv_encoder = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1, groups=self.channels[0]),
            nn.Conv2d(self.channels[0], 2 * self.channels[0], kernel_size=1))

        # 目标查询向量 (可学习的目标表示)
        self.object_queries = nn.Parameter(torch.randn(1, cls_objects, self.channels[0]))  # channels

        # 目标注意力
        self.object_attention = nn.MultiheadAttention(self.channels[0], 1, batch_first=True)
        self.object_update = nn.GRUCell(self.channels[0], self.channels[0])

    def motion_consistency_loss(self, object_tracks):
        B, T, num_objects, C = object_tracks.shape

        # 计算相邻时间步的目标表示相似度
        temp_consistency_loss = 0
        for t in range(T - 1):
            curr_objs = object_tracks[:, t]  # [B, num_objects, C]
            next_objs = object_tracks[:, t + 1]  # [B, num_objects, C]

            # 使用余弦相似度衡量目标表示的一致性
            similarity = F.cosine_similarity(curr_objs, next_objs, dim=2)  # [B, num_objects]
            temp_consistency_loss += (1 - similarity).mean()

        return temp_consistency_loss / (T - 1)

    def forward(self, x_enhanced):
        """
        或者三个尺度的特征 3 [B, C, T, H, W]
        x_enhanced: [B, C, T, H, W] - GMC_Att输出的浅层增强特征
        """
        B, C, T, H, W = x_enhanced.shape  # x_enhanced[0]
        object_states = self.object_queries.repeat(B, 1, 1)  # [B, 1, C]  self.object_queries.
        object_tracks = []

        for t in range(T):
            curr_feat = x_enhanced[:, :, t]  # x_enhanced[:, :, t]  # [B, C, H, W]
            feat_kv = self.kv_encoder(curr_feat)  # [B, 2C, H, W]
            feat_k, feat_v = torch.chunk(feat_kv, 2, dim=1)  # 各自形状: [B, C, H, W]

            feat_k = feat_k.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            feat_v = feat_v.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            attn_output, _ = self.object_attention(
                object_states,  # 查询：目标状态 [B, 1, C]
                feat_k,  # 键：特征图 [B, H*W, C]
                feat_v  # 值：特征图 [B, H*W, C]
            )  # [B, num_objects, C]

            object_states_flat = object_states.reshape(-1, C)  # [B*cls_objects, C]
            attn_output_flat = attn_output.reshape(-1, C)  # [B*cls_objects, C]

            updated_states = self.object_update(attn_output_flat, object_states_flat)  # [B*cls_objects, C]
            object_states = updated_states.reshape(B, self.cls_objects, C)  # [B, cls_objects, C]
            object_tracks.append(object_states)

            # 堆叠所有时间步的目标状态
        object_tracks = torch.stack(object_tracks, dim=1)  # [B, T, num_objects, C]
        object_sim_loss = self.motion_consistency_loss(object_tracks)

        return object_sim_loss, object_tracks


class ObjectGuidedEnhancement(nn.Module):
    def __init__(self, feat_dim, query_dim=16):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, feat_dim)
        self.key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.output_proj = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(feat_dim),
                                         nn.ReLU(inplace=True))

    def forward(self, features, object_queries):
        """
        features: [B, C, H, W] - 对齐后单尺度单帧特征
        object_queries: [B, 1, Q] - 目标查询状态
        """
        B, C, H, W = features.shape
        _, N, Q = object_queries.shape

        curr_feat = features  # [B, C, H, W]  [:,:,t]
        curr_queries = object_queries  # [B, N, Q]  [:, t]

        proj_queries = self.query_proj(curr_queries)  # [B, N, C]
        keys = self.key_conv(curr_feat)  # [B, C, H, W]
        keys_flat = keys.flatten(2)  # [B, C, H*W]

        attn = torch.bmm(
            proj_queries,  # [B, N, C]
            keys_flat  # [B, C, H*W]
        )  # [B, N, H*W]

        attn = attn / (C ** 0.5)  # 缩放
        attn = attn.view(B, N, H, W)
        mask = attn.sigmoid()  # [B, N, H, W]

        enhanced = curr_feat * (1.0 + mask)
        enhanced = self.output_proj(enhanced)

        # 堆叠所有增强的帧
        return enhanced, attn
