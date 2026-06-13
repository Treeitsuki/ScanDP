import copy
from typing import Dict, List, Optional, Tuple, Type, Union

import diffusion_policy_3d.model.vision_3d.point_process as point_process
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from termcolor import cprint

# from diffusion_policy_3d.model.vision_gridmap.model import PointTransformerV3, Point


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class ScanDPEncoder(nn.Module):
    def __init__(self,
                 observation_space: Dict,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 #  pointnet_type='dp3_encoder',
                 encoder_type='conv3d'
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.gridmap_key = 'gridmap'
        self.n_output_channels = pointcloud_encoder_cfg.out_channels

        # self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        self.gridmap_shape = observation_space[self.gridmap_key]

        cprint(f"[Encoder] gridmap shape: {self.gridmap_shape}", "yellow")
        cprint(f"[Encoder] state shape: {self.state_shape}", "yellow")

        if encoder_type == "conv3d":
            from .conv3d import Conv3DEncoder
            self.extractor = Conv3DEncoder()
        elif encoder_type == "pointnet2":
            from .pointnet2_utils import PointNet2Encoder
            self.extractor = PointNet2Encoder()
        elif encoder_type == "sparseconv3d":
            from .sparseconv3d import SparseConv3DEncoder

            # self.gridmap_shape = torch.tensor(self.gridmap_shape, device="cuda")
            input_channels = 1
            if pointcloud_encoder_cfg is not None and hasattr(pointcloud_encoder_cfg, "in_channels"):
                input_channels = pointcloud_encoder_cfg.in_channels
            self.extractor = SparseConv3DEncoder(
                self.gridmap_shape,
                input_channels=input_channels,
                output_channels=self.n_output_channels)
        elif encoder_type == "pointtransformerv3":
            # from model import PointTransformerV3, Point
            self.extractor = PointTransformerV3(cls_mode=True,
                                                in_channels=1,
                                                enc_channels=(
                                                    16, 32, 64, 128, 256),
                                                enable_flash=False,
                                                ).cuda()
        elif encoder_type == "sparseconv3d+pc":
            from .multi_stage_pointnet import MultiStagePointNetEncoder
            from .sparseconv3d import SparseConv3DEncoder
            self.extractor = SparseConv3DEncoder(
                self.gridmap_shape, output_channels=128)
            self.extractor_pc = MultiStagePointNetEncoder(out_channels=128)
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            raise NotImplementedError(f"pointnet_type: {encoder_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[Encoder] output dim: {self.n_output_channels}", "red")

        # self.resnet = torchvision.models.resnet18(pretrained=False)
        # self.resnet.fc = nn.Linear(512, 128)
        # self.resnet.cuda()

    def forward(self, observations: Dict) -> torch.Tensor:
        gridmap = observations[self.gridmap_key].unsqueeze(
            1)   # torch.Size([128, 1, 25, 25, 25])
        # assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # pn_feat = self.extractor(gridmap)  # B * out_channel
        if self.extractor.__class__.__name__ == "SparseConv3DEncoder":
            from .sparseconv3d import generate_sparse_representation
            features, coors, shape, batch_size, feat_dim = generate_sparse_representation(
                gridmap)
            features = features.cuda()
            coors = coors.cuda()
            shape = shape
            batch_size = torch.tensor(batch_size).cuda()
            pn_feat = self.extractor(features, coors, batch_size)

        elif self.extractor.__class__.__name__ == "PointTransformerV3":
            print("gridmap shape", gridmap.shape)
            gridmap.cuda()
            gridmap_dict = self.preprocess_ogm(gridmap)
            output = self.extractor(gridmap_dict)
            pn_feat = self.avg_pool_feats(output)  # B * out_channel
        else:
            pn_feat = self.extractor(gridmap)  # B * out_channel

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        pn_feat = pn_feat.to(state_feat.device)
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels

    def avg_pool_feats(self, point):
        # https://github.com/Pointcept/PointTransformerV3/issues/58
        feat = point.feat
        offsets = point.offset
        num_batch = offsets.shape[0]
        d = feat.shape[-1]
        avg_feats = torch.zeros(
            (num_batch, d), dtype=feat.dtype, device=feat.device)
        for point_num in range(num_batch):
            if point_num == 0:
                lb = 0
                ub = offsets[point_num]+1
            else:
                lb = ub-1
                ub = offsets[point_num]+1
            range_ = ub-lb
            if range_ > 1:
                avg_feats[point_num] = torch.mean(feat[lb:ub])
        return avg_feats

    def preprocess_ogm(self, ogm):
        """
        OGM (torch.Size([batch_size, 1, 25, 25, 25])) を
        PointTransformerV3 用の点群データに変換する
        """
        batch_size, _, D, H, W = ogm.shape  # (128, 1, 25, 25, 25)

        # 占有確率が 0.5 以上のボクセルの座標を取得 (各バッチごと)
        batch_coords = []
        batch_feats = []
        batch_indices = []

        for batch_idx in range(batch_size):
            # バッチごとの OGM を取得
            ogm_batch = ogm[batch_idx, 0]  # (25, 25, 25)

            # (x, y, z) 座標を取得
            coords = (ogm_batch > 0.95).nonzero(
                as_tuple=False).float()  # (N, 3)

            if coords.shape[0] == 0:
                continue  # 空ならスキップ

            # 各点の特徴量（占有確率）
            feats = ogm_batch[ogm_batch > 0.95].view(-1, 1).float()  # (N, 1)

            # バッチインデックス
            batch_ids = torch.full(
                (coords.shape[0],), batch_idx, dtype=torch.long)

            batch_coords.append(coords)
            batch_feats.append(feats)
            batch_indices.append(batch_ids)

        # リストをテンソルに変換
        batched_coords = torch.cat(batch_coords, dim=0).cuda()  # (M, 3)
        batched_feats = torch.cat(batch_feats, dim=0).cuda()  # (M, 1)
        batch_vals = torch.cat(batch_indices, dim=0).cuda()  # (M,)

        data = {
            "feat": batched_feats,  # 占有確率
            "batch": batch_vals,
            "coord": batched_coords,  # 点群座標 (x, y, z)
            "grid_size": 0.01
        }
        gridmap_dict = Point(data)

        return gridmap_dict
