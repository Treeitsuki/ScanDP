import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
# import diffusion_policy_3d.model.vision_3d.point_process as point_process

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

def r3m():
    from r3m import load_r3m
    model = load_r3m("resnet18", pretrained=True) # resnet18, resnet34
    model.eval()

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
        # self.point_cloud_key = 'point_cloud'
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
        else:
            raise NotImplementedError(f"pointnet_type: {encoder_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        # geometric features
        gridmap = observations[self.gridmap_key].unsqueeze(1)   # torch.Size([128, 1, 25, 25, 25])
        # assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # pn_feat = self.extractor(gridmap)  # B * out_channel
        geometric_feat = self.extractor(gridmap)  # B * out_channel
        # geometric_feat = torch.randn(128, 256).to("cuda")

        # TODO: semantic features
        # images = observations["image"].unsqueeze(1)
        # semantic_feat = torch.randn(128, 128).to("cuda")

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        # final_feat = torch.cat([geometric_feat, semantic_feat, state_feat], dim=-1)
        print(f"geometric_feat: {geometric_feat.shape}")
        final_feat = torch.cat([geometric_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels