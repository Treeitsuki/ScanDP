import time

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv3DEncoder(nn.Module):
    def __init__(self, shape, input_channels=1, output_channels=256):
        super().__init__()
        self.input_channels = int(input_channels)
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(self.input_channels, 32, 3).cuda(),
            nn.ReLU(),
            spconv.SparseConv3d(32, 64, 3).cuda(),
            nn.ReLU(),
            # convert spconv tensor to dense and convert it to NCHW format.
            spconv.ToDense().cuda(),
            nn.Conv3d(64, 128, 3).cuda(),
            nn.ReLU(),
        )
        self.shape = shape
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1)).cuda()
        self.flatten = nn.Flatten().cuda()
        self.fc = nn.Linear(128 * 1 * 1 * 1, output_channels).cuda()

    def _match_input_channels(self, features):
        if features.shape[1] == self.input_channels:
            return features
        if features.shape[1] > self.input_channels:
            if self.input_channels == 1:
                return features[:, :1]
            return features[:, :self.input_channels]
        pad = torch.zeros(
            (features.shape[0], self.input_channels - features.shape[1]),
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat([features, pad], dim=1)

    def forward(self, features, coors, batch_size):
        features = self._match_input_channels(features)
        coors = coors.int().to(features.device)
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        x = self.net(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def generate_sparse_representation(ogm):
    if ogm.dim() == 5:
        batch_size, num_channels, xdim, ydim, zdim = ogm.shape
        feat_dim = 1
        # Find non-zero elements
        batch_indices, x_indices, y_indices, z_indices = torch.nonzero(
            ogm[:, 0], as_tuple=True)
        # Extract features from nonzero positions
        features = ogm[batch_indices, 0, x_indices, y_indices,
                       z_indices].unsqueeze(-1)  # [num_points, 1]
    elif ogm.dim() == 6:
        batch_size, num_channels, xdim, ydim, zdim, feat_dim = ogm.shape
        # Find non-zero elements
        batch_indices, x_indices, y_indices, z_indices, feat_dim = torch.nonzero(
            ogm[:, 0], as_tuple=True)
        # Extract features from nonzero positions
        features = ogm[batch_indices, 0, x_indices, y_indices,
                       z_indices, :]  # [num_points, feat_dim]
    else:
        raise ValueError(
            "ogm must be either 5D [B,C,X,Y,Z] or 6D [B,C,X,Y,Z,F]")

    # Construct coordinate tensor
    coors = torch.stack([batch_indices, x_indices, y_indices,
                        z_indices], dim=1)  # [num_points, 4]

    return features, coors, [xdim, ydim, zdim], batch_size, feat_dim


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ogm = torch.load("/home/cvl/cvl/ScanDP/test/prob_map.pth").to(device)
    ogm = ogm.unsqueeze(0).unsqueeze(0).repeat(64, 1, 1, 1, 1)
    print(ogm.shape)
    batch_size = ogm.shape[0]
    channel = ogm.shape[1]
    shape = ogm.shape[2:]

    features, coors, shape, batch_size = generate_sparse_representation(ogm)
    print(shape)
    # dense_input = torch.randn(128, 1, 25, 25, 25).to(device)
    # batch_size = dense_input.shape[0]
    # channel = dense_input.shape[1]
    # shape = dense_input.shape[2:]

    # num_points = 1000  # Number of nonzero points
    # features = torch.rand((num_points, 1), device=device)  # Sparse feature vectors
    # coors = torch.randint(0, 25, (num_points, 4), device=device)  # Random coordinates
    # coors[:, 0] = torch.randint(0, batch_size, (num_points,), device=device)  # Batch indices

    # Define model
    model = SparseConv3DEncoder(shape, 128).to(device)

    print(features.shape, coors.shape)

    # Run forward pass
    t = time.time()
    output = model(features, coors, batch_size)

    # Print output shape
    print("Output shape:", output.shape)
    print("Time: {}s".format(time.time() - t))
