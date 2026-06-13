import numpy as np
import torch

import torch

import torch

def bresenham3D(pts_source, pts_target, map_size):
    if isinstance(map_size, list):
        assert len(map_size) == 3 and map_size[0] == map_size[1] == map_size[2], "map_size must be cubic"
        map_size = map_size[0]

    device = pts_source.device
    source_pts = pts_source.int().contiguous().unsqueeze(0)  # (3,) → (1,3)
    target_pts = pts_target.int().contiguous()
    num_rays = target_pts.shape[0]

    # ブロードキャストして (num_rays, 3) にする
    source_pts = source_pts.expand(num_rays, -1)

    # 差分計算
    deltas = torch.abs(target_pts - source_pts)
    steps = torch.max(deltas, dim=1)[0] + 1  # 各軸で最大のステップ数（開始点を含む）

    # ブロードキャストで方向計算
    direction = torch.sign(target_pts - source_pts)

    # 進行ステップのインデックスを作成 (最大ステップ数に合わせる)
    max_steps = steps.max().item()
    step_indices = torch.arange(max_steps, device=device).repeat(num_rays, 1)

    # 各レイの長さを超えるインデックスをマスク
    valid_mask = step_indices < steps[:, None]

    # Bresenham の進行処理 (ベクトル化)
    fractional_steps = step_indices / (steps[:, None] - 1).clamp(min=1)  # 0.0 ~ 1.0 の割合
    ray_coords = source_pts[:, None, :] + (fractional_steps[..., None] * deltas[:, None, :]).round().int() * direction[:, None, :]

    # マスク適用して不要な部分を除去
    results = ray_coords[valid_mask.unsqueeze(-1).expand(-1, -1, 3)].view(-1, 3)  # **明示的に (N, 3) にリシェイプ**

    return results.to(torch.long)


import plotly.graph_objects as go

def test_bresenham3D_pytorch():
    # Define test parameters
    pts_source = torch.tensor([0, 0, 0], device='cuda')
    pts_target = torch.tensor([
        [10, 10, 10],
        [4, 5, 2],
    ], device='cuda')
    map_size = 20

    # Call the function
    results = bresenham3D(pts_source, pts_target, map_size)

    # Convert results to CPU for plotting
    results_cpu = results.cpu().numpy()

    # Plot the results using plotly
    fig = go.Figure()

    # Add source point
    fig.add_trace(go.Scatter3d(
        x=[pts_source[0].item()],
        y=[pts_source[1].item()],
        z=[pts_source[2].item()],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Source'
    ))

    # Add target points
    fig.add_trace(go.Scatter3d(
        x=pts_target[:, 0].cpu().numpy(),
        y=pts_target[:, 1].cpu().numpy(),
        z=pts_target[:, 2].cpu().numpy(),
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Targets'
    ))

    # Add trajectory points
    fig.add_trace(go.Scatter3d(
        x=results_cpu[:, 0],
        y=results_cpu[:, 1],
        z=results_cpu[:, 2],
        mode='markers',
        marker=dict(size=3, color='green'),
        name='Trajectory'
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[0, map_size]),
            yaxis=dict(nticks=10, range=[0, map_size]),
            zaxis=dict(nticks=10, range=[0, map_size])
        ),
        title="Bresenham 3D Trajectory",
        showlegend=True
    )

    # Show plot
    fig.show()

if __name__ == "__main__":
    test_bresenham3D_pytorch()