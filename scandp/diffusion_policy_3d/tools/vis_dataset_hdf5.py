import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory_with_cmap(hdf5_file):

    with h5py.File(hdf5_file, "r") as f:
        action_data = f["pose"][:]
        drone_pos = action_data[:, :3]

    time_steps = np.arange(drone_pos.shape[0])
    norm = plt.Normalize(vmin=time_steps.min(), vmax=time_steps.max())
    cmap = plt.cm.magma

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # plot a trajectory
    for i in range(drone_pos.shape[0] - 1):
        ax.plot(
            drone_pos[i:i+2, 0],
            drone_pos[i:i+2, 1],
            drone_pos[i:i+2, 2],
            color=cmap(norm(time_steps[i]))
        )

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Time Step", fontsize=12)

    ax.set_title("Drone Trajectory", fontsize=15)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.axis('equal')
    ax.view_init(elev=30, azim=-45)

    plt.show()


def plot_trajectories_from_folder(folder_path):
    folder = Path(folder_path)
    h5_files = list(folder.glob("*.h5"))

    if not h5_files:
        print("No HDF5 files found in the specified folder.")
        return

    colors = pc.qualitative.Plotly  # 色のリスト（異なるファイルに適用）

    traces = []
    for i, hdf5_file in enumerate(h5_files):
        with h5py.File(hdf5_file, "r") as f:
            action_data = f["pose"][:]
            drone_pos = action_data[:, :3]

        trace = go.Scatter3d(
            x=drone_pos[:, 0],
            y=drone_pos[:, 1],
            z=drone_pos[:, 2],
            mode="lines",
            line=dict(
                color=colors[i % len(colors)],  # 異なるファイルごとに異なる色を適用
                width=6
            ),
            name=hdf5_file.stem  # ファイル名を凡例として表示
        )
        traces.append(trace)

    layout = go.Layout(
        title="Trajectories",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="manual",
        ),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

    # save the figure as an HTML file
    output_html = folder / "trajectories.html"
    fig.write_html(str(output_html))
    print(f"Figure saved as {output_html}")


if __name__ == "__main__":
    # hdf5_file = os.path.join(os.getcwd(), "demo_dir/test_long/1.h5")
    # plot_trajectory_with_cmap(hdf5_file)

    # フォルダパスを指定して実行
    folder_path = input("Enter the folder path containing HDF5 files: ")
    plot_trajectories_from_folder(folder_path)
