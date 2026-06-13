import os
import h5py
import numpy as np
import plotly.graph_objects as go
from termcolor import cprint
import open3d as o3d

def plot_trajectory_with_plotly(hdf5_file):
    with h5py.File(hdf5_file, "r") as f:
        action_data = f["qpos_array"][:]
        drone_pos = action_data[:, :3]

    # Calculate the total travel distance
    distances = np.linalg.norm(np.diff(drone_pos, axis=0), axis=1)
    total_distance = np.sum(distances)
    cprint(f"[Path Length]{total_distance:.2f} meters", "red")

    time_steps = np.arange(drone_pos.shape[0])

    trace = go.Scatter3d(
        x=drone_pos[:, 0],
        y=drone_pos[:, 1],
        z=drone_pos[:, 2],
        mode="lines",
        line=dict(
            color=time_steps,
            colorscale="Magma",
            colorbar=dict(
                title="Time Step",
                x=0.8,  # Adjust this value to move the colorbar closer to the figure
            ),
            width=6
        )
    )

    layout = go.Layout(
        title="Drone Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="manual",
        ),
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    
    # save a fig as html
    output_html = input("Enter the output HTML file name (with .html extension): ")
    if not output_html.endswith(".html"):
        output_html += ".html"
    fig.write_html(output_html)
    print(f"Figure saved as {output_html}")


def plot_trajectory_and_obj(hdf5_file, obj_path, scale, pos, quat, show_axes=True, show_background=True):
    with h5py.File(hdf5_file, "r") as f:
        action_data = f["qpos_array"][:]
        drone_pos = action_data[:, :3]

    # Calculate the total travel distance
    distances = np.linalg.norm(np.diff(drone_pos, axis=0), axis=1)
    total_distance = np.sum(distances)
    cprint(f"[Path Length]{total_distance:.2f} meters", "red")

    time_steps = np.arange(drone_pos.shape[0])

    trace = go.Scatter3d(
        x=drone_pos[:, 0],
        y=drone_pos[:, 1],
        z=drone_pos[:, 2],
        mode="lines",
        line=dict(
            color=time_steps,
            colorscale="Magma",
            colorbar=dict(
                title="Time Step",
                x=0.8,  # Adjust this value to move the colorbar closer to the figure
            ),
            width=6
        )
    )

    layout = go.Layout(
        title="Drone Trajectory",
        scene=dict(
            xaxis=dict(title="X", visible=show_axes),
            yaxis=dict(title="Y", visible=show_axes),
            zaxis=dict(title="Z", visible=show_axes),
            aspectmode="manual",
        ),
        paper_bgcolor="white" if show_background else "rgba(0,0,0,0)",
        plot_bgcolor="white" if show_background else "rgba(0,0,0,0)",
    )

    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.translate(-mesh.get_center())
    translation = pos
    mesh.translate(translation)
    mesh.scale(scale, center=mesh.get_center())
    quat = np.array(quat)
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    mesh.rotate(R, center=mesh.get_center())
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    plotly_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color='gray',
        opacity=1
    )

    fig = go.Figure(data=[trace, plotly_mesh], layout=layout)
    fig.show()
    
    # save a fig as html
    output_html = input("Enter the output HTML file name (with .html extension): ")
    if not output_html.endswith(".html"):
        output_html += ".html"
    fig.write_html(output_html)
    print(f"Figure saved as {output_html}")


def animate_trajectory_and_obj(hdf5_file, obj_path, scale, pos, quat, show_axes=True, show_background=True):
    with h5py.File(hdf5_file, "r") as f:
        action_data = f["qpos_array"][:]
        drone_pos = action_data[:, :3]

    distances = np.linalg.norm(np.diff(drone_pos, axis=0), axis=1)
    total_distance = np.sum(distances)
    cprint(f"[Path Length]{total_distance:.2f} meters", "red")

    # 読み込みと変換
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.translate(-mesh.get_center())
    mesh.translate(pos)
    mesh.scale(scale, center=mesh.get_center())
    R = o3d.geometry.get_rotation_matrix_from_quaternion(np.array(quat))
    mesh.rotate(R, center=mesh.get_center())
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # メッシュの描画要素（静的）
    plotly_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color='gray',
        opacity=1,
        name='Object'
    )

    # フレームごとの軌道線を作成
    frames = []
    for t in range(1, len(drone_pos) + 1):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=drone_pos[:t, 0],
                    y=drone_pos[:t, 1],
                    z=drone_pos[:t, 2],
                    mode='lines',
                    line=dict(
                        color=np.linspace(0, 1, t),
                        colorscale="Magma",
                        width=6
                    ),
                    name='Trajectory'
                ),
                plotly_mesh
            ],
            name=str(t)
        )
        frames.append(frame)

    # 最初のフレーム
    initial_trace = go.Scatter3d(
        x=drone_pos[:1, 0],
        y=drone_pos[:1, 1],
        z=drone_pos[:1, 2],
        mode='lines',
        line=dict(color='black', width=6),
        name='Trajectory'
    )

    layout = go.Layout(
        title="Drone Trajectory Animation",
        scene=dict(
            xaxis=dict(title="X", visible=show_axes),
            yaxis=dict(title="Y", visible=show_axes),
            zaxis=dict(title="Z", visible=show_axes),
            aspectmode="manual",
        ),
        paper_bgcolor="white" if show_background else "rgba(0,0,0,0)",
        plot_bgcolor="white" if show_background else "rgba(0,0,0,0)",
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 0, "redraw": True}, "fromcurrent": True}]
            }, {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}]
            }]
        }],
        sliders=[{
            "steps": [{
                "args": [[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}],
                "label": str(i),
                "method": "animate"
            } for i in range(1, len(drone_pos) + 1)],
            "transition": {"duration": 0},
            "x": 0.1,
            "len": 0.9
        }]
    )

    fig = go.Figure(
        data=[initial_trace, plotly_mesh],
        layout=layout,
        frames=frames
    )
    fig.show()

    output_html = input("Enter the output HTML file name (with .html extension): ")
    if not output_html.endswith(".html"):
        output_html += ".html"
    fig.write_html(output_html)
    print(f"Figure saved as {output_html}")


def visualize_html():
    import webbrowser
    input_html = input("Enter the HTML file name to display (with .html extension): ")
    if not os.path.exists(input_html):
        print(f"File {input_html} does not exist.")
        return

    # Open the HTML file in the default web browser
    webbrowser.open(f"file://{os.path.abspath(input_html)}")

if __name__ == "__main__":
    # visualize_html()
    
    # hdf5_file = os.path.join(os.getcwd(), "demo_dir/test_long/1.h5")
    hdf5_file = "/home/cvl/cvl/ScanDP/scandp/deploy_dir/armadillo_scandp_x1.h5"
    # plot_trajectory_with_plotly(hdf5_file)

    obj_path = "/home/cvl/cvl/ScanDP/scandp/diffusion_policy_3d/assets/armadillo.obj"
    # plot_trajectory_and_obj(hdf5_file, obj_path, scale=0.28)
    animate_trajectory_and_obj(hdf5_file, obj_path, scale=0.28, pos=[0, 0, 0.7], quat=[1, 1, 0, 0])