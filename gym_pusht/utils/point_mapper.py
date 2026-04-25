import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def generate_t_local_points(scale=30.0, length=4.0, point_spacing=6.0):
    x_vals = np.arange(-length * scale / 2, length * scale / 2 + 1e-9, point_spacing)
    y_vals = np.arange(0.0, length * scale + 1e-9, point_spacing)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    is_top_bar = (np.abs(grid_points[:, 0]) <= (length * scale / 2)) & (grid_points[:, 1] <= scale)
    is_stem = (np.abs(grid_points[:, 0]) <= (scale / 2)) & (grid_points[:, 1] >= scale) & (
        grid_points[:, 1] <= (length * scale)
    )
    return grid_points[is_top_bar | is_stem]


def transform_points_to_world(local_points, bx, by, ba):
    rotation = np.array(
        [
            [np.cos(ba), -np.sin(ba)],
            [np.sin(ba), np.cos(ba)],
        ],
        dtype=np.float64,
    )
    return local_points @ rotation.T + np.array([bx, by], dtype=np.float64)


def world_to_image(points, world_size=512.0, image_size=680.0):
    return (points / world_size) * image_size


def save_raw_image(image, output_path="raw.jpg"):
    Image.fromarray(image).convert("RGB").save(output_path, format="JPEG")
    print(f"Saved render to {output_path}")


def plot_t_coverage(image, center_image, agent_image, action_img, t_image_points, output_path="plot_bx_by.png"):
    fig, ax_plot = plt.subplots(1, 1, figsize=(7, 7))
    ax_plot.imshow(image)
    ax_plot.scatter(
        t_image_points[:, 0],
        t_image_points[:, 1],
        c="gray",
        s=18,
        alpha=0.9,
        zorder=6,
        label="Body",
    )
    ax_plot.scatter([center_image[0]], [center_image[1]], c="red", s=100, zorder=5, label="Center")
    ax_plot.scatter([agent_image[0]], [agent_image[1]], c="cyan", s=500, zorder=5, label="Agent")
    ax_plot.scatter([action_img[0]], [action_img[1]], c="lime", s=200, marker="*", zorder=7, label="Action")
    ax_plot.legend(loc="upper right")
    ax_plot.set_title("Block position and T coverage points")
    ax_plot.set_xlabel("X (pixels)")
    ax_plot.set_ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved matplotlib plot to {output_path} (center=({center_image[0]:.2f}, {center_image[1]:.2f}))")


def plot_3d_points(points_3d, center_point_3d=None, agent_point_3d=None, output_path="mapped_3d.png"):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap="viridis", s=8)
    if center_point_3d is not None:
        ax.scatter(
            [center_point_3d[0]],
            [center_point_3d[1]],
            [center_point_3d[2]],
            c="red",
            s=100,
            marker="o",
            label="Center 3D",
        )
    if agent_point_3d is not None:
        ax.scatter(
            [agent_point_3d[0]],
            [agent_point_3d[1]],
            [agent_point_3d[2]],
            c="cyan",
            s=140,
            marker="^",
            label="Agent 3D",
        )
    ax.set_title("Extruded T-body 3D points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # same limits for all axes to preserve spatial relationships
    ax.set_xlim(0, 680)
    ax.set_ylim(0, 680)
    ax.set_zlim(0, 680)
    if center_point_3d is not None or agent_point_3d is not None:
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved 3D matplotlib plot to {output_path} (num_points={len(points_3d)})")


def compute_image_points(observation, action, scale=30.0, length=4.0, point_spacing=6.0):
    """Return body coverage points, block center, agent point, and action point — all in image coordinates.

    Returns:
        body_image:   (N, 2) array of T-coverage points in image coords
        center_image: (2,)   block center in image coords
        agent_image:  (2,)   agent position in image coords
        action_image: (2,)   sampled action target in image coords
        t_world_points: (N, 2) body points in world coords (for diagnostics)
        t_local_points: (N, 2) body points in local coords (for diagnostics)
    """
    ax, ay, bx, by, ba = observation

    t_local_points = generate_t_local_points(scale=scale, length=length, point_spacing=point_spacing)
    t_world_points = transform_points_to_world(t_local_points, bx, by, ba)
    body_image = world_to_image(t_world_points)

    center_image = world_to_image(np.array([[bx, by]], dtype=np.float64))[0]
    agent_image = world_to_image(np.array([[ax, ay]], dtype=np.float64))[0]
    action_image = world_to_image(np.array([action], dtype=np.float64))[0]

    return body_image, center_image, agent_image, action_image, t_world_points, t_local_points


def extrude_points_to_3d(body_points_2d, num_layers=4, point_spacing=6.0):
    """Stack 2D body points into layered 3D points along the Z axis.

    Args:
        body_points_2d: (N, 2) array of XY points
        num_layers:     number of Z layers (default 4)
        point_spacing:  distance between layers, same as XY point spacing

    Returns:
        (N * num_layers, 3) array of XYZ points, z starting at 0
    """
    layers = [
        np.column_stack([body_points_2d, np.full(len(body_points_2d), i * point_spacing)])
        for i in range(num_layers)
    ]
    return np.vstack(layers)


def print_point_summary(t_local_points, t_world_points):
    print(f"Generated {len(t_local_points)} local points covering the T block")
    print("Sample world points (first 10):")
    for i, (px, py) in enumerate(t_world_points[:10]):
        print(f"  p{i}: ({px:.2f}, {py:.2f})")

