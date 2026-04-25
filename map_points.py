import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_env():
    return gym.make(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
    )


def run_single_step(env):
    observation, info = env.reset()
    print("Observation:", observation)
    print("Info:", info)

    action = env.action_space.sample()
    print("Action:", action)

    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()  # (680, 680, 3) numpy array
    print("Rendered image shape:", image.shape)
    return observation, image


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


def plot_t_coverage(image, bx_img, by_img, t_image_points, ba, output_path="plot_bx_by.png"):
    fig, ax_plot = plt.subplots(1, 1, figsize=(7, 7))
    ax_plot.imshow(image)
    ax_plot.scatter([bx_img], [by_img], c="red", s=100, zorder=5, label=f"center ({bx_img:.1f}, {by_img:.1f})")
    ax_plot.scatter(
        t_image_points[:, 0],
        t_image_points[:, 1],
        c="yellow",
        s=18,
        alpha=0.9,
        zorder=6,
        label="T coverage points",
    )
    ax_plot.legend(loc="upper right")
    ax_plot.set_title("Block position and T coverage points")
    ax_plot.set_xlabel("X (pixels)")
    ax_plot.set_ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved matplotlib plot to {output_path} (center=({bx_img:.2f}, {by_img:.2f}), angle={ba:.3f} rad)")


def print_point_summary(t_local_points, t_world_points):
    print(f"Generated {len(t_local_points)} local points covering the T block")
    print("Sample world points (first 10):")
    for i, (px, py) in enumerate(t_world_points[:10]):
        print(f"  p{i}: ({px:.2f}, {py:.2f})")


def main():
    env = create_env()
    try:
        observation, image = run_single_step(env)
        _, _, bx, by, ba = observation

        t_local_points = generate_t_local_points(scale=30.0, length=4.0, point_spacing=6.0)
        t_world_points = transform_points_to_world(t_local_points, bx, by, ba)

        center_world = np.array([[bx, by]], dtype=np.float64)
        center_image = world_to_image(center_world)
        bx_img, by_img = center_image[0]
        t_image_points = world_to_image(t_world_points)

        print_point_summary(t_local_points, t_world_points)
        save_raw_image(image, output_path="raw.jpg")
        plot_t_coverage(image, bx_img, by_img, t_image_points, ba, output_path="plot_bx_by.png")
    finally:
        env.close()
        pygame.quit()
        print("Done!")


if __name__ == "__main__":
    main()
