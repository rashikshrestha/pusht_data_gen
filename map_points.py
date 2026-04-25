import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#! Make environment
env = gym.make(
    "gym_pusht/PushT-v0",
    render_mode="rgb_array",
    observation_width=96,
    observation_height=96,
    visualization_width=680,
    visualization_height=680,
)

observation, info = env.reset()
print("Observation:", observation)
print("Info:", info)

action = env.action_space.sample()
print("Action:", action)

observation, reward, terminated, truncated, info = env.step(action)

image = env.render()  # (680, 680, 3) numpy array
print("Rendered image shape:", image.shape)

#! Plot center
# get: agent_x, agent_y, block_x, block_y, block_angle
ax, ay, bx, by, ba = observation

# T-block local points in environment coordinates
# (same dimensions used in PushTEnv.add_tee: length=4, scale=30)
scale = 30.0
length = 4.0
point_spacing = 6.0
x_vals = np.arange(-length * scale / 2, length * scale / 2 + 1e-9, point_spacing)
y_vals = np.arange(0.0, length * scale + 1e-9, point_spacing)
xx, yy = np.meshgrid(x_vals, y_vals)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# Union of two rectangles:
# 1) top bar: x in [-2s, 2s], y in [0, s]
# 2) stem:    x in [-s/2, s/2], y in [s, 4s]
is_top_bar = (np.abs(grid_points[:, 0]) <= (length * scale / 2)) & (grid_points[:, 1] <= scale)
is_stem = (np.abs(grid_points[:, 0]) <= (scale / 2)) & (grid_points[:, 1] >= scale) & (
    grid_points[:, 1] <= (length * scale)
)
t_local_points = grid_points[is_top_bar | is_stem]

# Rotate + translate corners to world coordinates
rotation = np.array(
    [
        [np.cos(ba), -np.sin(ba)],
        [np.sin(ba), np.cos(ba)],
    ],
    dtype=np.float64,
)
t_world_points = t_local_points @ rotation.T + np.array([bx, by], dtype=np.float64)

# Convert world [0, 512] -> image [0, 680]
bx_img = (bx / 512.0) * 680.0
by_img = (by / 512.0) * 680.0
t_image_points = (t_world_points / 512.0) * 680.0

print(f"Generated {len(t_local_points)} local points covering the T block")
print("Sample world points (first 10):")
for i, (px, py) in enumerate(t_world_points[:10]):
    print(f"  p{i}: ({px:.2f}, {py:.2f})")


Image.fromarray(image).convert("RGB").save('raw.jpg', format="JPEG")
print(f"Saved render to raw.jpg")

#! Plot bx, by on the image using matplotlib
fig, ax_plot = plt.subplots(1, 1, figsize=(7, 7))
ax_plot.imshow(image)
ax_plot.scatter([bx_img], [by_img], c='red', s=100, zorder=5, label=f'center ({bx_img:.1f}, {by_img:.1f})')
ax_plot.scatter(t_image_points[:, 0], t_image_points[:, 1], c='yellow', s=18, alpha=0.9, zorder=6, label='T coverage points')
ax_plot.legend(loc='upper right')
ax_plot.set_title('Block position and T coverage points')
ax_plot.set_xlabel('X (pixels)')
ax_plot.set_ylabel('Y (pixels)')
# ax_plot.axis('off')
plt.tight_layout()
plt.savefig('plot_bx_by.png', dpi=150)
plt.close(fig)
print(f"Saved matplotlib plot to plot_bx_by.png (center=({bx_img:.2f}, {by_img:.2f}), angle={ba:.3f} rad)")

env.close()
pygame.quit()
print("Done!")
