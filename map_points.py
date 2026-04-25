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

# T-block local corners in environment coordinates
# (same dimensions used in PushTEnv.add_tee: length=4, scale=30)
scale = 30.0
length = 4.0
t_local_corners = np.array(
    [
        [-length * scale / 2, scale],
        [length * scale / 2, scale],
        [length * scale / 2, 0],
        [-length * scale / 2, 0],
        [-scale / 2, scale],
        [-scale / 2, length * scale],
        [scale / 2, length * scale],
        [scale / 2, scale],
    ],
    dtype=np.float64,
)

# Rotate + translate corners to world coordinates
rotation = np.array(
    [
        [np.cos(ba), -np.sin(ba)],
        [np.sin(ba), np.cos(ba)],
    ],
    dtype=np.float64,
)
t_world_corners = t_local_corners @ rotation.T + np.array([bx, by], dtype=np.float64)

# Convert world [0, 512] -> image [0, 680]
bx_img = (bx / 512.0) * 680.0
by_img = (by / 512.0) * 680.0
t_image_corners = (t_world_corners / 512.0) * 680.0

print("T-block world corners (x, y):")
for i, (cx, cy) in enumerate(t_world_corners):
    print(f"  c{i}: ({cx:.2f}, {cy:.2f})")

print("T-block image corners (x, y):")
for i, (cx, cy) in enumerate(t_image_corners):
    print(f"  c{i}: ({cx:.2f}, {cy:.2f})")


Image.fromarray(image).convert("RGB").save('raw.jpg', format="JPEG")
print(f"Saved render to raw.jpg")

#! Plot bx, by on the image using matplotlib
fig, ax_plot = plt.subplots(1, 1, figsize=(7, 7))
ax_plot.imshow(image)
ax_plot.scatter([bx_img], [by_img], c='red', s=100, zorder=5, label=f'center ({bx_img:.1f}, {by_img:.1f})')
ax_plot.scatter(t_image_corners[:, 0], t_image_corners[:, 1], c='yellow', s=70, zorder=6, label='T corners')
for i, (cx, cy) in enumerate(t_image_corners):
    ax_plot.text(cx + 5, cy + 5, f'c{i}', color='black', fontsize=9, zorder=7)
ax_plot.legend(loc='upper right')
ax_plot.set_title('Block position (bx, by)')
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
