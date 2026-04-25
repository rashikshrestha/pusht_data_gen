import gymnasium as gym
import gym_pusht
import pygame

from gym_pusht.utils.point_mapper import *

def main():
    #! Create Env
    env = gym.make(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
    )

    try:
        #! Get observation, action, reward, and rendered image
        observation, info = env.reset()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        image = env.render()  # (680, 680, 3)
        point_spacing = 9.0

        #! Get 2D points
        body_image, center_image, agent_image, action_image, t_world_points, t_local_points = compute_image_points(
            observation, action,
            point_spacing=point_spacing
        )

        #! Points to 3D
        num_layers = 4
        body_points_3d = extrude_points_to_3d(
            t_world_points, num_layers=num_layers, point_spacing=point_spacing
        )
        z_layer = (num_layers/2) * point_spacing
        center_point_3d = np.array([center_image[0], center_image[1], z_layer], dtype=np.float64)
        agent_point_3d = np.array([agent_image[0], agent_image[1], z_layer], dtype=np.float64)
        action_point_3d = np.array([action_image[0], action_image[1], z_layer], dtype=np.float64)

        #! Print and Plots
        print_point_summary(t_local_points, t_world_points)
        plot_t_coverage(image, center_image, agent_image, action_image, body_image, output_path="mapped.png")
        plot_3d_points(
            body_points_3d,
            center_point_3d=center_point_3d,
            agent_point_3d=agent_point_3d,
            output_path="mapped_3d.png",
        )
    finally:
        env.close()
        pygame.quit()
        print("Done!")


if __name__ == "__main__":
    main()
