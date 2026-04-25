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
        body_points, origin_point, agent_point, action_point = compute_image_points(
            observation, action,
            point_spacing=point_spacing
        )

        #! Points to 3D
        num_layers = 4
        body_points_3d = extrude_points_to_3d(
            body_points, num_layers=num_layers, point_spacing=point_spacing
        )
        z_layer = (num_layers/2) * point_spacing
        center_point_3d = np.array([origin_point[0], origin_point[1], z_layer], dtype=np.float64)
        agent_point_3d = np.array([agent_point[0], agent_point[1], z_layer], dtype=np.float64)
        action_point_3d = np.array([action_point[0], action_point[1], z_layer], dtype=np.float64)

        #! Print and Plots
        print_point_summary(body_points)
        plot_t_coverage(image, origin_point, agent_point, action_point, body_points, output_path="mapped.png")
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
