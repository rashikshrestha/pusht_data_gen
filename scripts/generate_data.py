import argparse
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gym_pusht
import numpy as np
from PIL import Image

from gym_pusht.utils.point_mapper import compute_image_points, extrude_points_to_3d, plot_3d_points


def parse_args():
	parser = argparse.ArgumentParser(description="Collect random PushT frames and save image/observation/action.")
	parser.add_argument("--passes", type=int, default=6, help="Number of straight-line passes to collect.")
	parser.add_argument("--steps-per-pass", type=int, default=200, help="Number of actions in each straight-line pass.")
	parser.add_argument(
		"--reset-every-pass",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Whether to reset environment before each pass (default: true).",
	)
	parser.add_argument(
		"--target-noise",
		type=float,
		default=15.0,
		help="Uniform XY noise added to block center when planning pass target (range: [-n, +n]).",
	)
	parser.add_argument(
		"--action-noise",
		type=float,
		default=5.0,
		help="Uniform XY noise added to each action (range: [-n, +n]).",
	)
	parser.add_argument("--gif-duration-ms", type=int, default=50, help="Frame duration for output GIF in milliseconds.")
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Optional output directory. If omitted, uses data/run_<timestamp>.",
	)
	parser.add_argument("--seed", type=int, default=None, help="Optional random seed for pass directions.")
	parser.add_argument("--observation-width", type=int, default=96)
	parser.add_argument("--observation-height", type=int, default=96)
	parser.add_argument("--visualization-width", type=int, default=680)
	parser.add_argument("--visualization-height", type=int, default=680)
	return parser.parse_args()


def make_output_dir(output_dir_arg: str | None) -> Path:
	if output_dir_arg:
		out_dir = Path(output_dir_arg)
	else:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		out_dir = Path("data") / f"run_{timestamp}"

	out_dir.mkdir(parents=True, exist_ok=True)
	(out_dir / "images").mkdir(parents=True, exist_ok=True)
	(out_dir / "frames").mkdir(parents=True, exist_ok=True)
	(out_dir / "obs").mkdir(parents=True, exist_ok=True)
	(out_dir / "threed").mkdir(parents=True, exist_ok=True)
	return out_dir


def create_env(args):
	return gym.make(
		"gym_pusht/PushT-v0",
		render_mode="rgb_array",
		observation_width=args.observation_width,
		observation_height=args.observation_height,
		visualization_width=args.visualization_width,
		visualization_height=args.visualization_height,
	)


def _line_square_intersections(center: np.ndarray, direction: np.ndarray, low=0.0, high=512.0):
	cx, cy = float(center[0]), float(center[1])
	dx, dy = float(direction[0]), float(direction[1])
	ts = []
	eps = 1e-8

	if abs(dx) > eps:
		for x in (low, high):
			t = (x - cx) / dx
			y = cy + t * dy
			if low - 1e-6 <= y <= high + 1e-6:
				ts.append(t)

	if abs(dy) > eps:
		for y in (low, high):
			t = (y - cy) / dy
			x = cx + t * dx
			if low - 1e-6 <= x <= high + 1e-6:
				ts.append(t)

	if len(ts) < 2:
		raise ValueError("Could not find edge intersections for straight-line pass.")

	t_min = min(ts)
	t_max = max(ts)
	start = center + t_min * direction
	end = center + t_max * direction
	start = np.clip(start, low, high)
	end = np.clip(end, low, high)
	return start, end


def generate_pass_actions_through_block(block_center: np.ndarray, steps_per_pass: int, rng: np.random.Generator):
	theta = rng.uniform(0.0, 2.0 * np.pi)
	direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
	start, end = _line_square_intersections(block_center, direction)
	actions = np.linspace(start, end, num=steps_per_pass, dtype=np.float32)
	return actions, start, end


def add_target_noise(block_center: np.ndarray, noise_limit: float, rng: np.random.Generator):
	noise = rng.uniform(-noise_limit, noise_limit, size=2)
	noisy_target = np.clip(block_center + noise, 0.0, 512.0)
	return noisy_target


def add_action_noise(action: np.ndarray, noise_limit: float, rng: np.random.Generator):
	noise = rng.uniform(-noise_limit, noise_limit, size=2)
	noisy_action = np.clip(action + noise, 0.0, 512.0)
	return noisy_action.astype(np.float32)


def _to_python(obj):
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, np.generic):
		return obj.item()
	if isinstance(obj, dict):
		return {str(k): _to_python(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [_to_python(v) for v in obj]
	return obj


def _yaml_scalar(value):
	if isinstance(value, bool):
		return "true" if value else "false"
	if value is None:
		return "null"
	if isinstance(value, str):
		return json.dumps(value)
	return str(value)


def _write_yaml(f, data, indent=0):
	space = " " * indent
	if isinstance(data, dict):
		for key, value in data.items():
			if isinstance(value, (dict, list)):
				f.write(f"{space}{key}:\n")
				_write_yaml(f, value, indent + 2)
			else:
				f.write(f"{space}{key}: {_yaml_scalar(value)}\n")
		return

	if isinstance(data, list):
		for value in data:
			if isinstance(value, (dict, list)):
				f.write(f"{space}-\n")
				_write_yaml(f, value, indent + 2)
			else:
				f.write(f"{space}- {_yaml_scalar(value)}\n")
		return

	f.write(f"{space}{_yaml_scalar(data)}\n")


def save_frame(
	out_dir: Path,
	frame_idx: int,
	image: np.ndarray,
	action: np.ndarray,
	info: dict,
):
	image_path = out_dir / "images" / f"frame_{frame_idx:06d}.png"
	obs_path = out_dir / "obs" / f"frame_{frame_idx:06d}.yaml"

	Image.fromarray(image).save(image_path)

	info_clean = _to_python(info)
	action_values = [float(x) for x in np.asarray(action).reshape(-1)]
	yaml_payload = {
		"frame_index": frame_idx,
		"action": action_values,
	}
	if isinstance(info_clean, dict):
		yaml_payload.update(info_clean)
	else:
		yaml_payload["info"] = info_clean

	with obs_path.open("w", encoding="utf-8") as f:
		_write_yaml(f, yaml_payload, indent=0)

	return image_path, obs_path


def save_threed_frame(
	out_dir: Path,
	frame_idx: int,
	observation: np.ndarray,
	action: np.ndarray,
	point_spacing: float = 9.0,
	num_layers: int = 4,
):
	threed_path = out_dir / "threed" / f"frame_{frame_idx:06d}.png"
	body_points, origin_point, agent_point, action_point = compute_image_points(
		observation,
		action,
		point_spacing=point_spacing,
	)
	body_points_3d = extrude_points_to_3d(body_points, num_layers=num_layers, point_spacing=point_spacing)
	z_layer = (num_layers / 2.0) * point_spacing
	center_point_3d = np.array([origin_point[0], origin_point[1], z_layer], dtype=np.float64)
	agent_point_3d = np.array([agent_point[0], agent_point[1], z_layer], dtype=np.float64)
	action_point_3d = np.array([action_point[0], action_point[1], z_layer], dtype=np.float64)

	plot_3d_points(
		body_points_3d,
		center_point_3d=center_point_3d,
		agent_point_3d=agent_point_3d,
		action_point_3d=action_point_3d,
		output_path=str(threed_path),
	)

	return threed_path, body_points_3d


def save_gif(image_paths: list[Path], output_path: Path, duration_ms: int = 50):
	if not image_paths:
		return None

	frames = [Image.open(path).convert("RGB") for path in image_paths]
	try:
		frames[0].save(
			output_path,
			save_all=True,
			append_images=frames[1:],
			duration=duration_ms,
			loop=0,
		)
	finally:
		for frame in frames:
			frame.close()

	return output_path


def main():
	args = parse_args()
	out_dir = make_output_dir(args.output_dir)
	env = create_env(args)
	rng = np.random.default_rng(args.seed)

	manifest = {
		"passes": args.passes,
		"steps_per_pass": args.steps_per_pass,
		"reset_every_pass": args.reset_every_pass,
		"target_noise": args.target_noise,
		"action_noise": args.action_noise,
		"gif_duration_ms": args.gif_duration_ms,
		"seed": args.seed,
		"observation_shape": [5],
		"action_shape": [2],
		"frames": [],
	}

	global_frame_idx = 0
	saved_image_paths: list[Path] = []
	saved_threed_paths: list[Path] = []
	body_points_3d_steps: list[np.ndarray] = []

	try:
		observation, info = env.reset()
		should_stop = False
		for pass_idx in range(args.passes):
			if args.reset_every_pass and pass_idx > 0:
				observation, info = env.reset()

			block_center = np.array([observation[2], observation[3]], dtype=np.float64)
			target_point = add_target_noise(block_center, noise_limit=args.target_noise, rng=rng)
			actions, line_start, line_end = generate_pass_actions_through_block(
				block_center=target_point,
				steps_per_pass=args.steps_per_pass,
				rng=rng,
			)

			for step_idx, action in enumerate(actions):
				action_noisy = add_action_noise(action, noise_limit=args.action_noise, rng=rng)
				observation, reward, terminated, truncated, info = env.step(action_noisy)
				image = env.render()

				image_path, obs_path = save_frame(
					out_dir=out_dir,
					frame_idx=global_frame_idx,
					image=image,
					action=action_noisy,
					info=info,
				)
				threed_path, body_points_3d = save_threed_frame(
					out_dir=out_dir,
					frame_idx=global_frame_idx,
					observation=observation,
					action=action_noisy,
				)
				saved_image_paths.append(image_path)
				saved_threed_paths.append(threed_path)
				body_points_3d_steps.append(body_points_3d)

				manifest["frames"].append(
					{
						"frame_index": global_frame_idx,
						"pass_index": pass_idx,
						"step_index": step_idx,
						"block_center": [float(block_center[0]), float(block_center[1])],
						"target_point": [float(target_point[0]), float(target_point[1])],
						"line_start": [float(line_start[0]), float(line_start[1])],
						"line_end": [float(line_end[0]), float(line_end[1])],
						"image": str(image_path.relative_to(out_dir)),
						"obs_yaml": str(obs_path.relative_to(out_dir)),
						"threed": str(threed_path.relative_to(out_dir)),
						"body_points_3d_index": global_frame_idx,
					}
				)

				global_frame_idx += 1

				if terminated or truncated:
					should_stop = True
					break

			if should_stop:
				if args.reset_every_pass:
					print("Environment terminated/truncated; continuing with next pass after reset.")
					should_stop = False
					continue
				print("Environment terminated/truncated; stopping early because reset-every-pass is disabled.")
				break
	finally:
		env.close()

	body_points_3d_path = out_dir / "frames" / "body_points_3d.npy"
	if body_points_3d_steps:
		body_points_3d_tensor = np.stack(body_points_3d_steps, axis=0)
	else:
		body_points_3d_tensor = np.empty((0, 0, 3), dtype=np.float64)
	np.save(body_points_3d_path, body_points_3d_tensor)
	manifest["body_points_3d_npy"] = str(body_points_3d_path.relative_to(out_dir))
	manifest["body_points_3d_shape"] = list(body_points_3d_tensor.shape)

	manifest_path = out_dir / "manifest.json"
	with manifest_path.open("w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2)

	gif_path = save_gif(saved_image_paths, out_dir / "rollout.gif", duration_ms=args.gif_duration_ms)
	threed_gif_path = save_gif(saved_threed_paths, out_dir / "rollout_3d.gif", duration_ms=args.gif_duration_ms)

	print(f"Saved {global_frame_idx} frames to {out_dir}")
	print(f"Manifest: {manifest_path}")
	if gif_path is not None:
		print(f"GIF: {gif_path}")
	if threed_gif_path is not None:
		print(f"3D GIF: {threed_gif_path}")


if __name__ == "__main__":
	main()
