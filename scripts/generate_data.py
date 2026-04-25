import argparse
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gym_pusht
import numpy as np
from PIL import Image


def parse_args():
	parser = argparse.ArgumentParser(description="Collect random PushT frames and save image/observation/action.")
	parser.add_argument("--passes", type=int, default=6, help="Number of straight-line passes to collect.")
	parser.add_argument("--steps-per-pass", type=int, default=200, help="Number of actions in each straight-line pass.")
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


def save_frame(out_dir: Path, frame_idx: int, image: np.ndarray, observation: np.ndarray, action: np.ndarray):
	image_path = out_dir / "images" / f"frame_{frame_idx:06d}.png"
	frame_path = out_dir / "frames" / f"frame_{frame_idx:06d}.npz"

	Image.fromarray(image).save(image_path)
	np.savez_compressed(frame_path, observation=observation, action=action)

	return image_path, frame_path


def main():
	args = parse_args()
	out_dir = make_output_dir(args.output_dir)
	env = create_env(args)
	rng = np.random.default_rng(args.seed)

	manifest = {
		"passes": args.passes,
		"steps_per_pass": args.steps_per_pass,
		"seed": args.seed,
		"observation_shape": [5],
		"action_shape": [2],
		"frames": [],
	}

	global_frame_idx = 0

	try:
		for pass_idx in range(args.passes):
			observation, info = env.reset()
			block_center = np.array([observation[2], observation[3]], dtype=np.float64)
			actions, line_start, line_end = generate_pass_actions_through_block(
				block_center=block_center,
				steps_per_pass=args.steps_per_pass,
				rng=rng,
			)

			for step_idx, action in enumerate(actions):
				observation, reward, terminated, truncated, info = env.step(action)
				image = env.render()

				image_path, frame_path = save_frame(
					out_dir=out_dir,
					frame_idx=global_frame_idx,
					image=image,
					observation=observation,
					action=action,
				)

				manifest["frames"].append(
					{
						"frame_index": global_frame_idx,
						"pass_index": pass_idx,
						"step_index": step_idx,
						"line_start": [float(line_start[0]), float(line_start[1])],
						"line_end": [float(line_end[0]), float(line_end[1])],
						"image": str(image_path.relative_to(out_dir)),
						"data": str(frame_path.relative_to(out_dir)),
					}
				)

				global_frame_idx += 1

				if terminated or truncated:
					break
	finally:
		env.close()

	manifest_path = out_dir / "manifest.json"
	with manifest_path.open("w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2)

	print(f"Saved {global_frame_idx} frames to {out_dir}")
	print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
	main()
