# PushT Environment Overview

## Environment

**ID:** `gym_pusht/PushT-v0`

## Observation Space

The observation is a 5D vector:

```
[ pos_agent_x, pos_agent_y, block_x, block_y, block_angle ]
```

| Field | Description |
|---|---|
| `pos_agent` | Current (x, y) position of the agent (circular pusher) |
| `vel_agent` | Current (x, y) velocity of the agent |
| `block_pose` | (x, y, angle) pose of the T-shaped block |
| `goal_pose` | (x, y, angle) target pose the block should reach |
| `n_contacts` | Number of contacts between agent and block |
| `is_success` | Whether the block is sufficiently aligned with the goal |

## Action Space

**Type:** `Box(2,)` — continuous 2D vector

```
action = [target_x, target_y]
```

The action specifies the **target position** the agent should move toward, in pixel coordinates `[0, 512]`. The agent does not teleport — it moves incrementally each step via physics.

## pos_agent vs action

| | `pos_agent` | `action` |
|---|---|---|
| What it is | Current position of the agent | Target position for the agent |
| Role | State (observation) | Control input |
| Example | `[84., 169.]` | `[345.84, 410.22]` |

The environment applies velocity/physics to move the agent from `pos_agent` toward the `action` target each timestep.
