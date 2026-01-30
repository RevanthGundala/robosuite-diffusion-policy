"""Debug script to understand policy behavior."""
import numpy as np
from data.collect import ScriptedLiftPolicy
from envs.robosuite_wrapper import create_lift_env
import warnings
warnings.filterwarnings('ignore')

env = create_lift_env(use_images=False, render=False)
obs = env.reset()

# Check geometry
cube_body_id = env.env.cube_body_id
table_z = env.env.model.mujoco_arena.table_offset[2]

print(f"Success threshold: cube_z > {table_z + 0.04:.3f}")
print()

# Get gripper
robot = env.env.robots[0]
gripper = robot.gripper

scripted = ScriptedLiftPolicy()
scripted.reset()

print("Step | Phase | Z gap | Grasp? | Cube Z | Action[-1] | Fingers")
print("-" * 70)

for step in range(250):
    action = scripted(env, obs)
    
    gripper_pos = obs['gripper_pos']
    cube = env.env.sim.data.body_xpos[cube_body_id]
    z_gap = gripper_pos[2] - cube[2]
    grip_state = obs['proprio'][-2:]
    
    obs, reward, done, info = env.step(action)
    
    # Check if we have a grasp
    has_grasp = env.env._check_grasp(gripper, env.env.cube.contact_geoms)
    
    # Print more during critical phases
    show = step < 25 or (step >= 120 and step <= 170) or (step % 40 == 0)
    if show or scripted.phase != getattr(scripted, '_last_phase', -1):
        print(f"{step:4d} | {scripted.phase:5d} | {z_gap:.3f} | {str(has_grasp):5s} | {cube[2]:.3f} | {action[-1]:+.1f} | {grip_state[0]:+.3f}")
    scripted._last_phase = scripted.phase
    
    if info.get('success'):
        print(f"\n*** SUCCESS at step {step}! ***")
        break

final_cube = env.env.sim.data.body_xpos[cube_body_id]
print(f"\nFinal cube_z: {final_cube[2]:.3f}, threshold: {table_z + 0.04:.3f}")
env.close()
