action_names = ['left', 'idle', 'right', 'faster', 'slower']

default_config = {
    "lanes_count" : 4,
    "vehicles_count": 50,
    "duration": 40,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "initial_spacing": 2,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "collision_reward": -200,
    "right_lane_reward": 5,
    "high_speed_reward": 20,
    "lane_change_reward": 3,
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
    "screen_width": 800,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False
}