default_config = {
    "lanes_count" : 10,
    "vehicles_count": 50,
    "duration": 120,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "initial_spacing": 2, 
    "action": {"type": "DiscreteMetaAction",},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "collision_reward": -100, 
    "right_lane_reward": 0.1,
    "high_speed_reward": 100, 
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "screen_width": 800,  
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False
}


optimum_q_kinematics = {
    (0,0,0,0,0,0) : 2, 
    (0,0,0,0,1,0) : 3,
    (0,0,0,0,-1,0) : 2, 
    (1,0,0,0,0,0) : 2,
    (1,0,0,0,1,0) : 0,
    (1,0,0,0,-1,0) : 2,
    (0,0,1,0,0,0) : 2,
    (0,0,1,0,1,0) : 3,
    (0,0,0,1,0,0) : 3,
    (0,0,0,1,-1,0) : 3,
    (1,0,1,1,0,0) : 4,
    (0,0,1,1,0,0) : 3,
    (1,0,0,1,0,0) : 0,
    (1,0,0,1,-1,0) : 4,
    (1,0,1,0,0,0) : 2,
    (1,0,1,0,1,0) : 4,
    (0,0,0,0,0,1) : 3,    # !! ALL 3s could have to be substituted with 4s 
    (0,0,0,0,1,1) : 3, 
    (0,0,0,0,-1,1) : 3,  # Could have only turned right before, so faster 
    (1,0,0,0,0,1) : 3,   # Car in front and just turned, so the car will soon be on my left or right
    (1,0,0,0,1,1) : 3,  
    (1,0,0,0,-1,1) : 3,
    (0,0,1,0,0,1) : 3,
    (0,0,1,0,1,1) : 3,
    (0,0,0,1,0,1) : 3,
    (0,0,0,1,-1,1) : 4,   # Car on the right, couldnt have turned left, so soon will be in front
    (1,0,1,1,0,1) : 4,    # Terrible, brace
    (0,0,1,1,0,1) : 4,    # Terrible 
    (1,0,0,1,0,1) : 3,    # I have turned left
    (1,0,0,1,-1,1) : 4,   # On the right and in front, have turned right, terrible
    (1,0,1,0,0,1) : 3,    # On the left and in front, have turned right
    (1,0,1,0,1,1) : 4,    # On the left, cant turn right, could only have turned left, terrible
    #(0,1,0,0,0) : 2,
    #(0,1,0,0,1) : 3,
    #(0,1,0,0,-1) : 2,
    # (0,0,1,0,-1) : 2,
    # (0,0,0,1,1) : 3,
    #(1,1,0,0,0) : 2,
    #(1,1,0,0,1) : 0,
    #(1,1,0,0,-1) : 2,
    #(1,1,1,0,0) : 2,
    #(1,1,1,0,1) : 2,
    # (1,1,1,0,-1) : 1,
    #(1,1,1,1,0) : 1,
    # (1,1,1,1,1) : 1,
    # (1,1,1,1,-1) : 1,
    #(1,1,0,1,0) : 0,
    # (1,1,0,1,1) : 0,
    #(1,1,0,1,-1) : 1,
    # (1,0,1,1,1) : 1,
    # (1,0,1,1,-1) : 1,
    # (0,1,1,1,0) : 3,
    # (0,1,1,1,1) : 3,
    # (0,1,1,1,-1) : 3,
    # (0,0,1,1,1) : 3,
    # (0,0,1,1,-1) : 3,
    # (0,1,0,1,0) : 3,
    # (0,1,0,1,1) : 3,
    # (0,1,0,1,-1) : 3,
    # (0,1,1,0,0) : 2,
    # (0,1,1,0,1) : 3,
    # (0,1,1,0,-1) : 2,
    # (1,0,0,1,1) : 0,
    #(1,0,1,0,-1) : 2,
}