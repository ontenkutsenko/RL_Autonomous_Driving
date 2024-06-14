def fix_reward(reward, position, action, to_right_reward=5, to_right_skewness=2, change_lane_reward=-0.5):
    """
    This function is used to correct the reward function, which is not correctly outputted by the environment
    Params: 
        reward: float, the reward to fix    
        position: tuple, the position of the car
        action: int, the action taken by the car
        to_right_reward: float, the reward to give to the driver 
        to_right_skewness: float, the skewness to apply to the to_right_reward
        change_lane_reward: float, the reward to give when changing lanes
    """
    
    lane = position[1]
    lane_value = to_right_reward * ((lane/36)**to_right_skewness)
    lane_change = 1 if action in [0, 2] else 0
    reward = reward + lane_value + lane_change * change_lane_reward    
    return reward

def decode_meta_action(action):
    """
    Function to output the corresponding action in text-form
    """
    assert action in range(5), "The action must be between 0 and 4"
    if action == 0:
        return "LANE_LEFT"
    elif action == 1:
        return "IDLE"
    elif action == 2:
        return "LANE_RIGHT"
    elif action == 3:
        return "FASTER"
    elif action == 4:
        return "SLOWER"
    
def decode_danger(state):
    """
    Function to decode the danger state
    """
    state_meaning = ['front', 'back', 'left', 'right']
    to_return = ''
    for i in range(3): 
        if state[i] == 1:
            if to_return == '':
                to_return = 'Danger in '
            to_return += state_meaning[i] + ', '
    if to_return == '': 
        to_return = 'No danger'
    if state[4] == -1:
        to_return += '. Cant turn left'
    elif state[4] == 1:
        to_return += '. Cant turn right'
    return to_return
    

def decode_Q(Q): 
    """
    Function to decode the Q-values
    """
    return {(decode_danger(key[0]), decode_meta_action(key[1])) : value for key, value in Q.items()}
