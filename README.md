# RL_Autonomous_Driving

This project is focused on developing an RL agent capable of solving an environment for decision-making in Autonomous Driving. 

Autonomous Driving has long been considered a field in which RL algorithms excel, and this project aims to leverage the power of RL to create an intelligent agent that can solve the Farama’s foundation “highway-env” project, namely the Highway environment (refer to https://highway-env.farama.org/environments/highway/).

Report: https://www.overleaf.com/read/wqwxcfsxntff#f06c0b

____________
## TODO
- Use kinematics to substitute the current occupancy grid class

- He cannot know when he cant turn left or right because the lane is the final one

- Change the occupancy class to use occupancy grid with 5m grid size, 3 lanes and -30, 30 ahead

- Make a plot history to check the Q-function training learning   

- Check how many times each state is visited, and the distribution thereof

- Check the action distribution, so as to see if slowing down is the most chosen action and the one with the best Q value

- Tweak the reward for colision, so that the agent goes faster

- Dont use the argmax(Q(s,a)) policy always, allow for some temperature so that we don't get stuck in loops

______________
## Ideas
### Kinematics: 
- State space in this maner: (danger ahead, danger left, danger right, danger behind, lane position, (maybe) speed)

- Lane position can be 0 if in the middle, 1 if in the right, -1 if in the left

- Do we need speed? because if the speed is too fast, it might not be able to turn in time or we could just increase the safety distance, and then we wouldn't need speed

### Continuous actions with deep-Q learning:
- Read Sutton and Barto last parts 

