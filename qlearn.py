import gym
import numpy as np

# env = gym.make("MsPacman-v0")
env = gym.make("MountainCar-v0")
# env = gym.make("FlappyBird-v0")

env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # importance of future actions
EPISODES = 25000 # 25000 epochs 
SHOW_EVERY = 3000 # show something every 2000 episodes
epsilon = 0.4 # sets how "exploratory" (random) our move is going to be
epsilon_decay = 0.1*epsilon # decaus by 10% every time
START_DECAY = 2
END__DECAY = 3010
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)


# 20 buckets for the state space
DICRETE_OBS_SIZE = [20] * len(env.observation_space.high)
discrete_win_size = (env.observation_space.high - env.observation_space.low)/DICRETE_OBS_SIZE
# print(discrete_win_size)
q_table = np.random.uniform(low = -2, high = 1, size = (DICRETE_OBS_SIZE + [env.action_space.n]))
print(q_table.shape)

def get_discretized_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_win_size
    return tuple(discrete_state.astype(np.int))

# FOR DOCUMENTATION
# # we treat the discrete state as index for q_table
# # since q_table contains every such state possible
# print(discrete_state)
# # this should give us the values for the 3 actions possible
# print(q_table[discrete_state])
# # this gives us the max value (action to take)
# print(np.argmax(q_table[discrete_state]))

# note : cmd + [/] indents/un-indents blocks of code
for episode in range(EPISODES):
    discrete_state = get_discretized_state(env.reset())
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    done = False
    while not done:
        if np.random.random() < epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[discrete_state])
        next_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discretized_state(next_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            cur_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * cur_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q # update the q table values
        elif next_state[0] >= env.goal_position:
            print("we made it on episode : " , episode)
            q_table[discrete_state + (action,)] = 0
            break
        discrete_state = new_discrete_state
        if render:
            # print(reward)
            # env.render()
            render = False
        if END__DECAY >= episode >= START_DECAY:
            # env.render()
            epsilon -= epsilon_decay
        env.render()

env.close()
