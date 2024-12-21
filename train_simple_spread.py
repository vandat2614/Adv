from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3
import numpy as np
from matd3 import MATD3
from buffer import ReplayBuffer


env = simple_adversary_v3.parallel_env(render_mode="human", continuous_actions=True)
agents_names = ['adversary_0', 'agent_0', 'agent_1']
actor_dims = [env.observation_spaces[agent_name].shape[0] for agent_name in agents_names]
action_dims = [env.action_spaces[agent_name].shape[0] for agent_name in agents_names]
matd3_agent = MATD3(agents_names, actor_dims, action_dims)
buffer = ReplayBuffer(50000, actor_dims, action_dims, batch_size=512, agent_names=agents_names)

num_episodes = 30000 
max_episode_len = 100   
update_rate = 100
count = 0

for i in range(num_episodes):
    state, info = env.reset() # state = dict, key = agent_name, value = np.array(18,)
    critic_state = np.concatenate([s for s in state.values()]) # critic = np.array(54,)
    score = 0
    done = [False] * len(agents_names)
    episode_step = 0
    while not any(done):
        actions = matd3_agent.choose_action(state) # actions = dict, key = agent_name, value = np.array(5,)
        next_state, reward, termination, truncation, _  = env.step(actions) 
        # next_state === state
        # reward = dict, key = agent_name, value = 1 số (float)
        # termination === truncation = dict, key = agent_name, value = True/False
        critic_next_state = np.concatenate([s for s in next_state.values()]) # critic_next = np.array(54, )
        score += sum(reward.values())
        episode_step += 1
        if episode_step >= max_episode_len or any(termination.values()) or any(truncation.values()):
            done = [True] * len(agents_names)
        buffer.store_transition(state, critic_state, actions, list(reward.values()), next_state, critic_next_state, done) 
        state = next_state
        critic_state = critic_next_state
        
        if buffer.ready():
            matd3_agent.update_critic(buffer)
            count += 1
        if (count+1)%update_rate==0 and buffer.ready():
            matd3_agent.update_actor(buffer)
            matd3_agent.update_target_actor()
            matd3_agent.update_target_critic()
            count =0
    if (i+1)%100==0:
        print(f'Episode: {i+1} - score: {score} - num step: {episode_step}')

matd3_agent.save_checkpoint()
