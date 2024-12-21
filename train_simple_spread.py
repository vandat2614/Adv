from pettingzoo.mpe import simple_spread_v3
import numpy as np
from matd3 import MATD3
from buffer import ReplayBuffer


env = simple_spread_v3.parallel_env(render_mode="human", continuous_actions=True)
agents_names = ['agent_0', 'agent_1', 'agent_2']
actor_dims = [env.observation_spaces[agent_name].shape[0] for agent_name in agents_names]
action_dims = [env.action_spaces[agent_name].shape[0] for agent_name in agents_names]
matd3_agent = MATD3(agents_names, actor_dims, action_dims)
buffer = ReplayBuffer(50000, actor_dims, action_dims, batch_size=512, agent_names=agents_names)

num_episodes = 30000 
max_episode_len = 100   
update_rate = 100

for i in range(num_episodes):
    state, info = env.reset()
    critic_state = np.concatenate([s for s in state.values()])
    score = 0
    done = [False] * len(agents_names)
    episode_step = 0
    while not any(done):
        actions = matd3_agent.choose_action(state)
        next_state, reward, termination, truncation, _  = env.step(actions)
        critic_next_state = np.concatenate([s for s in next_state.values()])
        score += sum(reward.values())
        episode_step += 1
        if episode_step >= max_episode_len or any(termination.values()) or any(truncation.values()):
            done = [True] * len(agents_names)
        buffer.store_transition(state, critic_state, actions, list(reward.values()), next_state, critic_next_state, done) 
        state = next_state
        critic_state = critic_next_state
        
        if buffer.ready():
            matd3_agent.update_critic(buffer)
        if (i+1)%update_rate==0 and buffer.ready():
            print(f'UPDATE ACTOR')
            matd3_agent.update_actor(buffer)
            matd3_agent.update_target_actor()
            matd3_agent.update_target_critic()
    print(f'Episode: {i+1} - score: {score} - num step: {episode_step}')