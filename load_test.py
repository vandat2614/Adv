from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3
import numpy as np
from matd3 import MATD3

env = simple_adversary_v3.parallel_env(render_mode="human", continuous_actions=True)
agents_names = ['adversary_0', 'agent_0', 'agent_1']
actor_dims = [env.observation_spaces[agent_name].shape[0] for agent_name in agents_names]
action_dims = [env.action_spaces[agent_name].shape[0] for agent_name in agents_names]
matd3_agent = MATD3(agents_names, actor_dims, action_dims)
matd3_agent.load_checkpoint()

while True:
    state, info = env.reset()
    done = [False] * len(agents_names)
    while not any(done):
        actions = matd3_agent.choose_action(state)
        next_state, reward, termination, truncation, _  = env.step(actions) 

        if any(termination.values()) or any(truncation.values()):
            done = [True] * len(agents_names)