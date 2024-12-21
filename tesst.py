from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.parallel_env(render_mode="human", continuous_actions=True)
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    break
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()