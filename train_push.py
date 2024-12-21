import numpy as np
from matd3 import MATD3
from buffer import ReplayBuffer
import time
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1


if __name__ == '__main__':
    env = mamujoco_v1.parallel_env("Pusher", "3p", render_mode='human')

    agent_names = env.agents
    actor_dims = [env.observation_space(agent_name).shape[0] for agent_name in agent_names]
    action_dims = [env.action_space(agent_name).shape[0] for agent_name in agent_names]
    matd3_agent = MATD3(agent_names, actor_dims, action_dims)

    buffer = ReplayBuffer(10000, actor_dims, action_dims, batch_size=64, agent_names=agent_names)

    N_GAMES = 500000
    MAX_STEP_PER_EPISODE = 75

    for i in range(N_GAMES):
        state, info = env.reset()
        critic_state = np.concatenate([s for s in state.values()])

        score = 0
        done = [False] * len(agent_names)
        episode_step = 0

        while not any(done):
            actions = matd3_agent.choose_action(state)
            next_state, reward, termination, truncation, _ = env.step(actions)

            critic_next_state = np.concatenate([s for s in next_state.values()])

            score += sum(reward.values())
            episode_step += 1 

            if episode_step >= MAX_STEP_PER_EPISODE or any(termination.values()) or any(truncation.values()):
                done = [True] * len(agent_names)
            buffer.store_transition(state, critic_state, actions, list(reward.values()), next_state, critic_next_state, done) 

            state = next_state
            critic_state = critic_next_state

            matd3_agent.update_critic(buffer)
            if (i+1)%10==0:
                matd3_agent.update_target_critic()
            if (i+1)%2==0:
                matd3_agent.update_critic(buffer)
            if (i+1)%4==0:
                matd3_agent.update_target_actor()

        print(f'Episode: {i+1} - score: {score} - num step: {episode_step}')
    #     if not evaluate:

    #         if (avg_score > best_score) and (i > PRINT_INTERVAL):
    #             print(" avg_score, best_score", avg_score, best_score)
    #             # maddpg_agents.save_checkpoint() #!
    #             best_score = avg_score
    #     if i % PRINT_INTERVAL == 0 and i > 0:
    #         print('='*35)
    #         print('episode', i, 'average score {:.1f}'.format(avg_score))