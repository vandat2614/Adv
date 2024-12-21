import torch as T
import torch.nn.functional as F
from agent import Agent
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
import numpy as np


class MATD3:
    def __init__(self, agent_names, actor_dims, action_dims, 
                 alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/matd3/'):
        self.n_agents = len(agent_names)
        self.action_dims = action_dims
        self.agents = {}

        critic_dim = sum(actor_dims) + sum(action_dims)
        for agent_idx, agent_name in enumerate(agent_names): 
            self.agents[agent_name] = Agent(actor_dims[agent_idx], critic_dim, action_dims[agent_idx],
                                            agent_name = agent_name, alpha=alpha, beta=beta, chkpt_dir=chkpt_dir)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.load_models()

    def choose_action(self, raw_obs): # đầu vào là dict, key = agent_name, value = np.array(18,)
        actions = {agent_name : agent.choose_action(raw_obs[agent_name]) for agent_name, agent in self.agents.items()} 
        return actions # đầu ra là dict, key = agent_name, value = np.array(5,)

    def update_critic(self, memory):
        if not memory.ready(): 
            return

        actor_states, critic_states, actions, rewards, \
        actor_next_states, critic_next_states, dones = memory.sample_buffer()

        critic_states = T.tensor(critic_states, dtype=T.float).to(self.device) # (512, 54)
        # actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(self.device) # (512, 3)
        critic_next_states = T.tensor(critic_next_states, dtype=T.float).to(self.device) # (512, 54)
        dones = T.tensor(dones).to(self.device) # (512, 3)

        # Compute target actions
        critic_actions = []
        critic_next_actions = []
        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            critic_actions.append(T.tensor(actions[agent_idx], dtype=T.float))

            actor_next_state = T.tensor(actor_next_states[agent_idx], dtype=T.float).to(self.device) # (512, 18)
            critic_next_actions.append(agent.target_actor.forward(actor_next_state)) # (512, 5)

        critic_actions = T.cat(critic_actions, dim=-1).to(self.device) # (512, 15)
        critic_next_actions = T.cat(critic_next_actions, dim=-1).to(self.device) # (512, 15)

        # Update critic
        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            # Compute target Q-value
            q_target_value_1 = agent.target_critic_1.forward(critic_next_states, critic_next_actions).flatten() # (512 54), (512, 15) -> (512, 1) -> (512)
            q_target_value_2 = agent.target_critic_2.forward(critic_next_states, critic_next_actions).flatten()

            q_target_value = T.min(q_target_value_1, q_target_value_2) # (512)
            q_target_value[dones[:, agent_idx]] = 0.0 # (512)

            # Compute target and loss
            target = rewards[:, agent_idx].float() + agent.gamma * q_target_value.flatten() # (512) + gamma * (512) 

            q_value_1 = agent.critic_1.forward(critic_states, critic_actions).flatten() # (512, 54), (512, 15) -> (512, 1) -> (512)
            q_value_2 = agent.critic_2.forward(critic_states, critic_actions).flatten()
            critic_1_loss = F.mse_loss(q_value_1, target) 
            critic_2_loss = F.mse_loss(q_value_2, target)

            # update critic 1
            agent.critic_1.optimizer.zero_grad()
            critic_1_loss.backward(retain_graph=True)
            agent.critic_1.optimizer.step()

            # update critic 2
            agent.critic_2.optimizer.zero_grad()
            critic_2_loss.backward(retain_graph=True)
            agent.critic_2.optimizer.step()
    
    def update_actor(self, memory):
        if not memory.ready():
            return

        actor_states, critic_states, actions, rewards, \
        actor_next_states, critic_next_states, dones = memory.sample_buffer()

        critic_states = T.tensor(critic_states, dtype=T.float).to(self.device)
        
        # Calculate actions for each agent independently
        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            # Get current agent's action
            actor_state = T.tensor(actor_states[agent_idx], dtype=T.float).to(self.device)
            current_action = agent.actor.forward(actor_state)
            
            # Get other agents' actions
            other_actions = []
            for other_idx, (other_name, other_agent) in enumerate(self.agents.items()):
                if other_idx != agent_idx:
                    other_state = T.tensor(actor_states[other_idx], dtype=T.float).to(self.device)
                    other_action = other_agent.actor.forward(other_state).detach()
                    other_actions.append(other_action)
            
            # Combine actions in correct order
            all_actions = []
            for i in range(len(self.agents)):
                if i == agent_idx:
                    all_actions.append(current_action)
                else:
                    all_actions.append(other_actions.pop(0))
                    
            combined_actions = T.cat(all_actions, dim=-1)
            
            # Update actor
            agent.actor.optimizer.zero_grad()
            q_value = agent.critic_1.forward(critic_states, combined_actions)
            actor_loss = -T.mean(q_value)
            actor_loss.backward()
            agent.actor.optimizer.step()


    def update_target_critic(self):
        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            agent.update_target_critic()

    def update_target_actor(self):
        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            agent.update_target_actor()