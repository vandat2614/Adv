import numpy as np

class ReplayBuffer: # ok
    def __init__(self, max_size, actor_dims, 
            action_dims, batch_size,agent_names):
        
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = len(action_dims) # 3
        self.actor_dims = actor_dims # [18, 18, 18]
        self.batch_size = batch_size # 512
        self.action_dims = action_dims # [5, 5, 5]
        self.agent_names = agent_names # [agent_0, agent_1, agent_2]
        
        critic_dim = sum(actor_dims) # 54
        self.critic_state_memory = np.zeros((self.mem_size, critic_dim)) # (50k, 54)
        self.critic_next_state_memory = np.zeros((self.mem_size, critic_dim)) # (50k, 54)
        self.reward_memory = np.zeros((self.mem_size, self.n_agents), dtype=np.float64) # (50k, 3)
        self.terminal = np.zeros((self.mem_size, self.n_agents), dtype=bool) # (50k, 3)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = [] # len(list) = 3 -> (50k, 18)
        self.actor_next_state_memory = [] # len(list) = 3 -> (50k, 18)
        self.actor_action = []  # len(list) = 3 -> (50k, 5)

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i]))) 
            self.actor_next_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action.append(
                            np.zeros((self.mem_size, self.action_dims[i])))


    def store_transition(self, actor_states, critic_state, actions, rewards, 
                               actor_next_states, critic_next_state, done):
        index = self.mem_cntr % self.mem_size

        for agent_idx, agent_name in enumerate(self.agent_names):
            self.actor_state_memory[agent_idx][index] = actor_states[agent_name] 
            self.actor_next_state_memory[agent_idx][index] = actor_next_states[agent_name] 
            self.actor_action[agent_idx][index] = actions[agent_name]

        self.critic_state_memory[index] = critic_state 
        self.critic_next_state_memory[index] = critic_next_state
        self.reward_memory[index] = rewards 
        self.terminal[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_indicies = np.random.choice(max_mem, self.batch_size, replace=False)

        critic_states = self.critic_state_memory[batch_indicies] # (512, 54)
        rewards = self.reward_memory[batch_indicies]             # (512, 3)
        critic_next_states = self.critic_next_state_memory[batch_indicies]  # (512, 54)
        done = self.terminal[batch_indicies]                    # (512, 3)

        actor_states = [] # len(list) = 3, (512, 18)
        actor_next_states = [] # len(list) = 3, (512, 18)
        actions = [] # len(list) = 3, (512, 5)
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch_indicies])
            actor_next_states.append(self.actor_next_state_memory[agent_idx][batch_indicies])
            actions.append(self.actor_action[agent_idx][batch_indicies])

        return actor_states, critic_states, actions, rewards, \
               actor_next_states, critic_next_states, done

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True