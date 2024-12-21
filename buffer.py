import numpy as np

class ReplayBuffer: # ok
    def __init__(self, max_size, actor_dims, 
            action_dims, batch_size,agent_names):
        
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = len(action_dims)
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.action_dims = action_dims
        self.agent_names = agent_names
        
        critic_dim = sum(actor_dims)
        self.critic_state_memory = np.zeros((self.mem_size, critic_dim)) # for critic
        self.critic_next_state_memory = np.zeros((self.mem_size, critic_dim)) # for critic
        self.reward_memory = np.zeros((self.mem_size, self.n_agents)) # earch reward is number
        self.terminal = np.zeros((self.mem_size, self.n_agents), dtype=bool) # same reward

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = [] 
        self.actor_next_state_memory = []
        self.actor_action = [] 

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_next_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action.append(
                            np.zeros((self.mem_size, self.action_dims[i])))


    def store_transition(self, actor_states, critic_state, actions, rewards, 
                               actor_next_states, critic_next_state, done):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        #if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()
        
        index = self.mem_cntr % self.mem_size

        for agent_idx, agent_name in enumerate(self.agent_names):
            self.actor_state_memory[agent_idx][index] = actor_states[agent_name] # = np.array()
            self.actor_next_state_memory[agent_idx][index] = actor_next_states[agent_name] # = np.array)
            self.actor_action[agent_idx][index] = actions[agent_name] # = probs distribute = np.array()

        self.critic_state_memory[index] = critic_state # np vector with shape = (28, )
        self.critic_next_state_memory[index] = critic_next_state # same state
        self.reward_memory[index] = rewards 
        self.terminal[index] = done # done is list [F, F, F] or [T, T, T]
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch_indicies = np.random.choice(max_mem, self.batch_size, replace=False)
        critic_states = self.critic_state_memory[batch_indicies]
        rewards = self.reward_memory[batch_indicies]
        critic_next_states = self.critic_next_state_memory[batch_indicies]
        done = self.terminal[batch_indicies]

        actor_states = []
        actor_next_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch_indicies])
            actor_next_states.append(self.actor_next_state_memory[agent_idx][batch_indicies])
            actions.append(self.actor_action[agent_idx][batch_indicies])

        return actor_states, critic_states, actions, rewards, \
               actor_next_states, critic_next_states, done

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True