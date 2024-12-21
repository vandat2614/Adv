import torch as T
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_dim, critic_dim, action_dim, chkpt_dir,agent_name,
                    alpha=0.01, beta=0.01, fc1=64,
                    fc2=64, gamma=0.95, tau=0.01 ,
                    ):
        self.gamma = gamma
        self.tau = tau
        self.agent_name = agent_name
        self.action_dim = action_dim

        self.actor = ActorNetwork(alpha, actor_dim, fc1, fc2, action_dim,
                                    chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        
        self.target_actor = ActorNetwork(alpha, actor_dim, fc1, fc2, action_dim,
                                    chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')

        self.critic_1 = CriticNetwork(beta, critic_dim, fc1, fc2,
                                    chkpt_dir=chkpt_dir, name=self.agent_name+'_critic_1')
        
        self.target_critic_1 = CriticNetwork(beta, critic_dim, fc1, fc2,
                                    chkpt_dir=chkpt_dir,name=self.agent_name+'_target_critic_1')

        self.critic_2 = CriticNetwork(beta, critic_dim, fc1, fc2,
                                    chkpt_dir=chkpt_dir, name=self.agent_name+'_critic_2')
        
        self.target_critic_2 = CriticNetwork(beta, critic_dim, fc1, fc2,
                                    chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic_2')

        self.update_target_actor(tau=1)
        self.update_target_critic(tau=1)


    def choose_action(self, observation): # đầu vào là np.array(18,)
        state = T.tensor([observation], dtype=T.float).to(self.actor.device) # shape=(1, 18)
        actions = self.actor.forward(state) # (1, 5)

        noise = (T.rand(self.action_dim, device=self.actor.device) * (1 - actions).min())
        action = actions + noise
        return action.detach().cpu().numpy()[0] # (1, 5) - > (5, )

    def update_target_actor(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()
            
        self.target_actor.load_state_dict(target_actor_state_dict)

    def update_target_critic(self, tau=None):
        if tau is None:
            tau = self.tau

        target_critic_1_params = self.target_critic_1.named_parameters()
        critic_1_params = self.critic_1.named_parameters()

        target_critic_1_state_dict = dict(target_critic_1_params)
        critic_1_state_dict = dict(critic_1_params)
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)

        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_2_state_dict = dict(critic_2_params)
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        self.target_critic_2.load_state_dict(critic_2_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()