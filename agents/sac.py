from agents.base import Agent
import torch as T 
import torch.nn.functional as F

class SACAgent:
    def __init__(self, actor_net, critic_net_1, critic_net_2, value_net, 
                 target_value_net, memory, policy, gamma=0.99, actor_lr=3e-4,
                 critic_lr =3e-4, value_lr=3e-4,tau=0.005, reward_scale=2):
        self.reward_scale = reward_scale

        self.actor = actor_net
        self.critic_1 = critic_net_1
        self.critic_2 = critic_net_2
        self.value = value_net
        self.target_value = target_value_net

        self.networks = [net for net in [self.actor, self.critic_1, self.critic_2, self.value, self.target_value]]

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(),lr=critic_lr)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(),lr=critic_lr)
        self.value_optimizer = T.optim.Adam(self.value.parameters(),lr=value_lr)
        self.update_network_parameters(self.value,self.target_value, tau=1.0)

    def choose_actions(self, observation):
        state = T.tensor(observation, dtype=T.float, device=self.device)
        mu, sigma = self.actor(state)
        actions, _ = self.policy(mu, state)

        return actions.cpu().detach().numpy()
    
    def update(self):
        if not self.memory.ready():
            return
        states, actions, rewards, states_, dones = self.sample_memory()

        value = self.value(states).view(-1)
        value_ = self.target_value(states_).view(-1)
        value_[dones] = 0.0 

        # Calculate Actor Loss
        mu, sigma = self.actor(states)
        new_actions, log_probs = self.policy(mu,sigma, False)
        log_probs -= T.log(1- new_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1([states, new_actions])
        q2_new_policy = self.critic_2([states, new_actions])
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value_optimizer.zero_grad()
        value_target = critic_value-log_probs
        value_loss = 0.5*F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        #Calculate Critic Loss
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        q_hat = self.reward_scale*rewards + self.gamma * value_
        q1_old_policy = self. critic_1([states, actions]).view(-1)
        q2_old_policy = self.critic_2([states,actions]).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss +critic_2_loss
        critic_loss.backward()
        self.critic_2_optimizer.step()

        self.update_network_parameters(self.value, self.target_value)





       


        




    