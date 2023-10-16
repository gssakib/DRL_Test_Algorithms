import torch as T 

class GaussianPolicy:
    def __init__(self, n_actions, min_action=-1, max_action=1):
        self.min_action = min_action
        self.max_action = max_action
        self.n_actions = n_actions
    
    def __call__(self, mu, sigma, reparam=False):
        probs = T.distributions.Normal(mu,sigma)
        actions = probs.rsample() if reparam else probs.sample()
        log_probs = probs.log_prob(actions)
        actions=T.tanh(actions) / T.tensor(self.max_action, device=actions.device)

        return actions, log_probs