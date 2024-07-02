from network import Policy
import numpy as np
import torch
import torch.nn.functional as F


class Agent:
    def __init__(self, env_name, lr, input_dims, n_actions=4, gamma=0.99):
        self.env_name = env_name
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma

        self.reward_memory = []
        self.action_memory = []

        self.policy = Policy(
            input_dims=self.input_dims,
            n_actions=self.n_actions,
            lr=self.lr,
            chkpt_path=f"weights/{env_name}.pt",
        )

    def choose_action(self, state):
        # need to add batch dimension to state so pytorch doesn't freak out
        state = torch.Tensor(np.array(state)).to(self.policy.device)
        probs = F.softmax(self.policy(state), dim=-1)

        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        self.action_memory.append(log_prob)

        return action.detach().cpu().numpy()

    def learn(self):
        self.policy.optimizer.zero_grad()
        Gt = np.zeros_like(self.reward_memory, dtype=np.float64)

        for i in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for j in range(i, len(self.reward_memory)):
                G_sum += discount * self.reward_memory[j]
                discount *= self.gamma
            Gt[i] = G_sum

        Gt = torch.Tensor(Gt).to(self.policy.device)

        loss = 0
        for g, logprob in zip(Gt, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        self.clear_memory()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def clear_memory(self):
        self.reward_memory = []
        self.action_memory = []

    def save_checkpoints(self):
        self.policy.save_checkpoint()

    def load_checkpoints(self):
        self.policy.load_checkpoint()


if __name__ == "__main__":
    agent = Agent(0.0005, (8))
