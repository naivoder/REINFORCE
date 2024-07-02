from network import Policy
import numpy as np
import torch
import torch.nn.functional as F


class Agent:
    def __init__(
        self, env_name, lr, input_dims, n_actions=4, gamma=0.99, use_cnn=False
    ):
        self.env_name = env_name
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.use_cnn = use_cnn

        self.reward_memory = []
        self.action_memory = []

        self.policy = Policy(
            input_dims=self.input_dims,
            n_actions=self.n_actions,
            lr=self.lr,
            chkpt_path=f"weights/{env_name}.pt",
            use_cnn=use_cnn,
        )

    def choose_action(self, state):
        state = torch.Tensor(np.array(state)).to(self.policy.device)
        if self.use_cnn:
            state = state.unsqueeze(0)  # Add batch dimension for CNN
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
    agent = Agent(
        env_name="CartPole-v1", lr=0.0005, input_dims=(8,), n_actions=4, use_cnn=False
    )
