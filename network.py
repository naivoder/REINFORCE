import torch
from torch import nn


class Policy(nn.Module):
    def __init__(
        self, input_dims, n_actions, lr, chkpt_path="weights/policy.pt", use_cnn=False
    ):
        super(Policy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.chkpt_path = chkpt_path
        self.use_cnn = use_cnn

        if use_cnn:
            self.model = nn.Sequential(
                nn.Conv2d(input_dims[0], 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 9 * 9, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions),
            ).to(self.device)
        else:
            self.model = nn.Sequential(
                nn.Linear(*self.input_dims, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.n_actions),
            ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, x):
        if self.use_cnn:
            x = x / 255.0
        return self.model(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_path))


if __name__ == "__main__":
    model = Policy(input_dims=(8), n_actions=4, lr=0.0005)
