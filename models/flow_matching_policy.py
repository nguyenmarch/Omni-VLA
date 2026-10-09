"""Independent-pair conditional flow matching; does not claim OT matching."""
import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x)


class FlowMatchingActionPolicy(nn.Module):
    def __init__(self, action_dim=14, chunk_size=16, cond_dim=67, hidden_dim=256, direct_skip=False):
        super().__init__()
        self.action_dim, self.chunk_size = action_dim, chunk_size
        self.register_buffer('frequencies', torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.input = nn.Linear(action_dim*chunk_size+cond_dim+32, hidden_dim)
        self.body = nn.Sequential(nn.SiLU(), ResidualBlock(hidden_dim), ResidualBlock(hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.output = nn.Linear(hidden_dim, chunk_size*action_dim)
        self.direct_skip = direct_skip
        if direct_skip:
            self.noise_gate = nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,chunk_size*action_dim))
            self.condition_skip = nn.Linear(cond_dim,chunk_size*action_dim)

    def forward(self, x_t, t, condition):
        angles = t.reshape(-1, 1) * self.frequencies.reshape(1, -1)
        time_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        hidden = self.input(torch.cat([x_t.flatten(1), time_emb, condition], dim=-1))
        velocity = self.output(self.body(hidden))
        if self.direct_skip:
            velocity = velocity + self.noise_gate(time_emb)*x_t.flatten(1) + self.condition_skip(condition)
        return velocity.reshape(-1, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def sample_euler(self, condition, steps=6, noise=None):
        if steps < 1:
            raise ValueError('Euler steps must be positive')
        x = torch.randn(condition.shape[0], self.chunk_size, self.action_dim, device=condition.device) if noise is None else noise.clone()
        for i in range(steps):
            t = torch.full((condition.shape[0],), i/steps, device=condition.device)
            x = x + self(x, t, condition)/steps
        return x
