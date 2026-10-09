import torch
from torch import nn
from models.flow_matching_policy import ResidualBlock

class BehaviorCloningPolicy(nn.Module):
    def __init__(self, action_dim=14, chunk_size=16, cond_dim=67, hidden_dim=256, phase_heads=False):
        super().__init__()
        self.action_dim, self.chunk_size = action_dim, chunk_size
        self.phase_heads=phase_heads
        if phase_heads:
            self.backbone=nn.Sequential(nn.Linear(cond_dim,hidden_dim),nn.SiLU(),ResidualBlock(hidden_dim))
            self.heads=nn.ModuleList([nn.Linear(hidden_dim,chunk_size*action_dim) for _ in range(10)])
            return
        self.net = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.SiLU(), ResidualBlock(hidden_dim), ResidualBlock(hidden_dim), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, chunk_size*action_dim))

    def forward(self, condition):
        if self.phase_heads:
            features=self.backbone(condition)
            outputs=torch.stack([head(features) for head in self.heads],dim=1)
            phase=condition[:,-10:].argmax(dim=1)
            selected=(outputs*torch.nn.functional.one_hot(phase,10)[:,:,None]).sum(dim=1)
            return selected.reshape(-1,self.chunk_size,self.action_dim)
        return self.net(condition).reshape(-1, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def sample_euler(self, condition, steps=1, noise=None):
        return self(condition)
