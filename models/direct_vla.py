"""Compact recurrent RGB-D/language action policy; no object or phase input."""
import re
import numpy as np
import torch
from torch import nn

TASK_PHRASES = (
    ('stow the bottle in the drawer then close it', 'put the bottle into the drawer and shut it',
     'store the bottle and close the drawer'),
    ('open the drawer', 'pull the drawer open', 'open the drawer without moving the bottle'),
)
PARAPHRASES = ('put the bottle in the drawer then shut it', 'pull open the drawer')
VOCABULARY = sorted({word for group in TASK_PHRASES for text in group for word in re.findall('[a-z]+', text)})
ROBOT_KEYS = ('q', 'dq', 'gripper', 'wrench', 'ee')


def tokenize(texts, vocabulary=VOCABULARY):
    lookup = {word: i+1 for i, word in enumerate(vocabulary)}
    result = torch.zeros(len(texts), 24, dtype=torch.long)
    for i, text in enumerate(texts):
        words = re.findall('[a-z]+', text.lower())
        if not words or len(words) > 24 or any(word not in lookup for word in words):
            raise ValueError('Instruction outside the trained vocabulary')
        result[i, :len(words)] = torch.tensor([lookup[word] for word in words])
    return result


def measured_image(frame):
    """RGB + metric depth + calibrated XYZ; all channels measured from the image."""
    rgb = np.asarray(frame['rgb'])
    depth = np.asarray(frame['depth'], dtype=np.float32)
    k, transform = np.asarray(frame['K']), np.asarray(frame['T'])
    if rgb.shape != (*depth.shape, 3) or k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError('RGB-D/calibration contract mismatch')
    if not np.isfinite(k).all() or not np.isfinite(transform).all():
        raise ValueError('Nonfinite calibration')
    valid = np.isfinite(depth) & (depth > .05) & (depth < 5)
    if valid.mean() < .02:
        raise ValueError('Insufficient measured depth support')
    z = np.where(valid, depth, 0).astype(np.float32)
    v, u = np.mgrid[:z.shape[0], :z.shape[1]]
    pixels = np.stack((u, v, np.ones_like(u)), -1)
    xyz = (pixels @ np.linalg.inv(k).T * z[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    xyz = np.where(valid[..., None], (xyz-[.4, 0, .8]).clip(-2, 2), 0)
    return np.concatenate([rgb.astype(np.float32)/255, z[..., None]/3, xyz], -1).astype(np.float32).transpose(2, 0, 1)


def proprioception(observation):
    value = np.concatenate([observation[key] for key in ROBOT_KEYS]).astype(np.float32)
    if value.shape != (50,) or not np.isfinite(value).all():
        raise ValueError('Finite robot-only state [50] required')
    return value


class DirectVLA(nn.Module):
    def __init__(self, width=128, chunk_size=5, vocabulary=VOCABULARY):
        super().__init__()
        self.chunk_size, self.vocabulary = chunk_size, list(vocabulary)
        self.encoder = nn.Sequential(nn.Conv2d(7, 24, 5, 2, 2), nn.SiLU(),
                                     nn.Conv2d(24, 48, 3, 2, 1), nn.SiLU(),
                                     nn.Conv2d(48, 64, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)),
                                     nn.Flatten(), nn.Linear(1024, width), nn.SiLU())
        self.language = nn.Embedding(len(vocabulary)+1, 48, padding_idx=0)
        self.text_memory = nn.GRU(48, 48, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(width+50+48, width), nn.LayerNorm(width), nn.SiLU())
        self.memory = nn.GRU(width, width, batch_first=True)
        self.action = nn.Linear(width, chunk_size*14)
        self.phase = nn.Linear(width, 10)  # supervision only; never an input
        self.landmarks = nn.Linear(width, 6)  # auxiliary RGB-D regression

    def forward(self, images, robot, tokens, history_mask):
        b, t = images.shape[:2]
        visual = self.encoder(images.flatten(0, 1)).reshape(b, t, -1)
        text = nn.utils.rnn.pack_padded_sequence(self.language(tokens), tokens.ne(0).sum(1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
        _, text_hidden = self.text_memory(text)
        language = text_hidden[-1]
        fused = self.fusion(torch.cat([visual, robot, language[:, None].expand(-1, t, -1)], -1))
        # Right-padded windows: only actual history frames enter recurrent state.
        lengths = history_mask.sum(1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(fused, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.memory(packed)
        hidden = hidden[-1]
        return {'action': self.action(hidden).tanh().reshape(b, self.chunk_size, 14),
                'phase_logits': self.phase(hidden), 'landmarks': self.landmarks(hidden)}
