"""Small language-conditioned visual grounding model for finite-task experiments.

This is a trainable CNN/student, not a pretrained DINOv2 implementation.
Image tokens keep their spatial axis; ground-truth depth enables 3D decoding.
"""
import hashlib
import re
import torch
from torch import nn
from torch.nn import functional as F


class InstructionEncoder(nn.Module):
    def __init__(self, feature_dim=128, vocabulary=2048):
        super().__init__()
        self.vocabulary=vocabulary
        self.embedding=nn.Embedding(vocabulary,feature_dim,padding_idx=0)

    def tokenize(self,texts,max_length=24):
        ids=torch.zeros(len(texts),max_length,dtype=torch.long,device=self.embedding.weight.device)
        for i,text in enumerate(texts):
            words=re.findall(r'[a-z]+',text.lower())[:max_length]
            if not words:
                raise ValueError('Instruction must contain at least one supported word')
            for j,word in enumerate(words):
                ids[i,j]=1+int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%(self.vocabulary-1)
        return ids

    def forward(self,ids):
        return self.embedding(ids)


class VisualAffordanceHead(nn.Module):
    def __init__(self,feature_dim=128,latent_dim=128,n_heads=4,n_text_tokens=24,img_size=128,patch_size=16):
        super().__init__()
        self.feature_dim,self.latent_dim,self.img_size=feature_dim,latent_dim,img_size
        self.encoder=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),
                                   nn.Conv2d(64,feature_dim,3,2,1),nn.SiLU(),nn.Conv2d(feature_dim,feature_dim,3,2,1),nn.SiLU())
        self.language=InstructionEncoder(feature_dim)
        self.cross_attention=nn.MultiheadAttention(feature_dim,n_heads,batch_first=True)
        self.decoder=nn.Sequential(nn.Conv2d(feature_dim,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,1,1))
        self.latent_projector=nn.Sequential(nn.Linear(feature_dim,latent_dim),nn.LayerNorm(latent_dim))

    def encode_image(self,pixel_values):
        features=self.encoder(pixel_values)
        return features.flatten(2).transpose(1,2)

    def forward(self,image_tokens,text_tokens,camera_matrix=None,depth=None,world_from_camera=None,text_padding_mask=None):
        b,n,d=image_tokens.shape
        grid=int(n**.5)
        if grid*grid!=n:
            raise ValueError('Expected a square spatial token grid')
        fused,attention=self.cross_attention(image_tokens,text_tokens,text_tokens,key_padding_mask=text_padding_mask)
        fused=fused+image_tokens
        heatmap=F.interpolate(self.decoder(fused.transpose(1,2).reshape(b,d,grid,grid)),(self.img_size,self.img_size),mode='bilinear',align_corners=False)
        latent=self.latent_projector(fused.mean(1))
        points=None
        if camera_matrix is not None:
            if depth is None or world_from_camera is None:
                raise ValueError('3D grounding requires measured depth and camera extrinsics')
            peak=heatmap.flatten(1).argmax(1)
            u,v=peak%self.img_size,peak//self.img_size
            z=depth.reshape(b,-1).gather(1,peak[:,None])
            pixels=torch.stack([u,v,torch.ones_like(u)],dim=1).to(image_tokens.dtype)
            camera_points=torch.linalg.solve(camera_matrix,pixels[:,:,None]).squeeze(-1)*z
            homogeneous=torch.cat([camera_points,torch.ones(b,1,device=camera_points.device)],dim=1)
            points=(world_from_camera@homogeneous[:,:,None]).squeeze(-1)[:,:3]
        return heatmap,latent,points,attention

    def from_rgb(self,rgb,instruction_ids,**kwargs):
        return self(self.encode_image(rgb),self.language(instruction_ids),text_padding_mask=instruction_ids.eq(0),**kwargs)
