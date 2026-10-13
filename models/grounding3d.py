"""Language-conditioned heatmap + learned visible-surface-to-center depth offset."""
import numpy as np
import torch
from torch import nn
from models.affordance_head import VisualAffordanceHead
from simulation.vision import QUERIES
from utils.robust_depth import RobustDepthDecoder


class Grounding3D(nn.Module):
    def __init__(self,img_size=128,feature_dim=64,latent_dim=64,decoder_config=None):
        super().__init__();self.img_size=img_size
        self.head=VisualAffordanceHead(img_size=img_size,feature_dim=feature_dim,latent_dim=latent_dim)
        self.center_offset=nn.Sequential(nn.Linear(latent_dim,32),nn.SiLU(),nn.Linear(32,1),nn.Tanh())
        self.robust_decoder=RobustDepthDecoder(**decoder_config) if decoder_config is not None else None

    def decode_detailed(self,heat,offset,depth,k,transform,excluded=None):
        if self.robust_decoder is None:raise ValueError('Require patch-trained model for robust decoding')
        return self.robust_decoder(heat,depth,k,transform,offset,excluded)

    def forward(self,rgb,ids):
        heat,latent,_,_=self.head.from_rgb(rgb,ids)
        return heat,self.center_offset(latent).squeeze(1)*.1

    def decode(self,heat,offset,depth,k,transform):
        if self.robust_decoder is not None:
            result=self.decode_detailed(heat,offset,depth,k,transform)
            return result['points'],result['pixels'],result['center_depth']
        peak=heat.flatten(1).argmax(1)
        pixels=torch.stack([peak%self.img_size,peak//self.img_size,torch.ones_like(peak)],1).float()
        z=depth.flatten(1).gather(1,peak[:,None]).squeeze(1)+offset
        camera=torch.linalg.solve(k,pixels[:,:,None]).squeeze(-1)*z[:,None]
        homogeneous=torch.cat([camera,torch.ones_like(z[:,None])],1)
        points=(transform@homogeneous[:,:,None]).squeeze(-1)[:,:3]
        return points,pixels[:,:2],z


class GroundingRuntime:
    def __init__(self,checkpoint):
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if c['schema_version'] not in ('omni-grounding-rgbd-v2','omni-grounding-patch-v3'):raise ValueError('Require supported grounding checkpoint')
        if c['schema_version']=='omni-grounding-patch-v3' and not c['model_config'].get('decoder_config'):
            raise ValueError('Patch checkpoint missing decoder contract')
        self.model=Grounding3D(**c['model_config']);self.model.load_state_dict(c['model']);self.model.eval()
        self.image_size=self.model.img_size;self.checkpoint=c

    @torch.no_grad()
    def predict(self,frame,ablation='none'):
        if ablation not in ('none','drop-image','swap-text'):raise ValueError(ablation)
        rgb=np.asarray(frame['rgb'])
        if rgb.shape!=(self.image_size,self.image_size,3):raise ValueError('Image size mismatch')
        if ablation=='drop-image':rgb=np.zeros_like(rgb)
        queries=QUERIES[::-1] if ablation=='swap-text' else QUERIES
        ids=self.model.head.language.tokenize(queries)
        image=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].expand(2,-1,-1,-1)/255
        heat,offset=self.model(image,ids)
        depth=torch.from_numpy(np.asarray(frame['depth']).copy()).float()[None].expand(2,-1,-1)
        k=torch.from_numpy(np.asarray(frame['K']).copy()).float()[None].expand(2,-1,-1)
        t=torch.from_numpy(np.asarray(frame['T']).copy()).float()[None].expand(2,-1,-1)
        diagnostics={}
        if self.model.robust_decoder is not None:
            result=self.model.decode_detailed(heat,offset,depth,k,t)
            if not result['valid'].all():
                raise ValueError('Invalid patch grounding: support/IQR/depth/workspace gate; no fallback')
            points,pixels,z=result['points'],result['pixels'],result['center_depth']
            diagnostics={key:result[key].numpy().tolist() for key in ('support','iqr','surface_depth','valid')}
        else:points,pixels,z=self.model.decode(heat,offset,depth,k,t)
        if not torch.isfinite(points).all() or not ((z>.05)&(z<5)).all():
            raise ValueError('Invalid visual depth estimate; no oracle fallback')
        return points.numpy(),{'pixels':pixels.numpy().tolist(),'center_depth_m':z.numpy().tolist(),**diagnostics}
