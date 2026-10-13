"""Experimental batched patch-depth decoder; validity must gate action execution.

No single-pixel fallback. Clamped coordinates are diagnostics, not valid targets.
Center offsets must be trained/calibrated against this decoder before deployment.
"""
import math
import torch
from torch import nn


class RobustDepthDecoder(nn.Module):
    def __init__(self,patch_size=7,quantile=.5,min_support=6,min_fraction=.2,
                 max_iqr=.10,depth_range=(.05,5.),workspace=((-0.1,-.75,.4),(.95,.75,1.5))):
        super().__init__()
        if patch_size not in (5,7):raise ValueError('patch_size must be 5 or 7')
        if not 0<=quantile<=1 or not 0<min_fraction<=1:raise ValueError('Invalid quantile/support fraction')
        if not 1<=min_support<=patch_size**2 or not math.isfinite(max_iqr) or max_iqr<=0:
            raise ValueError('Invalid support/IQR configuration')
        if not 0<depth_range[0]<depth_range[1]:raise ValueError('Invalid depth range')
        self.patch_size,self.quantile,self.min_support=patch_size,quantile,min_support
        self.min_fraction,self.max_iqr,self.depth_range=min_fraction,max_iqr,depth_range
        lo=torch.tensor(workspace[0],dtype=torch.float32);hi=torch.tensor(workspace[1],dtype=torch.float32)
        if lo.shape!=(3,) or hi.shape!=(3,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (lo<hi).all():
            raise ValueError('Invalid workspace bounds')
        self.register_buffer('workspace_lower',lo);self.register_buffer('workspace_upper',hi)

    @staticmethod
    def _quantile(sorted_values,count,q):
        index=(count.clamp_min(1)-1).to(sorted_values.dtype)*q
        low=index.floor().long();high=index.ceil().long();weight=index-low
        a=sorted_values.gather(1,low[:,None]).squeeze(1)
        b=sorted_values.gather(1,high[:,None]).squeeze(1)
        # Avoid inf - inf for empty patches; empty patches are rejected below.
        return a*(1-weight)+b*weight

    def forward(self,heat,depth,k,world_from_camera,offset=None,excluded=None):
        if heat.ndim==4 and heat.shape[1]==1:heat=heat[:,0]
        if heat.ndim!=3 or depth.shape!=heat.shape:raise ValueError('Expected matching [B,H,W] heat/depth')
        batch,height,width=depth.shape
        if batch<1 or height<1 or width<1:raise ValueError('Empty input')
        if k.shape!=(batch,3,3) or world_from_camera.shape!=(batch,4,4):raise ValueError('Calibration shape mismatch')
        if not depth.is_floating_point() or not k.is_floating_point() or not world_from_camera.is_floating_point():
            raise ValueError('Depth/calibration must be floating point')
        if any(value.device!=depth.device for value in (heat,k,world_from_camera)):
            raise ValueError('Inputs must share a device')
        if not torch.isfinite(k).all() or not torch.isfinite(world_from_camera).all() or (torch.linalg.det(k).abs()<1e-8).any():
            raise ValueError('Invalid or singular calibration')
        if excluded is None:excluded=torch.zeros_like(depth,dtype=torch.bool)
        if excluded.shape!=depth.shape or excluded.dtype!=torch.bool:raise ValueError('Expected boolean exclusion mask')
        logits=heat.masked_fill(excluded|~torch.isfinite(heat),-torch.inf)
        has_peak=torch.isfinite(logits).flatten(1).any(1)
        peak=logits.flatten(1).argmax(1);u=peak%width;v=peak//width
        radius=self.patch_size//2
        axis=torch.arange(-radius,radius+1,device=depth.device)
        dy,dx=torch.meshgrid(axis,axis,indexing='ij')
        pu=u[:,None]+dx.flatten();pv=v[:,None]+dy.flatten()
        in_image=(pu>=0)&(pu<width)&(pv>=0)&(pv<height)
        indices=pv.clamp(0,height-1)*width+pu.clamp(0,width-1)
        values=depth.flatten(1).gather(1,indices)
        masked=excluded.flatten(1).gather(1,indices)
        usable=in_image&~masked&torch.isfinite(values)&(values>self.depth_range[0])&(values<self.depth_range[1])
        count=usable.sum(1);ordered=values.masked_fill(~usable,torch.inf).sort(1).values
        surface=self._quantile(ordered,count,self.quantile)
        iqr=self._quantile(ordered,count,.75)-self._quantile(ordered,count,.25)
        offset=torch.zeros_like(surface) if offset is None else offset
        if offset.shape!=(batch,):raise ValueError('Expected offset [B]')
        z=surface+offset
        pixels=torch.stack([u,v,torch.ones_like(u)],1).to(k.dtype)
        safe_z=torch.where(torch.isfinite(z),z,torch.zeros_like(z))
        camera=torch.linalg.solve(k,pixels[:,:,None]).squeeze(-1)*safe_z[:,None]
        homogeneous=torch.cat([camera,torch.ones_like(safe_z[:,None])],1)
        points=(world_from_camera@homogeneous[:,:,None]).squeeze(-1)[:,:3]
        lower=self.workspace_lower.to(points);upper=self.workspace_upper.to(points)
        in_workspace=torch.isfinite(points).all(1)&((points>=lower)&(points<=upper)).all(1)
        sufficient=(count>=self.min_support)&(count>=self.patch_size**2*self.min_fraction)
        stable=torch.isfinite(iqr)&(iqr<=self.max_iqr)
        depth_ok=torch.isfinite(z)&(z>self.depth_range[0])&(z<self.depth_range[1])
        calibration_ok=torch.isfinite(k).flatten(1).all(1)&torch.isfinite(world_from_camera).flatten(1).all(1)
        valid=has_peak&sufficient&stable&depth_ok&in_workspace&calibration_ok
        return {'points':points,'clamped_points':torch.maximum(lower,torch.minimum(upper,points)),
                'pixels':pixels[:,:2],'surface_depth':surface,'center_depth':z,'support':count,
                'iqr':iqr,'valid':valid,'in_workspace':in_workspace,'sufficient_support':sufficient,
                'stable_depth':stable,'has_peak':has_peak}
