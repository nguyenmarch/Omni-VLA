import torch
import pytest
from utils.robust_depth import RobustDepthDecoder


def sample():
    heat=torch.zeros(1,9,9);heat[0,4,4]=10
    depth=torch.ones_like(heat)
    k=torch.tensor([[[100.,0.,4.],[0.,100.,4.],[0.,0.,1.]]])
    return heat,depth,k,torch.eye(4)[None]


def test_median_ignores_center_outlier_and_invalid_depth():
    heat,depth,k,t=sample();depth[0,4,4]=4.;depth[0,2,2]=float('nan')
    result=RobustDepthDecoder()(heat,depth,k,t)
    assert result['valid'].item() and result['surface_depth'].item()==1
    assert result['support'].item()==48
    torch.testing.assert_close(result['points'],torch.tensor([[0.,0.,1.]]))


def test_no_single_pixel_fallback_and_no_clamping_into_validity():
    heat,depth,k,t=sample();depth.fill_(float('nan'));depth[0,4,4]=1.
    result=RobustDepthDecoder()(heat,depth,k,t)
    assert not result['valid'].item() and not result['sufficient_support'].item()
    depth.fill_(2.);result=RobustDepthDecoder()(heat,depth,k,t)
    assert not result['valid'].item() and not result['in_workspace'].item()
    assert result['points'][0,2]==2 and result['clamped_points'][0,2]==1.5


def test_mixed_surface_and_all_robot_mask_rejected():
    heat,depth,k,t=sample();depth[:,:,5:]=2.
    assert not RobustDepthDecoder()(heat,depth,k,t)['stable_depth'].item()
    depth.fill_(1.)
    result=RobustDepthDecoder()(heat,depth,k,t,excluded=torch.ones_like(depth,dtype=torch.bool))
    assert not result['valid'].item() and not result['has_peak'].item()


def test_border_support_and_low_quantile_are_batched():
    heat,depth,k,t=sample();heat.zero_();heat[0,0,0]=10
    result=RobustDepthDecoder(quantile=.2)(heat,depth,k,t)
    assert result['support'].item()==16 and result['valid'].item()
    with pytest.raises(ValueError):RobustDepthDecoder(patch_size=1)
    with pytest.raises(ValueError,match='calibration'):
        RobustDepthDecoder()(heat,depth,torch.zeros_like(k),t)
