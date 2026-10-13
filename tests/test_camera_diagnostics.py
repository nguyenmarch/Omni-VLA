import json
import struct
import zlib
import numpy as np
import pytest
from utils.camera_diagnostics import CameraDiagnostics,audit_frame,depth_at,png_bytes


def fixture_frame():
    return {'rgb':np.zeros((8,8,3),np.uint8),'depth':np.ones((8,8),np.float32),
            'K':np.eye(3,dtype=np.float32),'T':np.eye(4,dtype=np.float32)}


def test_depth_and_geometry_diagnostics_distinguish_center_from_surface():
    frame=fixture_frame();truth=np.array([[2.,2.,2.],[3.,3.,1.]])
    points=np.array([[1.,1.,1.],[3.,3.,1.]])
    metadata={'pixels':[[1,1],[3,3]],'center_depth_m':[1.,1.]}
    result=audit_frame(frame,truth,points,metadata,7,0,'approach')
    bottle,handle=result['targets']
    assert bottle['gt_uv']==[1.,1.] and bottle['pixel_error']==0
    assert bottle['gt_surface_depth_m']==1 and bottle['gt_center_camera_depth_m']==2
    assert bottle['world_error_m']==pytest.approx(3**.5)
    assert bottle['visibility_heuristic']=='possible_occlusion_or_center_surface_mismatch'
    assert handle['visibility_heuristic']=='depth_agreement_within_8cm'
    assert handle['gt_roundtrip_error_m']==0
    assert bottle['prediction_reprojection_error_px']==0
    assert depth_at(frame['depth'],[-1,2]) is None
    assert depth_at(frame['depth'],[2,2],-1) is None


def test_png_roundtrip_without_image_dependencies():
    rgb=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)
    png=png_bytes(rgb);assert png[:8]==b'\x89PNG\r\n\x1a\n'
    offset=8;compressed=b''
    while offset<len(png):
        length=struct.unpack('>I',png[offset:offset+4])[0];kind=png[offset+4:offset+8]
        data=png[offset+8:offset+8+length];offset+=12+length
        if kind==b'IDAT':compressed+=data
    rows=np.frombuffer(zlib.decompress(compressed),np.uint8).reshape(4,16)
    assert np.all(rows[:,0]==0)
    np.testing.assert_array_equal(rows[:,1:].reshape(4,5,3),rgb)


def test_saved_snapshot_is_self_contained_and_close_flushes_latest(tmp_path):
    frame=fixture_frame();points=np.array([[2.,2.,1.],[3.,3.,1.]])
    metadata={'pixels':[[2,2],[3,3]],'center_depth_m':[1.,1.]}
    monitor=CameraDiagnostics(tmp_path)
    for step in range(2):monitor.update(frame,points,points,metadata,7,step,'approach')
    monitor.close()
    saved=json.loads((tmp_path/'latest.json').read_text())
    assert saved['step']==1 and saved['targets'][0]['world_error_m']==0
    page=(tmp_path/'latest.html').read_text()
    assert 'data:image/png;base64,' in page and 'const live=false' in page
    assert '__INITIAL__' not in page
    assert len(json.loads((tmp_path/'snapshots.json').read_text()))==1
    np.testing.assert_array_equal(np.load(tmp_path/'latest_depth_m.npy'),frame['depth'])
