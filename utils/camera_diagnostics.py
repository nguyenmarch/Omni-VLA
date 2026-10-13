"""Camera/label audit and browser monitor; ground truth is display-only."""
import base64
import json
from pathlib import Path
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zlib
import numpy as np
from utils.camera import project, unproject


def png_bytes(rgb):
    rgb=np.asarray(rgb,dtype=np.uint8)
    height,width,channels=rgb.shape
    if channels!=3:raise ValueError('Expected RGB image')
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    pixels=b''.join(b'\x00'+row.tobytes() for row in rgb)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b''))


def depth_at(depth,uv,z=1):
    if not np.isfinite(uv).all() or z<=0:return None
    h,w=depth.shape
    if not (0<=uv[0]<w and 0<=uv[1]<h):return None
    u,v=np.rint(uv).astype(int);u=min(u,w-1);v=min(v,h-1)
    value=float(depth[v,u])
    return value if np.isfinite(value) and value>0 else None


def audit_frame(frame,truth,points,metadata,seed,step,phase):
    """Compare world-center labels with predicted peak/depth; never feeds policy."""
    k,t=frame['K'],frame['T'];depth=frame['depth'];targets=[]
    for i,name in enumerate(('bottle','handle')):
        gt=np.asarray(truth[i]);prediction=np.asarray(points[i])
        uv,z=project(gt,k,t);pred_uv=np.asarray(metadata['pixels'][i]);pred_z=float(metadata['center_depth_m'][i])
        surface_gt=depth_at(depth,uv,z);surface_pred=depth_at(depth,pred_uv,pred_z)
        projected_pred,projected_z=project(prediction,k,t)
        visible=surface_gt is not None and abs(float(z)-surface_gt)<=.08
        if surface_gt is None:visibility='out_of_view_or_invalid_depth'
        elif visible:visibility='depth_agreement_within_8cm'
        elif float(z)>surface_gt:visibility='possible_occlusion_or_center_surface_mismatch'
        else:visibility='label_in_front_of_surface_check_calibration_or_geometry'
        targets.append({'name':name,'gt_world_xyz_m':gt.tolist(),'pred_world_xyz_m':prediction.tolist(),
            'gt_uv':uv.tolist() if np.isfinite(uv).all() else None,'pred_uv':pred_uv.tolist(),
            'gt_center_camera_depth_m':float(z),'gt_surface_depth_m':surface_gt,
            'pred_surface_depth_m':surface_pred,'pred_center_camera_depth_m':pred_z,
            'pred_depth_offset_m':pred_z-surface_pred if surface_pred is not None else None,
            'center_depth_error_m':pred_z-float(z),
            'surface_depth_difference_m':surface_pred-surface_gt if surface_pred is not None and surface_gt is not None else None,
            'pixel_error':float(np.linalg.norm(pred_uv-uv)) if np.isfinite(uv).all() else None,
            'world_error_m':float(np.linalg.norm(prediction-gt)),
            'gt_roundtrip_error_m':float(np.linalg.norm(unproject(uv,z,k,t)-gt)) if np.isfinite(uv).all() else None,
            'prediction_reprojection_error_px':float(np.linalg.norm(projected_pred-pred_uv)),
            'prediction_reprojection_depth_error_m':float(projected_z-pred_z),
            'visibility_heuristic':visibility})
    return {'seed':int(seed),'step':int(step),'phase':phase,'camera':'head_cam',
            'width':int(depth.shape[1]),'height':int(depth.shape[0]),'K':np.asarray(k).tolist(),
            'T_world_from_camera':np.asarray(t).tolist(),'targets':targets,
            'notes':'GT markers are projected object centers, not segmentation or certified visible grasp points. Roundtrip checks mathematical consistency only, not independent calibration accuracy.'}


COLORS=((255,120,30),(20,220,240))


def overlay(rgb,targets):
    result=rgb.copy();h,w=result.shape[:2]
    for target,color in zip(targets,COLORS):
        for key,circle in [('gt_uv',True),('pred_uv',False)]:
            uv=target[key]
            if uv is None or not np.isfinite(uv).all() or not (0<=uv[0]<w and 0<=uv[1]<h):continue
            u,v=np.rint(uv).astype(int)
            for dy in range(-4,5):
                for dx in range(-4,5):
                    marked=9<=dx*dx+dy*dy<=20 if circle else abs(dx)==abs(dy)
                    if marked and 0<=u+dx<w and 0<=v+dy<h:result[v+dy,u+dx]=color
    return result


PAGE='''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Omni VLA — Camera diagnostics</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:24px}main{display:flex;gap:24px;flex-wrap:wrap}canvas{width:512px;max-width:90vw;image-rendering:pixelated;background:#000}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #475569;padding:8px;text-align:left}.note{color:#cbd5e1;max-width:1100px}button{padding:8px 16px}pre{white-space:pre-wrap}</style>
<h1>Head camera — chẩn đoán định vị</h1><p id="status">Waiting for frame…</p>
<button onclick="paused=!paused;this.textContent=paused?'Tiếp tục':'Tạm dừng'">Tạm dừng</button>
<p>Cam: head_cam · Chai: cam · Handle: cyan · ○ tọa độ đúng (GT) · × dự đoán</p>
<main><div>RGB<br><canvas id="rgb"></canvas></div><div>Depth: đen = 0.2m, trắng = 3.0m<br><canvas id="depth"></canvas></div></main>
<div id="metrics"></div><p class="note">GT chỉ dùng chẩn đoán. Điểm GT là tâm vật thể, có thể ở sau bề mặt hoặc bị che khuất. Depth agreement ≤8cm là heuristic, không chứng nhận visibility. Roundtrip gần zero chỉ xác nhận phép toán nhất quán; cần đối chiếu marker với ảnh để kiểm tra calibration/nhãn.</p>
<details><summary>Calibration K / T</summary><pre id="calibration"></pre></details>
<script>let paused=false; const initial=__INITIAL__; const live=__LIVE__;
function number(x,d=3){return x==null?'n/a':Number(x).toFixed(d)}
async function draw(id,url,frame){const image=new Image();image.src=url;await image.decode();const c=document.getElementById(id);c.width=frame.width;c.height=frame.height;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);frame.targets.forEach((t,i)=>{ctx.strokeStyle=i?'#14dcf0':'#ff781e';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=.8;ctx.font='5px sans-serif';[['gt_uv',true],['pred_uv',false]].forEach(([key,gt])=>{let p=t[key];if(!p||p[0]<0||p[0]>=c.width||p[1]<0||p[1]>=c.height)return;ctx.beginPath();if(gt)ctx.arc(p[0],p[1],3,0,Math.PI*2);else{ctx.moveTo(p[0]-3,p[1]-3);ctx.lineTo(p[0]+3,p[1]+3);ctx.moveTo(p[0]-3,p[1]+3);ctx.lineTo(p[0]+3,p[1]-3)}ctx.stroke();ctx.fillText(t.name+(gt?' GT':' pred'),p[0]+4,p[1]-4)})})}
async function show(f){if(!f)return;document.getElementById('status').textContent=`Seed ${f.seed} · Step ${f.step} · Phase ${f.phase}`;await Promise.all([draw('rgb',f.rgb_url,f),draw('depth',f.depth_url,f)]);let rows=f.targets.map(t=>`<tr><td>${t.name}</td><td>${number(t.world_error_m*100,2)}</td><td>${number(t.pixel_error,2)}</td><td>${number(t.gt_center_camera_depth_m)}</td><td>${number(t.gt_surface_depth_m)}</td><td>${number(t.pred_surface_depth_m)}</td><td>${number(t.pred_depth_offset_m)}</td><td>${number(t.pred_center_camera_depth_m)}</td><td>${number(t.center_depth_error_m)}</td><td>${t.visibility_heuristic}</td></tr>`).join('');document.getElementById('metrics').innerHTML='<table><tr><th>Target</th><th>XYZ error cm</th><th>Pixel error</th><th>GT center z m</th><th>Depth @ GT m</th><th>Depth @ pred m</th><th>Offset m</th><th>Pred center z m</th><th>z error m</th><th>Visibility heuristic</th></tr>'+rows+'</table>';document.getElementById('calibration').textContent=JSON.stringify({K:f.K,T:f.T_world_from_camera,checks:f.targets.map(t=>({target:t.name,roundtrip_m:t.gt_roundtrip_error_m,reprojection_px:t.prediction_reprojection_error_px}))},null,2)}
show(initial);async function poll(){try{if(!paused){let r=await fetch('/frame.json',{cache:'no-store'});if(!r.ok)throw Error(r.status);await show(await r.json())}}catch(e){document.getElementById('status').textContent+=' · Live disconnected; last frame retained'}finally{setTimeout(poll,250)}}if(live)poll();</script></html>'''


class CameraDiagnostics:
    def __init__(self,output,port=None):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.latest=None;self.latest_frame=None;self.latest_depth_rgb=None
        self.lock=threading.Lock();self.server=None;self.thread=None
        self.updates=0;self.saved_phases=set();self.snapshots=[]
        if port is not None:
            monitor=self
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_GET(self):
                    with monitor.lock:latest=monitor.latest
                    if self.path=='/frame.json':
                        content=json.dumps(latest,allow_nan=False).encode();kind='application/json'
                    elif self.path in ('/','/index.html'):
                        content=monitor.page(latest,True).encode();kind='text/html; charset=utf-8'
                    else:self.send_error(404);return
                    self.send_response(200);self.send_header('Content-Type',kind)
                    self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)
            self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
            self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
            print(f'Camera diagnostics: http://127.0.0.1:{self.server.server_port}',flush=True)

    @staticmethod
    def page(payload,live=False):
        return PAGE.replace('__INITIAL__',json.dumps(payload,allow_nan=False)).replace('__LIVE__','true' if live else 'false')

    def update(self,frame,truth,points,metadata,seed,step,phase,force_snapshot=False):
        payload=audit_frame(frame,truth,points,metadata,seed,step,phase)
        depth=np.asarray(frame['depth']);gray=(np.clip((np.nan_to_num(depth,nan=.2,posinf=3)-.2)/2.8,0,1)*255).astype(np.uint8)
        depth_rgb=np.repeat(gray[:,:,None],3,axis=2)
        for key,image in [('rgb',frame['rgb']),('depth',depth_rgb)]:
            payload[key+'_url']='data:image/png;base64,'+base64.b64encode(png_bytes(image)).decode()
        with self.lock:self.latest=payload
        self.latest_frame=frame;self.latest_depth_rgb=depth_rgb
        self.updates+=1
        key=(seed,phase,step) if force_snapshot else (seed,phase)
        if key not in self.saved_phases and len(self.snapshots)<24:
            self.saved_phases.add(key);prefix=f'snapshot_{len(self.snapshots):02d}'
            self._save(payload,frame,depth_rgb,prefix);self.snapshots.append({'file':prefix+'.html','seed':seed,'step':step,'phase':phase})
        if self.updates%5==1:self._save(payload,frame,depth_rgb,'latest')
        return payload

    def _save(self,payload,frame,depth_rgb,prefix):
        (self.output/(prefix+'.html')).write_text(self.page(payload))
        numeric={k:v for k,v in payload.items() if not k.endswith('_url')}
        (self.output/(prefix+'.json')).write_text(json.dumps(numeric,indent=2,allow_nan=False)+'\n')
        for key,image in [('rgb',frame['rgb']),('depth',depth_rgb)]:
            (self.output/(prefix+'_'+key+'.png')).write_bytes(png_bytes(overlay(image,payload['targets'])))
        np.save(self.output/(prefix+'_depth_m.npy'),frame['depth'])

    def close(self):
        (self.output/'snapshots.json').write_text(json.dumps(self.snapshots,indent=2)+'\n')
        if self.latest:self._save(self.latest,self.latest_frame,self.latest_depth_rgb,'latest')
        if self.server:self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
