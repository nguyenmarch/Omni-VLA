"""Download upstream Panda collision model and license (run once, network required)."""
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'simulation/assets/panda'
BASE = 'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda/'

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(BASE + 'panda.xml').read()
    (ROOT / 'upstream.xml').write_bytes(raw)
    root = ET.fromstring(raw)
    # Retain upstream kinematics, inertias and collision geometry; drop visual meshes.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'geom' and child.get('class') == 'visual':
                parent.remove(child)
    used = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    assets = root.find('asset')
    for mesh in list(assets.findall('mesh')):
        if mesh.get('name', Path(mesh.get('file')).stem) not in used:
            assets.remove(mesh)
            continue
        filename = mesh.get('file')
        dst = ROOT / 'assets' / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes(urllib.request.urlopen(BASE + 'assets/' + filename).read())
    (ROOT / 'LICENSE').write_bytes(urllib.request.urlopen(BASE + 'LICENSE').read())
    ET.ElementTree(root).write(ROOT / 'collision.xml')
    print('Downloaded', len(used), 'collision meshes to', ROOT)

if __name__ == '__main__':
    main()
