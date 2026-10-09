"""Compose two upstream Panda models into a portable MJCF scene."""
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as E

ASSETS = Path(__file__).resolve().parent / 'assets'

def build_scene():
    root = E.Element('mujoco', model='omni_drawer')
    E.SubElement(root, 'compiler', angle='radian', autolimits='true', meshdir='panda/assets')
    E.SubElement(root, 'option', timestep='0.002', integrator='implicitfast', gravity='0 0 -9.81')
    default = E.SubElement(root, 'default')
    E.SubElement(default, 'geom', friction='1.2 0.01 0.001', solref='0.008 1')
    assets = E.SubElement(root, 'asset')
    world = E.SubElement(root, 'worldbody')
    E.SubElement(world, 'light', pos='0.5 0 2', dir='0 0 -1')
    E.SubElement(world, 'geom', name='floor', type='plane', size='2 2 0.1', rgba='.15 .18 .2 1')
    E.SubElement(world, 'geom', name='table', type='box', pos='.4 0 .2', size='.8 .85 .2', rgba='.5 .4 .3 1')
    E.SubElement(world, 'camera', name='head_cam', pos='1.5 -1.4 1.5', xyaxes='.7 .7 0 -.4 .4 .8')
    actuators = E.SubElement(root, 'actuator')
    sensors = E.SubElement(root, 'sensor')
    contact = E.SubElement(root, 'contact')
    equality = E.SubElement(root, 'equality')
    upstream = E.parse(ASSETS / 'panda/collision.xml').getroot()
    for side, y in [('left', -.35), ('right', .35)]:
        model = deepcopy(upstream)
        names = {el.get('name') for el in model.iter() if el.get('name')}
        # Anonymous mesh names are inferred by MuJoCo from the file stem.
        for mesh in model.iter('mesh'):
            if not mesh.get('name'):
                mesh.set('name', Path(mesh.get('file')).stem)
                names.add(mesh.get('name'))
        classes = {el.get('class') for el in model.iter('default') if el.get('class')}
        for el in model.iter():
            for key, value in list(el.attrib.items()):
                if key in {'name', 'joint', 'joint1', 'joint2', 'body1', 'body2', 'tendon', 'material', 'mesh', 'site'} and value in names:
                    el.set(key, f'{side}_{value}')
                if key in {'class', 'childclass'} and value in classes:
                    el.set(key, f'{side}_{value}')
        for child in model.find('default'):
            default.append(child)
        for child in model.find('asset'):
            assets.append(child)
        base = model.find('worldbody/body')
        base.set('pos', f'0 {y} .4')
        # Sites are at center of fingertip grasp region.
        hand = next(el for el in base.iter('body') if el.get('name') == f'{side}_hand')
        E.SubElement(hand, 'site', name=f'{side}_ee', pos='0 0 .1034', size='.005', rgba='1 0 0 1')
        E.SubElement(hand, 'site', name=f'{side}_wrist', size='.002')
        if side == 'right':
            E.SubElement(hand, 'camera', name='right_wrist_cam', pos='.04 0 .03', xyaxes='0 -1 0 -1 0 -.55', fovy='65')
        for geom in base.iter('geom'):
            geom.set('rgba', '.75 .8 .9 1' if side == 'left' else '.9 .75 .6 1')
        world.append(base)
        for i in range(1, 8):
            limit = 87 if i <= 4 else 12
            E.SubElement(actuators, 'motor', name=f'{side}_motor{i}', joint=f'{side}_joint{i}', ctrlrange=f'-{limit} {limit}')
        for i in [1, 2]:
            E.SubElement(actuators, 'position', name=f'{side}_grip{i}', joint=f'{side}_finger_joint{i}', kp='500', kv='10', ctrlrange='0 .04', forcerange='-35 35')
        for child in model.find('equality'):
            equality.append(child)
        for child in model.find('contact'):
            contact.append(child)
        for tag in ['force', 'torque']:
            E.SubElement(sensors, tag, name=f'{side}_{tag}', site=f'{side}_wrist')
    drawer = E.SubElement(world, 'body', name='drawer', pos='.68 .10 .42')
    E.SubElement(drawer, 'joint', name='drawer_slide', type='slide', axis='-1 0 0', range='0 .22', damping='5', frictionloss='1')
    E.SubElement(drawer, 'geom', name='drawer_floor', type='box', size='.15 .11 .012', mass='.5', rgba='.3 .5 .65 1')
    for y in [-.115, .115]:
        E.SubElement(drawer, 'geom', type='box', pos=f'0 {y} .045', size='.15 .008 .045', mass='.1')
    E.SubElement(drawer, 'geom', type='box', pos='.15 0 .045', size='.008 .11 .045', mass='.1')
    E.SubElement(drawer, 'geom', type='box', pos='-.15 0 .035', size='.008 .11 .035', mass='.1')
    E.SubElement(drawer, 'geom', name='handle', type='box', pos='-.19 0 .09', size='.014 .012 .035', mass='.05', friction='2 .01 .001', rgba='.2 .2 .2 1')
    E.SubElement(drawer, 'site', name='handle_site', pos='-.19 0 .09', size='.005')
    E.SubElement(drawer, 'site', name='place_site', pos='0 0 .09', size='.005')
    bottle = E.SubElement(world, 'body', name='bottle', pos='.45 -.18 .462')
    E.SubElement(bottle, 'freejoint', name='bottle_free')
    E.SubElement(bottle, 'geom', name='bottle_geom', type='cylinder', size='.025 .06', mass='.12', friction='2 .01 .001', rgba='.2 .8 .5 1')
    E.SubElement(bottle, 'site', name='bottle_site', size='.005')
    E.indent(root)
    path = ASSETS / 'bottle_drawer_scene.xml'
    E.ElementTree(root).write(path, encoding='unicode')
    return path

if __name__ == '__main__':
    print(build_scene())
