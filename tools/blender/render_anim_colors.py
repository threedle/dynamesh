"""render_anim_colors.py — fixed-camera animated-color renders through Itai's
scene, for the DynaMesh webpage clips.

The webpage clips have a STATIC mesh whose vertex colors change every frame,
while render_single_mesh.py renders a static ply doing a camera turn. This
driver loads frame 1 of an export directory (adapted_f0001.ply ...), builds
the identical scene (blank.blend, "Default OBJ" material with the Col
attribute into a Principled BSDF, SUN light, floor shadow, white alpha-over),
then per frame swaps the Col data from that frame's ply and renders a still.

Camera: --az/--el degrees in the DynaMesh pipeline convention (Z-up object;
az 0 / el 0 is the training/video view looking along +Y), positioned at
--dist from the look point (the object's origin after it is dropped onto
the floor), matching Blender.look_at.

Run:
  blender blank.blend --background --python render_anim_colors.py -- \
      PLY_DIR OUT_DIR --pattern adapted_f{:04d}.ply --frames 1:150 \
      --az 0 --el 0 --dist 3.2 --scale 1.55 --resolution 768 768
"""
import argparse
import math
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Vector, Quaternion

argv = sys.argv[sys.argv.index('--') + 1:]
ap = argparse.ArgumentParser()
ap.add_argument('ply_dir')
ap.add_argument('out_dir')
ap.add_argument('--pattern', default='adapted_f{:04d}.ply')
ap.add_argument('--frames', default='1:150', help='START:END inclusive, or comma list')
ap.add_argument('--az', type=float, default=0.0)
ap.add_argument('--el', type=float, default=0.0)
ap.add_argument('--dist', type=float, default=3.2)
ap.add_argument('--scale', type=float, default=1.55)
ap.add_argument('--rotz', type=float, default=0.0, help='extra object yaw, degrees')
ap.add_argument('--rotx', type=float, default=0.0, help='extra object pitch, degrees (uprights Y-up-native meshes)')
ap.add_argument('--roll', type=float, default=0.0, help='camera roll about the view axis, degrees (positive rotates the image clockwise)')
ap.add_argument('--lift', type=float, default=0.0, help='raise the mesh above the floor by this much after any drop (loosens the shadow contact)')
ap.add_argument('--obj-roll', type=float, default=0.0, help='rotate the OBJECT about the camera view axis instead of the camera, then re-drop it to the floor: gives the rolled composition while the floor and its shadow stay level')
ap.add_argument('--resolution', nargs=2, type=int, default=[768, 768])
ap.add_argument('--samples', type=int, default=50)
ap.add_argument('--light-angle', type=float, default=0.1745329201221466)
ap.add_argument('--light-energy', type=float, default=2.0)
ap.add_argument('--sun-tilt', type=float, default=0.0, help='lean the sun off vertical (degrees); shadow escapes sideways for top-down cameras')
ap.add_argument('--sun-rotz', type=float, default=0.0, help='horizontal direction of the sun lean')
ap.add_argument('--spin', type=float, default=0.0,
                help='degrees of object yaw over the whole clip (rotating views)')
A = ap.parse_args(argv)

if ':' in A.frames:
    lo, hi = A.frames.split(':')
    FRAMES = list(range(int(lo), int(hi) + 1))
else:
    FRAMES = [int(x) for x in A.frames.split(',')]
N_TOTAL = len(FRAMES)


def read_ply_colors(path):
    """vertex uchar rgb of a binary_little_endian ply; the vertex layout is
    read from the header (trimesh writes xyz+rgba, the MeshNCA exporter
    xyz+rgb — a fixed dtype misreads the latter into confetti)."""
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.readline()
        nv = int(re.search(rb'element vertex (\d+)', header).group(1))
        vert_header = header.split(b'element face')[0]
        fields = []
        for typ, name in re.findall(rb'property (\w+) (\w+)', vert_header):
            t = {b'float': '<f4', b'double': '<f8', b'uchar': 'u1', b'int': '<i4'}[typ]
            fields.append((name.decode(), t))
        dt = np.dtype(fields)
        v = np.frombuffer(f.read(nv * dt.itemsize), dtype=dt)
    return np.stack([v['red'] if 'red' in v.dtype.names else v['r'],
                     v['green'] if 'green' in v.dtype.names else v['g'],
                     v['blue'] if 'blue' in v.dtype.names else v['b']], 1)


def srgb_to_linear(c):
    c = c.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ---- import frame 1 ------------------------------------------------------
first = os.path.join(A.ply_dir, A.pattern.format(FRAMES[0]))
bpy.ops.import_mesh.ply(filepath=first)
mesh = bpy.context.selected_objects[0]

# ---- mesh settings, as render_single_mesh.mesh_settings ------------------
bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = mesh
mesh.select_set(True)
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
# normalize to unit sphere so one --scale frames every object the same way
vs = np.array([v.co for v in mesh.data.vertices])
radius = np.linalg.norm(vs - vs.mean(0), axis=1).max()
mesh.scale = mesh.scale * (A.scale / radius)
mesh.rotation_euler.x = math.radians(A.rotx)
mesh.rotation_euler.z = math.radians(A.rotz)
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
mesh.data.update()
bpy.context.view_layer.update()
vertices = np.array([(mesh.matrix_world @ v.co) for v in mesh.data.vertices])
mesh.location = Vector((0.0, 0.0, mesh.location.z - vertices[:, 2].min()))

# material: Col attribute into the blend's Principled BSDF (import_colors)
mesh.data.materials.clear()
mat = bpy.data.materials['Default OBJ']
mesh.data.materials.append(mat)
mesh.active_material = mat
tree = mat.node_tree
attr = tree.nodes.new('ShaderNodeAttribute')
attr.attribute_name = 'Col'
tree.links.new(attr.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.8
tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.3
tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.1
mesh.data.use_auto_smooth = 1
bpy.ops.object.shade_smooth()

# ---- render settings, as render_single_mesh.render_settings --------------
scene = bpy.data.scenes['Scene']
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
try:
    import subprocess
    subprocess.check_output('nvidia-smi')
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for d in bpy.context.preferences.addons['cycles'].preferences.devices:
        d['use'] = 1
except Exception:
    scene.cycles.device = 'CPU'
scene.cycles.use_denoising = True
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
scene.cycles.samples = A.samples
scene.render.resolution_x, scene.render.resolution_y = A.resolution
scene.render.resolution_percentage = 100
scene.use_nodes = True

# camera at pipeline (az, el) about the look point; the pipeline's yaw 0
# looks from +Y (calibrated against the hand clips, sweep 2330430)
look = Vector((0.0, 0.0, mesh.location.z))
az, el = math.radians(A.az), math.radians(A.el)
cam = scene.objects['Camera']
cam.location = look + Vector((-A.dist * math.cos(el) * math.sin(az),
                              A.dist * math.cos(el) * math.cos(az),
                              A.dist * math.sin(el)))
direction = look - cam.location
if float(os.environ.get('OBJ_PITCH', 0) or 0):
    # rotate the OBJECT about the camera-right horizontal axis (screen X),
    # then re-drop: the object tips toward/away from this exact camera with
    # no apparent yaw or elevation change. Positive tips the top AWAY.
    from mathutils import Matrix
    _axis = direction.normalized().cross(Vector((0.0, 0.0, 1.0))).normalized()
    _pitch = math.radians(float(os.environ['OBJ_PITCH']))
    _center = mesh.matrix_world @ (0.125 * sum((Vector(c) for c in mesh.bound_box), Vector()))
    _R = (Matrix.Translation(_center) @ Matrix.Rotation(_pitch, 4, _axis)
          @ Matrix.Translation(-_center))
    mesh.matrix_world = _R @ mesh.matrix_world
    bpy.context.view_layer.update()
    import numpy as _np
    _n = len(mesh.data.vertices)
    _co = _np.empty(_n * 3, dtype=_np.float64)
    mesh.data.vertices.foreach_get('co', _co)
    _co = _co.reshape(-1, 3)
    _mw = _np.array(mesh.matrix_world)
    zmin = float((_co @ _mw[:3, :3].T + _mw[:3, 3]).min(axis=0)[2])
    mesh.location.z -= zmin
    bpy.context.view_layer.update()
    look = Vector((0.0, 0.0, (mesh.matrix_world @ (0.125 * sum((Vector(c) for c in mesh.bound_box), Vector()))).z))
    cam.location = look + Vector((-A.dist * math.cos(el) * math.sin(az),
                                  A.dist * math.cos(el) * math.cos(az),
                                  A.dist * math.sin(el)))
    direction = look - cam.location
if A.obj_roll:
    from mathutils import Matrix
    axis = direction.normalized()
    center = mesh.matrix_world @ (0.125 * sum((Vector(c) for c in mesh.bound_box), Vector()))
    R = (Matrix.Translation(center) @ Matrix.Rotation(math.radians(A.obj_roll), 4, axis)
         @ Matrix.Translation(-center))
    mesh.matrix_world = R @ mesh.matrix_world
    bpy.context.view_layer.update()
    # true vertex minimum: the transformed bounding-box corners overestimate
    # the extent under rotation and leave the mesh hovering above its shadow
    import numpy as _np
    _n = len(mesh.data.vertices)
    _co = _np.empty(_n * 3, dtype=_np.float64)
    mesh.data.vertices.foreach_get('co', _co)
    _co = _co.reshape(-1, 3)
    _mw = _np.array(mesh.matrix_world)
    zmin = float((_co @ _mw[:3, :3].T + _mw[:3, 3]).min(axis=0)[2])
    mesh.location.z -= zmin
    mesh.location.z += A.lift
    bpy.context.view_layer.update()
    look = Vector((0.0, 0.0, (mesh.matrix_world @ (0.125 * sum((Vector(c) for c in mesh.bound_box), Vector()))).z))
    cam.location = look + Vector((-A.dist * math.cos(el) * math.sin(az),
                                  A.dist * math.cos(el) * math.cos(az),
                                  A.dist * math.sin(el)))
    direction = look - cam.location
cam_quat = direction.to_track_quat('-Z', 'Y')
if A.roll:
    # roll about the view axis: rotating the camera by -roll about the
    # look direction turns the rendered image by +roll (clockwise)
    cam_quat = Quaternion(direction.normalized(), math.radians(-A.roll)) @ cam_quat
cam.rotation_euler = cam_quat.to_euler()

# lights (blank.blend path of render_single_mesh)
objs = [x.name for x in bpy.data.objects]
if 'light' in objs:
    L = bpy.data.objects['light']
    L.data.materials['Material'].node_tree.nodes['Emission'].inputs['Strength'].default_value = 1.8
    L.data.materials['Material'].node_tree.nodes['Emission'].inputs['Color'].default_value = [0.8, 0.77, 0.8, 1]
    L.scale = L.scale * 1.5
    L.location.x = 6
bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 5.24564))
bpy.data.objects['Sun'].data.color = (1.0, 0.95, 0.9626210331916809)
bpy.data.objects['Sun'].data.angle = A.light_angle
bpy.data.objects['Sun'].data.energy = A.light_energy
# optional shadow shaping via env: the side panel keeps lighting but stops
# casting (fill without streak), the sun grows softer, the world lifts fills
if os.environ.get('NO_SIDE_SHADOW') and 'light' in objs:
    bpy.data.objects['light'].visible_shadow = False
if os.environ.get('SUN_ANGLE'):
    bpy.data.objects['Sun'].data.angle = float(os.environ['SUN_ANGLE'])
if os.environ.get('SHADOW_ONLY'):
    # the mesh keeps casting but the camera cannot see it: the render is the
    # bare floor with the full cast shadow (nothing self-occluded), for
    # compositing a level shadow under an exact-silhouette 2D-rotated object
    mesh.visible_camera = False
if os.environ.get('SUN_TILT') or A.sun_tilt:
    # lean the sun off vertical so the shadow escapes sideways — needed for
    # top-down cameras where the object hides its own straight-down shadow.
    # Shadow extends toward world (-sin rotz, cos rotz).
    if A.sun_tilt:
        _t, _rz = A.sun_tilt, A.sun_rotz
    else:
        _t, _rz = (float(x) for x in os.environ['SUN_TILT'].split())
    for _o in bpy.data.objects:
        if _o.type == 'LIGHT':
            print('[LIGHTS]', _o.name, _o.data.type, _o.data.energy,
                  [c.type for c in _o.constraints], flush=True)
    for _o in bpy.data.objects:
        if _o.type == 'LIGHT' and _o.data.type == 'SUN':
            for _c in _o.constraints:
                _c.mute = True
            _o.rotation_euler = (math.radians(_t), 0.0, math.radians(_rz))
    print('[SUNTILT]', _t, _rz, flush=True)
if os.environ.get('CAM_FILL'):
    # a non-casting fill aligned with the camera: brightens exactly what
    # this view sees (for cameras that face the sun-shaded undersides)
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 3))
    cf = bpy.context.active_object
    cf.data.energy = float(os.environ['CAM_FILL'])
    cf.data.angle = 0.6
    cf.visible_shadow = False
    cf.rotation_euler = cam.rotation_euler
if os.environ.get('WORLD_FILL') and bpy.context.scene.world and bpy.context.scene.world.node_tree:
    for n in bpy.context.scene.world.node_tree.nodes:
        if n.type == 'BACKGROUND':
            n.inputs['Strength'].default_value = float(os.environ['WORLD_FILL'])

# below-horizon cameras cannot see past the floor: NO_FLOOR hides every
# scene mesh except the object and the emissive panel (no ground, no shadow)
if os.environ.get('NO_FLOOR'):
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name != mesh.name and o.name != 'light':
            o.hide_render = True

# white background through the compositor's Alpha Over
bpy.data.scenes['Scene'].node_tree.nodes['Alpha Over'].inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)

# ---- per-frame color update ----------------------------------------------
col_layer = mesh.data.vertex_colors['Col']
loop_vert = np.zeros(len(mesh.data.loops), dtype=np.int64)
mesh.data.loops.foreach_get('vertex_index', loop_vert)

# decide the byte->float convention by matching what the importer stored
cur = np.zeros(len(mesh.data.loops) * 4, dtype=np.float32)
col_layer.data.foreach_get('color', cur)
cur = cur.reshape(-1, 4)[:, :3]
f0 = read_ply_colors(first)
raw_err = np.abs(cur - (f0[loop_vert] / 255.0)).max()
lin_err = np.abs(cur - srgb_to_linear(f0)[loop_vert]).max()
to_float = (lambda c: c / 255.0) if raw_err <= lin_err else srgb_to_linear
print(f'[COLORS] importer convention: {"raw" if raw_err <= lin_err else "srgb->linear"} '
      f'(raw_err={raw_err:.4f} lin_err={lin_err:.4f})', flush=True)

# BAKE_SHADOW=1: instead of animating colors, render two orthographic
# top-down stills of the floor patch [-2,2]^2 — one with the object
# casting (hidden from camera) and one with the object gone — whose
# ratio is the object's exact Cycles floor shadow, for the live viewers.
if os.environ.get('BAKE_SHADOW'):
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 4.0
    cam.location = Vector((0.0, 0.0, 4.5))
    cam.rotation_euler = (0.0, 0.0, 0.0)
    os.makedirs(A.out_dir, exist_ok=True)
    mesh.visible_camera = False
    scene.render.filepath = os.path.join(A.out_dir, 'shadow.png')
    bpy.ops.render.render(write_still=True)
    mesh.hide_render = True
    scene.render.filepath = os.path.join(A.out_dir, 'empty.png')
    bpy.ops.render.render(write_still=True)
    print('[BAKED]', flush=True)
    import sys as _s; _s.exit(0)

os.makedirs(A.out_dir, exist_ok=True)
rgba = np.ones((len(loop_vert), 4), dtype=np.float32)
for i, fr in enumerate(FRAMES):
    cols = read_ply_colors(os.path.join(A.ply_dir, A.pattern.format(fr)))
    rgba[:, :3] = to_float(cols)[loop_vert]
    col_layer.data.foreach_set('color', rgba.ravel())
    mesh.data.update()
    if A.spin:
        mesh.rotation_euler.z = math.radians(A.spin) * i / max(1, N_TOTAL)
    scene.render.filepath = os.path.join(A.out_dir, f'{fr:04d}.png')
    bpy.ops.render.render(write_still=True)
    print(f'[FRAME] {fr}', flush=True)
print('[DONE]', flush=True)
