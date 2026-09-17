"""Shared machinery for the drawio figure re-render swaps.

extract <fig.xml> <outdir>    dump every embedded image + index.json
swap <fig.xml> <index.json>   apply prepared swaps: index entries with a
                              "new" path get their base64 replaced in place
align <old.png> <render.png> <out.jpg> [mode]  bbox-align render onto the
                              old panel canvas (mode: bbox|width|height)
"""
import re, os, sys, json, base64, io
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

IMG_RE = re.compile(r'image=data:image/(png|jpeg|jpg),([A-Za-z0-9+/=]+)')

def extract(xml_path, outdir):
    raw = open(xml_path).read()
    os.makedirs(outdir, exist_ok=True)
    # walk cells with ET for geometry, but keep raw b64 strings for swap keys
    root = ET.fromstring(raw)
    idx = []
    seen = {}
    for cell in root.iter('mxCell'):
        style = cell.get('style') or ''
        m = IMG_RE.search(style)
        if not m:
            continue
        b64 = m.group(2)
        geo = cell.find('mxGeometry')
        g = {k: float(geo.get(k, 0)) for k in ('x', 'y', 'width', 'height')} if geo is not None else {}
        key = b64[:24]
        n = len(idx)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        fn = f'{n:02d}_w{img.width}_h{img.height}.{m.group(1)}'
        img.save(os.path.join(outdir, fn))
        idx.append({'i': n, 'file': fn, 'fmt': m.group(1), 'b64head': key,
                    'b64len': len(b64), 'count': raw.count(b64),
                    'geo': g, 'imgsize': [img.width, img.height]})
    json.dump(idx, open(os.path.join(outdir, 'index.json'), 'w'), indent=1)
    print(f'{len(idx)} images -> {outdir}')

def find_b64(raw, head, length):
    for m in IMG_RE.finditer(raw):
        if m.group(2)[:24] == head and len(m.group(2)) == length:
            return m.group(2)
    raise KeyError(head)

def to_b64(path):
    img = Image.open(path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def swap(xml_path, index_path):
    raw = open(xml_path).read()
    idx = json.load(open(index_path))
    # group entries sharing one payload; entries are in document order, and
    # payload occurrences in raw follow the same order, so a shared payload
    # is replaced occurrence-by-occurrence with each entry's own new image
    from collections import defaultdict
    groups = defaultdict(list)
    for e in idx:
        if e.get('new'):
            groups[(e['b64head'], e['b64len'])].append(e)
    n = 0
    for (head, length), entries in groups.items():
        old = find_b64(raw, head, length)
        cnt = raw.count(old)
        assert cnt == entries[0]['count'] and len(entries) == cnt, (head, cnt, len(entries))
        for e in entries:
            nb64 = to_b64(e['new'])
            raw = raw.replace(old, nb64, 1)
            n += 1
    open(xml_path, 'w').write(raw)
    print(f'swapped {n} panels in {xml_path}')

def obj_bbox(im, thr=185):
    a = np.asarray(im.convert('RGB')).astype(int)
    m = a.min(axis=2) < thr
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

def align(old_path, new_path, out_path, mode='bbox', margin=3, shadow_thr=250):
    """Place the new render onto the old panel's canvas.

    The OBJECT (dark/colored pixels) is size-matched to the old panel's
    object bbox, but the pasted crop is the FULL non-white footprint —
    object plus the entire soft shadow (threshold `shadow_thr`, near
    white) — and the scale is reduced until that whole footprint fits
    inside the canvas. Shadows are therefore never clipped, neither by
    the crop nor by the panel border."""
    old = Image.open(old_path).convert('RGB')
    new = Image.open(new_path).convert('RGB')
    ob = obj_bbox(old)
    core = obj_bbox(new)                       # the object proper
    full = obj_bbox(new, thr=shadow_thr)       # object + entire soft shadow
    if full is None:
        full = core
    canvas = Image.new('RGB', old.size, 'white')
    W, H = old.size
    ox0, oy0, ox1, oy1 = ob
    cw, ch = core[2] - core[0], core[3] - core[1]
    ow, oh = ox1 - ox0, oy1 - oy0
    if mode == 'width':
        s = ow / cw
    elif mode == 'height':
        s = oh / ch
    else:
        s = min(ow / cw, oh / ch)
    fw, fh = full[2] - full[0], full[3] - full[1]
    s = min(s, (W - 2 * margin) / fw, (H - 2 * margin) / fh)
    crop = new.crop(full).resize((max(1, round(fw * s)), max(1, round(fh * s))), Image.LANCZOS)
    # position: object's core center lands on the old object center
    ccx = (core[0] + core[2]) / 2 - full[0]
    ccy = (core[1] + core[3]) / 2 - full[1]
    px = round((ox0 + ox1) / 2 - ccx * s)
    py = round((oy0 + oy1) / 2 - ccy * s)
    px = min(max(px, margin), W - margin - crop.width)
    py = min(max(py, margin), H - margin - crop.height)
    canvas.paste(crop, (px, py))
    canvas.save(out_path, quality=92)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'extract':
        extract(sys.argv[2], sys.argv[3])
    elif cmd == 'swap':
        swap(sys.argv[2], sys.argv[3])
    elif cmd == 'align':
        align(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else 'bbox')
