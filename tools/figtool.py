"""Shared machinery for the drawio figure re-render swaps.

extract <fig.xml> <outdir>    dump every embedded image + index.json
swap <fig.xml> <index.json>   apply prepared swaps: index entries with a
                              "new" path get their base64 replaced in place
align <old.png> <render.png> <out.jpg> [mode]  bbox-align render onto the
                              old panel canvas (mode: bbox|width|height)
"""
import re, os, sys, json, base64, io, hashlib
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
        n = len(idx)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        fn = f'{n:02d}_w{img.width}_h{img.height}.{m.group(1)}'
        img.save(os.path.join(outdir, fn))
        idx.append({'i': n, 'file': fn, 'fmt': m.group(1),
                    'b64md5': hashlib.md5(b64.encode()).hexdigest(),
                    'b64len': len(b64), 'count': raw.count(b64),
                    'geo': g, 'imgsize': [img.width, img.height]})
    json.dump(idx, open(os.path.join(outdir, 'index.json'), 'w'), indent=1)
    print(f'{len(idx)} images -> {outdir}')

def find_b64(raw, md5, length):
    import hashlib as _h
    for m in IMG_RE.finditer(raw):
        if len(m.group(2)) == length and _h.md5(m.group(2).encode()).hexdigest() == md5:
            return m.group(2)
    raise KeyError(md5)

def to_b64(path):
    img = Image.open(path)
    buf = io.BytesIO()
    if img.mode == 'RGBA':
        img.save(buf, 'PNG', optimize=True)
    else:
        img.convert('RGB').save(buf, 'JPEG', quality=90)
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
            groups[(e['b64md5'], e['b64len'])].append(e)
    n = 0
    for (head, length), entries in groups.items():
        old = find_b64(raw, head, length)  # head is the md5 now
        cnt = raw.count(old)
        assert cnt == entries[0]['count'] and len(entries) == cnt, (head, cnt, len(entries))
        for e in entries:
            nb64 = to_b64(e['new'])
            raw = raw.replace(old, nb64, 1)
            n += 1
    open(xml_path, 'w').write(raw)
    print(f'swapped {n} panels in {xml_path}')

def obj_bbox(im, thr=185, min_count=3):
    """Bounding box of non-background pixels, ignoring rows/columns with
    fewer than `min_count` qualifying pixels (stray denoiser dots)."""
    a = np.asarray(im.convert('RGB')).astype(int)
    m = a.min(axis=2) < thr
    rows = np.where(m.sum(axis=1) >= min_count)[0]
    cols = np.where(m.sum(axis=0) >= min_count)[0]
    if len(rows) == 0 or len(cols) == 0:
        ys, xs = np.where(m)
        if len(ys) == 0:
            return None
        return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return cols.min(), rows.min(), cols.max() + 1, rows.max() + 1

def fade_shadow(new, core, r0=0.30, r1=0.85):
    """Fade the cast shadow toward white with distance from the object's
    bbox, so long soft shadows taper out on their own instead of forcing
    the object smaller or getting cut by the panel border. Only grayish
    pixels outside the core bbox are touched."""
    a = np.asarray(new.convert('RGB')).astype(np.float32)
    H, W = a.shape[:2]
    x0, y0, x1, y1 = core
    size = max(x1 - x0, y1 - y0)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = np.maximum(np.maximum(x0 - xs, xs - x1), 0)
    dy = np.maximum(np.maximum(y0 - ys, ys - y1), 0)
    d = np.sqrt(dx * dx + dy * dy) / max(size, 1)
    w = np.clip((r1 - d) / max(r1 - r0, 1e-6), 0, 1)      # 1 near object, 0 far
    mn = a.min(axis=2); mx = a.max(axis=2)
    shadow = (mn < 250) & ((mx - mn) < 28) & (mn > 90)
    shadow[y0:y1, x0:x1] = False
    wmap = np.where(shadow, w, 1.0)[..., None]
    out = a * wmap + 255.0 * (1 - wmap)
    return Image.fromarray(out.astype(np.uint8))

RENDER_DIM = 1.0      # the webpage viewers run at bright 0.6; dimming a
                      # render by the same factor means scaling the light
                      # each object pixel reflects by 0.6 while the white
                      # background stays white

def apply_gamma(img, dim=None):
    dim = RENDER_DIM if dim is None else dim
    if abs(dim - 1.0) < 1e-3:
        return img
    a = np.asarray(img.convert('RGB')).astype(np.float32)
    w = (a.min(axis=2, keepdims=True) / 255.0) ** 6   # 1 at pure white bg
    s = dim + (1.0 - dim) * w
    return Image.fromarray(np.clip(a * s, 0, 255).astype(np.uint8))

def align(old_path, new_path, out_path, mode='bbox', margin=3, shadow_thr=250):
    """Place the new render onto the old panel's canvas.

    The OBJECT (dark/colored pixels) is size-matched to the old panel's
    object bbox, but the pasted crop is the FULL non-white footprint —
    object plus the entire soft shadow (threshold `shadow_thr`, near
    white) — and the scale is reduced until that whole footprint fits
    inside the canvas. Shadows are therefore never clipped, neither by
    the crop nor by the panel border."""
    old = Image.open(old_path).convert('RGB')
    new = apply_gamma(Image.open(new_path).convert('RGB'))
    ob = obj_bbox(old)
    core = obj_bbox(new)                       # the object proper
    new = fade_shadow(new, core)               # taper long shadows so the
    full = obj_bbox(new, thr=shadow_thr)       # object keeps full size
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
    # border fade: any grayish shadow pixel near a canvas edge ramps to
    # white so nothing ever ends in a hard line at the border
    a = np.asarray(canvas).astype(np.float32)
    Hc, Wc = a.shape[:2]
    band = max(6, int(0.06 * min(Wc, Hc)))
    ys, xs = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    dedge = np.minimum(np.minimum(xs, Wc - 1 - xs), np.minimum(ys, Hc - 1 - ys))
    w = np.clip(dedge / band, 0, 1)
    mn = a.min(axis=2); mx = a.max(axis=2)
    sh = (mn < 250) & ((mx - mn) < 28) & (mn > 90)
    wmap = np.where(sh, w, 1.0)[..., None]
    canvas = Image.fromarray((a * wmap + 255.0 * (1 - wmap)).astype(np.uint8))
    canvas.save(out_path, quality=92)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'extract':
        extract(sys.argv[2], sys.argv[3])
    elif cmd == 'swap':
        swap(sys.argv[2], sys.argv[3])
    elif cmd == 'align':
        align(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else 'bbox')
