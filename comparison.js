// DynaMesh interactive 3D grids: auto-rotating, fully interactive cells for
// the teaser (1 row of 8) and the comparison (3 methods x 4 timesteps),
// drawn by ONE WebGL renderer per grid container.
//
// Browsers cap the number of live WebGL contexts well below the ~32 cells on
// this page, so instead of one <model-viewer> per cell (as in the BSB page,
// which has only 6), each container gets a single transparent canvas and each
// cell's scene is rendered into its on-screen rectangle with a scissor test —
// the standard three.js "multiple elements" technique. Each cell still gets
// its own OrbitControls bound to its own div, so zoom / rotate / pan work per
// cell exactly like model-viewer's camera-controls.
//
// Containers: any element with class "dm-3d". Inside, each ".dm-grid" carries
//   data-method  asset prefix: assets/comparison/<method>_t<K>.glb
//   data-nt      cells in the row (default 4)
//   data-az/el   initial camera azimuth/elevation in degrees (default 45/25)
// Cells fall back to the untextured spot mesh when a GLB is missing.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

const ROTATE_DEG_PER_SEC = 20; // full turn in 18s; calmer than BSB's 60deg/s with this many cells
const ASSET_V = '9'; // bump when GLBs are re-exported, so cached copies don't linger

const glbCache = new Map(); // url -> Promise<scene template>; rows sharing a file share the load

// the GLBs are Draco-compressed (36 MB -> 3 MB); one shared loader + decoder
const _draco = new DRACOLoader();
_draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
const _gltf = new GLTFLoader();
_gltf.setDRACOLoader(_draco);

// one loader pass for the placeholder geometry, shared by every cell
const placeholderGeom = new Promise((resolve) => {
  new OBJLoader().load('assets/comparison/spot_render_frame.obj', (obj) => {
    let geom = null;
    obj.traverse((c) => { if (c.isMesh && !geom) geom = c.geometry; });
    geom.computeBoundingBox();
    const bb = geom.boundingBox;
    const center = bb.getCenter(new THREE.Vector3());
    const size = bb.getSize(new THREE.Vector3()).length();
    geom.translate(-center.x, -center.y, -center.z);
    geom.scale(1.35 / size, 1.35 / size, 1.35 / size);
    geom.computeVertexNormals();
    resolve(geom);
  }, undefined, () => resolve(new THREE.TorusKnotGeometry(0.35, 0.12, 120, 16)));
});

function loadAsset(url) {
  if (!glbCache.has(url)) {
    glbCache.set(url, new Promise((resolve, reject) => {
      _gltf.load(url, (gltf) => {
        const root = gltf.scene;
        root.traverse((child) => {
          if (!child.isMesh) return;
          // Our baked colors are sRGB values, but glTF defines COLOR_0 as
          // linear — left alone they render dark and muddy. Re-encode them to
          // linear so the renderer's output pass displays the original values.
          const col = child.geometry.getAttribute('color');
          if (col) {
            const c = new THREE.Color();
            for (let i = 0; i < col.count; i++) {
              c.setRGB(col.getX(i), col.getY(i), col.getZ(i)).convertSRGBToLinear();
              col.setXYZ(i, c.r, c.g, c.b);
            }
            col.needsUpdate = true;
          }
          // Unlit on purpose: the paper's renders sample the field directly
          // with no lighting, so a lit material reads darker than the figure.
          child.material = new THREE.MeshBasicMaterial({ vertexColors: !!col });
        });
        const box = new THREE.Box3().setFromObject(root);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        root.position.sub(center);
        root.scale.setScalar(1.35 / size);
        resolve(root);
      }, undefined, reject);
    }));
  }
  return glbCache.get(url);
}

function setupContainer(container) {
  // The canvas lives INSIDE the container and scrolls with it. A
  // viewport-fixed canvas redraws one rAF behind the scroll, so on a hard
  // scroll its stale frame slides into the surrounding text; an absolutely
  // positioned canvas moves with the page, and nothing can ever be drawn
  // outside the container's bounds.
  const canvas = document.createElement('canvas');
  canvas.className = 'dm-canvas';
  // belt and braces: set the critical styles inline so a stale cached
  // stylesheet can never leave the canvas in-flow (an in-flow canvas feeds
  // its own buffer size back into the container height, growing without bound)
  Object.assign(canvas.style, {
    position: 'absolute', inset: '0', width: '100%', height: '100%',
    pointerEvents: 'none', zIndex: '5',
  });
  container.style.position = 'relative';
  container.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0); // transparent — the page shows through
  renderer.setScissorTest(true);

  // a GPU reset (or mobile tab eviction) fires contextlost; without
  // preventDefault the context is never restored and every cell stays blank
  canvas.addEventListener('webglcontextlost', (e) => e.preventDefault());

  const cells = [];

  for (const grid of container.querySelectorAll('.dm-grid')) {
    const method = grid.dataset.method || grid.closest('.dm-block')?.dataset.method;
    const nT = Number(grid.dataset.nt || 4);
    const az = Number(grid.dataset.az ?? 45);
    const el = Number(grid.dataset.el ?? 25);
    const rx = Number(grid.dataset.rx ?? 0); // upright correction for meshes whose native up-axis differs
    for (let t = 1; t <= nT; t++) {
      const div = document.createElement('div');
      div.className = 'dm-cell';
      grid.appendChild(div);
      cells.push(makeCell(div, method, t, az, el, rx));
    }
  }

  function makeCell(div, method, t, azDeg, elDeg, rxDeg) {
    const scene = new THREE.Scene();
    // lights only matter for the gray placeholder; real cells are unlit
    scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d4cf, 1.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1.5, 3, 2);
    scene.add(key);

    const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
    const az = THREE.MathUtils.degToRad(azDeg);
    const el = THREE.MathUtils.degToRad(elDeg);
    const r = 2.4;
    camera.position.set(
      r * Math.cos(el) * Math.sin(az),
      r * Math.sin(el),
      r * Math.cos(el) * Math.cos(az),
    );

    const controls = new OrbitControls(camera, div);
    // OrbitControls forces touch-action:none on its element, which would eat
    // vertical swipes on phones and trap the page scroll inside the grid.
    // pan-y hands vertical swipes back to the browser; horizontal drags rotate.
    div.style.touchAction = 'pan-y';
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = ROTATE_DEG_PER_SEC / 6; // 2.0 == 30s/turn == 12deg/s
    controls.update();

    loadAsset(`assets/comparison/${method}_t${t}.glb?v=${ASSET_V}`).then(
      (root) => {
        const inst = root.clone();
        if (rxDeg) inst.rotation.x = THREE.MathUtils.degToRad(rxDeg);
        scene.add(inst);
      },
      () => {
        placeholderGeom.then((geom) => {
          const mat = new THREE.MeshStandardMaterial({ color: 0xb9b9b9, roughness: 0.85, metalness: 0.0 });
          const mesh = new THREE.Mesh(geom, mat);
          // the spot obj is Z-up; three.js is Y-up
          mesh.rotation.x = -Math.PI / 2;
          scene.add(mesh);
        });
      },
    );
    return { div, scene, camera, controls };
  }

  function render() {
    const contRect = container.getBoundingClientRect();
    const w = container.clientWidth;
    const h = container.clientHeight;
    // track browser-zoom / monitor changes so the buffer stays sharp
    const wantPR = Math.min(window.devicePixelRatio, 2);
    if (renderer.getPixelRatio() !== wantPR) renderer.setPixelRatio(wantPR);
    // compare with floor — setSize floors internally; rounding here would
    // mismatch forever at fractional zoom and reallocate the buffer every frame
    if (canvas.width !== Math.floor(w * wantPR) || canvas.height !== Math.floor(h * wantPR)) {
      renderer.setSize(w, h, false);
    }
    // skip work entirely while the whole block is off screen
    if (contRect.bottom < 0 || contRect.top > window.innerHeight) return;
    // wipe the whole canvas first, otherwise cells leave trails behind
    // when the layout changes (each render pass only clears its scissor rect)
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);

    for (const cell of cells) {
      const rect = cell.div.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
      const width = rect.right - rect.left;
      const height = rect.bottom - rect.top;
      // cell position relative to the container-bound canvas
      const left = rect.left - contRect.left;
      const bottom = h - (rect.bottom - contRect.top);

      cell.camera.aspect = width / height;
      cell.camera.updateProjectionMatrix();
      cell.controls.update();

      renderer.setViewport(left, bottom, width, height);
      renderer.setScissor(left, bottom, width, height);
      renderer.render(cell.scene, cell.camera);
    }
  }

  renderer.setAnimationLoop(render);
  return cells;
}

window.__dmCells = [];
for (const container of document.querySelectorAll('.dm-3d')) {
  window.__dmCells.push(...setupContainer(container));
}

// ---- time-synced teaser: the 3D result follows the reference video ------
// One mesh, constant geometry, per-frame vertex colors sampled from the
// sequence. Every displayed frame lerps between the two nearest sampled
// frames at the video's currentTime, so pausing or scrubbing the video
// freezes or scrubs the texture with it (FoldingAgent-style shared clock).
(async function initSyncTeaser() {
  const container = document.getElementById('teaser3d');
  const video = document.getElementById('teaserVideo');
  if (!container || !video) return;

  const startVideo = () => { try { video.play().catch(() => {}); } catch (e) {} };
  const bail = () => { container.textContent = ''; startVideo(); };

  // ---- streamed download -------------------------------------------------
  // The pack is laid out header, geometry, frame-0 colors, then delta
  // frames, so the mesh can appear and start as soon as the first ~0.7 MB
  // arrives; the remaining frames keep decoding in the background. If
  // playback ever outruns the download, the texture clamps to the newest
  // decoded frame and catches up silently.
  let resp;
  try {
    resp = await fetch(`assets/teaser_anim/plane_anim.bin?v=${ASSET_V}`);
    if (!resp.ok || !resp.body) throw new Error(resp.status);
  } catch (e) { bail(); return; }

  const rawReader = resp.body.getReader();
  const first = await rawReader.read().catch(() => ({}));
  if (!first.value) { bail(); return; }
  let stream = new ReadableStream({
    start(c) { c.enqueue(first.value); },
    async pull(c) {
      const r = await rawReader.read();
      if (r.done) c.close(); else c.enqueue(r.value);
    },
  });
  if (first.value[0] === 0x1f && first.value[1] === 0x8b) {
    // gzip on the wire (delta streams compress ~4x); ungzip incrementally
    stream = stream.pipeThrough(new DecompressionStream('gzip'));
  }
  const reader = stream.getReader();

  // sRGB -> linear, so the unlit material shows the field's own values
  const s2l = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    const l = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    s2l[i] = Math.round(l * 255);
  }

  let pre = new Uint8Array(0);          // bytes accumulated before the header parses
  let u8 = null, got = 0, total = 0;
  let meta = null, colOff = 0, n = 0;
  let colorsLin = null, absQ = null, lutQ = null, mask = 0;
  let framesReady = 0;
  let built = false;
  let setColorsAt = null;               // installed by buildScene
  let onNewFrames = null;

  function tryHeader() {
    if (meta || pre.length < 4) return;
    const headLen = new DataView(pre.buffer, pre.byteOffset, pre.length).getUint32(0, true);
    if (pre.length < 4 + headLen) return;
    meta = JSON.parse(new TextDecoder().decode(pre.subarray(4, 4 + headLen)));
    const { verts, tris, frames } = meta;
    const posOff = 4 + headLen;
    colOff = meta.v >= 3
      ? posOff + Math.ceil((verts * 6 + tris * 6) / 4) * 4
      : posOff + verts * 12 + Math.ceil((tris * 3 * 2) / 4) * 4;
    n = verts * 3;
    total = colOff + frames * n;
    u8 = new Uint8Array(total);
    const seed = pre.subarray(0, Math.min(pre.length, total));
    u8.set(seed);
    got = seed.length;
    pre = null;
    colorsLin = new Uint8Array(frames * n);
    const bits = meta.bits || 8;
    const shift = 8 - bits;
    mask = (1 << bits) - 1;
    lutQ = new Uint8Array(1 << bits);
    for (let i = 0; i < lutQ.length; i++) {
      lutQ[i] = meta.v >= 2 ? s2l[(i << shift) | (shift ? i >> (bits - shift) : 0)] : s2l[i];
    }
    absQ = new Uint8Array(n);
  }

  function decodeReadyFrames() {
    if (!u8) return;
    const { frames } = meta;
    while (framesReady < frames && got >= colOff + (framesReady + 1) * n) {
      const off = colOff + framesReady * n;
      const dst = framesReady * n;
      if (framesReady === 0 || meta.v < 2) {
        for (let i = 0; i < n; i++) { absQ[i] = u8[off + i] & mask; colorsLin[dst + i] = lutQ[absQ[i]]; }
      } else {
        for (let i = 0; i < n; i++) {
          absQ[i] = (absQ[i] + u8[off + i]) & mask;
          colorsLin[dst + i] = lutQ[absQ[i]];
        }
      }
      framesReady++;
    }
    if (onNewFrames) onNewFrames();
  }

  function buildScene() {
    built = true;
    const { verts, tris, frames, times } = meta;
    const posOff = 4 + new DataView(u8.buffer).getUint32(0, true);
    let positions;
    if (meta.v >= 3) {
      const posQ = new Uint16Array(u8.buffer, posOff, verts * 3);
      positions = new Float32Array(verts * 3);
      for (let i = 0; i < verts * 3; i++) {
        const k = i % 3;
        positions[i] = meta.pmin[k] + posQ[i] * meta.pscale[k];
      }
    } else {
      positions = new Float32Array(u8.buffer.slice(posOff, posOff + verts * 12));
    }
    const idxOff = posOff + (meta.v >= 3 ? verts * 6 : verts * 12);
    const indices = new Uint16Array(u8.buffer, idxOff, tris * 3);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const colAttr = new THREE.BufferAttribute(new Uint8Array(verts * 3), 3, true);
    colAttr.setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('color', colAttr);
    geom.setIndex(new THREE.BufferAttribute(indices, 1));
    geom.computeBoundingBox();
    const bb = geom.boundingBox;
    const center = bb.getCenter(new THREE.Vector3());
    const size = bb.getSize(new THREE.Vector3()).length();
    geom.translate(-center.x, -center.y, -center.z);
    geom.scale(1.5 / size, 1.5 / size, 1.5 / size);

    const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ vertexColors: true }));
    const scene = new THREE.Scene();
    scene.add(mesh);

    const canvas = document.createElement('canvas');
    Object.assign(canvas.style, {
      position: 'absolute', inset: '0', width: '100%', height: '100%',
      opacity: '0', transition: 'opacity 0.4s ease',
    });
    container.style.position = 'relative';
    container.appendChild(canvas);
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    canvas.addEventListener('webglcontextlost', (e) => e.preventDefault());

    const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
    // initial camera = the reference video's (supervised) viewpoint: the
    // pipeline's training camera is yaw 0 / elev 0, which lands on +Z
    // after the export's Z-up -> Y-up rotation
    const setView = (azDeg, elDeg, r = 2.6) => {
      const az = THREE.MathUtils.degToRad(azDeg), el = THREE.MathUtils.degToRad(elDeg);
      camera.position.set(r * Math.cos(el) * Math.sin(az), r * Math.sin(el), r * Math.cos(el) * Math.cos(az));
    };
    setView(0, 0);
    window.__syncView = setView;
    const controls = new OrbitControls(camera, container);
    container.style.touchAction = 'pan-y';
    container.style.cursor = 'grab';
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;

    // a plain click (no drag) on the shape toggles the shared clock
    let downX = 0, downY = 0, downT = 0;
    container.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; downT = performance.now(); });
    container.addEventListener('pointerup', (e) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) < 5 && performance.now() - downT < 400) {
        if (video.paused) video.play().catch(() => {}); else video.pause();
      }
    });

    let lastX = -1;
    setColorsAt = (tNorm) => {
      const maxF = framesReady - 1;
      if (maxF < 0) return;
      // bracket tNorm in the sampled times, clamped to what has decoded
      let b = 1;
      while (b <= maxF && times[b] < tNorm) b++;
      if (b > maxF) b = maxF;
      const a = Math.max(0, b - 1);
      const span = times[b] - times[a] || 1;
      const alpha = Math.min(1, Math.max(0, (tNorm - times[a]) / span));
      const x = a + alpha;
      if (Math.abs(x - lastX) < 0.01) return;
      lastX = x;
      const A = colorsLin.subarray(a * n, (a + 1) * n);
      const B = colorsLin.subarray(b * n, (b + 1) * n);
      const out = colAttr.array;
      for (let i = 0; i < out.length; i++) out[i] = A[i] + (B[i] - A[i]) * alpha;
      colAttr.needsUpdate = true;
    };
    onNewFrames = () => { lastX = -1; };   // a clamped time can now re-render
    setColorsAt(0);

    // ready: reveal the shape and start the shared clock from the beginning
    canvas.style.opacity = '1';
    try { video.currentTime = 0; } catch (e) {}
    startVideo();

    renderer.setAnimationLoop(() => {
      const w = container.clientWidth, h = container.clientHeight;
      if (!w || !h) return;
      const pr = Math.min(window.devicePixelRatio, 2);
      if (renderer.getPixelRatio() !== pr) renderer.setPixelRatio(pr);
      if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      const d = video.duration;
      if (d) setColorsAt(Math.min(1, video.currentTime / d));
      controls.update();
      renderer.render(scene, camera);
    });
  }

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (value && value.length) {
        if (!u8) {
          const merged = new Uint8Array(pre.length + value.length);
          merged.set(pre); merged.set(value, pre.length);
          pre = merged;
          tryHeader();
        } else if (got < total) {
          const take = Math.min(value.length, total - got);
          u8.set(value.subarray(0, take), got);
          got += take;
        }
        decodeReadyFrames();
        if (!built && meta && framesReady >= 1) buildScene();
      }
      if (done) break;
    }
  } catch (e) {
    if (!built) bail();                  // died before anything was shown
  }
})();
