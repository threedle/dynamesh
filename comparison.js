// DynaMesh interactive 3D viewers: the time-synced teaser (reference
// video beside the animated result) and the gallery's time-synced
// rotating cells. Vertex-color animation packs are described in the
// initSyncTeaser comment below.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const ROTATE_DEG_PER_SEC = 20; // full turn in 18s; calmer than BSB's 60deg/s with this many cells
const ASSET_V = '11'; // bump when GLBs are re-exported, so cached copies don't linger

// ---- lit stage -----------------------------------------------------------
// The clips are now rendered through the paper's Blender scene (sun from
// above casting a real shadow, a big soft side fill that lights but does
// not cast), so the live viewers use the same setup: a shadow-casting
// directional "sun", a non-casting side fill, ambient, and an invisible
// ground plane that shows only the received shadow.
function addStage(scene, mesh, opts) {
  const lift = opts && opts.lift !== undefined ? opts.lift : 0.17;
  const bright = opts && opts.bright !== undefined ? opts.bright : 1;
  mesh.castShadow = true;
  mesh.geometry.computeBoundingBox();
  const bb = mesh.geometry.boundingBox;
  const sun = new THREE.DirectionalLight(0xfff2ea, 6.5 * bright);
  sun.position.set(0, 6, 0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.radius = 16;
  sun.shadow.blurSamples = 16;
  sun.shadow.bias = -0.0004;
  const s = 2.2;
  sun.shadow.camera.left = -s; sun.shadow.camera.right = s;
  sun.shadow.camera.top = s; sun.shadow.camera.bottom = -s;
  sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 12;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xd6cdd6, 3.2 * bright);
  fill.position.set(6, 2, 1);           // the blend's side panel: light, no cast
  scene.add(fill);
  scene.add(new THREE.AmbientLight(0xffffff, 0.7 * bright));
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc4, 2.4 * bright));
  const groundMat = new THREE.ShadowMaterial({ opacity: 0.3 });
  // the received shadow fades with distance from the contact point, like
  // the Cycles clips' penumbra, instead of ending as a uniform block
  groundMat.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vPlanePos;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvPlanePos = position.xy;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vPlanePos;')
      .replace('#include <dithering_fragment>',
        'gl_FragColor.a *= 1.0 - smoothstep(0.45, 1.7, length(vPlanePos));\n#include <dithering_fragment>');
  };
  const ground = new THREE.Mesh(new THREE.CircleGeometry(4, 48), groundMat);
  ground.rotation.x = -Math.PI / 2;
  // the mesh may carry a fixed rotation (data-rx): place the floor under
  // its world-space lowest point
  mesh.updateMatrixWorld(true);
  const wbb = bb.clone().applyMatrix4(mesh.matrixWorld);
  ground.position.y = wbb.min.y - 0.005;
  ground.receiveShadow = true;
  scene.add(ground);
  scene.position.y = lift;
}

// small overlay button that puts a viewer's camera back where it started
function addResetButton(container, camera, controls) {
  const home = camera.position.clone();
  const homeTarget = controls.target.clone();
  const btn = document.createElement('button');
  btn.className = 'dm-reset';
  btn.type = 'button';
  btn.title = 'Reset view';
  btn.setAttribute('aria-label', 'Reset view');
  btn.innerHTML = '&#x21bb;';
  // the container itself listens for drags (OrbitControls) and clicks (the
  // teaser's play toggle): none of that may fire from the button
  for (const ev of ['pointerdown', 'pointerup', 'mousedown', 'touchstart']) {
    btn.addEventListener(ev, (e) => e.stopPropagation());
  }
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    camera.position.copy(home);
    controls.target.copy(homeTarget);
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.update();
  });
  if (!container.style.position) container.style.position = 'relative';
  container.appendChild(btn);
}

// ---- time-synced rotating cells (gallery) -------------------------------
// Each .dm-sync[data-pack] cell shows the object's 3D result with its
// vertex colors playing in sync with the row's videos, under the same
// auto-rotating, fully interactive camera as the old static cells. Packs
// load lazily when the cell scrolls near.
(function initSyncedCells() {
  const cells = document.querySelectorAll('.dm-sync[data-pack]');
  if (!cells.length) return;
  if (!('IntersectionObserver' in window)) { cells.forEach(setupSyncCell); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { io.unobserve(e.target); setupSyncCell(e.target); }
    });
  }, { rootMargin: '400px 0px' });
  cells.forEach((c) => io.observe(c));
})();

// fetch + parse a v3 animation pack into geometry-ready arrays
async function loadPackParsed(name) {
  const resp = await fetch(`assets/teaser_anim/${name}_anim.bin?v=${ASSET_V}`);
  if (!resp.ok) throw new Error(resp.status);
  let buf = await resp.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    buf = await new Response(
      new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip')),
    ).arrayBuffer();
  }
  const headLen = new DataView(buf).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headLen)));
  const { verts, tris, frames } = meta;
  const posOff = 4 + headLen;
  const n = verts * 3;
  const posQ = new Uint16Array(buf, posOff, n);
  const positions = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const k = i % 3;
    positions[i] = meta.pmin[k] + posQ[i] * meta.pscale[k];
  }
  const indices = new Uint16Array(buf, posOff + verts * 6, tris * 3);
  const colOff = posOff + Math.ceil((verts * 6 + tris * 6) / 4) * 4;
  const stored = new Uint8Array(buf, colOff, frames * n);

  // undo deltas, expand quantized values, convert sRGB -> linear
  const bits = meta.bits || 8;
  const shift = 8 - bits;
  const mask = (1 << bits) - 1;
  const lutQ = new Uint8Array(1 << bits);
  for (let i = 0; i < lutQ.length; i++) {
    const v8 = (i << shift) | (shift ? i >> (bits - shift) : 0);
    const c = v8 / 255;
    const l = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    lutQ[i] = Math.round(l * 255);
  }
  const colorsLin = new Uint8Array(frames * n);
  const absQ = new Uint8Array(n);
  for (let i = 0; i < n; i++) { absQ[i] = stored[i] & mask; colorsLin[i] = lutQ[absQ[i]]; }
  for (let f = 1; f < frames; f++) {
    const off = f * n;
    for (let i = 0; i < n; i++) {
      absQ[i] = (absQ[i] + stored[off + i]) & mask;
      colorsLin[off + i] = lutQ[absQ[i]];
    }
  }
  return { meta, positions, indices, colorsLin, n };
}

// build a mesh + nearest-frame color setter from a parsed pack
function makePackMesh(parsed, rxDeg, scale) {
  const { meta, positions, indices, colorsLin, n } = parsed;
  const { frames, times } = meta;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const colAttr = new THREE.BufferAttribute(new Uint8Array(n), 3, true);
  colAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('color', colAttr);
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  geom.computeBoundingBox();
  const bb = geom.boundingBox;
  const center = bb.getCenter(new THREE.Vector3());
  const size = bb.getSize(new THREE.Vector3()).length();
  geom.translate(-center.x, -center.y, -center.z);
  geom.scale(scale / size, scale / size, scale / size);
  geom.computeVertexNormals();
  const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.8, metalness: 0.3,
  }));
  if (rxDeg) mesh.rotation.x = THREE.MathUtils.degToRad(rxDeg);
  let lastA = -1;
  function setFrame(tNorm) {
    let b = 1;
    while (b < frames - 1 && times[b] < tNorm) b++;
    const a = (tNorm - times[b - 1] < times[b] - tNorm) ? b - 1 : b;
    if (a === lastA) return;
    lastA = a;
    colAttr.array.set(colorsLin.subarray(a * n, (a + 1) * n));
    colAttr.needsUpdate = true;
  }
  setFrame(0);
  return { mesh, setFrame };
}

async function setupSyncCell(container) {
  let parsed;
  try { parsed = await loadPackParsed(container.dataset.pack); } catch (e) { return; }
  const { mesh, setFrame } = makePackMesh(parsed, Number(container.dataset.rx ?? 0), 1.22);
  const scene = new THREE.Scene();
  scene.add(mesh);
  addStage(scene, mesh, { bright: 0.6 });

  const canvas = document.createElement('canvas');
  Object.assign(canvas.style, { position: 'absolute', inset: '0', width: '100%', height: '100%' });
  container.style.position = 'relative';
  container.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;
  canvas.addEventListener('webglcontextlost', (e) => e.preventDefault());

  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
  const az = THREE.MathUtils.degToRad(Number(container.dataset.az ?? 45));
  const el = THREE.MathUtils.degToRad(Number(container.dataset.el ?? 25));
  const r = 2.4;
  camera.position.set(r * Math.cos(el) * Math.sin(az), r * Math.sin(el), r * Math.cos(el) * Math.cos(az));
  const controls = new OrbitControls(camera, container);
  container.style.touchAction = 'pan-y';
  container.style.cursor = 'grab';
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = ROTATE_DEG_PER_SEC / 6;
  addResetButton(container, camera, controls);
  // real elapsed time keeps the angular speed constant across frame-rate dips
  const spinClock = new THREE.Clock();

  // the row's videos are the clock: follow the first live one
  const row = container.closest('.vgrid');
  const rowVideos = row ? Array.from(row.querySelectorAll('video')) : [];

  renderer.setAnimationLoop(() => {
    // skip everything while the cell is off screen; cap the accumulated
    // delta so the rotation resumes gently instead of jumping
    const rect = container.getBoundingClientRect();
    if (rect.bottom < 0 || rect.top > window.innerHeight) { spinClock.getDelta(); return; }
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    const pr = Math.min(window.devicePixelRatio, 2);
    if (renderer.getPixelRatio() !== pr) renderer.setPixelRatio(pr);
    if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    for (const v of rowVideos) {
      if (v.readyState >= 2 && v.duration && !v.paused) {
        setFrame(Math.min(1, v.currentTime / v.duration));
        break;
      }
    }
    controls.update(Math.min(spinClock.getDelta(), 0.1));
    renderer.render(scene, camera);
  });
}

// ---- time-synced rotating strips (generalization) -----------------------
// A strip shows the reference video followed by MANY shapes (the training
// shape and every unseen one). One renderer per strip draws all its cells
// with a scissor test, since a context per cell would blow the browser's
// WebGL context cap; each cell still has its own OrbitControls.
(function initSyncStrips() {
  const strips = document.querySelectorAll('.dm-syncstrip');
  if (!strips.length) return;
  if (!('IntersectionObserver' in window)) { strips.forEach(setupSyncStrip); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { io.unobserve(e.target); setupSyncStrip(e.target); }
    });
  }, { rootMargin: '400px 0px' });
  strips.forEach((st) => io.observe(st));
})();

async function setupSyncStrip(strip) {
  const wrap = strip.parentElement;           // .dm-syncstrip-wrap
  const video = strip.querySelector('video'); // the reference clip is the clock
  const cellDivs = Array.from(strip.querySelectorAll('.dm-synccell'));
  if (!cellDivs.length) return;

  // the canvas lives INSIDE the scrolled content, spanning its full width,
  // so a horizontal flick moves canvas and cells together natively — a
  // wrapper-pinned canvas repositions by measurement and lags a frame
  // behind fast scrolls, letting meshes slide out of their white cells
  const canvas = document.createElement('canvas');
  Object.assign(canvas.style, {
    position: 'absolute', left: '0', top: '0',
    pointerEvents: 'none', zIndex: '5',
  });
  strip.style.position = 'relative';
  strip.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;
  renderer.setScissorTest(true);
  canvas.addEventListener('webglcontextlost', (e) => e.preventDefault());

  const cells = [];
  cellDivs.forEach((div) => {
    loadPackParsed(div.dataset.pack).then((parsed) => {
      // training-shape cells render their static mesh larger than the
      // rotating neighbors; scale them down and raise them to match
      const isTrain = div.dataset.pack.endsWith('_train');
      const { mesh, setFrame } = makePackMesh(parsed, Number(div.dataset.rx ?? 0), isTrain ? 0.92 : 1.26);
      if (isTrain) mesh.position.y += 0.22;
      const scene = new THREE.Scene();
      scene.add(mesh);
      // the rorschach transfer strip reads too bright at full stage light
      addStage(scene, mesh, { bright: div.dataset.pack.startsWith('gx_') ? 0.6 : 1 });
      const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
      const az = THREE.MathUtils.degToRad(Number(div.dataset.az ?? 45));
      const el = THREE.MathUtils.degToRad(Number(div.dataset.el ?? 20));
      const r = 2.3;
      camera.position.set(r * Math.cos(el) * Math.sin(az), r * Math.sin(el), r * Math.cos(el) * Math.cos(az));
      const controls = new OrbitControls(camera, div);
      // both pan directions stay native on touch, so phones can scroll
      // the strip; rotation remains available with a mouse
      div.style.touchAction = 'pan-x pan-y';
      div.style.cursor = 'grab';
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.enablePan = true;
      controls.autoRotate = true;
      controls.autoRotateSpeed = ROTATE_DEG_PER_SEC / 6;
      addResetButton(div, camera, controls);
      cells.push({ div, scene, camera, controls, setFrame });
    }, () => {});
  });

  const spinClock = new THREE.Clock();
  // browsers pause offscreen autoplay videos, so the reference clip cannot
  // be the only clock: sync to it while it plays, and freewheel from the
  // last known time while it is paused, so every cell keeps animating and
  // looping no matter how far the strip is scrolled
  let clockT = 0, clockWall = performance.now();
  renderer.setAnimationLoop(() => {
    const wrapRect = wrap.getBoundingClientRect();
    if (wrapRect.bottom < 0 || wrapRect.top > window.innerHeight) { spinClock.getDelta(); return; }
    const w = strip.scrollWidth, h = strip.clientHeight;
    if (!w || !h) return;
    const pr = Math.min(window.devicePixelRatio, 2);
    if (renderer.getPixelRatio() !== pr) renderer.setPixelRatio(pr);
    if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) {
      renderer.setSize(w, h, false);
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
    }
    const dt = Math.min(spinClock.getDelta(), 0.1);
    const now = performance.now();
    const dur = (video && video.duration) || 6;
    let t;
    if (video && video.duration && video.readyState >= 2 && !video.paused) {
      t = Math.min(1, video.currentTime / video.duration);
      clockT = t; clockWall = now;
    } else {
      if (video && video.paused && video.readyState >= 2) video.play().catch(() => {});
      t = (clockT + (now - clockWall) / 1000 / dur) % 1;
    }
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);
    const canvasRect = canvas.getBoundingClientRect();
    for (const cell of cells) {
      const r = cell.div.getBoundingClientRect();
      // skip cells outside the strip's visible viewport
      if (r.right < wrapRect.left || r.left > wrapRect.right) { continue; }
      if (t !== null) cell.setFrame(t);
      cell.controls.update(dt);
      const width = r.width, height = r.height;
      const left = r.left - canvasRect.left;
      const bottom = h - (r.bottom - canvasRect.top);
      cell.camera.aspect = width / height;
      cell.camera.updateProjectionMatrix();
      renderer.setViewport(left, bottom, width, height);
      renderer.setScissor(left, bottom, width, height);
      renderer.render(cell.scene, cell.camera);
    }
  });
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
    geom.scale(1.38 / size, 1.38 / size, 1.38 / size);
    geom.computeVertexNormals();

    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.8, metalness: 0.3,
    }));
    const scene = new THREE.Scene();
    scene.add(mesh);
    addStage(scene, mesh, { lift: 0, bright: 0.52 });

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
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.VSMShadowMap;
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
    controls.autoRotate = true;
    controls.autoRotateSpeed = ROTATE_DEG_PER_SEC / 6;
    addResetButton(container, camera, controls);
    const spinClock = new THREE.Clock();

    // a plain click (no drag) on the shape toggles the shared clock
    let downX = 0, downY = 0, downT = 0;
    container.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; downT = performance.now(); });
    container.addEventListener('pointerup', (e) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) < 5 && performance.now() - downT < 400) {
        if (video.paused) video.play().catch(() => {}); else video.pause();
      }
    });

    // the pack carries every source frame, so nearest-frame playback is
    // exact; a memcpy ~30x/s replaces the per-vertex lerp at 60 fps
    let lastA = -1;
    setColorsAt = (tNorm) => {
      const maxF = framesReady - 1;
      if (maxF < 0) return;
      let b = 1;
      while (b <= maxF && times[b] < tNorm) b++;
      if (b > maxF) b = maxF;
      const lo = Math.max(0, b - 1);
      const a = (tNorm - times[lo] < times[b] - tNorm) ? lo : b;
      if (a === lastA) return;
      lastA = a;
      colAttr.array.set(colorsLin.subarray(a * n, (a + 1) * n));
      colAttr.needsUpdate = true;
    };
    onNewFrames = () => { lastA = -1; };   // a clamped time can now re-render
    setColorsAt(0);

    // ready: reveal the shape and start the shared clock from the beginning
    canvas.style.opacity = '1';
    try { video.currentTime = 0; } catch (e) {}
    startVideo();

    renderer.setAnimationLoop(() => {
      // skip render work while the teaser is scrolled out of view
      const rect = container.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) return;
      const w = container.clientWidth, h = container.clientHeight;
      if (!w || !h) return;
      const pr = Math.min(window.devicePixelRatio, 2);
      if (renderer.getPixelRatio() !== pr) renderer.setPixelRatio(pr);
      if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      const d = video.duration;
      if (d) setColorsAt(Math.min(1, video.currentTime / d));
      controls.update(Math.min(spinClock.getDelta(), 0.1));
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
