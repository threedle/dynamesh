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
const ASSET_V = '5'; // bump when GLBs are re-exported, so cached copies don't linger

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
    for (let t = 1; t <= nT; t++) {
      const div = document.createElement('div');
      div.className = 'dm-cell';
      grid.appendChild(div);
      cells.push(makeCell(div, method, t, az, el));
    }
  }

  function makeCell(div, method, t, azDeg, elDeg) {
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
      (root) => scene.add(root.clone()),
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
