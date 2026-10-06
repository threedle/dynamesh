// Click-to-enlarge popup (after threedle/crosslift's image lightbox).
// Clicking a clip or figure opens a large copy over a dimmed page; any
// click that is not a drag closes it, as do Escape and the close button.
// Figures zoom with the wheel and pan by dragging. The 3D viewers open
// through window.dmLightbox.openCustom (wired in comparison.js), which
// hands the builder a square stage and calls its disposer on close.
(function () {
  var css = [
    '.dm-lb{position:fixed;inset:0;z-index:1000;display:none;align-items:center;justify-content:center;',
    'flex-direction:column;background:rgba(20,22,30,.82);cursor:zoom-out;}',
    '.dm-lb.is-open{display:flex;}',
    '.dm-lb-frame{position:relative;background:#fff;border-radius:10px;box-shadow:0 10px 40px rgba(0,0,0,.45);',
    'overflow:hidden;cursor:default;display:flex;align-items:center;justify-content:center;}',
    '.dm-lb-frame video{display:block;width:100%;height:100%;object-fit:contain;background:#fff;}',
    '.dm-lb-frame img{display:block;max-width:none;user-select:none;-webkit-user-drag:none;transform-origin:0 0;}',
    '.dm-lb-cap{color:#fff;font:600 1rem/1.3 system-ui,-apple-system,Arial,sans-serif;text-align:center;',
    'margin:0 12px 10px;max-width:92vw;}',
    '.dm-lb-cap em{font-style:normal;opacity:.9;font-weight:400;color:inherit;}',
    '.dm-lb-hint{color:rgba(255,255,255,.7);font:400 .85rem system-ui,-apple-system,Arial,sans-serif;',
    'margin:10px 12px 0;text-align:center;}',
    '.dm-lb-close{position:absolute;top:10px;right:16px;border:0;background:none;color:#fff;',
    'font-size:2.4rem;line-height:1;cursor:pointer;padding:4px 10px;}',
    '.dm-lb-close:focus-visible{outline:2px solid #fff;border-radius:6px;}',
    '.dm-zoomable{cursor:zoom-in;}'
  ].join('');
  var st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);

  var lb = document.createElement('div');
  lb.className = 'dm-lb';
  lb.setAttribute('role', 'dialog');
  lb.setAttribute('aria-modal', 'true');
  lb.setAttribute('aria-hidden', 'true');
  lb.innerHTML = '<button type="button" class="dm-lb-close" aria-label="Close">&times;</button>' +
    '<div class="dm-lb-cap"></div><div class="dm-lb-frame"></div><div class="dm-lb-hint"></div>';
  document.body.appendChild(lb);
  var frame = lb.querySelector('.dm-lb-frame');
  var cap = lb.querySelector('.dm-lb-cap');
  var hint = lb.querySelector('.dm-lb-hint');
  var closeBtn = lb.querySelector('.dm-lb-close');

  var disposer = null, lastFocus = null, openedAt = 0;

  // square stage for clips and 3D views; figures size themselves
  function squareSide() {
    return Math.floor(Math.min(window.innerWidth * 0.92, window.innerHeight * 0.78));
  }

  function show(caption, hintText) {
    cap.innerHTML = caption || '';
    cap.style.display = caption ? '' : 'none';
    hint.textContent = hintText || '';
    lastFocus = document.activeElement;
    lb.classList.add('is-open');
    lb.setAttribute('aria-hidden', 'false');
    // hiding the scrollbar would shift the page behind; pad its width back
    var sbw = window.innerWidth - document.documentElement.clientWidth;
    if (sbw > 0) document.body.style.paddingRight = sbw + 'px';
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';
    openedAt = performance.now();
    closeBtn.focus();
  }

  function close() {
    if (!lb.classList.contains('is-open')) return;
    lb.classList.remove('is-open');
    lb.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    document.body.style.paddingRight = '';
    document.documentElement.style.overflow = '';
    if (disposer) { try { disposer(); } catch (e) {} disposer = null; }
    frame.innerHTML = '';
    frame.style.width = frame.style.height = '';
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  // a click closes unless it ended a drag (orbiting a 3D view, panning a
  // zoomed figure) or it is the click that opened the popup
  var down = null;
  lb.addEventListener('pointerdown', function (e) { down = { x: e.clientX, y: e.clientY }; });
  lb.addEventListener('click', function (e) {
    if (performance.now() - openedAt < 250) return;
    var moved = down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5;
    down = null;
    if (!moved) close();
  });
  closeBtn.addEventListener('click', function (e) { e.stopPropagation(); close(); });
  // the page behind must not scroll; viewers and figures handle their own
  // wheel before it bubbles here
  lb.addEventListener('wheel', function (e) { e.preventDefault(); }, { passive: false });
  lb.addEventListener('touchmove', function (e) { if (!e.target.closest('canvas')) e.preventDefault(); }, { passive: false });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });

  // ---- captions: the row's prompt plus the clip's column/method label
  function prevPrompt(el) {
    var node = el;
    while (node && node !== document.body) {
      var sib = node.previousElementSibling;
      while (sib) {
        if (sib.classList && sib.classList.contains('vprompt')) return sib.textContent.trim();
        var inner = sib.querySelectorAll ? sib.querySelectorAll('.vprompt') : [];
        if (inner.length) return inner[inner.length - 1].textContent.trim();
        sib = sib.previousElementSibling;
      }
      node = node.parentElement;
    }
    return '';
  }

  function viewLabel(v) {
    // main page: a cell may carry its own label
    var cell = v.closest('.vcell');
    if (cell && cell.querySelector('.vcell-label')) return cell.querySelector('.vcell-label').textContent.trim();
    // supp page: labels injected per clip, method names on divider rows
    var td = v.closest('td');
    if (td) {
      var parts = [];
      var tr = td.parentElement, prev = tr && tr.previousElementSibling;
      if (prev && prev.querySelector('.mname') && td.rowSpan <= 1) parts.push(prev.querySelector('.mname').textContent.trim());
      var lbl = td.querySelector('.cell-label');
      if (lbl) parts.push(lbl.textContent.trim());
      return parts.join(' · ');
    }
    // main page grids: the column header row names the view
    var grid = v.closest('.vgrid');
    if (grid) {
      var parts2 = [];
      var method = v.closest('.cmp-method');
      if (method && method.querySelector('.dm-block-label')) parts2.push(method.querySelector('.dm-block-label').textContent.trim());
      var head = null, scope = grid.parentElement;
      while (scope && !head) {
        head = scope.querySelector('.vgrid.vhead');
        scope = scope.parentElement;
      }
      if (head) {
        var idx = Array.prototype.indexOf.call(grid.children, v.closest('.vgrid > *'));
        if (head.children[idx]) parts2.push(head.children[idx].textContent.trim());
      }
      return parts2.join(' · ');
    }
    var block = v.parentElement && v.parentElement.querySelector('.dm-block-label');
    return block ? block.textContent.trim() : '';
  }

  function captionFor(el) {
    var p = prevPrompt(el), l = viewLabel(el);
    if (p && l) return p + ' <em>&middot; ' + l + '</em>';
    return p || l;
  }

  // ---- clips
  function openVideo(src) {
    var v = src;
    var url = v.__blob || v.dataset.src || v.currentSrc ||
      (v.querySelector('source') && v.querySelector('source').src);
    if (!url) return;
    var side = squareSide();
    frame.style.width = side + 'px';
    frame.style.height = side + 'px';
    var big = document.createElement('video');
    big.muted = true; big.loop = true; big.playsInline = true; big.autoplay = true;
    big.setAttribute('playsinline', '');
    // join the source clip's current moment so the popup continues it
    big.addEventListener('loadeddata', function () {
      if (v.readyState >= 2 && v.duration && big.duration) big.currentTime = v.currentTime % big.duration;
      big.play().catch(function () {});
    });
    big.src = url;
    frame.appendChild(big);
    disposer = function () { big.pause(); big.removeAttribute('src'); big.load(); };
    show(captionFor(v), 'Click anywhere to close');
  }

  // ---- figures: fitted, wheel to zoom, drag to pan
  function openImage(img) {
    var big = new Image();
    big.alt = img.alt || '';
    big.draggable = false;
    var scale = 1, ox = 0, oy = 0, fitW = 0, fitH = 0;
    function applyT() {
      var w = fitW * scale, h = fitH * scale;
      ox = Math.min(0, Math.max(fitW - w, ox));
      oy = Math.min(0, Math.max(fitH - h, oy));
      big.style.width = fitW + 'px';
      big.style.transform = 'translate(' + ox + 'px,' + oy + 'px) scale(' + scale + ')';
      frame.style.cursor = scale > 1 ? 'grab' : 'zoom-out';
    }
    big.onload = function () {
      var r = Math.min(window.innerWidth * 0.92 / big.naturalWidth, window.innerHeight * 0.8 / big.naturalHeight);
      fitW = big.naturalWidth * r; fitH = big.naturalHeight * r;
      frame.style.width = fitW + 'px';
      frame.style.height = fitH + 'px';
      applyT();
    };
    frame.addEventListener('wheel', onWheel, { passive: false });
    function onWheel(e) {
      e.preventDefault();
      var rect = frame.getBoundingClientRect();
      var mx = e.clientX - rect.left, my = e.clientY - rect.top;
      var next = Math.max(1, Math.min(6, scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      ox = mx - (mx - ox) * (next / scale);
      oy = my - (my - oy) * (next / scale);
      scale = next;
      applyT();
    }
    var pan = null;
    function onDown(e) { if (scale > 1) { pan = { x: e.clientX - ox, y: e.clientY - oy }; frame.style.cursor = 'grabbing'; } }
    function onMove(e) { if (pan) { ox = e.clientX - pan.x; oy = e.clientY - pan.y; applyT(); } }
    function onUp() { if (pan) { pan = null; applyT(); } }
    frame.addEventListener('pointerdown', onDown);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    big.src = img.currentSrc || img.src;
    frame.appendChild(big);
    disposer = function () {
      frame.removeEventListener('wheel', onWheel);
      frame.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      frame.style.cursor = '';
    };
    show('', 'Scroll to zoom · Drag to pan when zoomed · Click anywhere to close');
  }

  // ---- 3D views: build(stageDiv) returns a disposer
  function openCustom(build, fromEl) {
    var side = squareSide();
    frame.style.width = side + 'px';
    frame.style.height = side + 'px';
    var stage = document.createElement('div');
    stage.style.cssText = 'position:relative;width:100%;height:100%;';
    frame.appendChild(stage);
    show(fromEl ? captionFor(fromEl) : '',
      'Drag to rotate · Scroll to zoom · Right-drag to translate · Click outside to close');
    disposer = build(stage) || null;
  }

  window.dmLightbox = { openCustom: openCustom, close: close };

  // clicks inside the 3D stage drive the viewer; only the backdrop closes
  frame.addEventListener('click', function (e) {
    if (frame.querySelector('canvas')) e.stopPropagation();
  });

  // ---- triggers: every content clip and figure, except the teaser's own
  // player (it has controls) and the logo
  var clips = document.querySelectorAll('video:not(#teaserVideo)');
  clips.forEach(function (v) {
    v.classList.add('dm-zoomable');
    v.addEventListener('click', function () { openVideo(v); });
  });
  var figs = document.querySelectorAll('section img.halfwidthflex, img[data-zoom]');
  figs.forEach(function (img) {
    img.classList.add('dm-zoomable');
    img.addEventListener('click', function () { openImage(img); });
  });
})();
