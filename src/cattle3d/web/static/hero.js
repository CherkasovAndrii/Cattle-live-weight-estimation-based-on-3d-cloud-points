// Hero: the real merged point cloud of CowDatabase2 cow 70 (left, right and
// top Kinect views in one frame, img/cow70_points.bin: int16 x, y, z in mm,
// uint8 view, uint8 pad). Points fly in from the camera that saw them, then
// settle into one animal; a slice sweeps along the trunk the way the volume
// is measured. Drag to turn it. WebGL, with a plain 2-D canvas fallback.
(() => {
  "use strict";

  const canvas = document.getElementById("cow3d");
  if (!canvas) return;
  const stage = canvas.parentElement;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Data frame: x along the body, y across (left camera at -y), z up from the
  // floor. World: X = x, Y = z - 0.72, Z = y.
  const CAMS = [
    { name: "left", pos: [0, 0.2, -1.85], color: [0.33, 0.8, 1.0] },
    { name: "right", pos: [0, 0.2, 1.85], color: [0.95, 0.45, 1.0] },
    { name: "top", pos: [0, 1.55, 0], color: [1.0, 0.75, 0.35] },
  ];
  const ASSEMBLE_S = 2.6, SWEEP_S = 6.5;

  let n = 0, pos, view, seed;
  let yaw = -0.65, pitch = 0.32, auto = !still, dragging = null, idleAt = 0;
  let t0 = 0, visible = true, raf = 0;
  let draw = () => {};

  function project(p, w, h) {
    // Same transform as the vertex shader, for labels and the 2-D fallback.
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = cy * p[0] + sy * p[2], z1 = -sy * p[0] + cy * p[2];
    const y2 = cp * p[1] - sp * z1, z2 = sp * p[1] + cp * z1;
    const depth = 4.4 - z2;
    const f = 2.5 / depth, aspect = w / h;
    const s = aspect < 1 ? aspect : 1;
    return [w / 2 + (x1 * f * s / aspect) * w / 2, h / 2 - (y2 * f * s) * h / 2, depth];
  }

  function labels() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    for (const cam of CAMS) {
      const el = stage.querySelector(`[data-cam="${cam.name}"]`);
      if (!el) continue;
      const [x, y] = project(cam.pos, w, h);
      el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    }
  }

  // ---- WebGL -----------------------------------------------------------------
  const VS = `
    attribute vec3 a_pos; attribute float a_view; attribute float a_seed;
    uniform float u_yaw, u_pitch, u_aspect, u_time, u_assemble, u_sweep, u_dpr, u_size;
    uniform vec3 u_cam0, u_cam1, u_cam2, u_col0, u_col1, u_col2;
    varying vec3 v_col; varying float v_alpha;
    vec3 heat(float t) {
      vec3 a = vec3(0.32, 0.22, 0.95), b = vec3(0.25, 0.85, 0.95), c = vec3(1.0, 0.97, 0.92);
      return t < 0.55 ? mix(a, b, t / 0.55) : mix(b, c, (t - 0.55) / 0.45);
    }
    void main() {
      vec3 cam = a_view < 0.5 ? u_cam0 : (a_view < 1.5 ? u_cam1 : u_cam2);
      vec3 camCol = a_view < 0.5 ? u_col0 : (a_view < 1.5 ? u_col1 : u_col2);
      float start = a_seed * 0.55;
      float k = clamp((u_assemble - start) / 0.45, 0.0, 1.0);
      float e = 1.0 - pow(1.0 - k, 3.0);
      vec3 p = mix(cam, a_pos, e);
      float cy = cos(u_yaw), sy = sin(u_yaw), cp = cos(u_pitch), sp = sin(u_pitch);
      vec3 r = vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
      r = vec3(r.x, cp * r.y - sp * r.z, sp * r.y + cp * r.z);
      float depth = 4.4 - r.z;
      float f = 2.5 / depth;
      float s = u_aspect < 1.0 ? u_aspect : 1.0;
      gl_Position = vec4(r.x * f * s / u_aspect, r.y * f * s, 0.0, 1.0);
      float h = clamp((a_pos.y + 0.6) / 1.35, 0.0, 1.0);
      float settle = smoothstep(0.75, 1.0, u_assemble);
      vec3 col = mix(camCol, heat(h), settle * 0.85);
      float band = exp(-pow((a_pos.x - u_sweep) / 0.035, 2.0));
      col += vec3(0.9, 0.95, 1.0) * band * settle;
      float tw = 0.85 + 0.15 * sin(u_time * 2.0 + a_seed * 40.0);
      v_col = col * tw;
      v_alpha = k * (0.55 + 0.45 * band);
      gl_PointSize = u_size * u_dpr * f * (1.0 + band * 0.8);
    }`;
  const FS = `
    precision mediump float;
    varying vec3 v_col; varying float v_alpha;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      float r = dot(d, d) * 4.0;
      if (r > 1.0) discard;
      gl_FragColor = vec4(v_col, v_alpha * (1.0 - r) * (1.0 - r));
    }`;
  const LINE_VS = `
    attribute vec3 a_pos; attribute float a_k;
    uniform float u_yaw, u_pitch, u_aspect;
    varying float v_k;
    void main() {
      float cy = cos(u_yaw), sy = sin(u_yaw), cp = cos(u_pitch), sp = sin(u_pitch);
      vec3 r = vec3(cy * a_pos.x + sy * a_pos.z, a_pos.y, -sy * a_pos.x + cy * a_pos.z);
      r = vec3(r.x, cp * r.y - sp * r.z, sp * r.y + cp * r.z);
      float f = 2.5 / (4.4 - r.z);
      float s = u_aspect < 1.0 ? u_aspect : 1.0;
      gl_Position = vec4(r.x * f * s / u_aspect, r.y * f * s, 0.0, 1.0);
      v_k = a_k;
    }`;
  const LINE_FS = `
    precision mediump float;
    uniform vec3 u_col; uniform float u_alpha;
    varying float v_k;
    void main() { gl_FragColor = vec4(u_col, u_alpha * (1.0 - v_k * 0.85)); }`;

  function compile(gl, vs, fs) {
    const prog = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      gl.attachShader(prog, sh);
    }
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    return prog;
  }

  function setupGL() {
    const gl = canvas.getContext("webgl", { antialias: true, premultipliedAlpha: false, alpha: true });
    if (!gl) return false;
    let prog, lineProg;
    try { prog = compile(gl, VS, FS); lineProg = compile(gl, LINE_VS, LINE_FS); } catch { return false; }

    const data = new Float32Array(n * 5);
    for (let i = 0; i < n; i++) data.set([pos[3 * i], pos[3 * i + 1], pos[3 * i + 2], view[i], seed[i]], i * 5);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

    // Camera rays: each camera to the corners of the body box it faces.
    const box = [[-1.1, -0.6, -0.4], [1.1, 0.68, 0.4]];
    const rays = [];
    for (const cam of CAMS) {
      const face = cam.name === "top"
        ? [[box[0][0], box[1][1], box[0][2]], [box[1][0], box[1][1], box[0][2]], [box[0][0], box[1][1], box[1][2]], [box[1][0], box[1][1], box[1][2]]]
        : [[box[0][0], box[0][1], 0], [box[1][0], box[0][1], 0], [box[0][0], box[1][1], 0], [box[1][0], box[1][1], 0]];
      const z = cam.name === "left" ? box[0][2] : box[1][2];
      const lines = [];
      for (const c of face) lines.push(...cam.pos, 0, c[0], c[1], cam.name === "top" ? c[2] : z, 1);
      rays.push(new Float32Array(lines));
    }
    const rayBuf = rays.map((r) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, r, gl.STATIC_DRAW); return b; });

    const U = (p, name) => gl.getUniformLocation(p, name);
    const A = (p, name) => gl.getAttribLocation(p, name);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    draw = (time) => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const aspect = w / h;
      const assemble = still ? 2 : Math.min(2, time / ASSEMBLE_S);
      const sweep = still ? 9 : -1.4 + 2.8 * ((time % SWEEP_S) / SWEEP_S);

      gl.useProgram(lineProg);
      gl.uniform1f(U(lineProg, "u_yaw"), yaw); gl.uniform1f(U(lineProg, "u_pitch"), pitch);
      gl.uniform1f(U(lineProg, "u_aspect"), aspect);
      const fade = still ? 0.16 : 0.08 + 0.2 * Math.max(0, 1 - Math.abs(assemble - 0.6));
      rays.forEach((r, i) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, rayBuf[i]);
        const ap = A(lineProg, "a_pos"), ak = A(lineProg, "a_k");
        gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 3, gl.FLOAT, false, 16, 0);
        gl.enableVertexAttribArray(ak); gl.vertexAttribPointer(ak, 1, gl.FLOAT, false, 16, 12);
        gl.uniform3fv(U(lineProg, "u_col"), CAMS[i].color); gl.uniform1f(U(lineProg, "u_alpha"), fade);
        gl.drawArrays(gl.LINES, 0, r.length / 4);
        gl.disableVertexAttribArray(ap); gl.disableVertexAttribArray(ak);
      });

      gl.useProgram(prog);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      const ap = A(prog, "a_pos"), av = A(prog, "a_view"), as = A(prog, "a_seed");
      gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 3, gl.FLOAT, false, 20, 0);
      gl.enableVertexAttribArray(av); gl.vertexAttribPointer(av, 1, gl.FLOAT, false, 20, 12);
      gl.enableVertexAttribArray(as); gl.vertexAttribPointer(as, 1, gl.FLOAT, false, 20, 16);
      gl.uniform1f(U(prog, "u_yaw"), yaw); gl.uniform1f(U(prog, "u_pitch"), pitch);
      gl.uniform1f(U(prog, "u_aspect"), aspect); gl.uniform1f(U(prog, "u_time"), time);
      gl.uniform1f(U(prog, "u_assemble"), assemble); gl.uniform1f(U(prog, "u_sweep"), sweep);
      gl.uniform1f(U(prog, "u_dpr"), dpr);
      gl.uniform1f(U(prog, "u_size"), Math.max(5, Math.min(w, h * 1.4) / 85));
      CAMS.forEach((c, i) => { gl.uniform3fv(U(prog, `u_cam${i}`), c.pos); gl.uniform3fv(U(prog, `u_col${i}`), c.color); });
      gl.drawArrays(gl.POINTS, 0, n);
      gl.disableVertexAttribArray(ap); gl.disableVertexAttribArray(av); gl.disableVertexAttribArray(as);
      labels();
    };
    return true;
  }

  // ---- 2-D fallback ------------------------------------------------------------
  function setup2D() {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < n; i += 2) {
        const [x, y] = project([pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]], w, h);
        const t = Math.min(1, Math.max(0, (pos[3 * i + 1] + 0.6) / 1.35));
        ctx.fillStyle = `rgba(${80 + 175 * t | 0},${120 + 120 * t | 0},255,0.5)`;
        ctx.fillRect(x - 1, y - 1, 2, 2);
      }
      labels();
    };
  }

  // ---- loop and input -------------------------------------------------------------
  function frame(now) {
    if (!t0) t0 = now;
    const time = (now - t0) / 1000;
    if (auto && !dragging && now > idleAt) yaw += 0.0035;
    draw(time);
    raf = visible && !still ? requestAnimationFrame(frame) : 0;
  }
  const redraw = () => { if (!raf) raf = requestAnimationFrame(frame); };

  canvas.addEventListener("pointerdown", (e) => {
    dragging = { x: e.clientX, y: e.clientY, yaw, pitch, id: e.pointerId, moved: false };
  });
  window.addEventListener("pointermove", (e) => {
    if (!dragging || e.pointerId !== dragging.id) return;
    const dx = e.clientX - dragging.x, dy = e.clientY - dragging.y;
    if (!dragging.moved && Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
    if (!dragging.moved) { dragging.moved = true; canvas.setPointerCapture?.(e.pointerId); }
    yaw = dragging.yaw + dx * 0.008;
    pitch = Math.max(-0.1, Math.min(1.1, dragging.pitch + dy * 0.005));
    if (still) redraw();
  });
  const release = () => { if (dragging) { dragging = null; idleAt = performance.now() + 2500; } };
  window.addEventListener("pointerup", release);
  window.addEventListener("pointercancel", release);
  window.addEventListener("resize", redraw);
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible) redraw(); }).observe(canvas);
  document.addEventListener("visibilitychange", () => { visible = !document.hidden; if (visible) redraw(); });

  fetch(canvas.dataset.src)
    .then((res) => { if (!res.ok) throw new Error(res.status); return res.arrayBuffer(); })
    .then((buffer) => {
      const bytes = new DataView(buffer);
      n = Math.floor(buffer.byteLength / 8);
      pos = new Float32Array(n * 3); view = new Float32Array(n); seed = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const o = i * 8;
        pos[3 * i] = bytes.getInt16(o, true) / 1000;
        pos[3 * i + 1] = bytes.getInt16(o + 4, true) / 1000 - 0.72;
        pos[3 * i + 2] = bytes.getInt16(o + 2, true) / 1000;
        view[i] = bytes.getUint8(o + 6);
        seed[i] = Math.random();
      }
      if (!setupGL()) setup2D();
      stage.classList.add("ready");
      redraw();
    })
    .catch(() => stage.classList.add("failed"));
})();
