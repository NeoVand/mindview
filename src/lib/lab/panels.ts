// Panels: rectangles in the scene that show numbers straight from the GPU, one cell per number, at any distance.
//   ternary  a weight matrix as the model holds it (each weight -1, 0 or +1 times its group's scale), or each weight
//            times the input it multiplies (w x x[k], the products a matrix-vector product adds up along each row)
//   dense    activations (or the product of two activations, a[r, c] x b[r, c]) from the model's arena or the lab's
//            own scratch buffer
//   bars     a vector as a waveform: one signed bar per number
//   dial     rotary pairs: each cell a circle with the pair (x_i, x_i+64) before (faint) and after (bright) rotation
//   frame    only the outline of the rectangle, a pixel wide (the edges of a box)
// Far away a pixel covers many cells. It then shows the strongest of them, with its sign (a single busy column stays a
// line of light at any distance), scaled by how strong the strongest of that many ordinary cells would be, so the
// brightness does not change as you come closer. Ternary panels read this from a pyramid of 8 x 8 blocks (each block's
// peak and mean |v|, built on the GPU from the real values); other panels look at a few cells per pixel. Close up
// every cell is drawn apart, and when it is large enough, its value is printed in it.
import type { BonsaiLLM, Ternary } from '$lib/runtime/bonsai-llm';
import { DEPTH_FORMAT, FRAME_WGSL, HDR_FORMAT } from '$lib/engine/gpu';
import { DIGITS_WGSL, glyphTexture } from './digits';
import type { Stage, V3 } from './stage';

const STRIDE = 256;
const NONE = 0xffffffff;

export interface Src {
	offset: number;
	rs: number; // stride per displayed row
	cs: number; // stride per displayed column
	aux?: boolean; // read from the lab's scratch buffer instead of the arena
	dw?: boolean; // from the painter's dense weights
	cap?: boolean; // from a painting's capture
}

export interface PanelDesc {
	origin: V3; // top-left corner
	ax: V3; // along the columns, the whole width
	ay: V3; // down the rows, the whole height
	rows: number;
	cols: number;
	kind: 'ternary' | 'dense' | 'bars' | 'dial' | 'frame';
	tensor?: Ternary;
	transpose?: boolean; // displayed rows are the matrix's inputs
	m0?: number; // first output row of the view
	k0?: number; // first input column of the view
	x?: { offset: number; aux?: boolean; cap?: boolean } | null; // input vector: show w x x[k]
	painter?: boolean; // the tensor is one of the painter's
	permR?: number; // offset into the index buffer: displayed row -> row
	permC?: number;
	a?: Src & { div?: number; ds?: number }; // plus (column / div) x ds
	b?: Src;
	sweepStride?: number; // bars: added to a's index per sweep checkpoint (16 columns)
	gain: number; // the value shown at full brightness
	alpha?: number;
	tint?: V3;
	lift?: number; // extra glow above full brightness
	digits?: boolean;
	grid?: boolean;
	sweep?: number | null; // columns before this are done, those after it wait
	cut?: [number, number, number, number]; // visible part in cells: col0, row0, col1, row1
	additive?: boolean;
	pyramid?: boolean;
	visible?: boolean;
	/** At most this many cells (per side) looked at for a pixel that covers many (default 4). */
	samples?: number;
	/**
	 * A stack of copies: copy i stands at origin + i x step and reads its inputs shifted by i x the given strides
	 * (x: the input vector; a, b: the dense sources); causal: copy i shows only its first i + 1 rows. table: an
	 * offset into the index buffer of pairs (position, input shift): copy i then stands at origin + position x step
	 * and reads x and b (a with tableA) shifted by its own amount (for copies that are not evenly spaced).
	 */
	instances?: {
		count: number;
		step: V3;
		x?: number;
		a?: number;
		b?: number;
		causal?: boolean;
		table?: number;
		tableA?: boolean;
	};
}

interface Slot {
	d: PanelDesc;
	pyr: number; // offset in the pyramid buffer (u32s), NONE without
	w1: number;
	h1: number;
	dirty: boolean;
}

const COMMON = /* wgsl */ `
struct Panel {
  origin: vec3f, rows: f32,
  ax: vec3f, cols: f32,
  ay: vec3f, kind: f32,
  codes: u32, scales: u32, K: u32, flags: u32,
  xoff: u32, permR: u32, permC: u32, pyr: u32,
  a0: u32, ars: u32, acs: u32, b0: u32,
  brs: u32, bcs: u32, w1: u32, h1: u32,
  gain: f32, alpha: f32, sweep: f32, sweepStride: f32,
  tint: vec3f, lift: f32,
  cut0: vec2f, cut1: vec2f,
  m0: u32, k0: u32, maxSamples: u32, _p1: u32,
  istep: vec3f, icount: f32,
  ixs: u32, ias: u32, ibs: u32, iflags: u32,
  adiv: u32, ads: u32, itab: u32, _q1: u32,
};
// flags
const TRANSPOSE = 1u; const USE_X = 2u; const X_AUX = 4u; const PERM_R = 8u; const PERM_C = 16u;
const DIGITS = 32u; const GRID = 64u; const B_ON = 128u; const A_AUX = 256u; const B_AUX = 512u; const SWEEP = 1024u;
const W2 = 2048u; const A_DW = 4096u; const B_DW = 8192u; const X_CAP = 16384u; const A_CAP = 32768u; const B_CAP = 65536u;

// a number from one of the sources: the reader's arena, the lab's scratch, the painter's dense weights, the capture
fn readS(i: u32, aux: bool, dw: bool, cap: bool) -> f32 {
  if (cap) { return CAP[i]; }
  if (dw) { return DW[i]; }
  if (aux) { return AUX[i]; }
  return ARENA[i];
}
fn readA(i: u32, aux: bool) -> f32 { if (aux) { return AUX[i]; } return ARENA[i]; }

// the value of displayed cell (r, c)
fn cellValue(P: Panel, r: u32, c: u32) -> f32 {
  var rr = r; var cc = c;
  if ((P.flags & PERM_R) != 0u) { rr = IDX[P.permR + r]; }
  if ((P.flags & PERM_C) != 0u) { cc = IDX[P.permC + c]; }
  if (P.kind < 0.5) {
    var m = rr; var k = cc;
    if ((P.flags & TRANSPOSE) != 0u) { m = cc; k = rr; }
    m += P.m0; k += P.k0;
    let ci = P.codes + m * (P.K / 16u) + k / 16u;
    let si = P.scales + m * (P.K / 128u) + k / 128u;
    var word = 0u; var sc = 0.0;
    if ((P.flags & W2) != 0u) { word = CODES2[ci]; sc = SCALES2[si]; } else { word = CODES[ci]; sc = SCALES[si]; }
    let t = f32((word >> ((k % 16u) * 2u)) & 3u) - 1.0;
    var w = t * sc;
    if ((P.flags & USE_X) != 0u) { w *= readS(P.xoff + k, (P.flags & X_AUX) != 0u, false, (P.flags & X_CAP) != 0u); }
    return w;
  }
  var ia = P.a0 + rr * P.ars + cc * P.acs;
  if (P.adiv > 0u) { ia += (cc / P.adiv) * P.ads; }
  if ((P.flags & SWEEP) != 0u) { ia += u32(max(0.0, floor(P.sweep / 16.0))) * u32(P.sweepStride); }
  var v = readS(ia, (P.flags & A_AUX) != 0u, (P.flags & A_DW) != 0u, (P.flags & A_CAP) != 0u);
  if ((P.flags & B_ON) != 0u) {
    v *= readS(P.b0 + rr * P.brs + cc * P.bcs, (P.flags & B_AUX) != 0u, (P.flags & B_DW) != 0u, (P.flags & B_CAP) != 0u);
  }
  return v;
}
`;

const RENDER_WGSL = /* wgsl */ `${FRAME_WGSL}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> PU: Panel;
@group(0) @binding(2) var<storage, read> CODES: array<u32>;
@group(0) @binding(3) var<storage, read> SCALES: array<f32>;
@group(0) @binding(4) var<storage, read> ARENA: array<f32>;
@group(0) @binding(5) var<storage, read> AUX: array<f32>;
@group(0) @binding(6) var<storage, read> PYR: array<u32>;
@group(0) @binding(7) var<storage, read> IDX: array<u32>;
@group(0) @binding(8) var glyphs: texture_2d<f32>;
@group(0) @binding(9) var gsamp: sampler;
@group(0) @binding(10) var<storage, read> CODES2: array<u32>;
@group(0) @binding(11) var<storage, read> SCALES2: array<f32>;
@group(0) @binding(12) var<storage, read> DW: array<f32>;
@group(0) @binding(13) var<storage, read> CAP: array<f32>;
${COMMON}
${DIGITS_WGSL}

struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) inst: u32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let c = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1))[vi];
  var o: VO;
  // copy ii stands at istep x ii, or (with a table) at istep x its entry
  var k = f32(ii);
  if ((PU.iflags & 2u) != 0u) { k = f32(IDX[PU.itab + 2u * ii]); }
  o.pos = F.viewProj * vec4f(PU.origin + PU.istep * k + PU.ax * c.x + PU.ay * c.y, 1.0);
  o.uv = c;
  o.inst = ii;
  return o;
}

// pyramid level l (1..3) cell (x, y): (|peak|, peak), the peak being the block's value of largest size
fn pyrCell(l: u32, x: i32, y: i32) -> vec2f {
  var w = PU.w1; var h = PU.h1; var off = PU.pyr;
  for (var i = 1u; i < l; i++) { off += w * h; w = (w + 7u) / 8u; h = (h + 7u) / 8u; }
  let cx = u32(clamp(x, 0, i32(w) - 1)); let cy = u32(clamp(y, 0, i32(h) - 1));
  let pk = unpack2x16float(PYR[off + cy * w + cx]).x;
  return vec2f(abs(pk), pk);
}
// how much larger the largest of the cells a pixel covers is than a typical cell, for random values
fn boost(foot: f32) -> f32 { return 1.0 + 1.1 * sqrt(2.0 * log2(max(foot, 1.0))); }

fn hueOf(m: vec2f) -> vec3f {
  let r = clamp(m.y / max(m.x, 1e-30), -1.0, 1.0);
  let mid = BONE * 0.55;
  return select(mix(mid, PLUS, r), mix(mid, MINUS, -r), r < 0.0);
}

struct FO { @location(0) col: vec4f };

fn shade(i: VO) -> vec4f {
  // an instance (one of a stack of slices) reads its own part of the inputs
  var P = PU;
  P.xoff += i.inst * PU.ixs; P.a0 += i.inst * PU.ias; P.b0 += i.inst * PU.ibs;
  if ((PU.iflags & 2u) != 0u) {
    let d = IDX[PU.itab + 2u * i.inst + 1u];
    P.xoff += d;
    if ((PU.iflags & 4u) != 0u) { P.a0 += d; } else { P.b0 += d; }
  }
  if ((PU.iflags & 1u) != 0u) { P.cut1.y = min(P.cut1.y, f32(i.inst + 1u)); }
  let cell = vec2f(i.uv.x * P.cols, i.uv.y * P.rows); // (col, row)
  let fw = max(fwidth(cell), vec2f(1e-6));
  let dpx = dpdx(cell); let dpy = dpdy(cell);
  let foot = max(fw.x, fw.y);
  if (cell.x < P.cut0.x || cell.y < P.cut0.y || cell.x >= P.cut1.x || cell.y >= P.cut1.y) { return vec4f(-1.0); }
  let ci = vec2u(min(floor(cell), vec2f(P.cols - 1.0, P.rows - 1.0)));
  let local = fract(cell);
  let px = 1.0 / foot; // cell size in pixels

  // ---- bars: a waveform, one signed bar per column (rows are separate waveforms)
  if (P.kind > 1.5 && P.kind < 2.5) {
    let y = 1.0 - 2.0 * local.y; // up is positive, centre line at 0
    var lo = 0.0; var hi = 0.0; var v0 = 0.0;
    let n = u32(clamp(ceil(fw.x), 1.0, 8.0));
    for (var s = 0u; s < n; s++) {
      let cx = u32(clamp(cell.x + (f32(s) + 0.5) / f32(n) * fw.x - fw.x * 0.5, 0.0, P.cols - 1.0));
      let v = cellValue(P, ci.y, cx) / P.gain;
      if (s == 0u) { v0 = v; }
      lo = min(lo, v); hi = max(hi, v);
    }
    let top = clamp(hi, -1.0, 1.0); let bot = clamp(lo, -1.0, 1.0);
    let edge = max(fw.y * 2.0, 0.002);
    let inPos = smoothstep(-edge, edge, top - y) * smoothstep(-edge, edge, y);
    let inNeg = smoothstep(-edge, edge, y - bot) * smoothstep(-edge, edge, -y);
    var col = PLUS * inPos * (0.35 + 0.65 * min(1.0, hi)) + MINUS * inNeg * (0.35 + 0.65 * min(1.0, -lo));
    col += (PLUS * min(max(0.0, hi - 1.0), 2.0) * inPos + MINUS * min(max(0.0, -lo - 1.0), 2.0) * inNeg) * 0.3 * P.lift;
    // the centre line, and the cell edges when the bars are wide
    col += BONE * 0.06 * (1.0 - smoothstep(0.0, edge * 1.5, abs(y)));
    if (px > 6.0) { col *= 0.35 + 0.65 * smoothstep(0.0, 0.08, min(local.x, 1.0 - local.x)); }
    if ((P.flags & DIGITS) != 0u && px > 30.0) {
      // the value, in the half the bar does not reach
      let half = select(0.5, 0.0, v0 < 0.0);
      let bp = vec2f(local.x, (local.y - half) / 0.5);
      let aspect = (1.0 / fw.x) / (0.5 / fw.y);
      let t = printNum(v0 * P.gain, bp, aspect, 0.34, vec2f(dpx.x, dpx.y * 2.0), vec2f(dpy.x, dpy.y * 2.0));
      col = mix(col, BONE * 1.1, t);
    }
    return vec4f(col * P.alpha * P.tint, 1.0);
  }

  // ---- frames: the outline only
  if (P.kind > 3.5) {
    let uw = fw / vec2f(P.cols, P.rows); // (the derivatives are taken above, where control flow is uniform)
    let e = min(min(i.uv.x, 1.0 - i.uv.x) / uw.x, min(i.uv.y, 1.0 - i.uv.y) / uw.y);
    return vec4f(BONE * (1.0 - smoothstep(0.4, 1.4, e)) * P.alpha * P.tint, 1.0);
  }

  // ---- dials: rotary pairs
  if (P.kind > 2.5) {
    var rr = ci.y; var cc = ci.x;
    let ia = P.a0 + rr * P.ars + cc * P.acs;
    let ib = P.b0 + rr * P.brs + cc * P.bcs;
    let pre = vec2f(readA(ia, (P.flags & A_AUX) != 0u), readA(ia + 64u, (P.flags & A_AUX) != 0u)) / P.gain;
    let post = vec2f(readA(ib, (P.flags & B_AUX) != 0u), readA(ib + 64u, (P.flags & B_AUX) != 0u)) / P.gain;
    let q = (local - 0.5) * vec2f(2.0, -2.0); // -1..1, up positive
    let pw = 2.0 * foot; // one pixel in q units
    var col = BONE * 0.10 * (1.0 - smoothstep(0.0, pw * 1.5, abs(length(q) - 0.9)));
    let hand = fn_hand(q, pre, pw);
    let hand2 = fn_hand(q, post, pw);
    let size = 0.3 + 0.7 * sqrt(min(1.0, length(post)));
    col += BONE * 0.3 * size * hand + mix(MINUS, PLUS, 0.5 + 0.5 * sign(post.x)) * 1.2 * size * hand2;
    return vec4f(col * P.alpha * P.tint, 1.0);
  }

  // ---- matrices and dense grids: the strongest value this pixel covers, (|v|, v)
  var m = vec2f(0.0);
  var bf = foot; // how many cells (per side) the strongest was chosen from
  var exact = 0.0;
  if (foot <= 1.0) {
    exact = cellValue(P, ci.y, ci.x);
    m = vec2f(abs(exact), exact);
  } else if (P.kind < 0.5 && P.pyr != 0xffffffffu && foot >= 8.0) {
    // the strongest block among those this pixel covers, at the level whose blocks are about a pixel
    let l = u32(clamp(floor(log2(foot) / 3.0), 1.0, 3.0));
    let B = pow(8.0, f32(l));
    let k = foot / B;
    let n = u32(clamp(ceil(k), 1.0, 4.0));
    for (var sy = 0u; sy < n; sy++) {
      for (var sx = 0u; sx < n; sx++) {
        let o = (vec2f(f32(sx), f32(sy)) + 0.5) / f32(n) - 0.5;
        let b = vec2i(floor((cell + o * fw) / B));
        let q = pyrCell(l, b.x, b.y);
        if (q.x >= m.x) { m = q; }
      }
    }
  } else {
    let n = u32(clamp(ceil(foot), 1.0, f32(max(P.maxSamples, 1u))));
    bf = f32(n);
    for (var sy = 0u; sy < n; sy++) {
      for (var sx = 0u; sx < n; sx++) {
        let o = (vec2f(f32(sx), f32(sy)) + 0.5) / f32(n) - 0.5;
        let c2 = cell + o * fw;
        let cc = vec2u(clamp(floor(c2), vec2f(0.0), vec2f(P.cols - 1.0, P.rows - 1.0)));
        let v = cellValue(P, cc.y, cc.x);
        if (abs(v) >= m.x) { m = vec2f(abs(v), v); }
      }
    }
  }
  let mag = m.x / (P.gain * boost(bf));
  // a sign means something for one cell; mixed over many it does not: far away the light is neutral
  let hue = mix(hueOf(m), BONE * 0.9, smoothstep(0.8, 2.5, foot));
  var col = hue * (1.0 - exp(-mag * 1.6)) + hue * max(0.0, mag - 1.0) * 0.25 * P.lift;
  // before the sweep: done (full); after it: waiting (faint); at it: a bright edge
  if ((P.flags & SWEEP) != 0u) {
    let d = cell.x - P.sweep;
    col *= select(1.0, 0.16, d > 0.0);
    col += BONE * 0.9 * (1.0 - smoothstep(0.0, max(fw.x * 2.0, 0.2), abs(d)));
  }
  if (px > 5.0 && (P.flags & GRID) != 0u) {
    let e = min(min(local.x, 1.0 - local.x), min(local.y, 1.0 - local.y));
    col *= smoothstep(0.0, max(0.06, 1.0 * foot), e) * 0.85 + 0.15;
  }
  if ((P.flags & DIGITS) != 0u && px > 34.0) {
    let t = printNum(exact, local, 1.0, 0.3, dpx, dpy);
    col = mix(col * 0.55, BONE * 1.05, t);
  }
  return vec4f(col * P.alpha * P.tint, 1.0);
}

// a hand of full length along v (its direction is what the turning changes; its size shows as brightness)
fn fn_hand(q: vec2f, v: vec2f, pw: f32) -> f32 {
  if (length(v) < 1e-9) { return 0.0; }
  let L = 0.82;
  let dir = normalize(v);
  let t = clamp(dot(q, dir), 0.0, L);
  let d = length(q - dir * t);
  return 1.0 - smoothstep(pw * 0.6, pw * 1.8, d);
}

@fragment fn fsOpaque(i: VO) -> @location(0) vec4f {
  let c = shade(i);
  if (c.a < 0.0) { discard; }
  return vec4f(c.rgb, 1.0);
}
@fragment fn fsAdd(i: VO) -> @location(0) vec4f {
  let c = shade(i);
  if (c.a < 0.0) { discard; }
  return vec4f(c.rgb, 0.0);
}`;

// pyramid level 1 from the cells; then each level from the one before (8 x 8 blocks, means)
const PYR_WGSL = /* wgsl */ `
@group(0) @binding(1) var<uniform> PU: Panel;
@group(0) @binding(2) var<storage, read> CODES: array<u32>;
@group(0) @binding(3) var<storage, read> SCALES: array<f32>;
@group(0) @binding(4) var<storage, read> ARENA: array<f32>;
@group(0) @binding(5) var<storage, read> AUX: array<f32>;
@group(0) @binding(6) var<storage, read_write> PYR: array<u32>;
@group(0) @binding(7) var<storage, read> IDX: array<u32>;
@group(0) @binding(10) var<storage, read> CODES2: array<u32>;
@group(0) @binding(11) var<storage, read> SCALES2: array<f32>;
@group(0) @binding(12) var<storage, read> DW: array<f32>;
@group(0) @binding(13) var<storage, read> CAP: array<f32>;
${COMMON}
@compute @workgroup_size(8, 8)
fn level1(@builtin(global_invocation_id) g: vec3u) {
  let P = PU;
  if (g.x >= P.w1 || g.y >= P.h1) { return; }
  let R = u32(P.rows); let C = u32(P.cols);
  var sa = 0.0; var pk = 0.0; var n = 0.0;
  for (var y = 0u; y < 8u; y++) {
    let r = g.y * 8u + y;
    if (r >= R) { break; }
    for (var x = 0u; x < 8u; x++) {
      let c = g.x * 8u + x;
      if (c >= C) { break; }
      let v = cellValue(P, r, c);
      sa += abs(v); n += 1.0;
      if (abs(v) > abs(pk)) { pk = v; }
    }
  }
  // (peak, mean |v|)
  PYR[P.pyr + g.y * P.w1 + g.x] = pack2x16float(vec2f(pk, sa / max(n, 1.0)));
}
`;

const UP_WGSL = /* wgsl */ `
struct Up { src: u32, dst: u32, sw: u32, sh: u32, dw: u32, dh: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> U: Up;
@group(0) @binding(1) var<storage, read_write> PYR: array<u32>;
@compute @workgroup_size(8, 8)
fn up(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.dw || g.y >= U.dh) { return; }
  var pk = 0.0; var sa = 0.0; var n = 0.0;
  for (var y = 0u; y < 8u; y++) {
    let r = g.y * 8u + y;
    if (r >= U.sh) { break; }
    for (var x = 0u; x < 8u; x++) {
      let c = g.x * 8u + x;
      if (c >= U.sw) { break; }
      let q = unpack2x16float(PYR[U.src + r * U.sw + c]);
      if (abs(q.x) > abs(pk)) { pk = q.x; }
      sa += q.y; n += 1.0;
    }
  }
  PYR[U.dst + g.y * U.dw + g.x] = pack2x16float(vec2f(pk, sa / max(n, 1.0)));
}
`;

export class Panels {
	readonly aux: GPUBuffer;
	readonly idx: GPUBuffer;
	private painterBufs?: { codes: GPUBuffer; scales: GPUBuffer; dense: GPUBuffer };
	private capBuf?: GPUBuffer;
	private dummy: GPUBuffer;
	private device: GPUDevice;
	private slots: Slot[] = [];
	private uniforms: GPUBuffer;
	private pyrBuf?: GPUBuffer;
	private pyrSize = 0;
	private opaque: GPURenderPipeline;
	private additive: GPURenderPipeline;
	private level1: GPUComputePipeline;
	private upPipe: GPUComputePipeline;
	private bind?: GPUBindGroup;
	private computeBind?: GPUBindGroup;
	private upBind?: GPUBindGroup;
	private upUniforms: GPUBuffer;
	private glyphs: GPUTexture;
	private sampler: GPUSampler;
	private layout: GPUBindGroupLayout;
	private computeLayout: GPUBindGroupLayout;
	private data: ArrayBuffer;

	constructor(
		private stage: Stage,
		private llm: BonsaiLLM,
		opts: { capacity: number; aux?: number; idx?: number }
	) {
		const device = (this.device = stage.device);
		this.aux = device.createBuffer({
			label: 'lab scratch',
			size: Math.max(16, (opts.aux ?? 4) * 4),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});
		this.idx = device.createBuffer({
			label: 'lab index',
			size: Math.max(16, (opts.idx ?? 4) * 4),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.dummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
		this.uniforms = device.createBuffer({
			size: opts.capacity * STRIDE,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.upUniforms = device.createBuffer({
			size: 4096 * STRIDE,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.data = new ArrayBuffer(opts.capacity * STRIDE);
		this.glyphs = glyphTexture(device);
		this.sampler = device.createSampler({
			magFilter: 'linear',
			minFilter: 'linear',
			mipmapFilter: 'linear'
		});
		const ro = (binding: number, vis: number): GPUBindGroupLayoutEntry => ({
			binding,
			visibility: vis,
			buffer: { type: 'read-only-storage' }
		});
		const V = GPUShaderStage.VERTEX,
			Fr = GPUShaderStage.FRAGMENT,
			C = GPUShaderStage.COMPUTE;
		this.layout = device.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: V | Fr, buffer: { type: 'uniform' } },
				{
					binding: 1,
					visibility: V | Fr,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: STRIDE }
				},
				ro(2, Fr),
				ro(3, Fr),
				ro(4, Fr),
				ro(5, Fr),
				ro(6, Fr),
				ro(7, V | Fr),
				{ binding: 8, visibility: Fr, texture: {} },
				{ binding: 9, visibility: Fr, sampler: {} },
				ro(10, Fr),
				ro(11, Fr),
				ro(12, Fr),
				ro(13, Fr)
			]
		});
		const module = device.createShaderModule({ label: 'panels', code: RENDER_WGSL });
		const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
		const pipe = (entry: string, add: boolean) =>
			device.createRenderPipeline({
				layout: pl,
				vertex: { module, entryPoint: 'vs' },
				fragment: {
					module,
					entryPoint: entry,
					targets: [
						{
							format: HDR_FORMAT,
							blend: add
								? {
										color: { srcFactor: 'one', dstFactor: 'one' },
										alpha: { srcFactor: 'one', dstFactor: 'one' }
									}
								: undefined
						}
					]
				},
				primitive: { topology: 'triangle-list' },
				depthStencil: {
					format: DEPTH_FORMAT,
					depthWriteEnabled: !add,
					depthCompare: 'less'
				}
			});
		this.opaque = pipe('fsOpaque', false);
		this.additive = pipe('fsAdd', true);
		this.computeLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 1,
					visibility: C,
					buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: STRIDE }
				},
				ro(2, C),
				ro(3, C),
				ro(4, C),
				ro(5, C),
				{ binding: 6, visibility: C, buffer: { type: 'storage' } },
				ro(7, C),
				ro(10, C),
				ro(11, C),
				ro(12, C),
				ro(13, C)
			]
		});
		this.level1 = device.createComputePipeline({
			layout: device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }),
			compute: {
				module: device.createShaderModule({ label: 'pyramid', code: PYR_WGSL }),
				entryPoint: 'level1'
			}
		});
		this.upPipe = device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: device.createShaderModule({ label: 'pyramid up', code: UP_WGSL }),
				entryPoint: 'up'
			}
		});
	}

	get count() {
		return this.slots.length;
	}

	/** The painter's weights (ternary codes and scales, dense weights), for panels that show them. */
	setPainter(w: { codes: GPUBuffer; scales: GPUBuffer; dense: GPUBuffer }) {
		this.painterBufs = w;
		this.bind = this.computeBind = undefined;
	}

	/** A painting's capture (inputs of the painter's matrices), for panels that read it. */
	setCapture(buf: GPUBuffer | undefined) {
		this.capBuf = buf;
		this.bind = this.computeBind = undefined;
	}

	desc(i: number): PanelDesc {
		return this.slots[i].d;
	}

	/** Add a panel; returns its index. Call allocate() after adding panels that want a pyramid. */
	add(d: PanelDesc): number {
		const w1 = Math.ceil(d.cols / 8),
			h1 = Math.ceil(d.rows / 8);
		this.slots.push({ d, pyr: NONE, w1, h1, dirty: true });
		return this.slots.length - 1;
	}

	/** Change a panel. */
	set(i: number, d: Partial<PanelDesc>) {
		const s = this.slots[i];
		Object.assign(s.d, d);
		s.dirty = true;
	}

	/** Remove every panel (the buffers stay). */
	clear() {
		this.slots = [];
	}

	/** Make room for the pyramids of the panels that want one (keeps the buffer if it is big enough). */
	allocate() {
		let at = 0;
		for (const s of this.slots) {
			if (!s.d.pyramid || s.d.kind !== 'ternary') {
				s.pyr = NONE;
				continue;
			}
			s.pyr = at;
			let w = s.w1,
				h = s.h1;
			for (let l = 1; l <= 3; l++) {
				at += w * h;
				w = Math.ceil(w / 8);
				h = Math.ceil(h / 8);
			}
			s.dirty = true;
		}
		const size = Math.max(16, at * 4);
		if (!this.pyrBuf || this.pyrSize < size) {
			this.pyrBuf?.destroy();
			this.pyrBuf = this.device.createBuffer({
				label: 'pyramids',
				size,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
			});
			this.pyrSize = size;
			this.bind = this.computeBind = this.upBind = undefined;
		}
	}

	private ensureBinds() {
		if (this.bind) return;
		const { codes, scales } = this.llm.weightBuffers;
		this.pyrBuf ??= this.device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		});
		const buf = (binding: number, buffer: GPUBuffer, size?: number) => ({
			binding,
			resource: { buffer, size }
		});
		this.bind = this.device.createBindGroup({
			layout: this.layout,
			entries: [
				buf(0, this.stage.frame),
				buf(1, this.uniforms, STRIDE),
				buf(2, codes),
				buf(3, scales),
				buf(4, this.llm.arena),
				buf(5, this.aux),
				buf(6, this.pyrBuf),
				buf(7, this.idx),
				{ binding: 8, resource: this.glyphs.createView() },
				{ binding: 9, resource: this.sampler },
				buf(10, this.painterBufs?.codes ?? this.dummy),
				buf(11, this.painterBufs?.scales ?? this.dummy),
				buf(12, this.painterBufs?.dense ?? this.dummy),
				buf(13, this.capBuf ?? this.dummy)
			]
		});
		this.computeBind = this.device.createBindGroup({
			layout: this.computeLayout,
			entries: [
				buf(1, this.uniforms, STRIDE),
				buf(2, codes),
				buf(3, scales),
				buf(4, this.llm.arena),
				buf(5, this.aux),
				buf(6, this.pyrBuf),
				buf(7, this.idx),
				buf(10, this.painterBufs?.codes ?? this.dummy),
				buf(11, this.painterBufs?.scales ?? this.dummy),
				buf(12, this.painterBufs?.dense ?? this.dummy),
				buf(13, this.capBuf ?? this.dummy)
			]
		});
		this.upBind = this.device.createBindGroup({
			layout: this.upPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.upUniforms, size: 32 } },
				buf(1, this.pyrBuf)
			]
		});
	}

	private write(i: number) {
		const s = this.slots[i],
			d = s.d;
		const f = new Float32Array(this.data, i * STRIDE, STRIDE / 4);
		const u = new Uint32Array(this.data, i * STRIDE, STRIDE / 4);
		f.set([...d.origin, d.rows, ...d.ax, d.cols, ...d.ay], 0);
		f[11] = { ternary: 0, dense: 1, bars: 2, dial: 3, frame: 4 }[d.kind];
		const t = d.tensor;
		let flags = 0;
		if (d.transpose) flags |= 1;
		if (d.x) flags |= 2 | (d.x.aux ? 4 : 0);
		if (d.permR !== undefined) flags |= 8;
		if (d.permC !== undefined) flags |= 16;
		if (d.digits) flags |= 32;
		if (d.grid ?? true) flags |= 64;
		if (d.b) flags |= 128 | (d.b.aux ? 512 : 0) | (d.b.dw ? 8192 : 0) | (d.b.cap ? 65536 : 0);
		if (d.a?.aux) flags |= 256;
		if (d.a?.dw) flags |= 4096;
		if (d.a?.cap) flags |= 32768;
		if (d.x?.cap) flags |= 16384;
		if (d.painter) flags |= 2048;
		if (d.sweep !== undefined && d.sweep !== null) flags |= 1024;
		u.set([t?.codes ?? 0, t?.scales ?? 0, t?.cols ?? 0, flags], 12);
		u.set([d.x?.offset ?? 0, d.permR ?? 0, d.permC ?? 0, s.pyr], 16);
		u.set([d.a?.offset ?? 0, d.a?.rs ?? 0, d.a?.cs ?? 0, d.b?.offset ?? 0], 20);
		u.set([d.b?.rs ?? 0, d.b?.cs ?? 0, s.w1, s.h1], 24);
		f.set([d.gain || 1e-9, d.alpha ?? 1, d.sweep ?? 0, d.sweepStride ?? 0], 28);
		f.set([...(d.tint ?? [1, 1, 1]), d.lift ?? 1], 32);
		const cut = d.cut ?? [0, 0, d.cols, d.rows];
		f.set(cut, 36);
		u.set([d.m0 ?? 0, d.k0 ?? 0, d.samples ?? 4, 0], 40);
		const ins = d.instances;
		f.set([...(ins?.step ?? [0, 0, 0]), ins?.count ?? 1], 44);
		const iflags =
			(ins?.causal ? 1 : 0) | (ins?.table !== undefined ? 2 : 0) | (ins?.tableA ? 4 : 0);
		u.set([ins?.x ?? 0, ins?.a ?? 0, ins?.b ?? 0, iflags], 48);
		u.set([d.a?.div ?? 0, d.a?.ds ?? 0, ins?.table ?? 0, 0], 52);
		s.dirty = false;
	}

	private flush() {
		let lo = Infinity,
			hi = -1;
		this.slots.forEach((s, i) => {
			if (s.dirty) {
				this.write(i);
				lo = Math.min(lo, i);
				hi = Math.max(hi, i);
			}
		});
		if (hi >= 0)
			this.device.queue.writeBuffer(
				this.uniforms,
				lo * STRIDE,
				this.data,
				lo * STRIDE,
				(hi - lo + 1) * STRIDE
			);
	}

	/**
	 * Record the pyramids of the given panels (all with one by default), from the values as they are now (weights,
	 * or weights times the current inputs).
	 */
	buildPyramids(enc: GPUCommandEncoder, which?: number[]) {
		this.ensureBinds();
		this.flush();
		const list = (which ?? this.slots.map((_, i) => i)).filter((i) => this.slots[i].pyr !== NONE);
		if (!list.length) return;
		const pass = enc.beginComputePass();
		pass.setPipeline(this.level1);
		for (const i of list) {
			const s = this.slots[i];
			pass.setBindGroup(0, this.computeBind!, [i * STRIDE]);
			pass.dispatchWorkgroups(Math.ceil(s.w1 / 8), Math.ceil(s.h1 / 8));
		}
		pass.end();
		// levels 2 and 3, each from the one before (a uniform slot per dispatch)
		const ups: { src: number; dst: number; sw: number; sh: number; dw: number; dh: number }[] = [];
		for (const i of list) {
			const s = this.slots[i];
			let w = s.w1,
				h = s.h1,
				off = s.pyr;
			for (let l = 1; l < 3; l++) {
				const dw = Math.ceil(w / 8),
					dh = Math.ceil(h / 8);
				ups.push({ src: off, dst: off + w * h, sw: w, sh: h, dw, dh });
				off += w * h;
				w = dw;
				h = dh;
			}
		}
		for (let start = 0; start < ups.length; start += 4096) {
			const chunk = ups.slice(start, start + 4096);
			const data = new Uint32Array((chunk.length * STRIDE) / 4);
			chunk.forEach((u, j) => data.set([u.src, u.dst, u.sw, u.sh, u.dw, u.dh], (j * STRIDE) / 4));
			this.device.queue.writeBuffer(this.upUniforms, 0, data);
			// level 2 of every panel before level 3 (level 3 reads level 2)
			for (const step of [0, 1]) {
				const p = enc.beginComputePass();
				p.setPipeline(this.upPipe);
				chunk.forEach((u, j) => {
					if (j % 2 !== step) return;
					const bg = this.device.createBindGroup({
						layout: this.upPipe.getBindGroupLayout(0),
						entries: [
							{ binding: 0, resource: { buffer: this.upUniforms, offset: j * STRIDE, size: 32 } },
							{ binding: 1, resource: { buffer: this.pyrBuf! } }
						]
					});
					p.setBindGroup(0, bg);
					p.dispatchWorkgroups(Math.ceil(u.dw / 8), Math.ceil(u.dh / 8));
				});
				p.end();
			}
			if (start + 4096 < ups.length) throw new Error('Too many pyramids for one build.');
		}
	}

	/**
	 * Mean |v| of each listed panel's values, from its pyramid (level 2, or level 1 when level 2 is tiny). Resolves
	 * after the GPU has built them.
	 */
	async meanAbs(which: number[]): Promise<number[]> {
		const list = which.filter((i) => this.slots[i].pyr !== NONE);
		const regions = list.map((i) => {
			const s = this.slots[i];
			const w2 = Math.ceil(s.w1 / 8),
				h2 = Math.ceil(s.h1 / 8);
			return w2 * h2 >= 16
				? { off: s.pyr + s.w1 * s.h1, n: w2 * h2 }
				: { off: s.pyr, n: s.w1 * s.h1 };
		});
		const total = regions.reduce((a, r) => a + r.n, 0);
		if (!total) return which.map(() => 0);
		const rb = this.device.createBuffer({
			size: total * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = this.device.createCommandEncoder();
		let p = 0;
		for (const r of regions) {
			enc.copyBufferToBuffer(this.pyrBuf!, r.off * 4, rb, p * 4, r.n * 4);
			p += r.n;
		}
		this.device.queue.submit([enc.finish()]);
		await rb.mapAsync(GPUMapMode.READ);
		const u16 = new Uint16Array(rb.getMappedRange().slice(0));
		rb.destroy();
		const out = new Map<number, number>();
		p = 0;
		list.forEach((i, j) => {
			let s = 0;
			const n = regions[j].n;
			for (let k = 0; k < n; k++) s += half(u16[(p + k) * 2 + 1]);
			out.set(i, s / n);
			p += n;
		});
		return which.map((i) => out.get(i) ?? 0);
	}

	draw(pass: GPURenderPassEncoder) {
		if (!this.slots.length) return;
		this.ensureBinds();
		this.flush();
		for (const add of [false, true]) {
			pass.setPipeline(add ? this.additive : this.opaque);
			this.slots.forEach((s, i) => {
				if ((s.d.additive ?? false) !== add || s.d.visible === false || (s.d.alpha ?? 1) <= 0.001)
					return;
				pass.setBindGroup(0, this.bind!, [i * STRIDE]);
				pass.draw(6, s.d.instances?.count ?? 1);
			});
		}
	}

	/** The cell of a panel under a ray, nearest first: panel index and (row, col). */
	pick(ray: { o: V3; d: V3 }): { panel: number; row: number; col: number; dist: number } | null {
		let best: { panel: number; row: number; col: number; dist: number } | null = null;
		this.slots.forEach((s, i) => {
			const d = s.d;
			if (d.visible === false || (d.alpha ?? 1) <= 0.01) return;
			const n = cross(d.ax, d.ay);
			const den = dot(n, ray.d);
			if (Math.abs(den) < 1e-9) return;
			const t = dot(n, sub(d.origin, ray.o)) / den;
			if (t <= 0 || (best && t >= best.dist)) return;
			const p = sub(add3(ray.o, scale(ray.d, t)), d.origin);
			const u = dot(p, d.ax) / dot(d.ax, d.ax),
				v = dot(p, d.ay) / dot(d.ay, d.ay);
			if (u < 0 || u >= 1 || v < 0 || v >= 1) return;
			best = { panel: i, row: Math.floor(v * d.rows), col: Math.floor(u * d.cols), dist: t };
		});
		return best;
	}

	destroy() {
		this.aux.destroy();
		this.idx.destroy();
		this.dummy.destroy();
		this.uniforms.destroy();
		this.upUniforms.destroy();
		this.pyrBuf?.destroy();
		this.glyphs.destroy();
	}
}

function half(h: number) {
	const s = h & 0x8000 ? -1 : 1,
		e = (h >> 10) & 0x1f,
		f = h & 0x3ff;
	if (e === 0) return s * f * 2 ** -24;
	if (e === 31) return f ? NaN : s * Infinity;
	return s * (1 + f / 1024) * 2 ** (e - 15);
}
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const cross = (a: V3, b: V3): V3 => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0]
];
