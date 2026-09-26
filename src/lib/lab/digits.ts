// Numbers drawn inside shaders: a strip of glyphs (0-9 . − + e and a blank) rendered once with the piece's typeface,
// with smaller copies for distance, and WGSL that prints a float in a box with three significant figures. Any cell
// of any matrix can then show its exact value when it is big enough on screen.
import { FONT_FAMILY } from '$lib/engine/text';

export const GLYPHS = '0123456789.−+e ×';
const CW = 64,
	CH = 96;

/** The glyph strip (16 glyphs, CW x CH each) with mip levels, white on black. */
export function glyphTexture(device: GPUDevice): GPUTexture {
	const W = CW * 16,
		H = CH;
	const levels = 5;
	const tex = device.createTexture({
		size: [W, H],
		format: 'rgba8unorm',
		mipLevelCount: levels,
		usage:
			GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
	});
	const base = new OffscreenCanvas(W, H);
	const ctx = base.getContext('2d')!;
	ctx.fillStyle = '#000';
	ctx.fillRect(0, 0, W, H);
	ctx.fillStyle = '#fff';
	ctx.textAlign = 'center';
	ctx.textBaseline = 'middle';
	ctx.font = `400 ${Math.round(CH * 0.78)}px ${FONT_FAMILY}`;
	for (let i = 0; i < 16; i++) ctx.fillText(GLYPHS[i], i * CW + CW / 2, CH * 0.54);
	for (let l = 0; l < levels; l++) {
		const w = Math.max(1, W >> l),
			h = Math.max(1, H >> l);
		const c = new OffscreenCanvas(w, h);
		const g = c.getContext('2d')!;
		g.imageSmoothingQuality = 'high';
		g.drawImage(base, 0, 0, w, h);
		device.queue.copyExternalImageToTexture({ source: c }, { texture: tex, mipLevel: l }, [w, h]);
	}
	return tex;
}

/**
 * WGSL: glyph coverage and number printing. Needs `glyphs: texture_2d<f32>` and `gsamp: sampler` bound, and is
 * called with the derivatives of the box coordinates (taken in uniform control flow).
 */
export const DIGITS_WGSL = /* wgsl */ `
// coverage of glyph ch at p in [0,1]^2 of its box (y down); d = derivatives of p (x, y)
fn glyph(ch: u32, p: vec2f, dx: vec2f, dy: vec2f) -> f32 {
  if (ch >= 16u || ch == 14u || p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { return 0.0; }
  let uv = vec2f((f32(ch) + p.x) / 16.0, p.y);
  return textureSampleGrad(glyphs, gsamp, uv, dx / vec2f(16.0, 1.0), dy / vec2f(16.0, 1.0)).r;
}

// v with three significant figures: chars into s, returns how many
fn fmtNum(v: f32, s: ptr<function, array<u32, 9>>) -> u32 {
  let a = abs(v);
  (*s)[0] = select(select(14u, 12u, v > 0.0), 11u, v < 0.0);
  if (a < 1e-30) { (*s)[0] = 0u; return 1u; }
  var e = i32(floor(log2(a) * 0.30102999566));
  var m = u32(round(a / pow(10.0, f32(e - 2))));
  if (m < 100u) { e = e - 1; m = u32(round(a / pow(10.0, f32(e - 2)))); }
  if (m >= 1000u) { e = e + 1; m = u32(round(a / pow(10.0, f32(e - 2)))); }
  m = clamp(m, 100u, 999u);
  let d0 = m / 100u; let d1 = (m / 10u) % 10u; let d2 = m % 10u;
  var n = 1u;
  if (e >= 4 || e <= -3) {
    (*s)[1] = d0; (*s)[2] = 10u; (*s)[3] = d1; (*s)[4] = 13u; n = 5u;
    if (e < 0) { (*s)[n] = 11u; n++; }
    let ae = u32(abs(e));
    if (ae >= 10u) { (*s)[n] = ae / 10u; n++; }
    (*s)[n] = ae % 10u; n++;
    return n;
  }
  if (e == 3) { (*s)[1] = d0; (*s)[2] = d1; (*s)[3] = d2; (*s)[4] = 0u; return 5u; }
  if (e == 2) { (*s)[1] = d0; (*s)[2] = d1; (*s)[3] = d2; return 4u; }
  if (e == 1) { (*s)[1] = d0; (*s)[2] = d1; (*s)[3] = 10u; (*s)[4] = d2; return 5u; }
  if (e == 0) { (*s)[1] = d0; (*s)[2] = 10u; (*s)[3] = d1; (*s)[4] = d2; return 5u; }
  if (e == -1) { (*s)[1] = 10u; (*s)[2] = d0; (*s)[3] = d1; (*s)[4] = d2; return 5u; }
  (*s)[1] = 10u; (*s)[2] = 0u; (*s)[3] = d0; (*s)[4] = d1; (*s)[5] = d2; return 6u;
}

// coverage of v printed centred in a box: p in [0,1]^2 of the box (y down), aspect = box width / height,
// size = text height as a fraction of the box height
fn printNum(v: f32, p: vec2f, aspect: f32, size: f32, dx: vec2f, dy: vec2f) -> f32 {
  var s: array<u32, 9>;
  let n = fmtNum(v, &s);
  // glyph box: height size, width 0.62 of it (in box-height units), fitted to the box width
  var gh = size;
  var gw = gh * 0.62;
  let room = aspect * 0.92;
  if (gw * f32(n) > room) { gw = room / f32(n); gh = gw / 0.62; }
  let q = vec2f(p.x * aspect, p.y); // box-height units
  let x0 = (aspect - gw * f32(n)) * 0.5;
  let y0 = 0.5 - gh * 0.5;
  let gx = (q.x - x0) / gw;
  if (gx < 0.0 || gx >= f32(n)) { return 0.0; }
  let i = u32(gx);
  let local = vec2f(fract(gx), (q.y - y0) / gh);
  let k = vec2f(aspect / gw, 1.0 / gh);
  return glyph(s[i], local, dx * k, dy * k);
}
`;
