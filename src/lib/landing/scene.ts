// The landing's picture of one painting, drawn in 2D from a LandingRun at time t (seconds). Everything shown at t has
// been computed by then: a layer or a block appears when it completes, never while it is still to come.
//   reading   9 layers of the reader; lines join each word to the earlier words it attends to (mean over heads)
//   handing   the painter takes the reader's layers 3, 6 and 9 as its text; the picture starts as noise
//   sketch    pass 1, 25 blocks at 16 x 16 patches; under each word, where the picture attends to it
//   renoise   the sketch, enlarged and noised back for the second pass
//   paint     pass 2, 25 blocks at 32 x 32
//   done      the picture
import type { LandingRun } from './run';

export type Layout = 'line' | 'ring';

const LAYER = 1.2,
	HAND = 2.2,
	BLOCK = 0.36,
	REVEAL = 1.2,
	RENOISE = 1.2,
	HOLD = 4;
const T_READ = 9 * LAYER,
	T_PASS1 = T_READ + HAND,
	T_REVEAL = T_PASS1 + 25 * BLOCK,
	T_RENOISE = T_REVEAL + REVEAL,
	T_PASS2 = T_RENOISE + RENOISE,
	T_DONE = T_PASS2 + 25 * BLOCK;
export const DURATION = T_DONE + REVEAL + HOLD;

const C = {
	void: '#000000',
	bone: '#e8e2d6',
	ash: '#7d848c',
	glacier: [86, 200, 255],
	ember: [255, 176, 74]
};
const SERIF = "'Spectral', Georgia, serif";

type Phase =
	| { k: 'read'; layer: number; u: number }
	| { k: 'hand'; u: number }
	| { k: 'pass'; pass: 0 | 1; block: number; u: number }
	| { k: 'reveal'; u: number }
	| { k: 'renoise'; u: number }
	| { k: 'done'; u: number };

export function phaseAt(t: number): Phase {
	if (t < T_READ) {
		const l = Math.min(8, Math.floor(t / LAYER));
		return { k: 'read', layer: l, u: t / LAYER - l };
	}
	if (t < T_PASS1) return { k: 'hand', u: (t - T_READ) / HAND };
	if (t < T_REVEAL) {
		const b = Math.min(24, Math.floor((t - T_PASS1) / BLOCK));
		return { k: 'pass', pass: 0, block: b, u: (t - T_PASS1) / BLOCK - b };
	}
	if (t < T_RENOISE) return { k: 'reveal', u: (t - T_REVEAL) / REVEAL };
	if (t < T_PASS2) return { k: 'renoise', u: (t - T_RENOISE) / RENOISE };
	if (t < T_DONE) {
		const b = Math.min(24, Math.floor((t - T_PASS2) / BLOCK));
		return { k: 'pass', pass: 1, block: b, u: (t - T_PASS2) / BLOCK - b };
	}
	return { k: 'done', u: Math.min(1, (t - T_DONE) / REVEAL) };
}

const ease = (x: number) => {
	const c = Math.min(1, Math.max(0, x));
	return c * c * (3 - 2 * c);
};
const rgba = (c: number[], a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/** The prompt's tokens as shown: a leading space starts a word; a piece of a word gets a thin mark before it. */
function label(tok: string, first: boolean) {
	return tok.startsWith(' ') ? tok.slice(1) : !first && tok.match(/^[a-z0-9]/i) ? '·' + tok : tok;
}

/**
 * The reader's lines at a layer: from each prompt token to the earlier prompt tokens it attends to most (its attention
 * renormalised over them), at most three per token and none under 0.12.
 */
function readerEdges(run: LandingRun, layer: number) {
	const edges: { from: number; to: number; w: number }[] = [];
	for (let i = run.first + 1; i < run.end; i++) {
		const row = run.reader[layer][i];
		let sum = 0;
		for (let j = run.first; j < i; j++) sum += row[j];
		if (sum <= 0) continue;
		const mine = [];
		for (let j = run.first; j < i; j++) mine.push({ from: j, to: i, w: row[j] / sum });
		mine.sort((a, b) => b.w - a.w);
		edges.push(...mine.slice(0, 3).filter((e) => e.w >= 0.12));
	}
	return edges;
}

/** How much the picture reads each prompt token at (pass, block): mean attention over image rows, max 1. */
function masses(run: LandingRun, pass: 0 | 1, block: number) {
	const m: number[] = [];
	for (let k = run.first; k < run.end; k++) {
		const a = run.maps[pass][block][k];
		let s = 0;
		for (let i = 0; i < a.length; i++) s += a[i];
		m.push(s / a.length);
	}
	const top = Math.max(...m);
	return m.map((x) => 0.55 + 0.45 * Math.sqrt(x / top));
}

/** A token's map as an image (ember on black), normalised to its own 98th percentile. */
const mapCache = new WeakMap<Float32Array, OffscreenCanvas>();
function mapImage(a: Float32Array, side: number) {
	let c = mapCache.get(a);
	if (c) return c;
	const sorted = Float32Array.from(a).sort();
	const top = sorted[Math.floor(sorted.length * 0.98)] || 1;
	c = new OffscreenCanvas(side, side);
	const px = new ImageData(side, side);
	for (let i = 0; i < a.length; i++) {
		const v = Math.min(1, a[i] / top) ** 0.9;
		// black -> ember -> bone at the peaks
		const hot = Math.max(0, v - 0.7) / 0.3;
		px.data[i * 4] = C.ember[0] * v * (1 - hot) + 232 * hot;
		px.data[i * 4 + 1] = C.ember[1] * v * (1 - hot) + 226 * hot;
		px.data[i * 4 + 2] = C.ember[2] * v * (1 - hot) + 214 * hot;
		px.data[i * 4 + 3] = 255;
	}
	c.getContext('2d')!.putImageData(px, 0, 0);
	mapCache.set(a, c);
	return c;
}

interface Geometry {
	tokens: { x: number; y: number }[]; // label centres, prompt tokens only
	tiles: { x: number; y: number; s: number; round: boolean }[];
	picture: { x: number; y: number; s: number };
	font: number;
	arc: (ctx: CanvasRenderingContext2D, a: number, b: number) => void; // a path from token a to token b
}

/** Where things go; rise (0..1) moves the line of words up from the middle to make room for the painter. */
function geometry(run: LandingRun, layout: Layout, w: number, h: number, rise: number): Geometry {
	const n = run.end - run.first;
	if (layout === 'line') {
		const margin = Math.max(24, w * 0.05);
		const sw = (w - 2 * margin) / n;
		const font = Math.min(26, sw * 0.3);
		const settled = h * 0.3;
		const y = h * 0.52 + (settled - h * 0.52) * rise;
		const ts = Math.min(sw - 8, 96);
		const tokens = Array.from({ length: n }, (_, k) => ({ x: margin + sw * (k + 0.5), y }));
		const tiles = tokens.map((p) => ({
			x: p.x,
			y: y + font * 0.9 + ts / 2 + 8,
			s: ts,
			round: false
		}));
		const top = settled + font * 0.9 + ts + 44;
		const s = Math.min(440, h - top - 110, w * 0.6);
		return {
			tokens,
			tiles,
			picture: { x: w / 2, y: top + s / 2, s },
			font,
			arc: (ctx, a, b) => {
				const p = tokens[a],
					q = tokens[b];
				// higher for farther words, rising ever more slowly so the long lines do not flatten into a ceiling
				const H = h * 0.24;
				const lift = H * (1 - Math.exp(-Math.abs(q.x - p.x) / (H * 1.6)));
				ctx.moveTo(p.x, p.y - font * 0.95);
				ctx.quadraticCurveTo((p.x + q.x) / 2, p.y - font * 0.95 - lift * 2, q.x, q.y - font * 0.95);
			}
		};
	}
	const cx = w / 2,
		cy = h * 0.47;
	const s = Math.min(w, h) * 0.4;
	const R = s * 0.72 + 30,
		R2 = R + Math.min(w, h) * 0.075;
	const span = Math.PI * 1.55;
	const ang = (k: number) => -Math.PI / 2 - span / 2 + (span * k) / (n - 1);
	const font = Math.min(24, (R * span) / n / 3.2);
	const tokens = Array.from({ length: n }, (_, k) => ({
		x: cx + R * Math.cos(ang(k)),
		y: cy + R * Math.sin(ang(k))
	}));
	const ds = Math.min(84, ((R2 * span) / n) * 0.86);
	const tiles = Array.from({ length: n }, (_, k) => ({
		x: cx + R2 * Math.cos(ang(k)),
		y: cy + R2 * Math.sin(ang(k)),
		s: ds,
		round: true
	}));
	const Ri = R - font * 1.1;
	return {
		tokens,
		tiles,
		picture: { x: cx, y: cy, s },
		font,
		arc: (ctx, a, b) => {
			const pa = [cx + Ri * Math.cos(ang(a)), cy + Ri * Math.sin(ang(a))];
			const pb = [cx + Ri * Math.cos(ang(b)), cy + Ri * Math.sin(ang(b))];
			const mx = (pa[0] + pb[0]) / 2,
				my = (pa[1] + pb[1]) / 2;
			ctx.moveTo(pa[0], pa[1]);
			ctx.quadraticCurveTo(cx + (mx - cx) * 0.3, cy + (my - cy) * 0.3, pb[0], pb[1]);
		}
	};
}

function caption(p: Phase): string {
	switch (p.k) {
		case 'read':
			return `Reading your words, layer ${p.layer + 1} of 9. Each line joins a word to an earlier one it attends to.`;
		case 'hand':
			return 'The painter takes what the reader made of your words at layers 3, 6 and 9.';
		case 'pass':
			return p.pass === 0
				? `Sketching at a quarter size, block ${p.block + 1} of 25. Under each word: where the picture attends to it.`
				: `Painting over the sketch at full size, block ${p.block + 1} of 25.`;
		case 'reveal':
			return 'The sketch.';
		case 'renoise':
			return 'The sketch is enlarged and mixed with fresh noise: the second pass paints over it.';
		case 'done':
			return 'Painted in two passes. Change the words and paint it again.';
	}
}

/** Draw the scene at time t into a canvas of CSS size w x h (the context already scaled for the pixel ratio). */
export function drawScene(
	ctx: CanvasRenderingContext2D,
	run: LandingRun,
	t: number,
	layout: Layout,
	w: number,
	h: number
) {
	const p = phaseAt(t);
	const rise = p.k === 'read' ? 0 : p.k === 'hand' ? ease(p.u / 0.6) : 1;
	const g = geometry(run, layout, w, h, rise);
	const n = run.end - run.first;
	ctx.fillStyle = C.void;
	ctx.fillRect(0, 0, w, h);

	// ---- the reader's attention: this layer's lines, faded in from the last layer's as it completes. On the line, the
	// last layer's stay, faint, while the painter works (the ring's would cross the picture)
	const keep = layout === 'ring' ? 0 : 0.22;
	const readFade = p.k === 'read' ? 1 : p.k === 'hand' ? 1 - ease(p.u) * (1 - keep) : keep;
	const lastLayer = p.k === 'read' ? p.layer : 8;
	if (readFade > 0) {
		ctx.save();
		ctx.globalCompositeOperation = 'lighter';
		ctx.lineCap = 'round';
		const layers: [number, number][] =
			p.k === 'read' && p.layer > 0
				? [
						[p.layer - 1, 1 - ease(p.u / 0.3)],
						[p.layer, ease(p.u / 0.3)]
					]
				: [[lastLayer, 1]];
		for (const [l, a] of layers) {
			if (a <= 0) continue;
			for (const e of readerEdges(run, l)) {
				ctx.beginPath();
				g.arc(ctx, e.from - run.first, e.to - run.first);
				ctx.strokeStyle = rgba(C.glacier, Math.min(0.95, 0.2 + e.w * 1.3) * a * readFade);
				ctx.lineWidth = 0.8 + e.w * 4.5;
				ctx.stroke();
			}
		}
		ctx.restore();
	}

	// ---- the painter: pass, block, and how much it reads each word
	const pass: 0 | 1 | null =
		p.k === 'pass' ? p.pass : p.k === 'reveal' || p.k === 'renoise' ? 0 : p.k === 'done' ? 1 : null;
	const block = p.k === 'pass' ? p.block : 24;
	const mass = pass !== null ? masses(run, pass, block) : null;

	// ---- the words
	ctx.font = `300 ${g.font}px ${SERIF}`;
	ctx.textAlign = 'center';
	ctx.textBaseline = 'middle';
	for (let k = 0; k < n; k++) {
		const tok = run.tokens[run.first + k];
		ctx.fillStyle = rgba([232, 226, 214], mass ? mass[k] : 1);
		ctx.fillText(label(tok, k === 0), g.tokens[k].x, g.tokens[k].y);
	}

	// ---- under each word, where the picture attends to it (appears as the painter starts)
	const tileIn = p.k === 'read' ? 0 : p.k === 'hand' ? ease((p.u - 0.4) / 0.6) : 1;
	if (tileIn > 0) {
		for (let k = 0; k < n; k++) {
			const tl = g.tiles[k];
			ctx.save();
			ctx.beginPath();
			if (tl.round) ctx.arc(tl.x, tl.y, tl.s / 2, 0, Math.PI * 2);
			else ctx.rect(tl.x - tl.s / 2, tl.y - tl.s / 2, tl.s, tl.s);
			ctx.clip();
			ctx.fillStyle = '#070707';
			ctx.fillRect(tl.x - tl.s / 2, tl.y - tl.s / 2, tl.s, tl.s);
			if (pass !== null) {
				ctx.imageSmoothingEnabled = false;
				const side = run.sides[pass];
				const draw = (b: number, a: number) => {
					if (a <= 0) return;
					ctx.globalAlpha = a * tileIn * (mass?.[k] ?? 1);
					ctx.drawImage(
						mapImage(run.maps[pass][b][run.first + k], side),
						tl.x - tl.s / 2,
						tl.y - tl.s / 2,
						tl.s,
						tl.s
					);
				};
				if (p.k === 'pass' && block > 0) {
					draw(block - 1, 1 - ease(p.u / 0.3));
					draw(block, ease(p.u / 0.3));
				} else if (p.k === 'pass') draw(0, ease(p.u / 0.3));
				else draw(24, 1);
			}
			ctx.restore();
			ctx.globalAlpha = 1;
		}
	}

	// ---- the picture: the painter's latest finished guess. Before the first pass ends there is none, only the noise it
	// starts from (dim); then the sketch; while the second pass repaints it, the sketch under the fresh noise it was
	// mixed with (the second pass's input), dimmed; then the picture
	const pic = g.picture;
	const picIn = p.k === 'read' ? 0 : p.k === 'hand' ? ease((p.u - 0.3) / 0.5) : 1;
	if (picIn > 0) {
		const { noise, sketch, renoised, final } = run.pictures;
		const dim = 0.38;
		const layers: [ImageBitmap, number][] =
			p.k === 'hand' || (p.k === 'pass' && p.pass === 0)
				? [[noise, dim]]
				: p.k === 'reveal'
					? [
							[noise, dim * (1 - ease(p.u / 0.7))],
							[sketch, ease(p.u / 0.7)]
						]
					: p.k === 'renoise'
						? [
								[sketch, 1 - 0.45 * ease(p.u)],
								[renoised, 0.3 * Math.sin(Math.PI * Math.min(1, p.u))]
							]
						: p.k === 'pass'
							? [[sketch, 0.55]]
							: [
									[sketch, 0.55 * (1 - ease(p.u))],
									[final, ease(p.u)]
								];
		ctx.save();
		ctx.imageSmoothingEnabled = true;
		for (const [img, a] of layers) {
			ctx.globalAlpha = a * picIn;
			ctx.drawImage(img, pic.x - pic.s / 2, pic.y - pic.s / 2, pic.s, pic.s);
		}
		ctx.restore();
	}

	// ---- what is happening, and the count of it: 9 layers, 25 blocks, 25 blocks
	ctx.font = `italic 300 ${Math.max(14, Math.min(19, w / 80))}px ${SERIF}`;
	ctx.fillStyle = C.bone;
	ctx.textAlign = layout === 'ring' ? 'center' : 'left';
	ctx.textBaseline = 'alphabetic';
	ctx.fillText(caption(p), layout === 'ring' ? w / 2 : Math.max(24, w * 0.05), h - 58);
	spine(ctx, p, w, h, layout);
}

function spine(ctx: CanvasRenderingContext2D, p: Phase, w: number, h: number, layout: Layout) {
	const groups = [
		{ n: 9, name: 'reading' },
		{ n: 25, name: 'sketch' },
		{ n: 25, name: 'painting' }
	];
	const done = [
		p.k === 'read' ? p.layer + 1 : 9,
		p.k === 'read' || p.k === 'hand' ? 0 : p.k === 'pass' && p.pass === 0 ? p.block + 1 : 25,
		p.k === 'pass' && p.pass === 1 ? p.block + 1 : p.k === 'done' ? 25 : 0
	];
	const gap = 18,
		step = 7;
	const total = (9 + 25 + 25) * step + 2 * gap;
	let x = layout === 'ring' ? (w - total) / 2 : Math.max(24, w * 0.05);
	const y = h - 30;
	ctx.font = `italic 300 12px ${SERIF}`;
	ctx.textAlign = 'left';
	groups.forEach((gr, gi) => {
		ctx.fillStyle = C.ash;
		ctx.fillText(gr.name, x, y + 16);
		for (let i = 0; i < gr.n; i++) {
			ctx.fillStyle =
				i < done[gi] ? (gi === 0 ? rgba(C.glacier, 0.95) : rgba(C.ember, 0.95)) : '#262626';
			ctx.fillRect(x + i * step, y - 5, step - 3, 5);
		}
		x += gr.n * step + gap;
	});
}
