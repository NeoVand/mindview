// Your words in the painter, still threads: after the adapter they run on as a cable (each word its own place around
// it, as in the reader) beneath the row of readings. At every block, each word's thread reaches up out of the cable
// towards the place where the picture reads the word most (the peak of the word's map, smoothed a little; where the
// reading shows the word's colour): all the way, touching the reading, when the picture reads the word nearly as much
// as its most-read word; not at all when it hardly reads it. So a word's path is when, where and how hard it is being
// painted in; the picture that comes of it stands just above. Where a thread touches, a small ring marks the stitch.
// Between the threads, links: how the words read each other inside the painter (its attention runs over words and
// picture together, in both directions), drawn like the reader's, from the word read to the reader.
// Each stretch of thread is drawn once both its ends are known and never moves after (the curve through a point only
// uses the points before it). While the decoder works nothing is read, and the threads run on in the cable.
import { LineLayer } from './lines';

type V3 = [number, number, number];
const SUB = 8; // points per stretch between two knots
const TOUCH = 0.08; // how far off the reading a thread is where it touches
/** Within a pass (in ticks from its start): the threads' knots over the decoder's pictures, after the blocks. */
export const REST_TICKS = [26, 27, 28, 28.9];
const PER_PASS = 25 + REST_TICKS.length;

export interface Stitch {
	u: number; // across the picture, 0..1 from its left edge
	v: number; // down the picture, 0..1 from its top edge
	share: number; // how much the picture reads the word, 1 = as much as its most-read word
}

export class WordWeave {
	readonly threads: LineLayer;
	readonly links: LineLayer;
	readonly rings: LineLayer;
	private knots: (V3 | null)[][]; // [word][knot]: the entrance, then per pass 25 blocks and the rests
	private reaches: Float32Array[]; // [word][knot]: how far the thread reaches there (0 in the cable)
	private known = 0; // knots known so far
	private knotTick: number[];
	private per: number;
	private pts: Float32Array;
	private linkPts: number[] = [];
	private linkAttr: number[] = [];
	private ringPts: number[] = [];
	private ringAttr: number[] = [];
	private strand = 0;
	/** Where the threads come in (heading along +x). */
	readonly entrances: V3[];

	constructor(
		device: GPUDevice,
		frame: GPUBuffer,
		private palette: Float32Array,
		private words: number,
		steps: number,
		ticksPerStep: number,
		private picture: (tick: number) => [V3, V3, V3], // corner, across, down of a block's reading
		private lane: (tick: number, k: number) => V3 // word k's place in the cable at tick
	) {
		this.threads = new LineLayer(device, frame, true);
		this.links = new LineLayer(device, frame, true);
		this.rings = new LineLayer(device, frame, true);
		this.knotTick = [0];
		for (let s = 0; s < steps; s++) {
			for (let b = 0; b < 25; b++) this.knotTick.push(s * ticksPerStep + b + 1);
			for (const r of REST_TICKS) this.knotTick.push(s * ticksPerStep + r);
		}
		this.entrances = Array.from({ length: words }, (_, k) => lane(0, k));
		this.knots = Array.from({ length: words }, (_, k) => [
			this.entrances[k],
			...new Array(this.knotTick.length - 1).fill(null)
		]);
		this.reaches = Array.from({ length: words }, () => new Float32Array(this.knotTick.length));
		this.per = (this.knotTick.length - 1) * SUB + 1;
		const n = words * this.per;
		this.pts = new Float32Array(n * 4);
		const attr = new Uint32Array(n);
		for (let k = 0; k < words; k++)
			for (let i = 0; i < this.per; i++) {
				// hidden until reached: w beyond any cut until the stretch is known
				this.pts[(k * this.per + i) * 4 + 3] = 1e9;
				attr[k * this.per + i] = k | (k << 16) | (255 << 24);
			}
		for (let k = 0; k < words; k++) this.pts.set([...this.entrances[k], 0], k * this.per * 4);
		this.threads.set(this.pts, attr, palette);
	}

	/** How far a word's thread reaches from the cable to its place on the reading, 0..1, for its share. */
	static reach(share: number) {
		const t = Math.min(1, Math.max(0, (share - 0.3) / 0.6));
		return t * t * (3 - 2 * t);
	}

	/** The spot on block `tick`'s reading (just in front of it). */
	private spot(tick: number, st: Stitch, lift = TOUCH): V3 {
		const [c, ax, ay] = this.picture(tick);
		return [c[0] + ax[0] * st.u + ay[0] * st.v, c[1] + ax[1] * st.u + ay[1] * st.v, c[2] + lift];
	}

	private point(tick: number, st: Stitch, k: number): V3 {
		const to = this.spot(tick, st),
			from = this.lane(tick, k),
			r = WordWeave.reach(st.share);
		return [to[0], from[1] + (to[1] - from[1]) * r, from[2] + (to[2] - from[2]) * r];
	}

	/** Fill in the stretch that ends at knot i of word k (both ends known): a curve through the knots before it. */
	private stretch(k: number, i: number) {
		const K = this.knots[k];
		const P1 = K[i - 1]!,
			P2 = K[i]!,
			P0 = K[i - 2] ?? ([P1[0] - 1.6, P1[1], P1[2]] as V3); // the entrance is reached along +x
		// tangents from the past only: at the start, from the knot before it; at the end, along the stretch (short, so
		// a long jump does not throw the next stretch into a loop)
		const t1 = P1.map((v, d) => (v - P0[d]) * 0.4),
			t2 = P2.map((v, d) => (v - P1[d]) * 0.4);
		const w1 = this.knotTick[i - 1],
			w2 = this.knotTick[i];
		for (let q = 1; q <= SUB; q++) {
			const t = q / SUB,
				t2_ = t * t,
				t3 = t2_ * t;
			const h00 = 2 * t3 - 3 * t2_ + 1,
				h10 = t3 - 2 * t2_ + t,
				h01 = -2 * t3 + 3 * t2_,
				h11 = t3 - t2_;
			const o = (k * this.per + (i - 1) * SUB + q) * 4;
			for (let d = 0; d < 3; d++)
				this.pts[o + d] = h00 * P1[d] + h10 * t1[d] + h01 * P2[d] + h11 * t2[d];
			this.pts[o + 2] = Math.max(TOUCH * 0.75, this.pts[o + 2]); // never through the pictures
			this.pts[o + 3] = w1 + (w2 - w1) * t;
		}
	}

	/**
	 * One block's readings arrived: each word's stitch on the block's picture, and (optionally) the words' attention
	 * to each other [reader][read], averaged over heads.
	 */
	add(s: number, b: number, stitches: Stitch[], links?: Float32Array) {
		const i = 1 + s * PER_PASS + b,
			tick = this.knotTick[i];
		const at = stitches.map((st, k) => this.point(tick, st, k));
		this.known = i + 1 + (b === 24 ? REST_TICKS.length : 0);
		for (let k = 0; k < this.words; k++) {
			this.knots[k][i] = at[k];
			this.reaches[k][i] = WordWeave.reach(stitches[k].share);
			this.stretch(k, i);
			// after the last block, over the decoder's pictures (it never reads the words)
			if (b === 24)
				for (let r = i + 1; r <= i + REST_TICKS.length; r++) {
					this.knots[k][r] = this.lane(this.knotTick[r], k);
					this.stretch(k, r);
				}
		}
		this.threads.setPoints(this.pts);
		this.addRings(tick, stitches);
		if (links) this.addLinks(tick, at, links);
	}

	/** A ring on the reading where a word's thread touches it. */
	private addRings(tick: number, stitches: Stitch[]) {
		stitches.forEach((st, k) => {
			const reach = WordWeave.reach(st.share);
			if (reach < 0.9) return;
			const r = 0.06 + 0.08 * st.share;
			const o = this.spot(tick, st, 0.02);
			for (let q = 0; q <= 20; q++) {
				const a = (q / 20) * Math.PI * 2;
				this.ringPts.push(o[0] + r * Math.cos(a), o[1] + r * Math.sin(a), o[2], tick);
				this.ringAttr.push((this.strand & 0xffff) | (k << 16) | ((255 * reach) << 24));
			}
			this.strand++;
		});
		this.rings.set(new Float32Array(this.ringPts), new Uint32Array(this.ringAttr), this.palette);
	}

	/**
	 * Arcs from each word to the (at most two) other words it reads most in this block, in the colour of the word read,
	 * brighter the more it reads it; none weaker than a third of the block's strongest.
	 */
	private addLinks(tick: number, at: V3[], m: Float32Array) {
		const n = Math.round(Math.sqrt(m.length)),
			w = Math.min(n, this.words);
		let top = 1e-12;
		for (let i = 0; i < w; i++)
			for (let j = 0; j < w; j++) if (i !== j) top = Math.max(top, m[i * n + j]);
		for (let i = 0; i < w; i++) {
			const row = Array.from({ length: w }, (_, j) => [j, m[i * n + j] / top] as const)
				.filter(([j]) => j !== i)
				.sort((a, b) => b[1] - a[1])
				.slice(0, 2);
			for (const [j, p] of row) {
				if (p < 1 / 3) break;
				const A = at[j],
					B = at[i];
				const lift = 0.3 + 0.2 * Math.hypot(B[0] - A[0], B[1] - A[1]);
				for (let q = 0; q <= 24; q++) {
					const u = q / 24,
						bulge = Math.sin(Math.PI * u) * lift;
					// drawn from the word read to the reader, just before the threads reach the picture
					this.linkPts.push(
						A[0] + (B[0] - A[0]) * u,
						A[1] + (B[1] - A[1]) * u,
						A[2] + (B[2] - A[2]) * u + bulge,
						tick - 0.4 + 0.4 * u
					);
					this.linkAttr.push((this.strand & 0xffff) | (j << 16) | ((255 * p) << 24));
				}
				this.strand++;
			}
		}
		this.links.set(new Float32Array(this.linkPts), new Uint32Array(this.linkAttr), this.palette);
	}

	/** How far word k's thread reaches out of the cable at `cut`, 0..1 (between knots, in between). */
	reachAt(k: number, cut: number) {
		const T = this.knotTick;
		let i = 0;
		while (i + 1 < this.known && T[i + 1] <= cut) i++;
		const r = this.reaches[k];
		if (i + 1 >= this.known) return r[i];
		const f = Math.min(1, Math.max(0, (cut - T[i]) / (T[i + 1] - T[i])));
		return r[i] + (r[i + 1] - r[i]) * f;
	}

	/** Where word k's thread has reached by `cut`. */
	tip(k: number, cut: number): V3 {
		const P = this.pts,
			base = k * this.per;
		let lo = 0,
			hi = this.per - 1;
		while (lo < hi) {
			const m = (lo + hi + 1) >> 1;
			if (P[(base + m) * 4 + 3] <= cut) lo = m;
			else hi = m - 1;
		}
		const o = (base + lo) * 4;
		return [P[o], P[o + 1], P[o + 2]];
	}

	draw(pass: GPURenderPassEncoder, w: number, h: number, cut: number, gain = 1) {
		this.threads.now = cut;
		this.threads.width = 1;
		this.threads.gain = 0.9 * gain;
		this.threads.fresh = 2.5;
		this.threads.draw(pass, w, h);
		this.links.now = cut;
		this.links.width = 0.7;
		this.links.gain = 0.7 * gain;
		this.links.fresh = 2;
		this.links.fade = 0.6;
		this.links.draw(pass, w, h);
		this.rings.now = cut;
		this.rings.width = 0.7;
		this.rings.gain = 0.9 * gain;
		this.rings.fresh = 3;
		this.rings.draw(pass, w, h);
	}

	destroy() {
		this.threads.destroy();
		this.links.destroy();
		this.rings.destroy();
	}
}
