// Lab: the machine at 1:1. Every weight of Ternary Bonsai 1.7B on one wall, one cell each (256 cells to a unit), in
// the order the model uses them: the vocabulary on the left (the word's row is looked up), then the 28 layers left to
// right, each a column of its seven matrices, then the vocabulary again on the right (the same table, used a second
// time to score every possible next word). Every matrix is drawn so its columns are the 2,048 dimensions of the
// residual stream (q, k, v, gate and up read it; the output and down projections, shown transposed, write it), so a
// dimension is one vertical line through the whole wall.
//   weights  each cell is its weight: -1 (glacier), +1 (ember) or 0 (dark), times its group's scale
//   at work  each cell is its weight times the input it multiplies for the chosen word: what the matrix adds up
// Beside each matrix, its input and output as waveforms: the vector that comes in along the edge it is read from,
// the sums that come out along the other. Reordering the residual dimensions (and each layer's 6,144 neurons) by how
// active they are changes nothing the model computes; it only gathers the busy lines together.
// Then the painter (see machine-painter.ts): the adapter, 25 blocks and the output projection, lit by a real painting
// of the same prompt as it paints, with each block's picture above it and the finished picture at the end. Follow a
// word (it goes through the reader, the adapter and the painter's text weights) or a patch of the picture (the
// painter's image weights); both meet in the single blocks.
import type { BonsaiLLM, Ternary } from '$lib/runtime/bonsai-llm';
import { MAX_TOKENS } from '$lib/runtime/bonsai-llm';
import { chatPrompt } from '$lib/runtime/tokenizer';
import { clean } from '$lib/engine/text';
import type { Painter } from '$lib/runtime/painter';
import { ImagePlanes } from '$lib/viz/planes';
import { PainterWall } from './machine-painter';
import type { PaintingRun } from './painting-run';
import { Panels, type PanelDesc } from './panels';
import { FOLLOWED_PATCHES } from './shared';
import type { Scene, Stage, V3 } from './stage';

const U = 256; // cells per world unit
const GAP = 0.5; // between matrices in a column
const COL_GAP = 2.5; // between layer columns
const WAVE = 1.6; // waveform thickness
const WAVE_GAP = 0.35;

export interface MachineLabel {
	text: string;
	x: number;
	y: number;
	size: number; // css px
	alpha: number;
}

export interface CellInfo {
	title: string;
	lines: string[];
}

type Role = 'q' | 'k' | 'v' | 'o' | 'gate' | 'up' | 'down' | 'embed' | 'head';

interface MatrixPanel {
	panel: number;
	role: Role;
	layer: number;
	tensor: Ternary;
	name: string;
	transpose: boolean;
	m0: number; // first row of the tensor this panel shows (vocabulary folds)
	rows: number;
	origin: V3;
	w: number;
	h: number;
	start: number; // reveal: when this matrix is multiplied (seconds), and for how long
	dur: number;
}

interface Wave {
	panel: number;
	dim: 'resid' | 'neuron' | 'other'; // what its numbers are (for reordering)
	vec: () => Float32Array; // the values (for its gain and for pointing at)
	at: number; // reveal time
	name: string;
	layer: number;
}

export class Machine implements Scene {
	panels: Panels;
	private mats: MatrixPanel[] = [];
	private waves: Wave[] = [];
	private tokens: string[] = [];
	private ids: number[] = [];
	private words: number[] = []; // token positions of the prompt's own words
	token = 0;
	mode: 'weights' | 'work' = 'work';
	sorted = false;
	exposure = 0.15;
	/** Seconds into the reveal (the computation shown in order); Infinity = all done. */
	reveal = Infinity;
	revealLength = 0;
	playing = false;
	private gains = new Map<number, number>();
	private cpu?: {
		xn: Float32Array[];
		qraw: Float32Array[];
		kraw: Float32Array[];
		v: Float32Array[];
		attn: Float32Array[];
		mid: Float32Array[];
		resid: Float32Array[];
		xn2: Float32Array[];
		gate: Float32Array[];
		up: Float32Array[];
		act: Float32Array[];
		final: Float32Array;
		dAttn: Float32Array[];
		dMlp: Float32Array[];
	};
	private pending = false;
	private pred = '';
	busy = '';
	width = 0;
	height = 0;
	private x0 = 0;
	private layerX: number[] = [];
	private readId = 0;
	// the painter's part
	private wall?: PainterWall;
	private painter?: Painter;
	private planes: ImagePlanes;
	private run: PaintingRun | null = null;
	/** The painting step shown (0-based): while it paints, the one being painted, unless one is chosen. */
	step = 3;
	private stepChosen = false;
	/** The patch followed (0..1023), or null to follow the chosen word. */
	patch: number | null = null;
	private stateOf = new Map<number, string>();
	private pvec = new Map<string, Float32Array>();
	private refreshing = false;
	private refreshKey = '';
	private denseMean = new Map<number, number>();

	constructor(
		private stage: Stage,
		private llm: BonsaiLLM
	) {
		const c = llm.config;
		this.panels = new Panels(stage, llm, {
			capacity: 1024,
			aux: c.layers * c.dim * 2,
			idx: c.dim + c.layers * c.ffn
		});
		this.planes = new ImagePlanes(stage.device, stage.frame, true);
		this.build();
		stage.focusPlane = { n: [0, 0, 1], d: 0 };
	}

	/** The row followed through the painter: the chosen word's, or the patch's. */
	get row() {
		return this.patch === null ? this.token : 512 + this.patch;
	}

	/** The painter is here: lay its part of the wall out after the reader's. */
	attachPainter(painter: Painter, run: PaintingRun) {
		if (this.wall) return this.setRun(run);
		this.painter = painter;
		this.wall = new PainterWall(this.panels, this.planes, painter, this.x0 + this.width + 10);
		this.panels.setPainter(painter.weights);
		this.panels.allocate();
		this.width = this.wall.x1 - this.x0;
		this.height = Math.max(this.height, this.wall.height);
		// the dense weights' typical size (from a sample), for their brightness
		for (const m of this.wall.mats)
			if (m.dense) {
				const d = this.panels.desc(m.panel);
				this.denseMean.set(m.panel, 0.02 * Math.sqrt(3072 / Math.max(1, d.cols)));
			}
		this.setRun(run);
		void this.rebuild();
	}

	/** A (new) painting to light the painter's part with. */
	setRun(run: PaintingRun) {
		this.run = run;
		this.stepChosen = false;
		this.step = run.done ? run.steps - 1 : run.at.step;
		this.panels.setCapture(run.cap.buffer);
		this.wall?.forgetPictures();
		this.stateOf.clear();
		this.refreshKey = '';
	}

	get painting() {
		return this.run;
	}

	/** Follow a patch (the nearest one the painting follows) or, with null, the chosen word. */
	followPatch(p: number | null) {
		if (p !== null) {
			const r = (i: number) => [Math.floor(i / 32), i % 32];
			const [y, x] = r(p);
			p = FOLLOWED_PATCHES.reduce((best, q) => {
				const [qy, qx] = r(q),
					[by, bx] = r(best);
				return (qy - y) ** 2 + (qx - x) ** 2 < (by - y) ** 2 + (bx - x) ** 2 ? q : best;
			}, FOLLOWED_PATCHES[0]);
		}
		this.patch = p;
		void this.rebuild();
	}

	setStep(s: number) {
		this.step = s;
		this.stepChosen = true;
		this.refreshKey = '';
	}

	/** A click on the wall: on the finished picture, follow that patch. */
	clickAt(clientX: number, clientY: number) {
		const p = this.stage.pointUnder(clientX, clientY);
		if (!p || !this.wall) return;
		const patch = this.wall.patchAt(p);
		if (patch !== null) this.followPatch(patch);
	}

	/**
	 * The painter's panels for the row and step: point them at the capture (as far as it has painted), rebuild the
	 * pyramids of those that changed, and their brightness.
	 */
	private async refreshPainter() {
		const wall = this.wall,
			run = this.run;
		if (!wall || this.refreshing) return;
		this.refreshing = true;
		try {
			const work = this.mode === 'work';
			wall.light(run, this.step, this.row, work);
			const changed: number[] = [];
			for (const m of wall.mats) {
				const d = this.panels.desc(m.panel);
				const key = JSON.stringify(m.dense ? (d.b ?? null) : (d.x ?? null));
				if (this.stateOf.get(m.panel) !== key) {
					this.stateOf.set(m.panel, key);
					changed.push(m.panel);
				}
			}
			if (changed.length) {
				const enc = this.stage.device.createCommandEncoder();
				this.panels.buildPyramids(enc, changed);
				this.stage.device.queue.submit([enc.finish()]);
				const tern = changed.filter((p) => !wall.mats.find((m) => m.panel === p)!.dense);
				const means = await this.panels.meanAbs(tern);
				means.forEach((m, j) => this.gains.set(tern[j], m));
			}
			if (run) {
				const regions = wall.waveRegions(run, this.step, this.row);
				const vecs = await run.read(regions);
				// the painter's vectors carry a few numbers far larger than the rest: each waveform is scaled to the bulk
				// (a little above all but the top half percent) and the few beyond run off its edge and glow
				regions.forEach((r, j) => {
					const a = Float32Array.from(vecs[j], Math.abs).sort();
					this.gains.set(r.panel, a[Math.floor(a.length * 0.995)] * 1.4 || a[a.length - 1] || 1);
				});
				// the inputs, for pointing at a cell
				this.pvec.clear();
				const keys = this.inputKeys(run);
				const got = await run.read(keys.map((k) => k.region));
				keys.forEach((k, j) => this.pvec.set(k.key, got[j]));
				// dense panels: typical weight times typical input
				for (const m of wall.mats) {
					if (!m.dense) continue;
					const v = this.pvec.get(`${m.block}|${m.input}`);
					let mean = 0;
					if (v) for (const x of v) mean += Math.abs(x) / v.length;
					this.gains.set(m.panel, (this.denseMean.get(m.panel) ?? 0.02) * (v ? mean : 1));
				}
			}
			this.applyGains();
		} finally {
			this.refreshing = false;
		}
	}

	/** Where each painter matrix's input for the row and step is in the capture. */
	private inputKeys(run: PaintingRun) {
		const out: { key: string; region: { offset: number; count: number } }[] = [];
		const seen = new Set<string>();
		const { D, MLP, CTX, TAP, CIN } = { D: 3072, MLP: 9216, CTX: 7680, TAP: 6144, CIN: 128 };
		for (const m of this.wall!.mats) {
			const key = `${m.block}|${m.input}`;
			if (seen.has(key)) continue;
			seen.add(key);
			const off =
				m.block === -1 && (m.input === 'taps' || m.input === 'ctx')
					? run.cap.at(-1, -1, this.row, m.input)
					: m.block === -1
						? run.cap.at(this.step, -1, this.row, 'lat')
						: run.cap.at(this.step, m.block, this.row, m.input);
			if (off === undefined) continue;
			const n =
				m.input === 'taps'
					? TAP
					: m.input === 'ctx'
						? CTX
						: m.input === 'lat'
							? CIN
							: m.input === 'cat'
								? m.block < 5
									? MLP
									: D + MLP
								: D;
			out.push({ key, region: { offset: off, count: n } });
		}
		return out;
	}

	get tokenTexts() {
		return this.words.map((t) => clean(this.tokens[t]) || '·');
	}
	get wordIndex() {
		return this.words.indexOf(this.token);
	}
	get prediction() {
		return this.pred;
	}

	/** Lay the wall out. */
	private build() {
		const c = this.llm.config,
			D = c.dim,
			F = c.ffn,
			L = this.llm.layout,
			N = MAX_TOKENS;
		const KV = c.kvHeads * c.headDim,
			Q = c.heads * c.headDim;
		const p = this.panels;
		const colRows = Q + 2 * KV + Q + 3 * F; // 24,576 rows in a layer column
		const colH = colRows / U + 6 * GAP;
		const colW = D / U;
		const pitch = colW + WAVE * 2 + COL_GAP;
		const vocab = c.vocab;
		const folds = Math.ceil(vocab / colRows);
		const embd = this.llm.tensor('token_embd.weight');
		this.x0 = -(folds * (colW + COL_GAP) * 2 + c.layers * pitch) / 2;
		let x = this.x0;
		let time = 0;
		const tok = () => this.token;
		const mat = (
			role: Role,
			layer: number,
			tensor: Ternary,
			name: string,
			transpose: boolean,
			m0: number,
			rows: number,
			origin: V3,
			x: PanelDesc['x'],
			dur: number
		) => {
			const displayRows = transpose ? tensor.cols : rows;
			const w = D / U,
				h = displayRows / U;
			const panel = p.add({
				origin,
				ax: [w, 0, 0],
				ay: [0, -h, 0],
				rows: displayRows,
				cols: D,
				kind: 'ternary',
				tensor,
				transpose,
				m0,
				x,
				gain: 1,
				pyramid: true,
				digits: true,
				lift: 1.5
			});
			this.mats.push({
				panel,
				role,
				layer,
				tensor,
				name,
				transpose,
				m0,
				rows,
				origin,
				w,
				h,
				start: time,
				dur
			});
			time += dur;
			return panel;
		};
		const wave = (
			name: string,
			layer: number,
			origin: V3,
			ax: V3,
			ay: V3,
			n: number,
			offset: () => number,
			vec: () => Float32Array,
			aux = false,
			at = time,
			dim: Wave['dim'] = n === F ? 'neuron' : 'other'
		) => {
			const panel = p.add({
				origin,
				ax,
				ay,
				rows: 1,
				cols: n,
				kind: 'bars',
				a: { offset: 0, rs: 0, cs: 1, aux },
				gain: 1,
				digits: true,
				lift: 1
			});
			this.waves.push({ panel, vec, at, name, layer, dim });
			// the offset follows the chosen word
			this.waveOffsets.push({ panel, offset, aux });
			return panel;
		};

		// ---- the vocabulary, looked up (left)
		for (let f = 0; f < folds; f++) {
			const m0 = f * colRows,
				rows = Math.min(colRows, vocab - m0);
			mat('embed', -1, embd, 'token_embd.weight', false, m0, rows, [x, 0, 0], null, 0);
			x += colW + COL_GAP;
		}
		this.embedRow = p.add({
			origin: [0, 0, 0.001],
			ax: [colW, 0, 0],
			ay: [0, -1 / U, 0],
			rows: 1,
			cols: D,
			kind: 'ternary',
			tensor: embd,
			gain: 1,
			digits: true,
			lift: 3
		});
		time += 1.5;
		x += COL_GAP;

		// ---- the layers
		for (let l = 0; l < c.layers; l++) {
			const b = `blk.${l}.`;
			const cx = x + WAVE;
			this.layerX.push(cx + colW / 2);
			let y = 0;
			// input: the normed residual, along the top
			const tq = time;
			wave(
				`layer ${l + 1}: the word, normed`,
				l,
				[cx, y + WAVE_GAP + WAVE, 0],
				[colW, 0, 0],
				[0, -WAVE, 0],
				D,
				() => L.xn + (l * N + tok()) * D,
				() => this.cpu!.xn[l],
				false,
				tq,
				'resid'
			);
			const xin = { offset: 0 };
			const qT = this.llm.tensor(b + 'attn_q.weight'),
				kT = this.llm.tensor(b + 'attn_k.weight'),
				vT = this.llm.tensor(b + 'attn_v.weight');
			const pq = mat('q', l, qT, b + 'attn_q.weight', false, 0, Q, [cx, y, 0], { ...xin }, 1.1);
			this.xOffsets.push({ panel: pq, offset: () => L.xn + (l * N + tok()) * D });
			wave(
				`layer ${l + 1}: q (before its norm)`,
				l,
				[cx + colW + WAVE_GAP, y, 0],
				[0, -Q / U, 0],
				[WAVE, 0, 0],
				Q,
				() => L.qraw + (l * N + tok()) * Q,
				() => this.cpu!.qraw[l]
			);
			y -= Q / U + GAP;
			const pk = mat('k', l, kT, b + 'attn_k.weight', false, 0, KV, [cx, y, 0], { ...xin }, 0.6);
			this.xOffsets.push({ panel: pk, offset: () => L.xn + (l * N + tok()) * D });
			wave(
				`layer ${l + 1}: k (before its norm)`,
				l,
				[cx + colW + WAVE_GAP, y, 0],
				[0, -KV / U, 0],
				[WAVE, 0, 0],
				KV,
				() => L.kraw + (l * N + tok()) * KV,
				() => this.cpu!.kraw[l]
			);
			y -= KV / U + GAP;
			const pv = mat('v', l, vT, b + 'attn_v.weight', false, 0, KV, [cx, y, 0], { ...xin }, 0.6);
			this.xOffsets.push({ panel: pv, offset: () => L.xn + (l * N + tok()) * D });
			wave(
				`layer ${l + 1}: v`,
				l,
				[cx + colW + WAVE_GAP, y, 0],
				[0, -KV / U, 0],
				[WAVE, 0, 0],
				KV,
				() => L.v + (l * N + tok()) * KV,
				() => this.cpu!.v[l]
			);
			y -= KV / U + GAP;
			// attention mixes the words (not a matrix of weights); its result comes in along the left of the output
			// projection (shown transposed: its rows are the 2,048 attention values, its columns the residual)
			time += 0.4;
			wave(
				`layer ${l + 1}: attention's result`,
				l,
				[cx - WAVE_GAP, y, 0],
				[0, -Q / U, 0],
				[-WAVE, 0, 0],
				Q,
				() => L.attn + (l * N + tok()) * Q,
				() => this.cpu!.attn[l]
			);
			const oT = this.llm.tensor(b + 'attn_output.weight');
			const po = mat(
				'o',
				l,
				oT,
				b + 'attn_output.weight',
				true,
				0,
				D,
				[cx, y, 0],
				{ offset: 0 },
				1.1
			);
			this.xOffsets.push({ panel: po, offset: () => L.attn + (l * N + tok()) * Q });
			y -= Q / U;
			wave(
				`layer ${l + 1}: what attention adds`,
				l,
				[cx, y - WAVE_GAP * 0.4, 0],
				[colW, 0, 0],
				[0, -WAVE * 0.8, 0],
				D,
				() => l * 2 * D,
				() => this.cpu!.dAttn[l],
				true,
				time,
				'resid'
			);
			y -= GAP + WAVE * 0.8;
			// the MLP: its input along the top of gate and up
			wave(
				`layer ${l + 1}: the word, normed again`,
				l,
				[cx, y, 0],
				[colW, 0, 0],
				[0, -WAVE * 0.8, 0],
				D,
				() => L.xn2 + (l * N + tok()) * D,
				() => this.cpu!.xn2[l],
				false,
				time,
				'resid'
			);
			y -= WAVE * 0.8 + WAVE_GAP * 0.4;
			const gT = this.llm.tensor(b + 'ffn_gate.weight'),
				uT = this.llm.tensor(b + 'ffn_up.weight'),
				dT = this.llm.tensor(b + 'ffn_down.weight');
			const pg = mat(
				'gate',
				l,
				gT,
				b + 'ffn_gate.weight',
				false,
				0,
				F,
				[cx, y, 0],
				{ offset: 0 },
				2.2
			);
			this.xOffsets.push({ panel: pg, offset: () => L.xn2 + (l * N + tok()) * D });
			this.neuronRows.push({ panel: pg, layer: l });
			wave(
				`layer ${l + 1}: gate`,
				l,
				[cx + colW + WAVE_GAP, y, 0],
				[0, -F / U, 0],
				[WAVE, 0, 0],
				F,
				() => L.gate + (l * N + tok()) * F,
				() => this.cpu!.gate[l]
			);
			y -= F / U + GAP;
			const pu = mat('up', l, uT, b + 'ffn_up.weight', false, 0, F, [cx, y, 0], { offset: 0 }, 2.2);
			this.xOffsets.push({ panel: pu, offset: () => L.xn2 + (l * N + tok()) * D });
			this.neuronRows.push({ panel: pu, layer: l });
			wave(
				`layer ${l + 1}: up`,
				l,
				[cx + colW + WAVE_GAP, y, 0],
				[0, -F / U, 0],
				[WAVE, 0, 0],
				F,
				() => L.up + (l * N + tok()) * F,
				() => this.cpu!.up[l]
			);
			y -= F / U + GAP;
			time += 0.3;
			wave(
				`layer ${l + 1}: the neurons (SiLU(gate) × up)`,
				l,
				[cx - WAVE_GAP, y, 0],
				[0, -F / U, 0],
				[-WAVE, 0, 0],
				F,
				() => L.act + (l * N + tok()) * F,
				() => this.cpu!.act[l]
			);
			const pd = mat(
				'down',
				l,
				dT,
				b + 'ffn_down.weight',
				true,
				0,
				D,
				[cx, y, 0],
				{ offset: 0 },
				2.2
			);
			this.xOffsets.push({ panel: pd, offset: () => L.act + (l * N + tok()) * F });
			this.neuronRows.push({ panel: pd, layer: l });
			y -= F / U;
			wave(
				`layer ${l + 1}: what the neurons add`,
				l,
				[cx, y - WAVE_GAP * 0.4, 0],
				[colW, 0, 0],
				[0, -WAVE * 0.8, 0],
				D,
				() => (l * 2 + 1) * D,
				() => this.cpu!.dMlp[l],
				true,
				time,
				'resid'
			);
			x += pitch;
			time += 0.2;
		}
		void colH;

		// ---- the vocabulary again, scoring every next word (right)
		x += COL_GAP;
		const headStart = time;
		for (let f = 0; f < folds; f++) {
			const m0 = f * colRows,
				rows = Math.min(colRows, vocab - m0);
			const ph = mat(
				'head',
				c.layers,
				embd,
				'token_embd.weight',
				false,
				m0,
				rows,
				[x, 0, 0],
				{ offset: 0 },
				0.9
			);
			this.xOffsets.push({ panel: ph, offset: () => L.final + tok() * D });
			x += colW + COL_GAP;
		}
		void headStart;
		this.revealLength = time + 1;
		this.width = x - this.x0;
		this.height = colH;
		this.panels.allocate();
	}

	private embedRow = -1;
	private waveOffsets: { panel: number; offset: () => number; aux: boolean }[] = [];
	private xOffsets: { panel: number; offset: () => number }[] = [];
	private neuronRows: { panel: number; layer: number }[] = [];

	/** The whole wall in view. */
	overview(jump = false) {
		const c = this.stage.gpu.canvas;
		const aspect = c.clientWidth / Math.max(1, c.clientHeight);
		const t = Math.tan((38 * Math.PI) / 360);
		const v = {
			target: [this.x0 + this.width / 2, -this.height / 2 + 1, 0] as V3,
			dist: Math.max(this.width / (2 * t * aspect), (this.height + 8) / (2 * t)) * 1.06,
			yaw: 0,
			pitch: 0
		};
		if (jump) this.stage.jumpTo(v);
		else this.stage.flyTo(v);
	}

	/** Read a prompt: the forward pass, then everything the wall needs for the chosen word. */
	async read(prompt: string) {
		const id = ++this.readId;
		this.busy = 'Reading your words through all 28 layers';
		const ids = this.llm.tokenizer.encode(chatPrompt(prompt)).slice(0, MAX_TOKENS);
		const r = await this.llm.prefill(ids);
		if (id !== this.readId) return;
		this.ids = r.ids;
		this.tokens = r.tokens;
		const end = r.tokens.findIndex((t, i) => i > 3 && t.startsWith('<|im_end'));
		this.words = Array.from(
			{ length: Math.max(1, (end > 0 ? end : r.ids.length) - 3) },
			(_, i) => 3 + i
		);
		this.token = this.words[this.words.length - 1];
		await this.choose(this.token, true);
	}

	/** Show the computation for token position t (replaying it in order when `replay`). */
	async choose(t: number, replay = false) {
		const id = ++this.readId;
		this.token = t;
		this.busy = 'Collecting the numbers for this word';
		await this.readCpu();
		if (id !== this.readId) return;
		// the chosen word's row of the vocabulary, over the table
		const embed = this.mats.filter((m) => m.role === 'embed');
		const row = this.ids[t];
		const fold = embed.find((m) => row >= m.m0 && row < m.m0 + m.rows);
		if (fold) {
			const r = row - fold.m0;
			this.panels.set(this.embedRow, {
				origin: [fold.origin[0], fold.origin[1] - r / U, 0.002],
				m0: row,
				ay: [0, -Math.max(1 / U, 0.06), 0]
			});
		}
		this.pred = '';
		this.llm
			.readout(this.cpu!.resid[this.llm.config.layers], 1)
			.then(({ ids, probs }) => {
				if (id === this.readId)
					this.pred = `${clean(this.llm.tokenizer.decode([ids[0]]))} (${Math.round(probs[0] * 100)}%)`;
			})
			.catch(() => {});
		this.applySort();
		await this.rebuild();
		if (id !== this.readId) return;
		this.busy = '';
		if (replay) this.play();
	}

	play() {
		this.reveal = 0;
		this.playing = true;
	}

	skip() {
		this.reveal = Infinity;
		this.playing = false;
	}

	setMode(m: 'weights' | 'work') {
		this.mode = m;
		void this.rebuild();
	}

	setSorted(on: boolean) {
		this.sorted = on;
		this.applySort();
		void this.rebuild();
	}

	/** Everything the wall reads for the chosen word, onto the CPU (for gains, pointing, and the added parts). */
	private async readCpu() {
		const llm = this.llm,
			c = llm.config,
			L = llm.layout,
			N = MAX_TOKENS,
			t = this.token;
		const D = c.dim,
			F = c.ffn,
			Q = c.heads * c.headDim,
			KV = c.kvHeads * c.headDim;
		const regions: { offset: number; count: number }[] = [];
		const per = (base: number, width: number, layers = c.layers) =>
			Array.from(
				{ length: layers },
				(_, l) => regions.push({ offset: base + (l * N + t) * width, count: width }) - 1
			);
		const ix = {
			xn: per(L.xn, D),
			qraw: per(L.qraw, Q),
			kraw: per(L.kraw, KV),
			v: per(L.v, KV),
			attn: per(L.attn, Q),
			mid: per(L.mid, D),
			resid: per(L.resid, D, c.layers + 1),
			xn2: per(L.xn2, D),
			gate: per(L.gate, F),
			up: per(L.up, F),
			act: per(L.act, F)
		};
		const fi = regions.push({ offset: L.final + t * D, count: D }) - 1;
		const got = await llm.readMany(regions);
		const pick = (a: number[]) => a.map((i) => got[i]);
		const resid = pick(ix.resid),
			mid = pick(ix.mid);
		const dAttn = mid.map((m, l) => m.map((v, i) => v - resid[l][i]));
		const dMlp = mid.map((m, l) => resid[l + 1].map((v, i) => v - m[i]));
		this.cpu = {
			xn: pick(ix.xn),
			qraw: pick(ix.qraw),
			kraw: pick(ix.kraw),
			v: pick(ix.v),
			attn: pick(ix.attn),
			mid,
			resid,
			xn2: pick(ix.xn2),
			gate: pick(ix.gate),
			up: pick(ix.up),
			act: pick(ix.act),
			final: got[fi],
			dAttn,
			dMlp
		};
		const aux = new Float32Array(c.layers * 2 * D);
		for (let l = 0; l < c.layers; l++) {
			aux.set(dAttn[l], l * 2 * D);
			aux.set(dMlp[l], (l * 2 + 1) * D);
		}
		this.stage.device.queue.writeBuffer(this.panels.aux, 0, aux);
	}

	/** Reorder the residual dimensions and each layer's neurons by activity (or back to the model's own order). */
	private applySort() {
		const c = this.llm.config,
			D = c.dim,
			F = c.ffn;
		const cpu = this.cpu;
		const p = this.panels;
		if (!this.sorted || !cpu) {
			for (const m of this.mats) p.set(m.panel, { permR: undefined, permC: undefined });
			for (const w of this.waves) p.set(w.panel, { permC: undefined });
			p.set(this.embedRow, { permC: undefined });
			return;
		}
		// residual dimensions: by their mean size through the layers for this word
		const act = new Float64Array(D);
		for (const r of cpu.resid) for (let i = 0; i < D; i++) act[i] += Math.abs(r[i]);
		const dimPerm = Uint32Array.from(
			Array.from({ length: D }, (_, i) => i).sort((a, b) => act[b] - act[a])
		);
		const idx = new Uint32Array(D + c.layers * F);
		idx.set(dimPerm, 0);
		for (let l = 0; l < c.layers; l++) {
			const a = cpu.act[l];
			const perm = Array.from({ length: F }, (_, i) => i).sort(
				(x, y) => Math.abs(a[y]) - Math.abs(a[x])
			);
			idx.set(perm, D + l * F);
		}
		this.stage.device.queue.writeBuffer(p.idx, 0, idx);
		for (const m of this.mats) {
			const neuron = m.role === 'gate' || m.role === 'up' || m.role === 'down';
			p.set(m.panel, {
				permC: 0,
				permR: neuron ? D + m.layer * F : undefined
			});
		}
		p.set(this.embedRow, { permC: 0 });
		for (const w of this.waves)
			p.set(w.panel, {
				permC: w.dim === 'resid' ? 0 : w.dim === 'neuron' ? D + w.layer * F : undefined
			});
		this.permCache = undefined;
	}

	/** Point every panel at the chosen word's vectors, rebuild the pyramids, and set the brightnesses. */
	private async rebuild() {
		const p = this.panels;
		// the reader only ever sees words
		const work = this.mode === 'work' && this.patch === null;
		for (const o of this.xOffsets) p.set(o.panel, { x: work ? { offset: o.offset() } : null });
		for (const o of this.waveOffsets) {
			const d = p.desc(o.panel);
			p.set(o.panel, { a: { ...d.a!, offset: o.offset(), aux: o.aux } });
		}
		const enc = this.stage.device.createCommandEncoder();
		p.buildPyramids(enc);
		this.stage.device.queue.submit([enc.finish()]);
		const list = this.mats.map((m) => m.panel);
		const means = await p.meanAbs(list);
		means.forEach((m, j) => this.gains.set(list[j], m));
		for (const w of this.waves) {
			const v = w.vec();
			let mx = 0;
			for (const x of v) mx = Math.max(mx, Math.abs(x));
			this.gains.set(w.panel, mx || 1);
		}
		this.applyGains();
		this.stateOf.clear();
		this.refreshKey = '';
	}

	private applyGains() {
		const p = this.panels;
		for (const m of this.mats) {
			const g = this.gains.get(m.panel) ?? 1;
			p.set(m.panel, { gain: (g * 3.2) / this.exposure });
		}
		for (const w of this.waves)
			p.set(w.panel, { gain: (this.gains.get(w.panel) ?? 1) / Math.sqrt(this.exposure) });
		const e = this.mats.find((m) => m.role === 'embed');
		if (e) p.set(this.embedRow, { gain: ((this.gains.get(e.panel) ?? 1) * 1.2) / this.exposure });
		if (this.wall) {
			for (const m of this.wall.mats) {
				const g = this.gains.get(m.panel) ?? 1;
				p.set(m.panel, { gain: (g * (m.dense ? 6 : 3.2)) / this.exposure });
			}
			for (const w of this.wall.waves)
				p.set(w.panel, { gain: (this.gains.get(w.panel) ?? 1) / Math.sqrt(this.exposure) });
		}
	}

	setExposure(e: number) {
		this.exposure = e;
		this.applyGains();
	}

	update(dt: number) {
		if (this.playing) {
			// the whole computation in about 40 seconds
			this.reveal += (dt * this.revealLength) / 40;
			if (this.reveal >= this.revealLength) this.skip();
		}
		const r = this.reveal;
		const p = this.panels;
		for (const m of this.mats) {
			const done = r >= m.start + m.dur;
			const active = r >= m.start && !done;
			p.set(m.panel, {
				alpha: r < m.start ? 0.13 : 1,
				sweep: active && m.dur > 0 ? ((r - m.start) / m.dur) * D_(p, m.panel) : null
			});
		}
		for (const w of this.waves) p.set(w.panel, { alpha: r < w.at ? 0.0 : 1 });
		p.set(this.embedRow, { alpha: this.cpu ? 1 : 0 });
		// the painter: relight as it paints (and when the row, step or mode change)
		if (this.wall) {
			const run = this.run;
			if (run && !this.stepChosen) this.step = run.done ? run.steps - 1 : run.at.step;
			const key = JSON.stringify([
				run?.at,
				run?.encoded,
				run?.wordsKept,
				run?.done,
				this.row,
				this.step,
				this.mode,
				run?.prompt
			]);
			if (key !== this.refreshKey && !this.refreshing) {
				this.refreshKey = key;
				void this.refreshPainter();
			}
			// the matrices not reached yet (or not on the row's way) stay dim
			for (const m of this.wall.mats) {
				const lit = this.stateOf.get(m.panel);
				p.set(m.panel, { alpha: lit && lit !== 'null' ? 1 : 0.13 });
			}
			this.wall.place(run, this.step);
		}
	}

	draw(pass: GPURenderPassEncoder) {
		this.planes.draw(pass, this.stage.eye);
		this.panels.draw(pass);
	}

	/** Labels for the page: layer numbers, and matrix names when they are big enough to read. */
	labels(): MachineLabel[] {
		const out: MachineLabel[] = [];
		const s = this.stage;
		const upp = s.unitsPerPixel(s.view.dist);
		const px = 1 / upp; // css px per world unit, at the target's depth
		const colW = this.llm.config.dim / U;
		this.layerX.forEach((x, l) => {
			const q = s.project([x, 4.2, 0]);
			if (!q.front) return;
			out.push({
				text: `${l + 1}`,
				x: q.x,
				y: q.y,
				size: Math.min(28, Math.max(10, px * 1.8)),
				alpha: 0.8
			});
		});
		if (px * colW > 140) {
			for (const m of this.mats) {
				const q = s.project([m.origin[0] + m.w / 2, m.origin[1] - m.h / 2, 0]);
				if (
					!q.front ||
					q.x < -200 ||
					q.y < -200 ||
					q.x > s.gpu.canvas.clientWidth + 200 ||
					q.y > s.gpu.canvas.clientHeight + 200
				)
					continue;
				out.push({
					text: roleName(m),
					x: q.x,
					y: q.y,
					size: Math.min(34, px * 0.9),
					alpha: Math.min(0.85, (px * colW - 140) / 200)
				});
			}
		}
		const e = this.mats.find((m) => m.role === 'embed');
		const h = this.mats.find((m) => m.role === 'head');
		if (e && px * 1.4 >= 11) {
			const q = s.project([e.origin[0] + (e.w * 3.5 + COL_GAP * 3), 4.2, 0]);
			if (q.front)
				out.push({
					text: 'the vocabulary: your word is looked up',
					x: q.x,
					y: q.y,
					size: Math.min(24, Math.max(10, px * 1.4)),
					alpha: 0.7
				});
		}
		if (h && px * 1.4 >= 11) {
			const q = s.project([h.origin[0] + (h.w * 3.5 + COL_GAP * 3), 4.2, 0]);
			if (q.front)
				out.push({
					text: 'the same table: every next word is scored (the painter does not use it)',
					x: q.x,
					y: q.y,
					size: Math.min(24, Math.max(10, px * 1.4)),
					alpha: 0.7
				});
		}
		if (this.wall) {
			for (const l of this.wall.labelPoints()) {
				const q = s.project(l.at);
				// only once they are big enough not to run into each other
				if (!q.front || px * l.size < 11) continue;
				out.push({ text: l.text, x: q.x, y: q.y, size: Math.min(26, px * l.size), alpha: 0.75 });
			}
			// the patches a painting follows, on the finished picture (the one followed now, filled)
			if (this.run?.done)
				for (const pt of FOLLOWED_PATCHES) {
					const q = s.project(this.wall.patchPoint(pt));
					if (!q.front) continue;
					out.push({
						text: pt === this.patch ? '●' : '○',
						x: q.x,
						y: q.y,
						size: Math.min(22, Math.max(8, px * 0.8)),
						alpha: pt === this.patch ? 1 : 0.6
					});
				}
		}
		return out;
	}

	/** What is under the pointer (for the page to show), or null. */
	async inspect(clientX: number, clientY: number): Promise<CellInfo | null> {
		const hit = this.panels.pick(this.stage.ray(clientX, clientY));
		if (!hit || !this.cpu) return null;
		const m = this.mats.find((x) => x.panel === hit.panel);
		const p = this.panels.desc(hit.panel);
		const perm = async (off: number | undefined, i: number) =>
			off === undefined ? i : this.permAt(off + i);
		if (m) {
			const r = await perm(p.permR, hit.row),
				cc = await perm(p.permC, hit.col);
			const [row, col] = m.transpose ? [cc, r] : [r + m.m0, cc];
			const { code, scale } = await this.llm.weight(m.name, row, col);
			const x = this.inputOf(m, col);
			const w = code * scale;
			const lines = [
				`row ${row.toLocaleString()} (output), column ${col.toLocaleString()} (input)`,
				`weight: ${code > 0 ? '+1' : code < 0 ? '−1' : '0'} × ${fmt(scale)} = ${fmt(w)}`
			];
			if (x !== null && this.mode === 'work') lines.push(`× input ${fmt(x)} = ${fmt(w * x)}`);
			const out = this.outputOf(m, row);
			if (out !== null) lines.push(`this row adds up to ${fmt(out)}`);
			return {
				title: `${roleName(m)}${m.layer >= 0 && m.layer < this.llm.config.layers ? `, layer ${m.layer + 1}` : ''}`,
				lines
			};
		}
		const pm = this.wall?.mats.find((x) => x.panel === hit.panel);
		if (pm && this.painter) {
			const d = this.panels.desc(hit.panel);
			const x = this.pvec.get(`${pm.block}|${pm.input}`);
			const where =
				pm.block < 0
					? ''
					: pm.block < 5
						? `, double block ${pm.block + 1}`
						: pm.block < 25
							? `, single block ${pm.block + 1}`
							: '';
			let w: number, col: number, lines: string[];
			if (pm.dense) {
				col = hit.col;
				w = await this.painter.denseWeight(pm.name, hit.row * d.cols + hit.col);
				lines = [
					`row ${hit.row.toLocaleString()}, column ${col.toLocaleString()}`,
					`weight ${fmt(w)} (not ternary)`
				];
			} else {
				const transpose = d.transpose;
				const row = transpose ? hit.col : hit.row;
				col = transpose ? hit.row : hit.col;
				const { code, scale } = await this.painter.weight(pm.name, row, col);
				w = code * scale;
				lines = [
					`row ${row.toLocaleString()} (output), column ${col.toLocaleString()} (input)`,
					`weight: ${code > 0 ? '+1' : code < 0 ? '−1' : '0'} × ${fmt(scale)} = ${fmt(w)}`
				];
			}
			const lit = this.stateOf.get(hit.panel);
			if (x && lit && lit !== 'null') lines.push(`× input ${fmt(x[col])} = ${fmt(w * x[col])}`);
			else
				lines.push(
					this.patch === null && pm.stream === 'img'
						? 'a word never goes through these (the picture’s weights)'
						: this.patch !== null && pm.stream === 'txt'
							? 'a patch never goes through these (the words’ weights)'
							: 'not reached yet'
				);
			return { title: `${pm.label}${where}`, lines };
		}
		const w = this.waves.find((x) => x.panel === hit.panel);
		if (w) {
			const i = await perm(p.permC, hit.col);
			return {
				title: w.name,
				lines: [
					`number ${i.toLocaleString()} of ${w.vec().length.toLocaleString()}: ${fmt(w.vec()[i])}`
				]
			};
		}
		return null;
	}

	private permCache?: Uint32Array;
	private async permAt(i: number) {
		if (!this.permCache) {
			// the permutation was computed here; keep a copy
			const c = this.llm.config;
			const buf = this.stage.device.createBuffer({
				size: (c.dim + c.layers * c.ffn) * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
			});
			const enc = this.stage.device.createCommandEncoder();
			enc.copyBufferToBuffer(this.panels.idx, 0, buf, 0, buf.size);
			this.stage.device.queue.submit([enc.finish()]);
			await buf.mapAsync(GPUMapMode.READ);
			this.permCache = new Uint32Array(buf.getMappedRange().slice(0));
			buf.destroy();
		}
		return this.permCache[i];
	}

	private inputOf(m: MatrixPanel, col: number): number | null {
		const cpu = this.cpu!;
		const l = m.layer;
		switch (m.role) {
			case 'q':
			case 'k':
			case 'v':
				return cpu.xn[l][col];
			case 'o':
				return cpu.attn[l][col];
			case 'gate':
			case 'up':
				return cpu.xn2[l][col];
			case 'down':
				return cpu.act[l][col];
			case 'head':
				return cpu.final[col];
			default:
				return null;
		}
	}

	private outputOf(m: MatrixPanel, row: number): number | null {
		const cpu = this.cpu!;
		const l = m.layer;
		switch (m.role) {
			case 'q':
				return cpu.qraw[l][row];
			case 'k':
				return cpu.kraw[l][row];
			case 'v':
				return cpu.v[l][row];
			case 'o':
				return cpu.dAttn[l][row];
			case 'gate':
				return cpu.gate[l][row];
			case 'up':
				return cpu.up[l][row];
			case 'down':
				return cpu.dMlp[l][row];
			default:
				return null;
		}
	}

	/** Fly to a layer's column. */
	focusLayer(l: number) {
		const x = this.layerX[l];
		this.stage.flyTo({
			target: [x, -this.height / 2, 0],
			dist: this.height * 0.95,
			yaw: 0,
			pitch: 0
		});
	}

	destroy() {
		this.panels.destroy();
	}

	/** The painting's progress for the page: 0..1, and a line about it. */
	get paintStatus(): { fraction: number; text: string } | null {
		const r = this.run;
		if (!r) return null;
		if (r.done) return { fraction: 1, text: '' };
		if (!r.encoded)
			return {
				fraction: 0,
				text: 'The reader reads the prompt again for the painter (512 tokens), then the adapter'
			};
		const b = r.at.block;
		return {
			fraction: r.progress,
			text: `Painting: step ${r.at.step + 1} of ${r.steps}, block ${Math.min(25, b + 1)} of 25`
		};
	}
}

function D_(p: Panels, i: number) {
	return p.desc(i).cols;
}

function roleName(m: MatrixPanel) {
	switch (m.role) {
		case 'q':
			return 'queries (q)';
		case 'k':
			return 'keys (k)';
		case 'v':
			return 'values (v)';
		case 'o':
			return 'attention output (transposed)';
		case 'gate':
			return 'gate';
		case 'up':
			return 'up';
		case 'down':
			return 'down (transposed)';
		case 'embed':
			return 'vocabulary';
		case 'head':
			return 'vocabulary, scoring';
	}
}

export function fmt(v: number) {
	if (v === 0) return '0';
	const a = Math.abs(v);
	const s = a >= 1000 || a < 0.001 ? v.toExponential(2) : v.toPrecision(3);
	return s.replace('-', '−');
}
