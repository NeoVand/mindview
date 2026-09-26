// The score: maps playback time (seconds) to what every act shows and where the camera looks.
import type { ReadingState } from './acts/reading';
import type { PaintingState } from './acts/painting';
import { TOWER } from './acts/painting';
import { LOOM } from './acts/reading';

export interface Shape {
	tokens: number;
	layers: number; // hidden states incl. embeddings
	taps: number[];
	steps: number;
	blocks: number;
	words: string[]; // the prompt's own words, for the focus cycle
	decoder: string[]; // captions of the VAE decoder stages, coarse to fine
}

export interface Moment {
	act: 0 | 1 | 2 | 3 | 4; // intro, reading, handoff, painting, finale
	caption: string;
	finale: number; // 0..1 how far into the final reveal
	reading: ReadingState;
	handoff: { progress: number; fade: number };
	painting: PaintingState;
	eye: [number, number, number];
	target: [number, number, number];
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const ease = (x: number) => {
	const t = clamp01(x);
	return t * t * (3 - 2 * t);
};
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const lerp3 = (a: number[], b: number[], t: number): [number, number, number] => [
	lerp(a[0], b[0], t),
	lerp(a[1], b[1], t),
	lerp(a[2], b[2], t)
];

export class Timeline {
	readonly intro = 3;
	readonly readingDur = 26;
	readonly handoffDur = 8;
	readonly stepDur = [24, 10, 9, 9];
	readonly hold = 1.8;
	readonly decodeStage = 2.4; // seconds per decoder stage
	readonly finaleHold = 10;
	readonly finaleDur: number;
	readonly readingStart: number;
	readonly handoffStart: number;
	readonly paintStart: number;
	readonly stepStart: number[];
	readonly finaleStart: number;
	readonly duration: number;

	constructor(private s: Shape) {
		this.readingStart = this.intro;
		this.handoffStart = this.readingStart + this.readingDur;
		this.paintStart = this.handoffStart + this.handoffDur;
		this.stepStart = [];
		let t = this.paintStart;
		for (let i = 0; i < s.steps; i++) {
			this.stepStart.push(t);
			t += (this.stepDur[i] ?? this.stepDur[this.stepDur.length - 1]) + this.hold;
		}
		this.finaleStart = t;
		this.finaleDur = 1.5 + Math.max(1, s.decoder.length) * this.decodeStage + this.finaleHold;
		this.duration = t + this.finaleDur;
	}

	at(time: number): Moment {
		const s = this.s;
		const t = ((time % this.duration) + this.duration) % this.duration;
		const topLayer = s.layers - 1;
		const loomH = topLayer * LOOM.sy;

		// ---- Act I
		const r = t - this.readingStart;
		const reading: ReadingState = {
			tokenReveal: s.tokens * ease(r / 7),
			layerFront: topLayer * ease((r - 3) / 17) + 0.001,
			tapGlow: ease((r - 21) / 4) * (1 - ease((t - this.handoffStart - 5) / 3)),
			fade: 1 - ease((t - this.handoffStart - 2) / 5)
		};

		// ---- Act II
		const h = (t - this.handoffStart) / this.handoffDur;
		const handoff = { progress: clamp01(h), fade: h > 0 && h < 1.15 ? 1 : 0 };

		// ---- Act III
		let step = 0,
			scan = 0;
		for (let i = 0; i < s.steps; i++) {
			if (t >= this.stepStart[i]) {
				step = i;
				const dur = this.stepDur[i] ?? this.stepDur[this.stepDur.length - 1];
				scan = s.blocks * clamp01((t - this.stepStart[i]) / dur);
			}
		}
		const fin = t - this.finaleStart;
		if (fin >= 0) scan = s.blocks;
		// decoding: -1 while the painter works, then climbs through the decoder stages (1.5 s to hand over)
		const stages = Math.max(1, s.decoder.length);
		const decoder =
			fin < 0
				? -1
				: Math.min(stages - 1, -1 + fin / 1.5 + Math.max(0, fin - 1.5) / this.decodeStage);
		const paintFade = ease((t - this.paintStart + 4) / 4);
		const words = s.words.length;
		const painting: PaintingState = {
			step,
			scan,
			fade: paintFade,
			focus: words ? Math.floor(Math.max(0, t - this.paintStart) / 1.4) % words : 0,
			beam: t >= this.paintStart && fin < 0 ? ease((t - this.paintStart) / 2) : 0,
			palette: 0,
			canvasLens: step * s.blocks + Math.min(scan, s.blocks - 0.001),
			decoder,
			canvasSize: TOWER.canvasSize,
			canvasOpacity: 1.15,
			towerFade: 1 - ease((fin + 1) / 3)
		};

		// ---- camera
		const loomWide = { eye: [0, loomH * 0.35, 21], target: [0, loomH * 0.35, 0] };
		const loomLeft = { eye: [-9, loomH * 0.55, 17], target: [0.5, loomH * 0.5, 0] };
		const loomTop = { eye: [6, loomH * 0.85, 16], target: [0, loomH * 0.6, 0] };
		const zScan = TOWER.z0 - Math.min(scan, s.blocks) * TOWER.dz;
		const tower = {
			eye: [7.2, TOWER.y + 1.8, zScan + 8.0],
			target: [TOWER.x - 0.9, TOWER.y - 0.2, zScan - 3.2]
		};
		const canvasZ = TOWER.z0 - s.blocks * TOWER.dz - TOWER.canvasGap;
		// the finished image sits right of centre so the prompt can title it from the left
		const finale = {
			eye: [-2.2, TOWER.y + 0.4, canvasZ + 16],
			target: [-2.2, TOWER.y + 0.4, canvasZ]
		};
		const travel = { eye: [9, 9, TOWER.z0 + 10], target: [0, TOWER.y, TOWER.z0 - 6] };
		let cam = loomWide;
		if (t < this.readingStart + 10) cam = loomWide;
		const mixCam = (a: typeof cam, b: typeof cam, k: number) => ({
			eye: lerp3(a.eye, b.eye, k),
			target: lerp3(a.target, b.target, k)
		});
		cam = mixCam(loomWide, loomLeft, ease((t - this.readingStart - 6) / 12));
		cam = mixCam(cam, loomTop, ease((t - this.readingStart - 17) / 10));
		cam = mixCam(cam, travel, ease((t - this.handoffStart - 1) / (this.handoffDur - 1)));
		cam = mixCam(cam, tower, ease((t - this.paintStart + 1) / 5));
		cam = mixCam(cam, finale, ease((t - this.finaleStart + 0.5) / 5));
		// a slow breath so the scene never freezes
		const eye = cam.eye.map((v, i) => v + Math.sin(t * 0.13 + i * 2.1) * 0.25) as [
			number,
			number,
			number
		];

		let act: Moment['act'] = 0;
		let caption = '';
		if (t >= this.readingStart) {
			act = 1;
			caption =
				r < 3
					? 'Reading your words'
					: `Reading your words: layer ${Math.min(topLayer, Math.floor(reading.layerFront))} of ${topLayer}`;
		}
		if (t >= this.handoffStart) {
			act = 2;
			caption = `Handing layers ${s.taps.slice(0, -1).join(', ')} and ${s.taps[s.taps.length - 1]} to the painter`;
		}
		if (t >= this.paintStart) {
			act = 3;
			caption = `Painting: step ${step + 1} of ${s.steps}, block ${Math.min(s.blocks, Math.floor(scan) + 1)} of ${s.blocks}`;
		}
		if (fin >= 0) {
			act = 4;
			// the canvas dwells on a stage for ~55% of its time, then crossfades; switch the caption mid-fade
			const k = Math.floor(Math.max(0, decoder) + 0.25);
			caption = decoder < 0 ? 'Handing the finished latent to the decoder' : (s.decoder[k] ?? '');
			if (fin > this.finaleDur - this.finaleHold) caption = '';
		}
		return {
			act,
			caption,
			finale: fin >= 0 ? ease(fin / 3) : 0,
			reading,
			handoff,
			painting,
			eye,
			target: cam.target as [number, number, number]
		};
	}
}
