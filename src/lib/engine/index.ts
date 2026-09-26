// mindview engine: replays a recorded generation trace as three acts (reading, handoff, painting).
import { mat4, vec3 } from 'wgpu-matrix';
import { FRAME_BYTES, HDR_FORMAT, initGPU, type GPU } from './gpu';
import { Post } from './post';
import { loadTrace, type Trace } from './trace';
import { ensureFonts, TextAtlas } from './text';
import { WordLayer } from './words';
import { Reading } from './acts/reading';
import { Painting } from './acts/painting';
import { Handoff } from './acts/handoff';
import { Timeline, type Moment } from './timeline';

export interface EngineStatus {
	caption: string;
	act: Moment['act'];
	time: number;
	duration: number;
	paused: boolean;
	prompt: string;
}

const FOV = (40 * Math.PI) / 180;

export class Engine {
	private gpu!: GPU;
	private post!: Post;
	private frame!: GPUBuffer;
	private reading!: Reading;
	private painting!: Painting;
	private handoff!: Handoff;
	private words!: WordLayer;
	private timeline!: Timeline;
	private trace!: Trace;
	private raf = 0;
	private last = 0;
	private observer?: ResizeObserver;
	time = 0;
	speed = 1;
	paused = false;
	palette = 0;

	constructor(
		private canvas: HTMLCanvasElement,
		private onStatus: (s: EngineStatus) => void
	) {}

	async start(traceBase: string, onProgress?: (f: number, what: string) => void) {
		onProgress?.(0, 'Starting the graphics card');
		this.gpu = await initGPU(this.canvas);
		const { device } = this.gpu;
		onProgress?.(0.02, 'Loading the recorded thoughts');
		const [trace] = await Promise.all([
			loadTrace(traceBase, (f) => onProgress?.(0.02 + f * 0.9, 'Loading the recorded thoughts')),
			ensureFonts()
		]);
		this.trace = trace;
		onProgress?.(0.95, 'Arranging the scene');

		this.frame = device.createBuffer({
			size: FRAME_BYTES,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.post = new Post(this.gpu);
		this.reading = new Reading(device, this.frame, trace, HDR_FORMAT);
		this.painting = new Painting(device, this.frame, trace, HDR_FORMAT);

		// handoff: every token at each tapped layer flies from the loom to its word in the painter; padding follows as dust
		const starts: {
			pos: [number, number, number];
			tap: number;
			real: boolean;
			strength: number;
		}[] = [];
		const ctx = trace.arrays.ctx_norm.data;
		const realMax = Math.max(...Array.from(ctx.slice(0, this.reading.n)));
		const wordOfToken = new Map(this.painting.words.map((w, i) => [w.token, i]));
		this.reading.taps.forEach((layer, tap) => {
			for (let t = 0; t < this.reading.n; t++)
				starts.push({
					pos: this.reading.anchor(t, layer),
					tap,
					real: true,
					strength: Math.min(1.5, ctx[t] / realMax + 0.3)
				});
		});
		const padCount = Math.min(ctx.length - this.reading.n, 480);
		for (let i = 0; i < padCount; i++) {
			const [x, y] = this.reading.anchor(
				Math.random() * (this.reading.n - 1),
				this.reading.taps[i % 3]
			);
			starts.push({
				pos: [x, y, (Math.random() - 0.5) * 1.8],
				tap: i % 3,
				real: false,
				strength: 1
			});
		}
		this.handoff = new Handoff(device, this.frame, HDR_FORMAT, starts, (i, real) => {
			if (!real)
				return [
					(Math.random() - 0.5) * 7,
					4.2 + (Math.random() - 0.5) * 5,
					this.painting.planeZ(0) + 1.5 + Math.random() * 2
				];
			const token = i % this.reading.n;
			const w = wordOfToken.get(token);
			return w === undefined
				? [
						-3.5 + (Math.random() - 0.5),
						4.2 + (Math.random() - 0.5) * 3,
						this.painting.planeZ(0) + 0.6
					]
				: this.painting.wordPos(w, 0);
		});

		const atlasWords = [...this.reading.allWords(), ...this.painting.words.map((w) => w.text)];
		this.words = new WordLayer(
			device,
			this.frame,
			new TextAtlas(device, atlasWords, { px: 72 }),
			600,
			HDR_FORMAT
		);
		this.timeline = new Timeline({
			tokens: this.reading.n,
			layers: this.reading.layers,
			taps: this.reading.taps,
			steps: this.painting.steps,
			blocks: this.painting.blocks,
			words: this.painting.words.map((w) => w.text),
			decoder: trace.decoder.map((d) => d.label)
		});

		this.observer = new ResizeObserver(() => this.resize());
		this.observer.observe(this.canvas);
		this.resize();
		onProgress?.(1, 'Ready');
		this.last = performance.now();
		this.raf = requestAnimationFrame(this.loop);
	}

	get duration() {
		return this.timeline?.duration ?? 0;
	}

	seek(t: number) {
		this.time = Math.max(0, Math.min(this.duration - 0.01, t));
	}

	destroy() {
		cancelAnimationFrame(this.raf);
		this.observer?.disconnect();
		this.gpu?.device.destroy();
	}

	private resize() {
		const dpr = Math.min(window.devicePixelRatio || 1, 2);
		const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
		const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
		if (w === this.canvas.width && h === this.canvas.height && this.post.sceneView) return;
		this.canvas.width = w;
		this.canvas.height = h;
		this.post.resize(w, h);
	}

	private loop = (now: number) => {
		const dt = Math.min(0.1, (now - this.last) / 1000);
		this.last = now;
		if (!this.paused) this.time = (this.time + dt * this.speed) % this.duration;
		this.render();
		this.raf = requestAnimationFrame(this.loop);
	};

	private render() {
		const { device } = this.gpu;
		const m = this.timeline.at(this.time);
		m.painting.palette = this.palette;
		const aspect = this.canvas.width / this.canvas.height;
		const proj = mat4.perspective(FOV, aspect, 0.05, 400);
		const view = mat4.lookAt(m.eye, m.target, [0, 1, 0]);
		const viewProj = mat4.multiply(proj, view);
		const fwd = vec3.normalize(vec3.subtract(m.target, m.eye));
		const right = vec3.normalize(vec3.cross(fwd, [0, 1, 0]));
		const up = vec3.cross(right, fwd);
		const pixel = (2 * Math.tan(FOV / 2)) / this.canvas.height;
		const f = new Float32Array(FRAME_BYTES / 4);
		f.set(viewProj, 0);
		f.set([...right, 0], 16);
		f.set([...up, 0], 20);
		f.set([...m.eye, 1], 24);
		f.set([this.time, aspect, pixel, 0], 28);
		device.queue.writeBuffer(this.frame, 0, f);

		this.reading.update(m.reading);
		this.handoff.update(m.handoff.progress, m.handoff.fade);
		this.painting.update(m.painting);
		const words = [
			...(m.reading.fade > 0.01 ? this.reading.words(m.reading) : []),
			...(m.painting.fade > 0.01 ? this.painting.wordInstances(m.painting) : [])
		];
		this.words.set(words);
		this.post.fade =
			Math.min(1, this.time / 1.5) * (1 - Math.max(0, (this.time - (this.duration - 1.5)) / 1.5));
		// the final image is shown as itself: less bloom, exposure that undoes the tonemap's shoulder
		this.post.bloomStrength = 0.9 - 0.65 * m.finale;
		this.post.exposure = 1 + 0.3 * m.finale;

		const enc = device.createCommandEncoder();
		const pass = enc.beginRenderPass({
			colorAttachments: [
				{ view: this.post.sceneView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }
			]
		});
		if (m.reading.fade > 0.01) this.reading.draw(pass);
		if (m.handoff.fade > 0) this.handoff.draw(pass);
		if (m.painting.fade > 0.01) this.painting.draw(pass, m.painting);
		this.words.draw(pass);
		pass.end();
		this.post.finish(enc, this.time);
		device.queue.submit([enc.finish()]);

		this.onStatus({
			caption: m.caption,
			act: m.act,
			time: this.time,
			duration: this.duration,
			paused: this.paused,
			prompt: this.trace.manifest.prompt
		});
	}
}
