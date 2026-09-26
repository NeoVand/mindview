// One real painting for the labs: the reader re-reads the prompt at 512 tokens, the adapter turns its layers 7, 14
// and 21 into the painter's conditioning, and the painter runs its steps, block by block, through the scheduler (a
// few milliseconds per frame). While it paints it keeps, for the prompt's words and a few image patches, the input of
// every matrix at every block of every step (and, for one patch and one word, everything the blocks compute; see
// PainterCapture), plus the picture each block has in mind and the finished picture.
import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
import { Painter, PainterCapture } from '$lib/runtime/painter';
import type { GpuScheduler } from '$lib/runtime/scheduler';
import { chatPrompt } from '$lib/runtime/tokenizer';

const { NT, NI } = Painter.dims;

export interface RunOptions {
	seed: number;
	steps: number;
	/** Image patches to follow (0..1023, row-major on the 32 x 32 grid). */
	patches: number[];
	/**
	 * Which followed rows get everything (joint indices: text 0..511, image 512 + patch), or 'followed' for all of
	 * them (every word and every followed patch).
	 */
	full: number[] | 'followed';
	/** The step at which they get it (the other steps keep what every followed row keeps); -1 the last. */
	fullStep: number;
}

export class PaintingRun {
	readonly cap: PainterCapture;
	/** The picture after each block: [step][block] (64 x 64), filled in as it paints. */
	readonly pictures: (GPUTexture | null)[][];
	/** The finished picture (512 x 512) once done. */
	final: GPUTexture | null = null;
	/** Joint rows of the prompt's own words. */
	readonly words: number[];
	readonly tokens: string[];
	/** How far it has got: the last block done (step, block); block 25 = the step is done. */
	at = { step: 0, block: -1 };
	encoded = false;
	done = false;
	readonly steps: number;
	/** The step at which the full rows keep everything. */
	readonly fullStep: number;
	private textures: GPUTexture[] = [];
	private dead = false;

	constructor(
		private device: GPUDevice,
		llm: BonsaiLLM,
		readonly painter: Painter,
		scheduler: GpuScheduler,
		readonly prompt: string,
		readonly opts: RunOptions
	) {
		painter.setSteps(opts.steps);
		this.steps = painter.steps;
		painter.setNoise(undefined, opts.seed);
		const ids = llm.tokenizer.encode(chatPrompt(prompt)).slice(0, NT);
		this.tokens = ids.map((i) => llm.tokenizer.decode([i]));
		const end = this.tokens.findIndex((t, i) => i > 3 && t.startsWith('<|im_end'));
		this.words = Array.from(
			{ length: Math.max(1, (end > 0 ? end : ids.length) - 3) },
			(_, i) => 3 + i
		);
		const patchRows = opts.patches.map((p) => NT + p);
		const full = opts.full === 'followed' ? [...this.words, ...patchRows] : opts.full;
		this.fullStep = opts.fullStep < 0 ? this.steps - 1 : Math.min(opts.fullStep, this.steps - 1);
		this.cap = new PainterCapture(device, this.steps, this.words, patchRows, full, [this.fullStep]);
		this.pictures = Array.from({ length: this.steps }, () => new Array(25).fill(null));
		const { tasks } = painter.encodeTasks(llm, prompt, scheduler.slice);
		tasks.push({ cost: 0, record: () => {}, done: () => void (this.encoded = true) });
		scheduler.push(...tasks);
		scheduler.push(
			...painter.captureTasks(this.cap, {
				budget: scheduler.slice,
				onPicture: (enc, s, b) => {
					const src = painter.taef2.earlyTexture;
					const tex = this.device.createTexture({
						size: [src.width, src.height],
						format: 'rgba8unorm',
						usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
					});
					enc.copyTextureToTexture({ texture: src }, { texture: tex }, [src.width, src.height]);
					this.textures.push(tex);
					this.pictures[s][b] = tex;
				},
				onBlock: (s, b) => {
					if (!this.dead) this.at = { step: s, block: b };
				},
				onStep: (s) => {
					if (!this.dead) this.at = { step: s, block: 25 };
				},
				onDone: () => {
					if (this.dead) return;
					const src = painter.taef2.texture;
					const tex = this.device.createTexture({
						size: [src.width, src.height],
						format: 'rgba8unorm',
						usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
					});
					const enc = this.device.createCommandEncoder();
					enc.copyTextureToTexture({ texture: src }, { texture: tex }, [src.width, src.height]);
					this.device.queue.submit([enc.finish()]);
					this.textures.push(tex);
					this.final = tex;
					this.done = true;
				}
			})
		);
	}

	/** How much of the painting is done, 0..1. */
	get progress() {
		if (this.done) return 1;
		return (this.at.step * 26 + this.at.block + 1) / (this.steps * 26);
	}

	/** Copy regions of the capture back (floats). */
	async read(regions: { offset: number; count: number }[]): Promise<Float32Array[]> {
		const total = regions.reduce((a, r) => a + r.count, 0);
		if (!total) return regions.map(() => new Float32Array(0));
		const buf = this.device.createBuffer({
			size: total * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = this.device.createCommandEncoder();
		let p = 0;
		for (const r of regions) {
			enc.copyBufferToBuffer(this.cap.buffer, r.offset * 4, buf, p * 4, r.count * 4);
			p += r.count;
		}
		this.device.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const all = new Float32Array(buf.getMappedRange().slice(0));
		buf.destroy();
		p = 0;
		return regions.map((r) => all.subarray(p, (p += r.count)));
	}

	/** Whether block b of step s has run. */
	has(s: number, b: number) {
		return this.done || s < this.at.step || (s === this.at.step && b <= this.at.block);
	}

	/** The joint row of an image patch. */
	static patchRow(p: number) {
		return NT + p;
	}

	static readonly grid = Math.sqrt(NI);

	destroy() {
		this.dead = true;
		this.cap.destroy();
		for (const t of this.textures) t.destroy();
	}
}
