// One graphics device and one copy of the reader for every lab: moving from one lab to another keeps them, so a
// switch costs a new canvas, not a new download (the model stays on the GPU).
import { asset } from '$app/paths';
import { type GPU } from '$lib/engine/gpu';
import { ensureFonts } from '$lib/engine/text';
import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
import { Painter, type PainterProgress } from '$lib/runtime/painter';
import { GpuScheduler } from '$lib/runtime/scheduler';
import { PaintingRun, type RunOptions } from './painting-run';

let device: Promise<GPUDevice> | undefined;
let llm: Promise<BonsaiLLM> | undefined;
let painter: Promise<Painter> | undefined;
const painterListeners = new Set<(p: PainterProgress) => void>();
const listeners = new Set<(f: number) => void>();

function makeDevice(): Promise<GPUDevice> {
	return (async () => {
		if (!navigator.gpu)
			throw new Error(
				'This browser has no WebGPU. Open the piece in a recent Chrome, Edge or Safari.'
			);
		const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
		if (!adapter) throw new Error('No WebGPU adapter is available on this machine.');
		const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query')
			? ['timestamp-query']
			: [];
		const dev = await adapter.requestDevice({
			requiredFeatures,
			requiredLimits: {
				maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
				maxBufferSize: adapter.limits.maxBufferSize,
				maxStorageBuffersPerShaderStage: Math.min(
					16,
					adapter.limits.maxStorageBuffersPerShaderStage
				)
			}
		});
		dev.lost.then((info) => {
			console.error('WebGPU device lost:', info.message);
			device = undefined;
			llm = undefined;
			painter = undefined;
		});
		return dev;
	})();
}

/** The shared device, with this canvas set up to show it. */
export async function labGPU(canvas: HTMLCanvasElement): Promise<GPU> {
	device ??= makeDevice();
	const dev = await device;
	const context = canvas.getContext('webgpu');
	if (!context) throw new Error('Could not create a WebGPU canvas context.');
	const format = navigator.gpu.getPreferredCanvasFormat();
	context.configure({ device: dev, format, alphaMode: 'opaque' });
	return { device: dev, context, format, canvas };
}

/** The shared reader (Ternary Bonsai 1.7B), loaded once. onProgress hears the download (0..1). */
export async function labModel(
	dev: GPUDevice,
	onProgress?: (f: number) => void
): Promise<BonsaiLLM> {
	if (onProgress) listeners.add(onProgress);
	try {
		llm ??= Promise.all([
			BonsaiLLM.load(dev, asset('/models/ternary-bonsai-1.7b/model.gguf'), (f) =>
				listeners.forEach((l) => l(f))
			),
			ensureFonts()
		])
			.then(([m]) => m)
			.catch((e) => {
				llm = undefined;
				throw e;
			});
		return await llm;
	} finally {
		if (onProgress) listeners.delete(onProgress);
	}
}

/** The shared painter (Bonsai Image 4B and its decoder, about 1.1 GB), loaded once. */
export async function labPainter(
	dev: GPUDevice,
	onProgress?: (p: PainterProgress) => void
): Promise<Painter> {
	if (onProgress) painterListeners.add(onProgress);
	try {
		painter ??= Painter.load(dev, asset('/models/bonsai-image-4b'), (p) =>
			painterListeners.forEach((l) => l(p))
		).catch((e) => {
			painter = undefined;
			throw e;
		});
		return await painter;
	} finally {
		if (onProgress) painterListeners.delete(onProgress);
	}
}

let scheduler: GpuScheduler | undefined;

/** One queue for all the painting work, whichever lab is showing (each lab's stage pumps it every frame). */
export function labScheduler(dev: GPUDevice): GpuScheduler {
	scheduler ??= new GpuScheduler(dev);
	return scheduler;
}

let run: { key: string; run: PaintingRun } | undefined;

/** The patches every painting follows: a 4 x 4 grid, and the middle of the picture. */
export const FOLLOWED_PATCHES = [
	...Array.from({ length: 16 }, (_, i) => (4 + 8 * Math.floor(i / 4)) * 32 + 4 + 8 * (i % 4)),
	16 * 32 + 16
];

/**
 * The painting of a prompt (kept, and reused by every lab, while the prompt, seed and steps stay the same; it goes on
 * painting while you move between labs). By default every word and every followed patch keeps everything at the last
 * step (about 500 MB of capture in all), so every lab can share the one painting.
 */
export function labPainting(
	dev: GPUDevice,
	llm: BonsaiLLM,
	p: Painter,
	prompt: string,
	opts: Partial<RunOptions> = {}
): PaintingRun {
	const steps = opts.steps ?? 4;
	const fullStep = opts.fullStep ?? -1;
	const o: RunOptions = {
		seed: opts.seed ?? 7,
		steps,
		patches: opts.patches ?? FOLLOWED_PATCHES,
		full: opts.full ?? 'followed',
		fullStep: fullStep >= steps - 1 ? -1 : fullStep // -1: the last
	};
	const key = JSON.stringify([prompt, o]);
	if (run?.key === key) return run.run;
	const sch = labScheduler(dev);
	sch.clear();
	run?.run.destroy();
	run = { key, run: new PaintingRun(dev, llm, p, sch, prompt, o) };
	return run.run;
}

/** Give the painter up (a lab that paints on its own, like Threads, uses its activations): the shared painting stops. */
export function releasePainting() {
	if (!run) return;
	scheduler?.clear();
	run.run.destroy();
	run = undefined;
}
