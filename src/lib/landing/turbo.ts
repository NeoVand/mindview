// The landing's models: the one-file mindview-t2i-turbo, read once and kept while the site is open. The reader (its
// first part, 158 MB) arrives first and can start reading while the painter (the rest) is still downloading.
import { packedUrl } from '$lib/models';
import { ensureFonts } from '$lib/engine/text';
import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
import { PackedModel } from '$lib/runtime/packed';
import { Painter } from '$lib/runtime/painter';
import type { ThreadsModel } from '$lib/viz/threads';

export interface TurboProgress {
	stage: 'reader' | 'painter';
	fraction: number; // of that part
	megabytes: number; // downloaded so far, both parts
	total: number; // megabytes, both parts
}

let packed: Promise<PackedModel> | undefined;
let reader: Promise<BonsaiLLM> | undefined;
let painter: Promise<Painter> | undefined;
const listeners = new Set<(p: TurboProgress) => void>();

function file() {
	packed ??= PackedModel.open(packedUrl()).catch((e) => {
		packed = undefined;
		throw e;
	});
	return packed;
}

/** Sizes of the two parts, in megabytes. */
async function sizes() {
	const p = await file();
	const size = (part: 'reader' | 'painter') =>
		p.tensors(part).reduce((a, t) => a + p.bytes(t), 0) / 1e6;
	return { reader: size('reader'), painter: size('painter') };
}

/** What the threads piece needs to know about this model. */
export async function turboModel(): Promise<ThreadsModel> {
	const p = await file();
	const text = p.painterConfig.text as { taps?: number[] };
	return {
		taps: text.taps ?? [3, 6, 9],
		readerWeights: '764 million',
		painterWeights: '3.7 billion',
		textRows: 256,
		fast: true
	};
}

/** The reader (Ternary Bonsai 1.7B, its first layers), on the device. */
export async function turboReader(
	device: GPUDevice,
	onProgress?: (p: TurboProgress) => void
): Promise<BonsaiLLM> {
	if (onProgress) listeners.add(onProgress);
	try {
		reader ??= (async () => {
			const p = await file();
			const s = await sizes();
			const total = s.reader + s.painter;
			const [gguf] = await Promise.all([
				p.readerGGUF((f) =>
					listeners.forEach((l) =>
						l({ stage: 'reader', fraction: f, megabytes: f * s.reader, total })
					)
				),
				ensureFonts()
			]);
			return BonsaiLLM.fromGGUF(device, gguf);
		})().catch((e) => {
			reader = undefined;
			throw e;
		});
		return await reader;
	} finally {
		if (onProgress) listeners.delete(onProgress);
	}
}

/** The painter (the ternary DiT, its side branches and the decoder), on the device. */
export async function turboPainter(
	device: GPUDevice,
	onProgress?: (p: TurboProgress) => void
): Promise<Painter> {
	if (onProgress) listeners.add(onProgress);
	try {
		painter ??= (async () => {
			const p = await file();
			const s = await sizes();
			const total = s.reader + s.painter;
			return Painter.fromPacked(device, p, (e) =>
				listeners.forEach((l) =>
					l({
						stage: 'painter',
						fraction: e.fraction,
						megabytes: s.reader + e.fraction * s.painter,
						total
					})
				)
			);
		})().catch((e) => {
			painter = undefined;
			throw e;
		});
		return await painter;
	} finally {
		if (onProgress) listeners.delete(onProgress);
	}
}
