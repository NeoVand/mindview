// The landing's models: the one-file mindview-t2i-turbo, read once and kept while the site is open. The reader (its
// first part, 158 MB) arrives first and can start reading while the painter (the rest) is still downloading. Beside it,
// the painter's readouts for the visuals (20 MB, from mindview-painter): the tuned lens, which shows the picture each
// block has in mind, as /lab/threads does.
import { packedUrl, painterUrl, TURBO_REVISION } from '$lib/models';
import { ensureFonts } from '$lib/engine/text';
import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
import { PackedModel } from '$lib/runtime/packed';
import { LEAN, Painter } from '$lib/runtime/painter';
import { lean } from '$lib/lab/device';
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

/** What the threads piece needs to know about this model (pack_model.py taps the reader after layers 3, 6 and 9). */
export const TURBO: ThreadsModel = {
	taps: [3, 6, 9],
	readerWeights: '764 million',
	painterWeights: '3.7 billion',
	textRows: 256,
	fast: true,
	name: `mohsenvand/mindview-t2i-turbo@${TURBO_REVISION}`
};

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
			const [p, viz] = await Promise.all([file(), Painter.loadViz(painterUrl()).catch(() => null)]);
			const s = await sizes();
			const total = s.reader + s.painter;
			return Painter.fromPacked(
				device,
				p,
				(e) =>
					listeners.forEach((l) =>
						l({
							stage: 'painter',
							fraction: e.fraction,
							megabytes: s.reader + e.fraction * s.painter,
							total
						})
					),
				viz,
				// on a phone, memory held tightly (the landing paints Fast: its branch is the one on the GPU)
				lean() ? LEAN : null,
				'lora'
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
