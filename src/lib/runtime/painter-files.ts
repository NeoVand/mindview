// Reading the painter's exported files (research/scripts/export_painter.py -> static/models/bonsai-image-4b/):
// manifest.json lists every tensor with its file and byte offsets; ternary tensors use the same packing as the
// language model (16 two-bit codes per u32, one f32 scale per 128 inputs).
import { fetchModelFile, fetchModelJson } from './cache';
import { halfToFloat } from './gguf';

export interface DenseEntry {
	kind: 'dense';
	dtype: 'f16' | 'f32';
	shape: number[];
	file: string;
	offset: number;
	bytes: number;
}
export interface TernaryEntry {
	kind: 'ternary';
	shape: [number, number];
	file: string;
	codes: { offset: number; bytes: number };
	scales: { offset: number; bytes: number };
}
export interface PainterManifest {
	tensors: Record<string, DenseEntry | TernaryEntry>;
	config: Record<string, unknown>;
	[key: string]: unknown;
}

export class PainterFiles {
	private files = new Map<string, ArrayBuffer>();

	private constructor(
		readonly base: string,
		readonly manifest: PainterManifest
	) {}

	static async open(base: string): Promise<PainterFiles> {
		return new PainterFiles(base, await fetchModelJson<PainterManifest>(`${base}/manifest.json`));
	}

	/** Load the named files (from this computer when kept there, else downloaded), with progress over their size. */
	async fetch(names: string[], onProgress?: (fraction: number) => void) {
		const todo = names.filter((n) => !this.files.has(n));
		const listed = this.manifest.files as Record<string, { bytes: number }> | undefined;
		const sizes = todo.map((n) => listed?.[n]?.bytes ?? 1);
		const total = sizes.reduce((a, b) => a + b, 0) || 1;
		let got = 0;
		for (const [i, n] of todo.entries()) {
			const buf = await fetchModelFile(`${this.base}/${n}`, (f) =>
				onProgress?.((got + f * sizes[i]) / total)
			);
			got += sizes[i];
			this.files.set(n, buf);
		}
	}

	/** Forget a downloaded file (after its tensors are on the GPU). */
	release(name: string) {
		this.files.delete(name);
	}

	names(prefix: string) {
		return Object.keys(this.manifest.tensors).filter((k) => k.startsWith(prefix));
	}

	entry(name: string) {
		const e = this.manifest.tensors[name];
		if (!e) throw new Error(`The painter has no tensor ${name}.`);
		return e;
	}

	/** A dense tensor as f32. */
	dense(name: string): Float32Array {
		const e = this.entry(name);
		if (e.kind !== 'dense') throw new Error(`${name} is not dense.`);
		const buf = this.file(e.file);
		const n = e.bytes / (e.dtype === 'f16' ? 2 : 4);
		if (e.dtype === 'f32') return new Float32Array(buf.slice(e.offset, e.offset + e.bytes));
		const h = new Uint16Array(buf, e.offset, n);
		const out = new Float32Array(n);
		for (let i = 0; i < n; i++) out[i] = halfToFloat(h[i]);
		return out;
	}

	/** A ternary tensor's packed codes and scales (views into the downloaded file). */
	ternary(name: string): { rows: number; cols: number; codes: Uint32Array; scales: Float32Array } {
		const e = this.entry(name);
		if (e.kind !== 'ternary') throw new Error(`${name} is not ternary.`);
		const buf = this.file(e.file);
		return {
			rows: e.shape[0],
			cols: e.shape[1],
			codes: new Uint32Array(buf, e.codes.offset, e.codes.bytes / 4),
			scales: new Float32Array(buf, e.scales.offset, e.scales.bytes / 4)
		};
	}

	private file(name: string) {
		const f = this.files.get(name);
		if (!f) throw new Error(`${name} has not been downloaded.`);
		return f;
	}
}
