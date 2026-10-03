// The one-file text-to-image model, mindview-t2i (research/scripts/pack_model.py). One GGUF file holds:
//   the reader     Ternary Bonsai 1.7B cut to the layers the conditioning reads, with its tokenizer (a JSON blob)
//   cond           the map from the reader's taps to the DiT's text stream (the adapter and the context embedder in one)
//   the painter    the ternary DiT of Bonsai Image 4B, and TAEF2
// Ternary weights are trits, 5 to a byte, with f16 scales beside them (GGML_TRIT5); scales are stored as byte planes;
// tensors are deflated where that helps (GGML_DEFLATE, inflated by the browser). The file is read in parts, each a
// range request kept on this computer, so the whole of it is never in memory at once.
import { fetchModelRange, modelFileTag } from './cache';
import {
	GGML_DEFLATE,
	GGML_F16,
	GGML_F16_PLANES,
	GGML_F32,
	GGML_JSON,
	GGML_TRIT5,
	halvesToFloats,
	inflateRaw,
	parseGGUF,
	type GGUF,
	type GGUFTensor
} from './gguf';

const READER_PREFIXES = ['token_embd.', 'blk.', 'output_norm.', 'tokenizer.'];

/** A tensor as the runtime wants it: its type after unwrapping (TRIT5, F16, F32 or JSON) and its bytes. */
export interface Payload {
	type: number;
	data: Uint8Array;
}

export class PackedModel {
	private stored: Record<string, number>;

	private constructor(
		readonly url: string,
		readonly header: GGUF,
		private tag: { tag: string; online: boolean }
	) {
		this.stored = JSON.parse(String(header.meta['mindview.stored'] ?? '{}'));
	}

	/** Read the file's header (metadata and tensor table). */
	static async open(url: string): Promise<PackedModel> {
		const tag = await modelFileTag(url);
		for (let len = 4 << 20; ; len *= 4) {
			const buf = await fetchModelRange(url, 0, len, tag);
			try {
				return new PackedModel(url, parseGGUF(buf), tag);
			} catch (e) {
				// the header did not fit in what was read: read more
				if (!(e instanceof RangeError) || len >= 256 << 20) throw e;
			}
		}
	}

	get meta() {
		return this.header.meta;
	}

	/** The painter's settings: text template and taps, image layout, schedule, decoder graph. */
	get painterConfig(): Record<string, unknown> {
		return JSON.parse(String(this.meta['mindview.painter']));
	}

	tensor(name: string): GGUFTensor | undefined {
		return this.header.tensors.get(name);
	}

	/** The tensors of one part, in file order. */
	tensors(part: 'reader' | 'painter'): GGUFTensor[] {
		return [...this.header.tensors.values()]
			.filter((t) => READER_PREFIXES.some((p) => t.name.startsWith(p)) === (part === 'reader'))
			.sort((a, b) => a.offset - b.offset);
	}

	static count(t: GGUFTensor) {
		return t.dims.reduce((a, b) => a * b, 1);
	}

	/** The bytes of a tensor as the runtime uses it (unwrapped). */
	static rawBytes(type: number, n: number) {
		if (type === GGML_TRIT5) return Math.ceil(n / 80) * 16;
		if (type === GGML_F16 || type === GGML_F16_PLANES) return n * 2;
		if (type === GGML_F32) return n * 4;
		if (type === GGML_JSON) return n;
		throw new Error(`Unknown tensor type ${type}.`);
	}

	/** The bytes a tensor takes in the file. */
	bytes(t: GGUFTensor) {
		return t.type === GGML_DEFLATE
			? this.stored[t.name]
			: PackedModel.rawBytes(t.type, PackedModel.count(t));
	}

	/** Unwrap a tensor's stored bytes: inflate, and put byte planes back together. */
	static async payload(t: GGUFTensor, stored: Uint8Array): Promise<Payload> {
		let type = t.type,
			data = stored;
		if (type === GGML_DEFLATE) {
			const h = new DataView(stored.buffer, stored.byteOffset, 16);
			type = h.getUint32(0, true);
			const raw = h.getUint32(4, true),
				packed = h.getUint32(8, true);
			data = await inflateRaw(stored.subarray(16, 16 + packed));
			if (data.length !== raw)
				throw new Error(`${t.name} inflated to ${data.length} bytes, not ${raw}.`);
		}
		if (type === GGML_F16_PLANES) {
			const n = data.length / 2;
			const out = new Uint8Array(data.length);
			for (let i = 0; i < n; i++) {
				out[2 * i] = data[n + i]; // little-endian: the low byte first
				out[2 * i + 1] = data[i];
			}
			type = GGML_F16;
			data = out;
		}
		return { type, data };
	}

	/**
	 * The reader's tensors as a GGUF of their own (unwrapped, in one buffer), with the tokenizer (and its chat template)
	 * put back into its metadata, to load as the Qwen3 it is.
	 */
	async readerGGUF(onProgress?: (fraction: number) => void): Promise<GGUF> {
		const ts = this.tensors('reader');
		const start = ts[0].offset,
			end = Math.max(...ts.map((t) => t.offset + this.bytes(t)));
		const buf = await fetchModelRange(this.url, start, end, this.tag, onProgress);
		const parts: [GGUFTensor, Payload][] = [];
		for (const t of ts)
			parts.push([
				t,
				await PackedModel.payload(t, new Uint8Array(buf, t.offset - start, this.bytes(t)))
			]);
		// the file as a whole is mindview-t2i; this part of it is a Qwen3 (its settings are the qwen3.* keys)
		const meta = { ...this.meta, 'general.architecture': 'qwen3' };
		const size = parts.reduce((a, [, p]) => a + Math.ceil(p.data.length / 32) * 32, 0);
		const out = new Uint8Array(size);
		const tensors = new Map<string, GGUFTensor>();
		let o = 0;
		for (const [t, p] of parts) {
			if (p.type === GGML_JSON) {
				Object.assign(meta, JSON.parse(new TextDecoder().decode(p.data)));
				continue;
			}
			out.set(p.data, o);
			tensors.set(t.name, { ...t, type: p.type, offset: o });
			o += Math.ceil(p.data.length / 32) * 32;
		}
		return { meta, tensors, buffer: out.buffer };
	}

	/**
	 * Read tensors in parts of about `chunk` bytes, in file order, handing each part's tensors (unwrapped) to `each`; a
	 * ternary tensor always comes with its `.scale` in the same part.
	 */
	async read(
		ts: GGUFTensor[],
		each: (part: Map<string, Payload>) => void,
		onProgress?: (fraction: number) => void,
		chunk = 96 << 20,
		// only these tensors: parts with none of them are skipped, the others keep the same ranges (and so the same
		// copies in the browser's cache)
		want?: (t: GGUFTensor) => boolean
	) {
		const units: GGUFTensor[][] = [];
		for (const t of ts) {
			const last = units[units.length - 1];
			if (t.name.endsWith('.scale') && last?.[0].name === t.name.slice(0, -6)) last.push(t);
			else units.push([t]);
		}
		const total = ts.reduce((a, t) => a + this.bytes(t), 0);
		const end = (u: GGUFTensor[]) => Math.max(...u.map((t) => t.offset + this.bytes(t)));
		let done = 0;
		for (let i = 0; i < units.length;) {
			const group: GGUFTensor[] = [...units[i++]];
			const start = group[0].offset;
			let stop = end(group);
			while (i < units.length && end(units[i]) - start <= chunk) {
				group.push(...units[i]);
				stop = end(units[i++]);
			}
			if (want && !group.some(want)) {
				done += group.reduce((a, t) => a + this.bytes(t), 0);
				continue;
			}
			const buf = await fetchModelRange(this.url, start, stop, this.tag, (f) =>
				onProgress?.((done + f * (stop - start)) / total)
			);
			const part = new Map<string, Payload>();
			for (const t of group)
				if (!want || want(t))
					part.set(
						t.name,
						await PackedModel.payload(t, new Uint8Array(buf, t.offset - start, this.bytes(t)))
					);
			each(part);
			done += group.reduce((a, t) => a + this.bytes(t), 0);
			onProgress?.(done / total);
		}
	}

	/** A dense payload's values as f32. */
	static floats(p: Payload, name = ''): Float32Array {
		const copy = p.data.slice().buffer;
		if (p.type === GGML_F32) return new Float32Array(copy);
		if (p.type === GGML_F16) return halvesToFloats(new Uint16Array(copy));
		throw new Error(`${name} is not dense.`);
	}
}
