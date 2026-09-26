// Minimal GGUF v3 reader: metadata, tensor table, and raw tensor bytes. Tolerates PrismML's custom quant types.

export type GGUFValue = string | number | bigint | boolean | GGUFValue[];

export interface GGUFTensor {
	name: string;
	dims: number[]; // ne0 (contiguous) first
	type: number;
	offset: number; // absolute byte offset in the file
}

export interface GGUF {
	meta: Record<string, GGUFValue>;
	tensors: Map<string, GGUFTensor>;
	buffer: ArrayBuffer;
}

export const GGML_F32 = 0;
export const GGML_F16 = 1;
export const GGML_Q2_0_G128_LEGACY = 42; // PrismML ternary g128 in 2-bit slots (older files)
export const GGML_PQ2_0 = 142; // same block layout under its current id

export function parseGGUF(buffer: ArrayBuffer): GGUF {
	const dv = new DataView(buffer);
	const dec = new TextDecoder();
	let p = 0;
	const u32 = () => ((p += 4), dv.getUint32(p - 4, true));
	const u64 = () => ((p += 8), Number(dv.getBigUint64(p - 8, true)));
	const str = () => {
		const n = u64();
		const s = dec.decode(new Uint8Array(buffer, p, n));
		p += n;
		return s;
	};
	const scalar = (t: number): GGUFValue => {
		switch (t) {
			case 0:
				return ((p += 1), dv.getUint8(p - 1));
			case 1:
				return ((p += 1), dv.getInt8(p - 1));
			case 2:
				return ((p += 2), dv.getUint16(p - 2, true));
			case 3:
				return ((p += 2), dv.getInt16(p - 2, true));
			case 4:
				return u32();
			case 5:
				return ((p += 4), dv.getInt32(p - 4, true));
			case 6:
				return ((p += 4), dv.getFloat32(p - 4, true));
			case 7:
				return ((p += 1), dv.getUint8(p - 1) !== 0);
			case 8:
				return str();
			case 10:
				return ((p += 8), dv.getBigUint64(p - 8, true));
			case 11:
				return ((p += 8), dv.getBigInt64(p - 8, true));
			case 12:
				return ((p += 8), dv.getFloat64(p - 8, true));
			default:
				throw new Error(`Unknown GGUF value type ${t}`);
		}
	};
	const value = (t: number): GGUFValue => {
		if (t !== 9) return scalar(t);
		const at = u32();
		const n = u64();
		const arr: GGUFValue[] = new Array(n);
		for (let i = 0; i < n; i++) arr[i] = scalar(at);
		return arr;
	};

	const magic = dec.decode(new Uint8Array(buffer, 0, 4));
	if (magic !== 'GGUF') throw new Error('This file is not a GGUF model.');
	p = 4;
	const version = u32();
	if (version < 2) throw new Error(`GGUF version ${version} is too old.`);
	const nTensors = u64();
	const nKV = u64();
	const meta: Record<string, GGUFValue> = {};
	for (let i = 0; i < nKV; i++) {
		const key = str();
		meta[key] = value(u32());
	}
	const infos: GGUFTensor[] = [];
	for (let i = 0; i < nTensors; i++) {
		const name = str();
		const nd = u32();
		const dims: number[] = [];
		for (let d = 0; d < nd; d++) dims.push(u64());
		const type = u32();
		const offset = u64();
		infos.push({ name, dims, type, offset });
	}
	const align = Number(meta['general.alignment'] ?? 32);
	const dataStart = Math.ceil(p / align) * align;
	const tensors = new Map<string, GGUFTensor>();
	for (const t of infos) tensors.set(t.name, { ...t, offset: dataStart + t.offset });
	return { meta, tensors, buffer };
}

export function halfToFloat(h: number): number {
	const s = h & 0x8000 ? -1 : 1;
	const e = (h >> 10) & 0x1f;
	const f = h & 0x3ff;
	if (e === 0) return s * f * 2 ** -24;
	if (e === 31) return f ? NaN : s * Infinity;
	return s * (1 + f / 1024) * 2 ** (e - 15);
}
