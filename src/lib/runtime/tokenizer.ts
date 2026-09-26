// Byte-level BPE tokenizer (GPT-2 / Qwen2 style) built from the vocabulary and merges stored in the GGUF.

const PRETOKENIZE =
	/(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function byteToUnicode(): string[] {
	const bs: number[] = [];
	for (let b = 33; b <= 126; b++) bs.push(b);
	for (let b = 161; b <= 172; b++) bs.push(b);
	for (let b = 174; b <= 255; b++) bs.push(b);
	const cs = [...bs];
	let n = 0;
	for (let b = 0; b < 256; b++)
		if (!bs.includes(b)) {
			bs.push(b);
			cs.push(256 + n++);
		}
	const map: string[] = new Array(256);
	bs.forEach((b, i) => (map[b] = String.fromCodePoint(cs[i])));
	return map;
}

export class Tokenizer {
	private vocab = new Map<string, number>();
	private ranks = new Map<string, number>();
	private byteMap = byteToUnicode();
	private unicodeByte = new Map<string, number>();
	private special: RegExp;
	private specialSet: Set<string>;
	private cache = new Map<string, number[]>();

	constructor(
		readonly tokens: string[],
		merges: string[],
		tokenTypes: number[]
	) {
		tokens.forEach((t, i) => this.vocab.set(t, i));
		merges.forEach((m, i) => this.ranks.set(m, i));
		this.byteMap.forEach((u, b) => this.unicodeByte.set(u, b));
		// control / user-defined tokens (type != 1) are matched whole, before BPE
		const specials = tokens
			.filter((_, i) => tokenTypes[i] !== 1 && tokens[i].length > 1)
			.sort((a, b) => b.length - a.length);
		this.specialSet = new Set(specials);
		const escaped = specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
		this.special = new RegExp(`(${escaped.join('|') || '(?!)'})`);
	}

	encode(text: string): number[] {
		const out: number[] = [];
		for (const part of text.split(this.special)) {
			if (!part) continue;
			if (this.specialSet.has(part)) {
				out.push(this.vocab.get(part)!);
				continue;
			}
			for (const m of part.matchAll(PRETOKENIZE)) out.push(...this.bpe(m[0]));
		}
		return out;
	}

	decode(ids: number[]): string {
		const bytes: number[] = [];
		for (const id of ids)
			for (const ch of this.tokens[id] ?? '') {
				const b = this.unicodeByte.get(ch);
				if (b !== undefined) bytes.push(b);
				else bytes.push(...new TextEncoder().encode(ch));
			}
		return new TextDecoder().decode(new Uint8Array(bytes));
	}

	private bpe(piece: string): number[] {
		const hit = this.cache.get(piece);
		if (hit) return hit;
		let word = Array.from(new TextEncoder().encode(piece), (b) => this.byteMap[b]);
		while (word.length > 1) {
			let best = -1,
				bestRank = Infinity;
			for (let i = 0; i < word.length - 1; i++) {
				const r = this.ranks.get(word[i] + ' ' + word[i + 1]);
				if (r !== undefined && r < bestRank) {
					bestRank = r;
					best = i;
				}
			}
			if (best < 0) break;
			const merged = word[best] + word[best + 1];
			const next: string[] = [];
			for (let i = 0; i < word.length; i++) {
				if (i < word.length - 1 && word[i] === word[best] && word[i + 1] === word[best + 1]) {
					next.push(merged);
					i++;
				} else next.push(word[i]);
			}
			word = next;
		}
		const ids = word.map((w) => {
			const id = this.vocab.get(w);
			if (id === undefined) throw new Error(`Token piece not in vocabulary: ${w}`);
			return id;
		});
		this.cache.set(piece, ids);
		return ids;
	}
}

/** The chat framing the models were trained with (thinking disabled), as the image pipeline uses it. */
export function chatPrompt(user: string): string {
	return `<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}
