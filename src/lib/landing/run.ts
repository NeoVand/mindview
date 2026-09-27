// One painting as the landing shows it: the reader's attention per layer, then the painter's two passes (a sketch at
// 16 x 16 patches, the picture at 32 x 32), block by block. Loaded from a recorded run (research/scripts/
// capture_fast.py, export_landing_capture.py); a live run will fill the same shape.

export interface LandingRun {
	prompt: string;
	/** The tokens the reader reads (chat template included) and the range that is the prompt itself. */
	tokens: string[];
	first: number;
	end: number;
	/** Reader attention per layer (1..9), mean over heads: [layer][to][from]. */
	reader: number[][][];
	/** Painter attention per pass and block: maps[pass][block][token] = the image rows' attention to it (side²). */
	maps: [Float32Array[][], Float32Array[][]];
	sides: [number, number];
	/** Pictures: the noise it starts from, the sketch, the sketch noised back, the picture. */
	pictures: { noise: ImageBitmap; sketch: ImageBitmap; renoised: ImageBitmap; final: ImageBitmap };
}

function f16(bits: number) {
	const s = bits & 0x8000 ? -1 : 1,
		e = (bits >> 10) & 0x1f,
		m = bits & 0x3ff;
	if (e === 0) return s * m * 2 ** -24;
	if (e === 31) return m ? NaN : s * Infinity;
	return s * (1 + m / 1024) * 2 ** (e - 15);
}

async function image(url: string) {
	return createImageBitmap(await (await fetch(url)).blob());
}

/** A recorded run from a folder written by export_landing_capture.py. */
export async function loadRun(base: string): Promise<LandingRun> {
	const meta = await (await fetch(`${base}/meta.json`)).json();
	const reader = await (await fetch(`${base}/reader.json`)).json();
	const n: number = meta.n_real;
	const sides: [number, number] = meta.sides;
	const maps = (await Promise.all(
		[0, 1].map(async (s) => {
			const raw = new Uint16Array(await (await fetch(`${base}/maps_s${s}.bin`)).arrayBuffer());
			const ni = sides[s] * sides[s];
			return Array.from({ length: 25 }, (_, b) =>
				Array.from({ length: n }, (_, k) => {
					const out = new Float32Array(ni);
					const at = (b * n + k) * ni;
					for (let i = 0; i < ni; i++) out[i] = f16(raw[at + i]);
					return out;
				})
			);
		})
	)) as [Float32Array[][], Float32Array[][]];
	const [noise, sketch, renoised, final] = await Promise.all(
		['state_s0', 'sketch', 'state_s1', 'final'].map((f) => image(`${base}/${f}.png`))
	);
	const tokens: string[] = meta.words;
	return {
		prompt: meta.prompt,
		tokens,
		first: 3, // <|im_start|> user \n come first
		end: tokens.indexOf('<|im_end|>'),
		reader,
		maps,
		sides,
		pictures: { noise, sketch, renoised, final }
	};
}
