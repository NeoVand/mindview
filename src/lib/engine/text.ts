// Renders words into a single-channel atlas (Canvas2D) so WGSL can draw them as glowing quads.

export interface Glyph {
	u0: number;
	v0: number;
	u1: number;
	v1: number;
	aspect: number; // width / height of the word box
}

export const FONT_FAMILY = `Spectral, "Noto Serif SC", "Noto Serif JP", serif`;

export async function ensureFonts() {
	// Ghost words drift into Chinese and Japanese, so the CJK fallback has to be ready before we rasterise.
	await Promise.allSettled([
		document.fonts.load(`300 64px Spectral`),
		document.fonts.load(`italic 300 64px Spectral`),
		document.fonts.load(`400 64px "Noto Serif SC"`, '在日本灯光保存')
	]);
}

export class TextAtlas {
	texture: GPUTexture;
	private glyphs = new Map<string, Glyph>();

	constructor(device: GPUDevice, words: string[], opts: { px?: number; italic?: boolean } = {}) {
		const px = opts.px ?? 64;
		const pad = Math.ceil(px * 0.35);
		const W = 4096;
		const font = `${opts.italic ? 'italic ' : ''}300 ${px}px ${FONT_FAMILY}`;
		const unique = [...new Set(words.map(clean))].filter(Boolean);
		const measure = new OffscreenCanvas(8, 8).getContext('2d')!;
		measure.font = font;
		const lineH = px + pad * 2;
		let x = 0,
			y = 0;
		const place: { word: string; x: number; y: number; w: number }[] = [];
		for (const word of unique) {
			const w = Math.ceil(measure.measureText(word).width) + pad * 2;
			if (x + w > W) {
				x = 0;
				y += lineH;
			}
			place.push({ word, x, y, w });
			x += w;
		}
		const H = Math.max(64, 2 ** Math.ceil(Math.log2(y + lineH)));
		const canvas = new OffscreenCanvas(W, H);
		const ctx = canvas.getContext('2d')!;
		ctx.font = font;
		ctx.fillStyle = '#fff';
		ctx.textBaseline = 'middle';
		for (const p of place) {
			ctx.fillText(p.word, p.x + pad, p.y + lineH / 2);
			this.glyphs.set(p.word, {
				u0: p.x / W,
				v0: p.y / H,
				u1: (p.x + p.w) / W,
				v1: (p.y + lineH) / H,
				aspect: p.w / lineH
			});
		}
		this.texture = device.createTexture({
			size: [W, H],
			format: 'rgba8unorm',
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.RENDER_ATTACHMENT
		});
		device.queue.copyExternalImageToTexture({ source: canvas }, { texture: this.texture }, [W, H]);
	}

	get(word: string): Glyph | undefined {
		return this.glyphs.get(clean(word));
	}
}

/** Token strings carry leading spaces and control characters; show them the way a reader would. */
export function clean(word: string): string {
	const printable = Array.from(word.replace(/\s+/g, ' '))
		.filter((ch) => ch.charCodeAt(0) >= 32)
		.join('');
	return printable.trim();
}
