// Where the models come from. In development they are read from static/models when they are there (see the README);
// otherwise, and on the published site, from Hugging Face, each pinned to a revision so a copy kept in the browser
// stays valid:
//   the reader   PrismML's Ternary Bonsai 1.7B, as they publish it (GGUF, 2-bit ternary)
//   the painter  Bonsai Image 4B's DiT, its decoder (TAEF2), the adapter from the reader, and the per-block lens, in the
//                layout research/scripts/export_painter.py writes
import { asset } from '$app/paths';

declare const __LOCAL_MODELS__: boolean;

const READER_HUB =
	'https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/983b5dec2ff16aab79990711ba0f828a499a7e6a/Ternary-Bonsai-1.7B-Q2_0.gguf';
const PAINTER_HUB = 'https://huggingface.co/NeoVand/mindview-painter/resolve/main';

/** The reader's GGUF file. */
export function readerUrl(): string {
	return __LOCAL_MODELS__ ? asset('/models/ternary-bonsai-1.7b/model.gguf') : READER_HUB;
}

/** The folder of the painter's files (manifest.json and the files it lists). */
export function painterUrl(): string {
	return __LOCAL_MODELS__ ? asset('/models/bonsai-image-4b') : PAINTER_HUB;
}
