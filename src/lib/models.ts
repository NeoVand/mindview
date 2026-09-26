// Where the models come from. In development they are read from static/models when they are there (see the README);
// otherwise, and on the published site, from Hugging Face (a copy kept in the browser is checked against the file's
// ETag):
//   the reader   PrismML's Ternary Bonsai 1.7B, as they publish it (GGUF, 2-bit ternary)
//   the painter  Bonsai Image 4B's DiT, its decoder (TAEF2), the adapter from the reader, and the per-block lens, in the
//                layout research/scripts/export_painter.py writes
//   mindview-t2i all of it for painting in one file (research/scripts/pack_model.py): the reader cut to 9 layers, the
//                map from it, the DiT and TAEF2
import { asset } from '$app/paths';

declare const __LOCAL_MODELS__: boolean;
declare const __LOCAL_PACKED__: boolean;

const READER_HUB =
	'https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/983b5dec2ff16aab79990711ba0f828a499a7e6a/Ternary-Bonsai-1.7B-Q2_0.gguf';
const PAINTER_HUB = 'https://huggingface.co/mohsenvand/mindview-painter/resolve/main';
const PACKED_HUB =
	'https://huggingface.co/mohsenvand/mindview-t2i/resolve/b14079d41508022a5f1a807612c9054c142a8ff5/mindview-t2i.gguf';

/** The reader's GGUF file. */
export function readerUrl(): string {
	return __LOCAL_MODELS__ ? asset('/models/ternary-bonsai-1.7b/model.gguf') : READER_HUB;
}

/** The folder of the painter's files (manifest.json and the files it lists). */
export function painterUrl(): string {
	return __LOCAL_MODELS__ ? asset('/models/bonsai-image-4b') : PAINTER_HUB;
}

/** The one-file text-to-image model (research/scripts/pack_model.py). */
export function packedUrl(): string {
	return __LOCAL_PACKED__ ? asset('/models/mindview-t2i/model.gguf') : PACKED_HUB;
}
