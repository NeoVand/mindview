// Whether this device can hold a page's models before it starts downloading them. A phone gives a web page far less
// graphics memory than a computer (Safari on an iPhone reloads a tab at about 1.5 to 3 GB). Measured in Chrome on a Mac
// (2026-10-03): the labs keep about 3 GB on the GPU; a live run of the landing about 2.9 GB, Paint about 2.7 GB, and
// with the painter held tightly as on a phone (Painter's lean mode) 2.1 and 2.0 GB, no buffer over 256 MB.

export interface Fit {
	ok: boolean;
	why?: string; // what stands in the way, in a sentence
}

/**
 * Whether to hold the models' memory tightly (Painter's lean mode): on a phone, and in development with ?phone (which
 * also limits buffers as a phone may, see lab/shared.ts), to try the phone's path on a computer.
 */
export function lean(): boolean {
	return phoneLike() || (import.meta.env.DEV && new URLSearchParams(location.search).has('phone'));
}

/** A phone (iPadOS calls itself a Mac and is not counted: iPads have more memory). */
export function phoneLike(): boolean {
	const ua = navigator.userAgent;
	const data = (navigator as Navigator & { userAgentData?: { mobile?: boolean } }).userAgentData;
	return data?.mobile === true || /iPhone|iPod|Android.*Mobile|Mobile.*Firefox/.test(ua);
}

/** Can this device hold `gigabytes` of models on its GPU, with no buffer larger than `largest` MB? */
export async function gpuFit(gigabytes: number, largest: number): Promise<Fit> {
	if (!navigator.gpu)
		return {
			ok: false,
			why: 'This browser has no WebGPU. Open the piece in a recent Chrome, Edge or Safari on a computer.'
		};
	const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
	if (!adapter) return { ok: false, why: 'No WebGPU graphics adapter is available here.' };
	const most = Math.min(adapter.limits.maxBufferSize, adapter.limits.maxStorageBufferBindingSize);
	if (most < largest * 2 ** 20)
		return {
			ok: false,
			why: `This graphics card lets a web page make blocks of memory of up to ${Math.floor(most / 2 ** 20)} MB; the models need one of ${largest} MB.`
		};
	if (phoneLike() && !(import.meta.env.DEV && new URLSearchParams(location.search).has('phone')))
		return {
			ok: false,
			why:
				gigabytes <= 2.5
					? `This needs about ${gigabytes} GB of graphics memory, held as tightly as it can be on a phone. A recent phone with 8 GB of memory may manage it; on others the tab may reload part way.`
					: `This needs about ${gigabytes} GB of graphics memory, and a phone gives a web page less than that: the tab would most likely reload part way.`
		};
	return { ok: true };
}

let chose = false; // the visitor chose to try once already (in this visit)

/**
 * Resolves when the page may load its models: at once where they fit (or the visitor already chose to try), otherwise
 * once the visitor chooses to try anyway (onBlocked hears why, with the function that lets them through).
 */
export async function mayLoad(
	gigabytes: number,
	largest: number,
	onBlocked: (why: string, go: () => void) => void
): Promise<void> {
	if (chose) return;
	const fit = await gpuFit(gigabytes, largest);
	if (fit.ok) return;
	await new Promise<void>((go) => onBlocked(fit.why!, go));
	chose = true;
}
