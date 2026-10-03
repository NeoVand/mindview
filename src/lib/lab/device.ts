// Whether this device can hold a page's models before it starts downloading them. A phone gives a web page far less
// graphics memory than a computer (Safari on an iPhone reloads a tab at about 1.5 to 3 GB), and the labs keep about
// 3 GB on the GPU, a live run of the landing or of Paint about 2.8 GB (measured in Chrome on a Mac, 2026-10-02).

export interface Fit {
	ok: boolean;
	why?: string; // what stands in the way, in a sentence
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
	if (phoneLike())
		return {
			ok: false,
			why: `This needs about ${gigabytes} GB of graphics memory, and a phone gives a web page less than that: the tab would most likely reload part way.`
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
