// Model files, kept on this computer after the first download. Chrome's HTTP cache will not keep files this large (the
// reader alone is 460 MB), so they go into Cache Storage. A copy is used as long as the server's ETag (size and
// modification time) still matches, so a changed file is downloaded again; without a network the copy is used as is.

const CACHE = 'mindview-models-v1';
const TAG = 'x-mindview-tag';

async function open(): Promise<Cache | null> {
	try {
		return typeof caches === 'undefined' ? null : await caches.open(CACHE);
	} catch {
		return null;
	}
}

/** The file at url, from this computer when it is up to date there, else downloaded (and kept). */
export async function fetchModelFile(
	url: string,
	onProgress?: (fraction: number) => void
): Promise<ArrayBuffer> {
	let tag = '';
	let online = true;
	try {
		const head = await fetch(url, { method: 'HEAD', cache: 'no-store' });
		if (!head.ok) throw new Error(`Could not find ${url} (${head.status}).`);
		tag = head.headers.get('etag') ?? head.headers.get('last-modified') ?? '';
	} catch (e) {
		if (e instanceof Error && e.message.startsWith('Could not find')) throw e;
		online = false;
	}
	const cache = await open();
	if (cache) {
		const kept = await cache.match(url);
		if (kept && (!online || (tag && kept.headers.get(TAG) === tag))) {
			const buf = await kept.arrayBuffer();
			onProgress?.(1);
			return buf;
		}
	}
	const res = await fetch(url, { cache: 'no-store' });
	if (!res.ok || !res.body) throw new Error(`Could not download ${url} (${res.status}).`);
	const total = Number(res.headers.get('content-length')) || 0;
	const reader = res.body.getReader();
	const chunks: Uint8Array[] = [];
	let got = 0;
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		got += value.length;
		if (total) onProgress?.(got / total);
	}
	const buf = new Uint8Array(got);
	let p = 0;
	for (const c of chunks) {
		buf.set(c, p);
		p += c.length;
	}
	if (cache && tag)
		try {
			await cache.put(url, new Response(buf, { headers: { [TAG]: tag } }));
		} catch {
			// out of room: it will simply be downloaded again next time
		}
	onProgress?.(1);
	return buf.buffer;
}

/** A small JSON file, the same way. */
export async function fetchModelJson<T>(url: string): Promise<T> {
	return JSON.parse(new TextDecoder().decode(await fetchModelFile(url))) as T;
}
