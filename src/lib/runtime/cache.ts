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

/** A file's ETag (or last-modified) on the server, or '' offline. */
async function tagOf(url: string): Promise<{ tag: string; online: boolean }> {
	try {
		const head = await fetch(url, { method: 'HEAD', cache: 'no-store' });
		if (!head.ok) throw new Error(`Could not find ${url} (${head.status}).`);
		return {
			tag: head.headers.get('etag') ?? head.headers.get('last-modified') ?? '',
			online: true
		};
	} catch (e) {
		if (e instanceof Error && e.message.startsWith('Could not find')) throw e;
		return { tag: '', online: false };
	}
}

/**
 * Bytes [start, end) of the file at url (a range request), kept on this computer like whole files are, so a large file
 * can be read in parts without ever holding all of it. `tag` is the file's tag from modelFileTag (checked once).
 */
export async function fetchModelRange(
	url: string,
	start: number,
	end: number,
	tag: { tag: string; online: boolean },
	onProgress?: (fraction: number) => void
): Promise<ArrayBuffer> {
	const key = `${url}${url.includes('?') ? '&' : '?'}mindview-range=${start}-${end}`;
	const cache = await open();
	if (cache) {
		const kept = await cache.match(key);
		if (kept && (!tag.online || (tag.tag && kept.headers.get(TAG) === tag.tag))) {
			const buf = await kept.arrayBuffer();
			onProgress?.(1);
			return buf;
		}
	}
	const res = await fetch(url, {
		cache: 'no-store',
		headers: { Range: `bytes=${start}-${end - 1}` }
	});
	if (!res.ok || !res.body) throw new Error(`Could not download ${url} (${res.status}).`);
	const total = end - start;
	const buf = new Uint8Array(total);
	const reader = res.body.getReader();
	let got = 0;
	if (res.status === 200) {
		// the server sent the whole file: keep only the part asked for
		let pos = 0;
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			const a = Math.max(start, pos),
				b = Math.min(end, pos + value.length);
			if (b > a) buf.set(value.subarray(a - pos, b - pos), a - start);
			pos += value.length;
			got = Math.max(0, Math.min(total, pos - start));
			onProgress?.(got / total);
			if (pos >= end) {
				await reader.cancel();
				break;
			}
		}
	} else {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			buf.set(value, got);
			got += value.length;
			onProgress?.(got / total);
		}
	}
	if (got !== total) throw new Error(`Downloaded ${got} of ${total} bytes of ${url}.`);
	if (cache && tag.tag)
		try {
			await cache.put(key, new Response(buf, { headers: { [TAG]: tag.tag } }));
		} catch {
			// out of room: it will simply be downloaded again next time
		}
	onProgress?.(1);
	return buf.buffer;
}

/** Look a model file up once (for fetchModelRange). */
export const modelFileTag = tagOf;
