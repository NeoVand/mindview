// Small dense linear algebra for per-prompt layouts (n <= 128), all on the CPU.

/** Eigen-decomposition of a symmetric n x n matrix (cyclic Jacobi). Returns eigenvalues descending with vectors as columns. */
export function symEig(
	Ain: Float64Array,
	n: number
): { values: Float64Array; vectors: Float64Array } {
	const A = Float64Array.from(Ain);
	const V = new Float64Array(n * n);
	for (let i = 0; i < n; i++) V[i * n + i] = 1;
	for (let sweep = 0; sweep < 60; sweep++) {
		let off = 0;
		for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] ** 2;
		if (off < 1e-18) break;
		for (let p = 0; p < n; p++)
			for (let q = p + 1; q < n; q++) {
				const apq = A[p * n + q];
				if (Math.abs(apq) < 1e-15) continue;
				const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
				const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
				const c = 1 / Math.sqrt(t * t + 1),
					s = t * c;
				for (let k = 0; k < n; k++) {
					const akp = A[k * n + p],
						akq = A[k * n + q];
					A[k * n + p] = c * akp - s * akq;
					A[k * n + q] = s * akp + c * akq;
				}
				for (let k = 0; k < n; k++) {
					const apk = A[p * n + k],
						aqk = A[q * n + k];
					A[p * n + k] = c * apk - s * aqk;
					A[q * n + k] = s * apk + c * aqk;
				}
				for (let k = 0; k < n; k++) {
					const vkp = V[k * n + p],
						vkq = V[k * n + q];
					V[k * n + p] = c * vkp - s * vkq;
					V[k * n + q] = s * vkp + c * vkq;
				}
			}
	}
	const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b * n + b] - A[a * n + a]);
	const values = new Float64Array(n),
		vectors = new Float64Array(n * n);
	order.forEach((o, j) => {
		values[j] = A[o * n + o];
		for (let k = 0; k < n; k++) vectors[k * n + j] = V[k * n + o];
	});
	return { values, vectors };
}

/** Classical MDS of rows (direction only, centred) into `dims` coordinates. */
export function mds(X: Float32Array, n: number, d: number, dims = 3): Float64Array {
	const Y = new Float64Array(n * d);
	for (let i = 0; i < n; i++) {
		let norm = 0;
		for (let k = 0; k < d; k++) norm += X[i * d + k] ** 2;
		norm = Math.sqrt(norm) || 1;
		for (let k = 0; k < d; k++) Y[i * d + k] = X[i * d + k] / norm;
	}
	for (let k = 0; k < d; k++) {
		let m = 0;
		for (let i = 0; i < n; i++) m += Y[i * d + k];
		m /= n;
		for (let i = 0; i < n; i++) Y[i * d + k] -= m;
	}
	const G = new Float64Array(n * n);
	for (let i = 0; i < n; i++)
		for (let j = i; j < n; j++) {
			let s = 0;
			for (let k = 0; k < d; k++) s += Y[i * d + k] * Y[j * d + k];
			G[i * n + j] = G[j * n + i] = s;
		}
	const { values, vectors } = symEig(G, n);
	const P = new Float64Array(n * dims);
	for (let j = 0; j < dims; j++) {
		const s = Math.sqrt(Math.max(values[j], 1e-12));
		for (let i = 0; i < n; i++) P[i * dims + j] = vectors[i * n + j] * s;
	}
	return P;
}

/** Rotate P (n x 3) to best match Q (Kabsch, rotation only, no reflection flip beyond what best fits). */
export function align3(P: Float64Array, Q: Float64Array, n: number): Float64Array {
	const M = new Float64Array(9);
	for (let i = 0; i < n; i++)
		for (let a = 0; a < 3; a++)
			for (let b = 0; b < 3; b++) M[a * 3 + b] += P[i * 3 + a] * Q[i * 3 + b];
	// polar decomposition R = M (M^T M)^-1/2 via eigen of M^T M
	const MtM = new Float64Array(9);
	for (let a = 0; a < 3; a++)
		for (let b = 0; b < 3; b++)
			for (let k = 0; k < 3; k++) MtM[a * 3 + b] += M[k * 3 + a] * M[k * 3 + b];
	const { values, vectors } = symEig(MtM, 3);
	const inv = new Float64Array(9);
	for (let a = 0; a < 3; a++)
		for (let b = 0; b < 3; b++)
			for (let k = 0; k < 3; k++)
				inv[a * 3 + b] +=
					(vectors[a * 3 + k] * vectors[b * 3 + k]) / Math.sqrt(Math.max(values[k], 1e-12));
	const R = new Float64Array(9);
	for (let a = 0; a < 3; a++)
		for (let b = 0; b < 3; b++)
			for (let k = 0; k < 3; k++) R[a * 3 + b] += M[a * 3 + k] * inv[k * 3 + b];
	const out = new Float64Array(n * 3);
	for (let i = 0; i < n; i++)
		for (let b = 0; b < 3; b++)
			for (let a = 0; a < 3; a++) out[i * 3 + b] += P[i * 3 + a] * R[a * 3 + b];
	return out;
}
