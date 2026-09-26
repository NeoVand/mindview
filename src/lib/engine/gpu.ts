// WebGPU device setup and small resource helpers.

export interface GPU {
	device: GPUDevice;
	context: GPUCanvasContext;
	format: GPUTextureFormat;
	canvas: HTMLCanvasElement;
}

export const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
/** Depth for scenes where pictures should hide what is behind them. */
export const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

export async function initGPU(canvas: HTMLCanvasElement): Promise<GPU> {
	if (!navigator.gpu)
		throw new Error(
			'This browser has no WebGPU. Open the piece in a recent Chrome, Edge or Safari.'
		);
	const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
	if (!adapter) throw new Error('No WebGPU adapter is available on this machine.');
	// timestamps let the compute scheduler measure how long its work really takes on this GPU
	const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query')
		? ['timestamp-query']
		: [];
	const device = await adapter.requestDevice({
		requiredFeatures,
		requiredLimits: {
			maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
			maxBufferSize: adapter.limits.maxBufferSize
		}
	});
	device.lost.then((info) => console.error('WebGPU device lost:', info.message));
	const context = canvas.getContext('webgpu');
	if (!context) throw new Error('Could not create a WebGPU canvas context.');
	const format = navigator.gpu.getPreferredCanvasFormat();
	context.configure({ device, format, alphaMode: 'opaque' });
	return { device, context, format, canvas };
}

export function storageBuffer(
	device: GPUDevice,
	data: Float32Array | Uint32Array,
	label?: string
): GPUBuffer {
	const buf = device.createBuffer({
		label,
		size: Math.max(16, Math.ceil(data.byteLength / 4) * 4),
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
	return buf;
}

export function uniformBuffer(device: GPUDevice, bytes: number, label?: string): GPUBuffer {
	return device.createBuffer({
		label,
		size: Math.ceil(bytes / 16) * 16,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});
}

export function textureArrayFromBitmaps(
	device: GPUDevice,
	bitmaps: ImageBitmap[],
	size: number,
	label?: string
): GPUTexture {
	const tex = device.createTexture({
		label,
		size: [size, size, bitmaps.length],
		format: 'rgba8unorm-srgb',
		usage:
			GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
	});
	bitmaps.forEach((bm, i) => {
		if (bm.width !== size || bm.height !== size)
			throw new Error(`Image ${i} is ${bm.width}x${bm.height}, expected ${size}²`);
		device.queue.copyExternalImageToTexture({ source: bm }, { texture: tex, origin: [0, 0, i] }, [
			size,
			size
		]);
	});
	return tex;
}

/** Shared WGSL: per-frame camera uniforms and the ternary palette. */
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProj: mat4x4f,
  camRight: vec4f,
  camUp: vec4f,
  camPos: vec4f,
  time: f32,
  aspect: f32,
  pixel: f32,     // world units per pixel at distance 1 (for line widths)
  _pad: f32,
};
// linear-light versions of glacier #56C8FF (-1), ember #FFB04A (+1), bone #E8E2D6 (type)
const MINUS = vec3f(0.093, 0.578, 1.0);
const PLUS  = vec3f(1.0, 0.434, 0.069);
const BONE  = vec3f(0.807, 0.761, 0.672);
fn signColor(v: f32) -> vec3f { return select(MINUS, PLUS, v > 0.0); }
`;

export const FRAME_BYTES = 64 + 16 * 3 + 16;
