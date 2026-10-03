// A stage for the labs: the canvas, an HDR scene with bloom, and a camera that can be turned (drag), moved (right
// drag, or shift + drag) and brought closer to whatever is under the pointer (scroll), from a whole model down to
// one weight. A lab is a scene: it updates itself every frame and draws into the pass the stage opens.
import { mat4, vec3 } from 'wgpu-matrix';
import { FRAME_BYTES, type GPU } from '$lib/engine/gpu';
import { Post } from '$lib/engine/post';

export type V3 = [number, number, number];

export interface Scene {
	/** Before drawing (dt in seconds). */
	update(dt: number, stage: Stage): void;
	/** Compute work for this frame, before the scene is drawn. */
	compute?(enc: GPUCommandEncoder, stage: Stage): void;
	/** Draw into the scene pass (colour + depth). */
	draw(pass: GPURenderPassEncoder, stage: Stage): void;
}

export interface View {
	target: V3;
	dist: number;
	yaw: number;
	pitch: number;
}

const FOV = (38 * Math.PI) / 180; // on a landscape screen; see Stage.fov

export class Stage {
	readonly device: GPUDevice;
	readonly post: Post;
	readonly frame: GPUBuffer;
	/** Where the camera is, and where it is going (it eases towards `want`). */
	view: View = { target: [0, 0, 0], dist: 10, yaw: 0, pitch: 0 };
	want: View = { target: [0, 0, 0], dist: 10, yaw: 0, pitch: 0 };
	/** How quickly the camera follows `want` (per second). */
	ease = 3;
	/** Distance limits for zooming. */
	minDist = 0.002;
	maxDist = 4000;
	/** A plane the pointer zooms towards when nothing else is under it: normal and offset (n . p = d). */
	focusPlane: { n: V3; d: number } | null = { n: [0, 0, 1], d: 0 };
	time = 0;
	width = 1;
	height = 1;
	eye: V3 = [0, 0, 10];
	right: V3 = [1, 0, 0];
	up: V3 = [0, 1, 0];
	fwd: V3 = [0, 0, -1];
	private viewProj = mat4.identity();
	private raf = 0;
	private last = 0;
	private observer: ResizeObserver;
	private scene?: Scene;
	/** Called after each frame (for page overlays that follow the scene). */
	onFrame?: () => void;
	/** Seconds since the visitor last touched the camera. */
	idle = 0;

	constructor(readonly gpu: GPU) {
		this.device = gpu.device;
		this.post = new Post(gpu);
		this.post.bloomStrength = 0.55;
		this.frame = gpu.device.createBuffer({
			size: FRAME_BYTES,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.observer = new ResizeObserver(() => this.resize());
		this.observer.observe(gpu.canvas);
		this.resize();
	}

	/** Show another scene (the loop keeps running). */
	setScene(scene: Scene) {
		this.scene = scene;
	}

	start(scene: Scene) {
		this.scene = scene;
		this.last = performance.now();
		this.raf = requestAnimationFrame(this.loop);
	}

	destroy() {
		cancelAnimationFrame(this.raf);
		this.observer.disconnect();
		this.frame.destroy();
	}

	/** Go to a view (smoothly); anything left out stays as it is. */
	flyTo(v: Partial<View>) {
		this.want = {
			target: v.target ?? this.want.target,
			dist: v.dist ?? this.want.dist,
			yaw: v.yaw ?? this.want.yaw,
			pitch: v.pitch ?? this.want.pitch
		};
	}

	/** Jump to a view at once. */
	jumpTo(v: Partial<View>) {
		this.flyTo(v);
		this.view = { ...this.want, target: [...this.want.target] as V3 };
	}

	/** The vertical field of view: FOV on a landscape screen, wider on a narrow one (a phone held upright) so that
	 * the same width fits. */
	get fov() {
		const c = this.gpu.canvas;
		const aspect = c.clientWidth / Math.max(1, c.clientHeight);
		return 2 * Math.atan(Math.tan(FOV / 2) * Math.max(1, Math.pow(1.45 / aspect, 0.85)));
	}

	/**
	 * Pointer controls on the canvas: drag to turn, right-drag (or shift) to move, scroll to come closer; on a touch
	 * screen, two fingers pinch to come closer and move together to move. Returns a function that removes them.
	 */
	attachControls(canvas: HTMLCanvasElement, onClick?: (x: number, y: number) => void) {
		let drag: { x: number; y: number; id: number; pan: boolean; moved: number } | null = null;
		const touches = new Map<number, { x: number; y: number }>();
		let pinch: { d: number; x: number; y: number } | null = null;
		const pair = () => {
			const [a, b] = [...touches.values()];
			return { d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
		};
		const down = (e: PointerEvent) => {
			canvas.setPointerCapture(e.pointerId);
			if (e.pointerType === 'touch') {
				touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
				if (touches.size === 2) {
					pinch = pair();
					drag = null; // a second finger turns a turn into a pinch
					return;
				}
			}
			drag = {
				x: e.clientX,
				y: e.clientY,
				id: e.pointerId,
				pan: e.button === 2 || e.shiftKey,
				moved: 0
			};
		};
		const move = (e: PointerEvent) => {
			const t = touches.get(e.pointerId);
			if (t) {
				t.x = e.clientX;
				t.y = e.clientY;
			}
			if (pinch && touches.size === 2) {
				const p = pair();
				if (p.d > 0 && pinch.d > 0) this.zoomAt(p.x, p.y, pinch.d / p.d);
				this.pan(p.x - pinch.x, p.y - pinch.y);
				pinch = p;
				return;
			}
			if (!drag || e.pointerId !== drag.id) return;
			const dx = e.clientX - drag.x,
				dy = e.clientY - drag.y;
			drag.x = e.clientX;
			drag.y = e.clientY;
			drag.moved += Math.abs(dx) + Math.abs(dy);
			if (drag.pan) this.pan(dx, dy);
			else this.orbit(dx, dy);
		};
		const up = (e: PointerEvent) => {
			touches.delete(e.pointerId);
			if (touches.size < 2) pinch = null;
			if (drag && drag.id === e.pointerId) {
				if (drag.moved < 4 && onClick) onClick(e.clientX, e.clientY);
				drag = null;
			}
		};
		const wheel = (e: WheelEvent) => {
			e.preventDefault();
			// pinch on a trackpad arrives as ctrl + wheel; both zoom
			const f = Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
			this.zoomAt(e.clientX, e.clientY, f);
		};
		const menu = (e: Event) => e.preventDefault();
		canvas.addEventListener('pointerdown', down);
		canvas.addEventListener('pointermove', move);
		canvas.addEventListener('pointerup', up);
		canvas.addEventListener('pointercancel', up);
		canvas.addEventListener('wheel', wheel, { passive: false });
		canvas.addEventListener('contextmenu', menu);
		return () => {
			canvas.removeEventListener('pointerdown', down);
			canvas.removeEventListener('pointermove', move);
			canvas.removeEventListener('pointerup', up);
			canvas.removeEventListener('pointercancel', up);
			canvas.removeEventListener('wheel', wheel);
			canvas.removeEventListener('contextmenu', menu);
		};
	}

	orbit(dx: number, dy: number) {
		this.want.yaw -= dx * 0.005;
		this.want.pitch = Math.max(-1.45, Math.min(1.45, this.want.pitch + dy * 0.005));
		this.touch();
	}

	/** Move the target with the pointer (dx, dy in CSS pixels). */
	pan(dx: number, dy: number) {
		const k = (this.want.dist * 2 * Math.tan(this.fov / 2)) / this.gpu.canvas.clientHeight;
		const t = this.want.target;
		for (let i = 0; i < 3; i++) t[i] += (-this.right[i] * dx + this.up[i] * dy) * k;
		this.view.target = [...t] as V3;
		this.touch();
	}

	/** Come closer (f < 1) or go back (f > 1), keeping the point under the pointer where it is. */
	zoomAt(clientX: number, clientY: number, f: number) {
		const dist = Math.max(this.minDist, Math.min(this.maxDist, this.want.dist * f));
		f = dist / this.want.dist;
		const hit = this.pointUnder(clientX, clientY);
		if (hit) {
			const t = this.want.target;
			for (let i = 0; i < 3; i++) t[i] = hit[i] + (t[i] - hit[i]) * f;
		}
		this.want.dist = dist;
		this.touch();
	}

	private touch() {
		this.idle = 0;
	}

	/** The world point under the pointer: on the focus plane, or on the plane through the target facing the camera. */
	pointUnder(clientX: number, clientY: number): V3 | null {
		const { o, d } = this.ray(clientX, clientY);
		const plane = this.focusPlane ?? {
			n: this.fwd,
			d: vec3.dot(this.fwd, this.want.target)
		};
		const den = vec3.dot(plane.n, d);
		if (Math.abs(den) < 1e-6) return null;
		const s = (plane.d - vec3.dot(plane.n, o)) / den;
		if (s <= 0) return null;
		return [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
	}

	/** The ray from the eye through a point on the canvas (client coordinates). */
	ray(clientX: number, clientY: number): { o: V3; d: V3 } {
		const box = this.gpu.canvas.getBoundingClientRect();
		const nx = ((clientX - box.left) / box.width) * 2 - 1,
			ny = 1 - ((clientY - box.top) / box.height) * 2;
		const t = Math.tan(this.fov / 2),
			aspect = box.width / box.height;
		const d = vec3.normalize(
			vec3.add(
				vec3.add(this.fwd, vec3.scale(this.right, nx * t * aspect)),
				vec3.scale(this.up, ny * t)
			)
		);
		return { o: [...this.eye] as V3, d: [d[0], d[1], d[2]] };
	}

	/** Where a world point falls on the canvas (CSS pixels), and whether it is in front of the camera. */
	project(p: V3): { x: number; y: number; front: boolean; depth: number } {
		const m = this.viewProj;
		const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
			y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
			w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
		const c = this.gpu.canvas;
		return {
			x: ((x / w) * 0.5 + 0.5) * c.clientWidth,
			y: (0.5 - (y / w) * 0.5) * c.clientHeight,
			front: w > 0,
			depth: w
		};
	}

	/** World units per CSS pixel at a distance from the eye. */
	unitsPerPixel(depth: number) {
		return (depth * 2 * Math.tan(this.fov / 2)) / this.gpu.canvas.clientHeight;
	}

	private resize() {
		const c = this.gpu.canvas;
		const dpr = Math.min(window.devicePixelRatio || 1, 2);
		const w = Math.max(1, Math.floor(c.clientWidth * dpr));
		const h = Math.max(1, Math.floor(c.clientHeight * dpr));
		if (w === c.width && h === c.height && this.post.sceneView) return;
		c.width = w;
		c.height = h;
		this.width = w;
		this.height = h;
		this.post.resize(w, h);
	}

	private loop = (now: number) => {
		const dt = Math.min(0.1, (now - this.last) / 1000);
		this.last = now;
		this.time += dt;
		this.idle += dt;
		this.frameOnce(dt);
		this.raf = requestAnimationFrame(this.loop);
	};

	/** A compute queue fed a slice of work every frame (the painting), if any. */
	scheduler?: { pump(): void };

	private frameOnce(dt: number) {
		const scene = this.scene!;
		this.scheduler?.pump();
		scene.update(dt, this);
		// ease the camera (zoom in log space, so a long zoom moves at an even pace)
		const k = 1 - Math.exp(-dt * this.ease);
		const v = this.view,
			w = this.want;
		for (let i = 0; i < 3; i++) v.target[i] += (w.target[i] - v.target[i]) * k;
		v.dist = Math.exp(Math.log(v.dist) + (Math.log(w.dist) - Math.log(v.dist)) * k);
		v.yaw += (w.yaw - v.yaw) * k;
		v.pitch += (w.pitch - v.pitch) * k;

		const { canvas } = this.gpu;
		const aspect = canvas.width / canvas.height;
		const eye: V3 = [
			v.target[0] + v.dist * Math.cos(v.pitch) * Math.sin(v.yaw),
			v.target[1] + v.dist * Math.sin(v.pitch),
			v.target[2] + v.dist * Math.cos(v.pitch) * Math.cos(v.yaw)
		];
		const near = Math.max(1e-5, v.dist * 0.01),
			far = v.dist * 60 + 3000;
		const proj = mat4.perspective(this.fov, aspect, near, far);
		const view = mat4.lookAt(eye, v.target, [0, 1, 0]);
		this.viewProj = mat4.multiply(proj, view);
		const fwd = vec3.normalize(vec3.subtract(v.target, eye));
		const right = vec3.normalize(vec3.cross(fwd, [0, 1, 0]));
		const up = vec3.cross(right, fwd);
		this.eye = eye;
		this.fwd = [fwd[0], fwd[1], fwd[2]];
		this.right = [right[0], right[1], right[2]];
		this.up = [up[0], up[1], up[2]];
		const f = new Float32Array(FRAME_BYTES / 4);
		f.set(this.viewProj, 0);
		f.set([...this.right, 0], 16);
		f.set([...this.up, 0], 20);
		f.set([...eye, 1], 24);
		f.set([this.time, aspect, (2 * Math.tan(this.fov / 2)) / canvas.height, 0], 28);
		this.device.queue.writeBuffer(this.frame, 0, f);

		const enc = this.device.createCommandEncoder();
		scene.compute?.(enc, this);
		const pass = enc.beginRenderPass({
			colorAttachments: [
				{ view: this.post.sceneView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }
			],
			depthStencilAttachment: {
				view: this.post.depthView,
				depthClearValue: 1,
				depthLoadOp: 'clear',
				depthStoreOp: 'discard'
			}
		});
		scene.draw(pass, this);
		pass.end();
		this.post.finish(enc, this.time);
		this.device.queue.submit([enc.finish()]);
		this.onFrame?.();
	}
}
