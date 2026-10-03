// One ordered queue for all the compute the piece does (reading, painting, decoding), fed to the GPU a slice at a
// time from the render loop, so frames keep flowing while the models work. Tasks run strictly in the order they
// were queued; each is a small batch of GPU commands (a few milliseconds). Two batches are kept in flight so the GPU
// never waits for the page between them; how much goes in a batch follows the frame rate (more while frames come on
// time, less after a late one). clear() drops everything queued and makes the callbacks of work already on the GPU
// harmless (a new prompt replaces the old run).

export interface GpuTask {
	/** Estimated GPU time in milliseconds (used to pace the queue). */
	cost: number;
	/**
	 * Record the task's commands. Uniform writes must be done here (they land before this submission). `sub`
	 * numbers the submission: tasks in the same submission must not reuse each other's uniform slots.
	 */
	record(enc: GPUCommandEncoder, sub: number): void;
	/** Runs once the task's commands have finished on the GPU (skipped if the queue was cleared meanwhile). */
	done?: () => void | Promise<void>;
}

const IN_FLIGHT = 2;

/**
 * Measures GPU time from the start of one pass to the end of another (timestamp queries, where the device has them;
 * both must be passes that do real work, since empty passes record nothing). Results arrive a frame or two later, as
 * a smoothed average per key in `ms`.
 */
export class GpuTimer {
	private qs?: GPUQuerySet;
	private resolve?: GPUBuffer;
	private pool: GPUBuffer[] = [];
	private next = 0;
	private readonly slots = 32;
	/** Smoothed milliseconds per key. */
	readonly ms: Record<string, number> = {};

	constructor(private device: GPUDevice) {
		if (!device.features.has('timestamp-query')) return;
		this.qs = device.createQuerySet({ type: 'timestamp', count: this.slots * 2 });
		this.resolve = device.createBuffer({
			size: this.slots * 256,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
		});
	}

	/**
	 * A span: pass `first` as the timestampWrites of the first pass and `last` of the last one, then call
	 * finish(enc) after the last pass and the returned function after submitting.
	 */
	span(key: string) {
		const qs = this.qs,
			resolve = this.resolve;
		if (!qs || !resolve) return { first: undefined, last: undefined, finish: () => () => {} };
		const slot = this.next++ % this.slots;
		return {
			first: { querySet: qs, beginningOfPassWriteIndex: slot * 2 },
			last: { querySet: qs, endOfPassWriteIndex: slot * 2 + 1 },
			finish: (enc: GPUCommandEncoder) => {
				enc.resolveQuerySet(qs, slot * 2, 2, resolve, slot * 256);
				const rb =
					this.pool.pop() ??
					this.device.createBuffer({
						size: 16,
						usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
					});
				enc.copyBufferToBuffer(resolve, slot * 256, rb, 0, 16);
				return () => {
					rb.mapAsync(GPUMapMode.READ).then(
						() => {
							const t = new BigInt64Array(rb.getMappedRange());
							const ms = Number(t[1] - t[0]) / 1e6;
							rb.unmap();
							this.pool.push(rb);
							if (!(ms > 0 && ms < 1000)) return;
							const m = this.ms[key];
							this.ms[key] = m === undefined ? ms : m + (ms - m) * 0.1;
						},
						() => {}
					);
				};
			}
		};
	}
}

export class GpuScheduler {
	private queue: GpuTask[] = [];
	private inflight = 0;
	private gen = 0;
	/** Milliseconds of estimated GPU work to submit per frame. */
	budget = 6;
	/** The least work per frame, even after late frames: owners raise it when nothing on screen needs to stay smooth. */
	floor = 3;
	readonly timer: GpuTimer;
	/** Size tasks are cut to (owners read this when they build tasks); one task holds up a frame at most this long. */
	readonly slice = 6;
	private sub = 0;
	private lastPump = 0;
	private interval = 16.7; // smoothed time between frames
	private recent: number[] = []; // the last frames' times, to find the pace the page keeps without our work
	private callbacks: Promise<void> = Promise.resolve();
	/** Milliseconds of GPU work done so far in this run (estimate). */
	spent = 0;

	constructor(private device: GPUDevice) {
		this.timer = new GpuTimer(device);
	}

	get generation() {
		return this.gen;
	}

	/** The time between frames, smoothed (ms). */
	get frameMs() {
		return this.interval;
	}

	get pending() {
		return this.queue.length + this.inflight;
	}

	push(...tasks: GpuTask[]) {
		this.queue.push(...tasks);
	}

	/** Drop all queued work; callbacks of work already submitted will not run. */
	clear() {
		this.queue = [];
		this.gen++;
		this.spent = 0;
	}

	/** Resolves when everything queued so far has run (or the queue was cleared). */
	idle(): Promise<void> {
		const gen = this.gen;
		return new Promise((resolve) => {
			const check = () =>
				this.gen !== gen || this.pending === 0 ? resolve() : setTimeout(check, 16);
			check();
		});
	}

	/** Call once per frame: submits about `budget` ms of work unless two batches are still on the GPU. */
	pump() {
		const now = performance.now();
		const frame = this.lastPump ? Math.min(100, now - this.lastPump) : 16.7;
		this.lastPump = now;
		this.interval += (frame - this.interval) * 0.25;
		this.recent.push(frame);
		if (this.recent.length > 90) this.recent.shift();
		if (!this.queue.length) return;
		// find the most work per frame that keeps frames on time: a little more after every frame that came on time, a
		// good deal less after one that came late. On time is the pace the page keeps anyway: display frames are 16.7
		// ms apart, but a heavy scene may draw only every second one (33 ms), and then a 33 ms frame is not our doing.
		// The fastest recent frame tells that pace (it comes back each time the work is cut).
		const pace = Math.max(6, Math.min(...this.recent));
		if (frame > Math.max(24, pace * 1.35)) this.budget = Math.max(this.floor, this.budget * 0.8);
		else this.budget = Math.min(40, Math.max(this.floor, this.budget * 1.02 + 0.15));
		if (this.inflight >= IN_FLIGHT) return;
		const gen = this.gen;
		const sub = ++this.sub;
		const enc = this.device.createCommandEncoder();
		const batch: GpuTask[] = [];
		let cost = 0;
		while (this.queue.length && (batch.length === 0 || cost + this.queue[0].cost <= this.budget)) {
			const task = this.queue.shift()!;
			task.record(enc, sub);
			batch.push(task);
			cost += task.cost;
		}
		this.device.queue.submit([enc.finish()]);
		this.inflight++;
		this.spent += cost;
		const finished = this.device.queue.onSubmittedWorkDone().then(() => {
			this.inflight--;
		});
		// callbacks run in the order the work was queued
		this.callbacks = this.callbacks.then(async () => {
			await finished;
			if (gen !== this.gen) return;
			for (const t of batch)
				try {
					await t.done?.();
				} catch (e) {
					console.error('GPU task callback failed', e);
				}
		});
	}
}

/** Estimated milliseconds for a matrix multiply of M x N x K on this class of GPU (about 1 TFLOP/s). */
export const gemmMs = (M: number, N: number, K: number) => (2 * M * N * K) / 1.1e9;
