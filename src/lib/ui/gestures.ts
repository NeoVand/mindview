// Turning and zooming a canvas by hand: drag to turn, scroll or pinch to come closer, and on a touch screen two
// fingers to pinch. Returns a function that removes the listeners.

export interface Orbitable {
	orbit(dx: number, dy: number): void;
	zoom(f: number): void; // f < 1 comes closer
}

export function attachOrbit(canvas: HTMLCanvasElement, target: () => Orbitable | undefined) {
	const pointers = new Map<number, { x: number; y: number }>();
	let pinch = 0; // the distance between two fingers, while two are down

	const spread = () => {
		const [a, b] = [...pointers.values()];
		return Math.hypot(a.x - b.x, a.y - b.y);
	};
	const down = (e: PointerEvent) => {
		pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
		canvas.setPointerCapture(e.pointerId);
		if (pointers.size === 2) pinch = spread();
	};
	const move = (e: PointerEvent) => {
		const p = pointers.get(e.pointerId);
		const t = target();
		if (!p || !t) return;
		const dx = e.clientX - p.x,
			dy = e.clientY - p.y;
		p.x = e.clientX;
		p.y = e.clientY;
		if (pointers.size === 1) t.orbit(dx, dy);
		else if (pointers.size === 2) {
			const d = spread();
			if (pinch > 0 && d > 0) t.zoom(pinch / d);
			pinch = d;
		}
	};
	const up = (e: PointerEvent) => {
		pointers.delete(e.pointerId);
		pinch = pointers.size === 2 ? spread() : 0;
	};
	const wheel = (e: WheelEvent) => {
		const t = target();
		if (!t) return;
		e.preventDefault();
		// a trackpad pinch arrives as ctrl + wheel
		t.zoom(Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
	};
	canvas.addEventListener('pointerdown', down);
	canvas.addEventListener('pointermove', move);
	canvas.addEventListener('pointerup', up);
	canvas.addEventListener('pointercancel', up);
	canvas.addEventListener('wheel', wheel, { passive: false });
	return () => {
		canvas.removeEventListener('pointerdown', down);
		canvas.removeEventListener('pointermove', move);
		canvas.removeEventListener('pointerup', up);
		canvas.removeEventListener('pointercancel', up);
		canvas.removeEventListener('wheel', wheel);
	};
}
