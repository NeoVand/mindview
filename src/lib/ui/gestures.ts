// Turning, moving and zooming a canvas by hand. With a mouse: drag to turn, right-drag (or shift-drag) to move, scroll
// to come closer. On a touch screen: one finger turns, two fingers pinch to come closer and move together to move.
// Returns a function that removes the listeners.

export interface Orbitable {
	orbit(dx: number, dy: number): void;
	zoom(f: number): void; // f < 1 comes closer
	pan(dx: number, dy: number): void; // the view follows the pointer (CSS pixels)
}

export function attachOrbit(canvas: HTMLCanvasElement, target: () => Orbitable | undefined) {
	const pointers = new Map<number, { x: number; y: number }>();
	let pinch: { d: number; x: number; y: number } | null = null; // while two fingers are down
	let moving = false; // a mouse drag that moves rather than turns

	const pair = () => {
		const [a, b] = [...pointers.values()];
		return { d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
	};
	const down = (e: PointerEvent) => {
		pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
		canvas.setPointerCapture(e.pointerId);
		moving = e.pointerType === 'mouse' && (e.button === 2 || e.shiftKey);
		if (pointers.size === 2) pinch = pair();
	};
	const move = (e: PointerEvent) => {
		const p = pointers.get(e.pointerId);
		const t = target();
		if (!p || !t) return;
		const dx = e.clientX - p.x,
			dy = e.clientY - p.y;
		p.x = e.clientX;
		p.y = e.clientY;
		if (pointers.size === 1) {
			if (moving) t.pan(dx, dy);
			else t.orbit(dx, dy);
		} else if (pointers.size === 2 && pinch) {
			const q = pair();
			if (pinch.d > 0 && q.d > 0) t.zoom(pinch.d / q.d);
			t.pan(q.x - pinch.x, q.y - pinch.y);
			pinch = q;
		}
	};
	const up = (e: PointerEvent) => {
		pointers.delete(e.pointerId);
		pinch = pointers.size === 2 ? pair() : null;
		if (!pointers.size) moving = false;
	};
	const wheel = (e: WheelEvent) => {
		const t = target();
		if (!t) return;
		e.preventDefault();
		// a trackpad pinch arrives as ctrl + wheel; shift + wheel moves sideways
		if (e.shiftKey && !e.ctrlKey) t.pan(-(e.deltaX || e.deltaY), 0);
		else t.zoom(Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
	};
	const menu = (e: Event) => e.preventDefault(); // the right button moves
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
