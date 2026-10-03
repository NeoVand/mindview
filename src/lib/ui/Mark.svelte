<script lang="ts">
	// The mark: the three values a ternary weight can take, −1 (glacier), 0 (the void, a ring) and +1 (ember).
	let { size = 22 }: { size?: number } = $props();
</script>

<svg width={size} height={size * 0.5} viewBox="0 0 32 16" aria-hidden="true">
	<circle cx="5" cy="8" r="4.4" fill="var(--glacier)" />
	<circle cx="16" cy="8" r="3.5" fill="none" stroke="var(--bone)" stroke-width="1.6" />
	<circle cx="27" cy="8" r="4.4" fill="var(--ember)" />
</svg>
