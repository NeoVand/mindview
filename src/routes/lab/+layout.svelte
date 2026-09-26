<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';

	let { children } = $props();

	const labs = [
		{ href: '/lab/threads', name: 'Threads' },
		{ href: '/lab/layer', name: 'One layer' },
		{ href: '/lab/machine', name: 'The machine' },
		{ href: '/lab/cube', name: 'Compute cube' }
	] as const;
</script>

{@render children()}

<nav aria-label="Labs">
	<a
		href={resolve('/lab')}
		class="home"
		aria-current={page.url.pathname === resolve('/lab') ? 'page' : undefined}>Labs</a
	>
	{#each labs as lab (lab.href)}
		<a
			href={resolve(lab.href)}
			aria-current={page.url.pathname === resolve(lab.href) ? 'page' : undefined}>{lab.name}</a
		>
	{/each}
</nav>

<style>
	nav {
		position: fixed;
		top: 1.1rem;
		right: 1.4rem;
		z-index: 20;
		display: flex;
		gap: 1.1rem;
		font-size: 0.82rem;
		letter-spacing: 0.01em;
	}
	a {
		color: var(--bone);
		opacity: 0.45;
		text-decoration: none;
		padding: 0.15rem 0;
		border-bottom: 1px solid transparent;
		transition: opacity 0.2s;
	}
	a:hover {
		opacity: 0.85;
	}
	a[aria-current='page'] {
		opacity: 0.95;
		border-bottom-color: var(--ember);
	}
	a:focus-visible {
		outline: 1px solid var(--ember);
		outline-offset: 3px;
	}
	.home {
		margin-right: 0.4rem;
	}
</style>
