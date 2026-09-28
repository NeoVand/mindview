<script lang="ts">
	import { resolve } from '$app/paths';
	import Paint from '$lib/paint/Paint.svelte';
</script>

<svelte:head>
	<title>Paint</title>
	<meta
		name="description"
		content="A text-to-image model in one file of about 1 GB, running in your browser: type something and it paints it."
	/>
</svelte:head>

<Paint
	elsewhere={[
		{ href: resolve('/'), label: 'The first piece' },
		{ href: resolve('/lab'), label: 'The labs' }
	]}
	compute={{ href: resolve('/lab'), label: 'See it compute, in the labs' }}
/>
