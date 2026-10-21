import '@testing-library/jest-dom/vitest';

// jsdom has no layout, so no ResizeObserver (Svelte's bind:clientHeight uses it).
globalThis.ResizeObserver ??= class {
	observe() {}
	unobserve() {}
	disconnect() {}
};

// jsdom has <dialog> but not its modal methods.
if (typeof HTMLDialogElement !== 'undefined') {
	HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
		this.open = true;
	};
	HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
		this.open = false;
	};
}
