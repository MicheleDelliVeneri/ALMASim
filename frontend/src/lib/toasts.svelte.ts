/**
 * Short notices ("Job submitted", "Cancelled 01J…"), shown by `Toasts.svelte`
 * in the layout. Errors stay until dismissed; the rest fade after a while.
 */

export type ToastTone = 'info' | 'ok' | 'warn' | 'danger';

export interface Toast {
	id: number;
	tone: ToastTone;
	message: string;
}

const LINGER_MS = 6000;

export class Toasts {
	items = $state<Toast[]>([]);
	#next = 1;

	show(message: string, tone: ToastTone = 'info', lingerMs: number = LINGER_MS): number {
		const id = this.#next++;
		this.items = [...this.items, { id, tone, message }];
		if (tone !== 'danger') setTimeout(() => this.dismiss(id), lingerMs);
		return id;
	}

	dismiss(id: number): void {
		this.items = this.items.filter((toast) => toast.id !== id);
	}
}

export const toasts = new Toasts();
