import { fireEvent, render, screen } from '@testing-library/svelte';
import { createRawSnippet } from 'svelte';
import { describe, expect, it, vi } from 'vitest';

import Dialog from './Dialog.svelte';

const body = createRawSnippet(() => ({ render: () => '<p>Are you sure?</p>' }));

describe('Dialog', () => {
	it('opens as a modal named by its title', () => {
		render(Dialog, { title: 'Cancel run', onClose: vi.fn(), children: body });
		const dialog = screen.getByRole('dialog', { name: 'Cancel run' });
		expect(dialog).toHaveProperty('open', true);
		expect(screen.getByText('Are you sure?')).toBeInTheDocument();
	});

	it('asks its owner to close on Escape, the close button and the backdrop', async () => {
		const onClose = vi.fn();
		render(Dialog, { title: 'Cancel run', onClose, children: body });
		const dialog = screen.getByRole('dialog');

		const escape = new Event('cancel', { cancelable: true });
		dialog.dispatchEvent(escape);
		expect(escape.defaultPrevented).toBe(true); // the owner closes it, by unmounting
		expect(onClose).toHaveBeenCalledTimes(1);

		await fireEvent.click(screen.getByRole('button', { name: 'Close' }));
		expect(onClose).toHaveBeenCalledTimes(2);

		await fireEvent.click(dialog); // outside the content: the backdrop
		expect(onClose).toHaveBeenCalledTimes(3);

		await fireEvent.click(screen.getByText('Are you sure?'));
		expect(onClose).toHaveBeenCalledTimes(3);
	});

	it('closes the native dialog when unmounted', () => {
		const { unmount } = render(Dialog, { title: 'X', onClose: vi.fn(), children: body });
		const dialog = screen.getByRole('dialog') as HTMLDialogElement;
		unmount();
		expect(dialog.open).toBe(false);
	});
});
