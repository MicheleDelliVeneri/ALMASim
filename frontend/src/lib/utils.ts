/** Shared utility functions */

export const formatDateStamp = () => {
	const date = new Date();
	return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}-${String(date.getHours()).padStart(2, '0')}${String(date.getMinutes()).padStart(2, '0')}`;
};

export const supportsFilePicker = () =>
	typeof window !== 'undefined' &&
	typeof (window as Window & { showSaveFilePicker?: () => unknown }).showSaveFilePicker ===
		'function';
