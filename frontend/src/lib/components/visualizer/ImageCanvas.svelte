<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	interface ImageData {
		image: number[][];
		stats: {
			shape: number[];
			integrated_shape: number[];
			min: number;
			max: number;
			mean: number;
			std: number;
			cube_name: string;
		};
		method: string;
	}

	interface SelectionStats {
		mean: number;
		std: number;
		sum: number;
		min: number;
		max: number;
		count: number;
	}

	interface SavedSelection {
		id: string;
		type: 'rectangle' | 'circle';
		start: { x: number; y: number };
		end: { x: number; y: number };
		stats: SelectionStats;
	}

	interface Props {
		imageData: ImageData | null;
		scale: number;
		panX: number;
		panY: number;
		onScaleChange: (scale: number) => void;
		onPanChange: (x: number, y: number) => void;
		onReset: () => void;
	}

	let { imageData, scale, panX, panY, onScaleChange, onPanChange, onReset }: Props = $props();

	let canvasRef: HTMLCanvasElement;
	let isDragging = $state(false);
	let dragStart = $state({ x: 0, y: 0 });

	// Selection tool state
	let selectionMode = $state<'none' | 'rectangle' | 'circle'>('none');
	let isSelecting = $state(false);
	let selectionStart = $state({ x: 0, y: 0 });
	let selectionEnd = $state({ x: 0, y: 0 });
	let savedSelections = $state<SavedSelection[]>([]);
	let selectedSelectionId = $state<string | null>(null);

	// Move/resize state
	let isDraggingSelection = $state(false);
	let isResizingSelection = $state(false);
	let resizeHandle = $state<'tl' | 'tr' | 'bl' | 'br' | 'n' | 's' | 'e' | 'w' | null>(null);
	let dragOffset = $state({ x: 0, y: 0 });
	let cursorStyle = $state('grab');

	function screenToImageCoords(screenX: number, screenY: number): { x: number; y: number } | null {
		const canvas = canvasRef;
		if (!canvas || !imageData) return null;

		const image = imageData.image;
		const height = image.length;
		const width = image[0]?.length || 0;

		const displayWidth = canvas.width / (window.devicePixelRatio || 1);
		const displayHeight = canvas.height / (window.devicePixelRatio || 1);
		const centerX = displayWidth / 2;
		const centerY = displayHeight / 2;

		const fitScale = Math.min(displayWidth / width, displayHeight / height);
		const totalScale = scale * fitScale;

		// Reverse the transform
		const worldX = (screenX - centerX - panX) / totalScale + width / 2;
		const worldY = (screenY - centerY - panY) / totalScale + height / 2;

		return { x: Math.floor(worldX), y: Math.floor(worldY) };
	}

	function calculateSelectionStats(): SelectionStats | null {
		if (!imageData || !isSelecting) return null;

		const image = imageData.image;
		const height = image.length;
		const width = image[0]?.length || 0;

		const start = screenToImageCoords(selectionStart.x, selectionStart.y);
		const end = screenToImageCoords(selectionEnd.x, selectionEnd.y);

		if (!start || !end) return null;

		let values: number[] = [];

		if (selectionMode === 'rectangle') {
			const x1 = Math.max(0, Math.min(start.x, end.x));
			const x2 = Math.min(width - 1, Math.max(start.x, end.x));
			const y1 = Math.max(0, Math.min(start.y, end.y));
			const y2 = Math.min(height - 1, Math.max(start.y, end.y));

			for (let y = y1; y <= y2; y++) {
				for (let x = x1; x <= x2; x++) {
					values.push(image[y][x]);
				}
			}
		} else if (selectionMode === 'circle') {
			const centerX = (start.x + end.x) / 2;
			const centerY = (start.y + end.y) / 2;
			const radius = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2)) / 2;

			const x1 = Math.max(0, Math.floor(centerX - radius));
			const x2 = Math.min(width - 1, Math.ceil(centerX + radius));
			const y1 = Math.max(0, Math.floor(centerY - radius));
			const y2 = Math.min(height - 1, Math.ceil(centerY + radius));

			for (let y = y1; y <= y2; y++) {
				for (let x = x1; x <= x2; x++) {
					const dist = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
					if (dist <= radius) {
						values.push(image[y][x]);
					}
				}
			}
		}

		if (values.length === 0) return null;

		const sum = values.reduce((a, b) => a + b, 0);
		const mean = sum / values.length;
		const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
		const std = Math.sqrt(variance);
		const min = Math.min(...values);
		const max = Math.max(...values);

		return { mean, std, sum, min, max, count: values.length };
	}

	function drawImage() {
		const canvas = canvasRef;
		const data = imageData;
		if (!canvas || !data) return;

		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const image = data.image;
		const height = image.length;
		const width = image[0]?.length || 0;

		if (width === 0 || height === 0) return;

		// Get container size - adaptive height
		const container = canvas.parentElement;
		const containerWidth = container ? container.clientWidth - 2 : 600;
		const containerHeight = 400; // Smaller height for grid layout

		// Calculate display size (fit to container while maintaining aspect ratio)
		const aspectRatio = width / height;
		let displayWidth = containerWidth;
		let displayHeight = containerWidth / aspectRatio;

		// If height exceeds container, scale by height instead
		if (displayHeight > containerHeight) {
			displayHeight = containerHeight;
			displayWidth = containerHeight * aspectRatio;
		}

		// Set canvas size (account for high DPI)
		const dpr = window.devicePixelRatio || 1;
		canvas.width = displayWidth * dpr;
		canvas.height = displayHeight * dpr;
		canvas.style.width = `${displayWidth}px`;
		canvas.style.height = `${displayHeight}px`;

		// Scale context for high DPI
		ctx.scale(dpr, dpr);

		// Clear canvas
		ctx.clearRect(0, 0, displayWidth, displayHeight);

		// Create ImageData at original image resolution
		const imgData = ctx.createImageData(width, height);
		const pixels = imgData.data;

		// Convert 2D array to ImageData
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const idx = (y * width + x) * 4;
				const value = image[y][x];
				// Grayscale
				pixels[idx] = value; // R
				pixels[idx + 1] = value; // G
				pixels[idx + 2] = value; // B
				pixels[idx + 3] = 255; // A
			}
		}

		// Create temporary canvas to hold the image
		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = width;
		tempCanvas.height = height;
		const tempCtx = tempCanvas.getContext('2d');
		if (!tempCtx) return;

		tempCtx.putImageData(imgData, 0, 0);

		// Calculate scale to fit image in display area
		const fitScale = Math.min(displayWidth / width, displayHeight / height);

		// Apply user transform (pan and zoom)
		ctx.save();

		// Center the image initially
		const centerX = displayWidth / 2;
		const centerY = displayHeight / 2;

		// Apply transforms: translate to center, scale, then pan
		ctx.translate(centerX + panX, centerY + panY);
		ctx.scale(scale * fitScale, scale * fitScale);
		ctx.translate(-width / 2, -height / 2);

		// Draw the image
		ctx.drawImage(tempCanvas, 0, 0);

		ctx.restore();

		// Draw all saved selections
		for (const selection of savedSelections) {
			ctx.save();
			const isSelected = selection.id === selectedSelectionId;
			ctx.strokeStyle = isSelected ? 'rgba(255, 165, 0, 0.9)' : 'rgba(0, 123, 255, 0.8)';
			ctx.fillStyle = isSelected ? 'rgba(255, 165, 0, 0.15)' : 'rgba(0, 123, 255, 0.1)';
			ctx.lineWidth = isSelected ? 3 : 2;

			const x = Math.min(selection.start.x, selection.end.x);
			const y = Math.min(selection.start.y, selection.end.y);
			const w = Math.abs(selection.end.x - selection.start.x);
			const h = Math.abs(selection.end.y - selection.start.y);

			if (selection.type === 'rectangle') {
				ctx.fillRect(x, y, w, h);
				ctx.strokeRect(x, y, w, h);
			} else if (selection.type === 'circle') {
				const centerX = (selection.start.x + selection.end.x) / 2;
				const centerY = (selection.start.y + selection.end.y) / 2;
				const radius = Math.sqrt(w * w + h * h) / 2;

				ctx.beginPath();
				ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
				ctx.fill();
				ctx.stroke();
			}

			ctx.restore();

			// Draw resize handles for selected selection
			if (isSelected) {
				drawResizeHandles(ctx, selection);
			}
		}

		// Draw current selection being drawn
		if (isSelecting && selectionMode !== 'none') {
			ctx.save();
			ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
			ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
			ctx.lineWidth = 2;
			ctx.setLineDash([5, 5]);

			const x = Math.min(selectionStart.x, selectionEnd.x);
			const y = Math.min(selectionStart.y, selectionEnd.y);
			const w = Math.abs(selectionEnd.x - selectionStart.x);
			const h = Math.abs(selectionEnd.y - selectionStart.y);

			if (selectionMode === 'rectangle') {
				ctx.fillRect(x, y, w, h);
				ctx.strokeRect(x, y, w, h);
			} else if (selectionMode === 'circle') {
				const centerX = (selectionStart.x + selectionEnd.x) / 2;
				const centerY = (selectionStart.y + selectionEnd.y) / 2;
				const radius = Math.sqrt(w * w + h * h) / 2;

				ctx.beginPath();
				ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
				ctx.fill();
				ctx.stroke();
			}

			ctx.restore();
		}
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const canvas = canvasRef;
		if (!canvas || !imageData) return;

		const rect = canvas.getBoundingClientRect();
		const mouseX = e.clientX - rect.left;
		const mouseY = e.clientY - rect.top;

		// Calculate zoom factor
		const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
		const oldScale = scale;
		const newScale = Math.max(0.1, Math.min(10, oldScale * zoomFactor));

		// Get container dimensions
		const displayWidth = canvas.width / (window.devicePixelRatio || 1);
		const displayHeight = canvas.height / (window.devicePixelRatio || 1);
		const centerX = displayWidth / 2;
		const centerY = displayHeight / 2;

		// Calculate mouse position relative to center
		const mouseOffsetX = mouseX - centerX;
		const mouseOffsetY = mouseY - centerY;

		// Adjust pan to zoom toward mouse position
		const scaleDelta = newScale / oldScale;
		const newPanX = panX + mouseOffsetX * (1 - scaleDelta);
		const newPanY = panY + mouseOffsetY * (1 - scaleDelta);

		onPanChange(newPanX, newPanY);
		onScaleChange(newScale);
	}

	function handleMouseDown(e: MouseEvent) {
		if (e.button === 0) {
			const canvas = canvasRef;
			if (!canvas) return;

			const rect = canvas.getBoundingClientRect();
			const mouseX = e.clientX - rect.left;
			const mouseY = e.clientY - rect.top;

			// Check if clicking on resize handle of selected selection
			if (selectedSelectionId) {
				const selectedSelection = savedSelections.find((s) => s.id === selectedSelectionId);
				if (selectedSelection) {
					const handle = getResizeHandle(mouseX, mouseY, selectedSelection);
					if (handle) {
						// Start resizing
						isResizingSelection = true;
						resizeHandle = handle as any;
						return;
					}
				}
			}

			// Check if clicking on an existing selection (in reverse order to get topmost)
			let clickedSelection: SavedSelection | null = null;
			for (let i = savedSelections.length - 1; i >= 0; i--) {
				if (isPointInSelection(mouseX, mouseY, savedSelections[i])) {
					clickedSelection = savedSelections[i];
					break;
				}
			}

			if (clickedSelection) {
				// Select the clicked selection and prepare to move it
				selectedSelectionId = clickedSelection.id;
				isDraggingSelection = true;
				dragOffset = {
					x: mouseX - clickedSelection.start.x,
					y: mouseY - clickedSelection.start.y
				};
				drawImage();
			} else if (selectionMode !== 'none') {
				// Start new selection
				isSelecting = true;
				selectionStart = { x: mouseX, y: mouseY };
				selectionEnd = { x: mouseX, y: mouseY };
				selectedSelectionId = null;
			} else {
				// Start panning
				isDragging = true;
				dragStart = { x: e.clientX - panX, y: e.clientY - panY };
				selectedSelectionId = null;
				drawImage();
			}
		}
	}

	function handleMouseMove(e: MouseEvent) {
		const canvas = canvasRef;
		if (!canvas) return;

		const rect = canvas.getBoundingClientRect();
		const mouseX = e.clientX - rect.left;
		const mouseY = e.clientY - rect.top;

		// Update cursor based on what's being hovered
		if (!isDragging && !isDraggingSelection && !isResizingSelection && !isSelecting) {
			if (selectedSelectionId) {
				const selectedSelection = savedSelections.find((s) => s.id === selectedSelectionId);
				if (selectedSelection) {
					const handle = getResizeHandle(mouseX, mouseY, selectedSelection);
					if (handle) {
						// Set cursor based on resize handle
						const cursors: Record<string, string> = {
							tl: 'nw-resize',
							tr: 'ne-resize',
							bl: 'sw-resize',
							br: 'se-resize',
							n: 'n-resize',
							s: 's-resize',
							e: 'e-resize',
							w: 'w-resize'
						};
						cursorStyle = cursors[handle] || 'grab';
					} else if (isPointInSelection(mouseX, mouseY, selectedSelection)) {
						cursorStyle = 'move';
					} else {
						cursorStyle = selectionMode !== 'none' ? 'crosshair' : 'grab';
					}
				} else {
					cursorStyle = selectionMode !== 'none' ? 'crosshair' : 'grab';
				}
			} else {
				cursorStyle = selectionMode !== 'none' ? 'crosshair' : 'grab';
			}
		} else if (isDragging) {
			cursorStyle = 'grabbing';
		} else if (isDraggingSelection) {
			cursorStyle = 'move';
		}

		if (isDraggingSelection && selectedSelectionId) {
			// Move selection
			const selection = savedSelections.find((s) => s.id === selectedSelectionId);
			if (selection) {
				const newStartX = mouseX - dragOffset.x;
				const newStartY = mouseY - dragOffset.y;
				const dx = newStartX - selection.start.x;
				const dy = newStartY - selection.start.y;

				savedSelections = savedSelections.map((s) =>
					s.id === selectedSelectionId
						? {
								...s,
								start: { x: s.start.x + dx, y: s.start.y + dy },
								end: { x: s.end.x + dx, y: s.end.y + dy }
							}
						: s
				);
				drawImage();
			}
		} else if (isResizingSelection && selectedSelectionId && resizeHandle) {
			// Resize selection
			const selection = savedSelections.find((s) => s.id === selectedSelectionId);
			if (selection) {
				savedSelections = savedSelections.map((s) => {
					if (s.id !== selectedSelectionId) return s;

					let newStart = { ...s.start };
					let newEnd = { ...s.end };

					if (s.type === 'rectangle') {
						// Handle rectangle resize
						if (resizeHandle.includes('n')) newStart.y = mouseY;
						if (resizeHandle.includes('s')) newEnd.y = mouseY;
						if (resizeHandle.includes('w')) newStart.x = mouseX;
						if (resizeHandle.includes('e')) newEnd.x = mouseX;
						if (resizeHandle === 'tl') {
							newStart.x = mouseX;
							newStart.y = mouseY;
						}
						if (resizeHandle === 'tr') {
							newEnd.x = mouseX;
							newStart.y = mouseY;
						}
						if (resizeHandle === 'bl') {
							newStart.x = mouseX;
							newEnd.y = mouseY;
						}
						if (resizeHandle === 'br') {
							newEnd.x = mouseX;
							newEnd.y = mouseY;
						}
					} else if (s.type === 'circle') {
						// Handle circle resize (adjust radius)
						const centerX = (s.start.x + s.end.x) / 2;
						const centerY = (s.start.y + s.end.y) / 2;
						const newRadius = Math.sqrt(
							Math.pow(mouseX - centerX, 2) + Math.pow(mouseY - centerY, 2)
						);
						newStart = { x: centerX - newRadius, y: centerY - newRadius };
						newEnd = { x: centerX + newRadius, y: centerY + newRadius };
					}

					// Recalculate stats
					const tempSelection = { ...s, start: newStart, end: newEnd };
					const imageCoords1 = screenToImageCoords(newStart.x, newStart.y);
					const imageCoords2 = screenToImageCoords(newEnd.x, newEnd.y);

					if (imageCoords1 && imageCoords2) {
						// Create temporary selection for stats calculation
						selectionStart = newStart;
						selectionEnd = newEnd;
						const stats = calculateSelectionStats();
						if (stats) {
							return { ...s, start: newStart, end: newEnd, stats };
						}
					}

					return { ...s, start: newStart, end: newEnd };
				});
				drawImage();
			}
		} else if (isSelecting && selectionMode !== 'none') {
			// Update selection
			selectionEnd = { x: mouseX, y: mouseY };
			drawImage();
		} else if (isDragging) {
			// Pan
			onPanChange(e.clientX - dragStart.x, e.clientY - dragStart.y);
		}
	}

	function handleMouseUp() {
		if (isSelecting && selectionMode !== 'none') {
			// Finalize selection and save it
			const stats = calculateSelectionStats();
			if (stats) {
				const newSelection: SavedSelection = {
					id: `selection-${Date.now()}`,
					type: selectionMode,
					start: { ...selectionStart },
					end: { ...selectionEnd },
					stats
				};
				savedSelections = [...savedSelections, newSelection];
				selectedSelectionId = newSelection.id;
			}
			isSelecting = false;
			drawImage();
		}

		if (isDraggingSelection || isResizingSelection) {
			// Stop moving/resizing
			isDraggingSelection = false;
			isResizingSelection = false;
			resizeHandle = null;
			drawImage();
		}

		isDragging = false;
	}

	function isPointInSelection(x: number, y: number, selection: SavedSelection): boolean {
		const sx = Math.min(selection.start.x, selection.end.x);
		const sy = Math.min(selection.start.y, selection.end.y);
		const w = Math.abs(selection.end.x - selection.start.x);
		const h = Math.abs(selection.end.y - selection.start.y);

		if (selection.type === 'rectangle') {
			return x >= sx && x <= sx + w && y >= sy && y <= sy + h;
		} else if (selection.type === 'circle') {
			const centerX = (selection.start.x + selection.end.x) / 2;
			const centerY = (selection.start.y + selection.end.y) / 2;
			const radius = Math.sqrt(w * w + h * h) / 2;
			const dist = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
			return dist <= radius;
		}
		return false;
	}

	function deleteSelection(id: string) {
		savedSelections = savedSelections.filter((s) => s.id !== id);
		if (selectedSelectionId === id) {
			selectedSelectionId = null;
		}
		drawImage();
	}

	function deleteSelectedSelection() {
		if (selectedSelectionId) {
			deleteSelection(selectedSelectionId);
		}
	}

	function getResizeHandle(x: number, y: number, selection: SavedSelection): string | null {
		const handleSize = 8;
		const sx = Math.min(selection.start.x, selection.end.x);
		const sy = Math.min(selection.start.y, selection.end.y);
		const ex = Math.max(selection.start.x, selection.end.x);
		const ey = Math.max(selection.start.y, selection.end.y);
		const w = ex - sx;
		const h = ey - sy;

		if (selection.type === 'rectangle') {
			// Corner handles
			if (Math.abs(x - sx) < handleSize && Math.abs(y - sy) < handleSize) return 'tl';
			if (Math.abs(x - ex) < handleSize && Math.abs(y - sy) < handleSize) return 'tr';
			if (Math.abs(x - sx) < handleSize && Math.abs(y - ey) < handleSize) return 'bl';
			if (Math.abs(x - ex) < handleSize && Math.abs(y - ey) < handleSize) return 'br';

			// Edge handles
			if (Math.abs(x - sx - w / 2) < handleSize && Math.abs(y - sy) < handleSize) return 'n';
			if (Math.abs(x - sx - w / 2) < handleSize && Math.abs(y - ey) < handleSize) return 's';
			if (Math.abs(x - sx) < handleSize && Math.abs(y - sy - h / 2) < handleSize) return 'w';
			if (Math.abs(x - ex) < handleSize && Math.abs(y - sy - h / 2) < handleSize) return 'e';
		} else if (selection.type === 'circle') {
			const centerX = (selection.start.x + selection.end.x) / 2;
			const centerY = (selection.start.y + selection.end.y) / 2;
			const radius = Math.sqrt(w * w + h * h) / 2;

			// Four cardinal direction handles on circle perimeter
			if (Math.abs(x - (centerX + radius)) < handleSize && Math.abs(y - centerY) < handleSize)
				return 'e';
			if (Math.abs(x - (centerX - radius)) < handleSize && Math.abs(y - centerY) < handleSize)
				return 'w';
			if (Math.abs(x - centerX) < handleSize && Math.abs(y - (centerY - radius)) < handleSize)
				return 'n';
			if (Math.abs(x - centerX) < handleSize && Math.abs(y - (centerY + radius)) < handleSize)
				return 's';
		}

		return null;
	}

	function drawResizeHandles(ctx: CanvasRenderingContext2D, selection: SavedSelection) {
		const handleSize = 6;
		const sx = Math.min(selection.start.x, selection.end.x);
		const sy = Math.min(selection.start.y, selection.end.y);
		const ex = Math.max(selection.start.x, selection.end.x);
		const ey = Math.max(selection.start.y, selection.end.y);
		const w = ex - sx;
		const h = ey - sy;

		ctx.fillStyle = 'white';
		ctx.strokeStyle = 'rgba(255, 165, 0, 1)';
		ctx.lineWidth = 2;

		const drawHandle = (x: number, y: number) => {
			ctx.fillRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize);
			ctx.strokeRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize);
		};

		if (selection.type === 'rectangle') {
			// Corner handles
			drawHandle(sx, sy);
			drawHandle(ex, sy);
			drawHandle(sx, ey);
			drawHandle(ex, ey);

			// Edge handles
			drawHandle(sx + w / 2, sy);
			drawHandle(sx + w / 2, ey);
			drawHandle(sx, sy + h / 2);
			drawHandle(ex, sy + h / 2);
		} else if (selection.type === 'circle') {
			const centerX = (selection.start.x + selection.end.x) / 2;
			const centerY = (selection.start.y + selection.end.y) / 2;
			const radius = Math.sqrt(w * w + h * h) / 2;

			// Four cardinal handles
			drawHandle(centerX + radius, centerY);
			drawHandle(centerX - radius, centerY);
			drawHandle(centerX, centerY - radius);
			drawHandle(centerX, centerY + radius);
		}
	}

	onMount(() => {
		const canvas = canvasRef;
		if (canvas) {
			canvas.addEventListener('wheel', handleWheel, { passive: false });
			canvas.addEventListener('mousedown', handleMouseDown);
			window.addEventListener('mousemove', handleMouseMove);
			window.addEventListener('mouseup', handleMouseUp);

			if (imageData) {
				drawImage();
			}
		}
	});

	onDestroy(() => {
		const canvas = canvasRef;
		if (canvas) {
			canvas.removeEventListener('wheel', handleWheel);
			canvas.removeEventListener('mousedown', handleMouseDown);
			window.removeEventListener('mousemove', handleMouseMove);
			window.removeEventListener('mouseup', handleMouseUp);
		}
	});

	// Redraw when image data or transform changes
	$effect(() => {
		if (imageData) {
			// Trigger redraw when data, scale, pan, or selections change
			const _ = scale;
			const __ = panX;
			const ___ = panY;
			const ____ = imageData;
			const _____ = selectionEnd;
			const ______ = savedSelections;
			const _______ = selectedSelectionId;
			setTimeout(() => drawImage(), 0);
		}
	});
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<h2 class="text-lg font-semibold text-gray-900">Integrated Image</h2>
		<div class="flex items-center space-x-4 text-sm text-gray-600">
			<span>Zoom: {(scale * 100).toFixed(0)}%</span>
			<button
				onclick={onReset}
				class="rounded-md bg-gray-100 px-3 py-1 transition-colors hover:bg-gray-200"
			>
				Reset View
			</button>
		</div>
	</div>

	<!-- Selection Tools -->
	<div class="flex items-center justify-between rounded-md border border-gray-200 bg-gray-50 p-3">
		<span class="text-sm font-medium text-gray-700">Selection Tool:</span>
		<div class="flex items-center space-x-2">
			<button
				onclick={() => {
					selectionMode = 'none';
				}}
				class="rounded px-3 py-1 text-sm transition-colors {selectionMode === 'none'
					? 'bg-white text-gray-900 shadow-sm ring-1 ring-gray-300'
					: 'text-gray-600 hover:text-gray-900'}"
				title="Pan mode"
			>
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11"
					></path>
				</svg>
			</button>
			<button
				onclick={() => {
					selectionMode = 'rectangle';
				}}
				class="rounded px-3 py-1 text-sm transition-colors {selectionMode === 'rectangle'
					? 'bg-white text-gray-900 shadow-sm ring-1 ring-blue-500'
					: 'text-gray-600 hover:text-gray-900'}"
				title="Rectangle selection"
			>
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<rect x="4" y="6" width="16" height="12" rx="2" stroke-width="2"></rect>
				</svg>
			</button>
			<button
				onclick={() => {
					selectionMode = 'circle';
				}}
				class="rounded px-3 py-1 text-sm transition-colors {selectionMode === 'circle'
					? 'bg-white text-gray-900 shadow-sm ring-1 ring-blue-500'
					: 'text-gray-600 hover:text-gray-900'}"
				title="Circle selection"
			>
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<circle cx="12" cy="12" r="8" stroke-width="2"></circle>
				</svg>
			</button>
			{#if selectedSelectionId}
				<button
					onclick={deleteSelectedSelection}
					class="rounded bg-red-100 px-3 py-1 text-xs text-red-700 transition-colors hover:bg-red-200"
					title="Delete selected region"
				>
					Delete
				</button>
			{/if}
			{#if savedSelections.length > 0}
				<button
					onclick={() => {
						savedSelections = [];
						selectedSelectionId = null;
						drawImage();
					}}
					class="rounded bg-gray-200 px-3 py-1 text-xs text-gray-700 transition-colors hover:bg-gray-300"
					title="Clear all selections"
				>
					Clear All
				</button>
			{/if}
		</div>
	</div>

	<!-- Selection Statistics -->
	{#if selectedSelectionId}
		{@const selectedSelection = savedSelections.find((s) => s.id === selectedSelectionId)}
		{#if selectedSelection}
			<div class="rounded-md border border-orange-200 bg-orange-50 p-4">
				<h3 class="mb-3 text-sm font-semibold text-orange-900">
					Selected Region Statistics ({selectedSelection.type})
				</h3>
				<div class="grid grid-cols-2 gap-3 text-sm md:grid-cols-3">
					<div>
						<span class="text-gray-600">Pixels:</span>
						<span class="ml-2 font-medium text-gray-900">{selectedSelection.stats.count}</span>
					</div>
					<div>
						<span class="text-gray-600">Sum:</span>
						<span class="ml-2 font-medium text-gray-900"
							>{selectedSelection.stats.sum.toFixed(2)}</span
						>
					</div>
					<div>
						<span class="text-gray-600">Mean:</span>
						<span class="ml-2 font-medium text-gray-900"
							>{selectedSelection.stats.mean.toFixed(2)}</span
						>
					</div>
					<div>
						<span class="text-gray-600">Std Dev:</span>
						<span class="ml-2 font-medium text-gray-900"
							>{selectedSelection.stats.std.toFixed(2)}</span
						>
					</div>
					<div>
						<span class="text-gray-600">Min:</span>
						<span class="ml-2 font-medium text-gray-900"
							>{selectedSelection.stats.min.toFixed(2)}</span
						>
					</div>
					<div>
						<span class="text-gray-600">Max:</span>
						<span class="ml-2 font-medium text-gray-900"
							>{selectedSelection.stats.max.toFixed(2)}</span
						>
					</div>
				</div>
			</div>
		{/if}
	{/if}

	<div
		class="flex items-center justify-center overflow-hidden rounded-md border border-gray-300 bg-gray-100"
		style="touch-action: none; height: 400px; min-height: 400px;"
	>
		<canvas bind:this={canvasRef} class="block" style="cursor: {cursorStyle}; user-select: none;">
		</canvas>
	</div>

	<div class="space-y-1 text-xs text-gray-500">
		<p>• Scroll to zoom in/out</p>
		<p>• Click and drag to pan</p>
		<p>• Right-click and drag (or middle mouse button) to pan</p>
	</div>
</div>
