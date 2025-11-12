import { Show, onMount, onCleanup, createEffect, createSignal } from "solid-js";

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

interface ImageCanvasProps {
  imageData: ImageData | null;
  scale: number;
  panX: number;
  panY: number;
  onScaleChange: (scale: number) => void;
  onPanChange: (x: number, y: number) => void;
  onReset: () => void;
}

export function ImageCanvas(props: ImageCanvasProps) {
  let canvasRef: HTMLCanvasElement | undefined;
  const [rawImageData, setRawImageData] = createSignal<ImageData | null>(null);

  // Update rawImageData when props change
  createEffect(() => {
    if (props.imageData) {
      setRawImageData(props.imageData);
    }
  });

  const drawImage = () => {
    const canvas = canvasRef;
    const data = rawImageData();
    if (!canvas || !data) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const image = data.image;
    const height = image.length;
    const width = image[0]?.length || 0;

    if (width === 0 || height === 0) return;

    // Get container size for responsive display
    const container = canvas.parentElement;
    const containerWidth = container ? container.clientWidth - 2 : width;
    const containerHeight = container ? container.clientHeight - 2 : height;
    
    // Calculate display size (fit to container while maintaining aspect ratio)
    const aspectRatio = width / height;
    let displayWidth = width;
    let displayHeight = height;
    
    if (width > containerWidth || height > containerHeight) {
      if (containerWidth / aspectRatio <= containerHeight) {
        displayWidth = containerWidth;
        displayHeight = containerWidth / aspectRatio;
      } else {
        displayHeight = containerHeight;
        displayWidth = containerHeight * aspectRatio;
      }
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
    const imageData = ctx.createImageData(width, height);
    const imgData = imageData.data;

    // Convert 2D array to ImageData
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const value = image[y][x];
        // Grayscale
        imgData[idx] = value;     // R
        imgData[idx + 1] = value; // G
        imgData[idx + 2] = value; // B
        imgData[idx + 3] = 255;   // A
      }
    }
    
    // Create temporary canvas to hold the image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;
    
    tempCtx.putImageData(imageData, 0, 0);
    
    // Calculate scale to fit image in display area
    const fitScale = Math.min(displayWidth / width, displayHeight / height);
    
    // Apply user transform (pan and zoom)
    ctx.save();
    
    // Center the image initially
    const centerX = displayWidth / 2;
    const centerY = displayHeight / 2;
    
    // Apply transforms: translate to center, scale, then pan
    ctx.translate(centerX + props.panX, centerY + props.panY);
    ctx.scale(props.scale * fitScale, props.scale * fitScale);
    ctx.translate(-width / 2, -height / 2);
    
    // Draw the image
    ctx.drawImage(tempCanvas, 0, 0);
    
    ctx.restore();
  };

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const oldScale = props.scale;
    const newScale = Math.max(0.1, Math.min(10, oldScale * zoomFactor));

    // Zoom towards mouse position
    const worldX = (mouseX - props.panX) / oldScale;
    const worldY = (mouseY - props.panY) / oldScale;
    
    props.onPanChange(mouseX - worldX * newScale, mouseY - worldY * newScale);
    props.onScaleChange(newScale);
  };

  const [isDragging, setIsDragging] = createSignal(false);
  const [dragStart, setDragStart] = createSignal({ x: 0, y: 0 });

  const handleMouseDown = (e: MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - props.panX, y: e.clientY - props.panY });
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging()) {
      props.onPanChange(e.clientX - dragStart().x, e.clientY - dragStart().y);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  onMount(() => {
    const canvas = canvasRef;
    if (canvas) {
      canvas.addEventListener("wheel", handleWheel, { passive: false });
      canvas.addEventListener("mousedown", handleMouseDown);
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      
      if (rawImageData()) {
        drawImage();
      }
    }
  });

  onCleanup(() => {
    const canvas = canvasRef;
    if (canvas) {
      canvas.removeEventListener("wheel", handleWheel);
      canvas.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    }
  });

  // Redraw when image data or transform changes
  createEffect(() => {
    if (rawImageData()) {
      // Trigger redraw when data, scale, or pan changes
      const _ = props.scale;
      const __ = props.panX;
      const ___ = props.panY;
      const ____ = rawImageData();
      setTimeout(() => drawImage(), 0);
    }
  });

  return (
    <div class="space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-lg font-semibold text-gray-900">Integrated Image</h2>
        <div class="flex items-center space-x-4 text-sm text-gray-600">
          <span>Zoom: {(props.scale * 100).toFixed(0)}%</span>
          <button
            onClick={props.onReset}
            class="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
          >
            Reset View
          </button>
        </div>
      </div>
      
      <div class="border border-gray-300 rounded-md overflow-hidden bg-gray-100">
        <canvas
          ref={(el) => (canvasRef = el)}
          class="block cursor-move"
          style={{
            "max-width": "100%",
            "height": "auto",
          }}
        />
      </div>
      
      <div class="text-xs text-gray-500 space-y-1">
        <p>• Scroll to zoom in/out</p>
        <p>• Click and drag to pan</p>
        <p>• Right-click and drag (or middle mouse button) to pan</p>
      </div>
    </div>
  );
}

