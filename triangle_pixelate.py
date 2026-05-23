import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import shared_memory


def triangle_grid(image_size, triangle_size):
    """
    Generates the coordinates of equilateral triangles to cover the image.
    Creates a hexagonal grid where each vertex is shared by 6 triangles.
    """
    width, height = image_size
    triangles = []
    h = triangle_size * np.sqrt(3) / 2  # height of an equilateral triangle

    # Number of rows and columns needed
    rows = int(np.ceil(height / h)) + 1
    cols = int(np.ceil(width / (triangle_size / 2))) + 2

    for row in range(rows):
        y_base = row * h
        for col in range(cols):
            # Start grid slightly before x=0 to ensure proper coverage
            x_base = (col - 1) * (triangle_size / 2)

            # Alternate triangles to create a hexagonal grid
            if (row + col) % 2 == 0:
                # Upward-pointing triangle
                pts = [
                    (x_base, y_base + h),
                    (x_base + triangle_size / 2, y_base),
                    (x_base + triangle_size, y_base + h),
                ]
            else:
                # Downward-pointing triangle
                pts = [
                    (x_base, y_base),
                    (x_base + triangle_size / 2, y_base + h),
                    (x_base + triangle_size, y_base),
                ]

            triangles.append(pts)

    return triangles


# Shared-memory globals used by worker processes
_shared_img = None
_shared_shm = None
_img_shape = None
_img_dtype = None
_img_width = None
_img_height = None


def _init_worker(shm_name, shape, dtype_str, width, height):
    """Initializer for worker processes: attach to existing shared memory."""
    global _shared_img, _shared_shm, _img_shape, _img_dtype, _img_width, _img_height
    existing = shared_memory.SharedMemory(name=shm_name)
    _shared_shm = existing
    _img_dtype = np.dtype(dtype_str)
    _img_shape = tuple(shape)
    _shared_img = np.ndarray(_img_shape, dtype=_img_dtype, buffer=existing.buf)
    _img_width = width
    _img_height = height


def _compute_color(pts, grayscale_flag, dominant_flag, lum_weights_list):
    """Compute fill color for a single triangle using the shared image."""
    from PIL import Image, ImageDraw

    pts = [(float(x), float(y)) for x, y in pts]
    mask = Image.new("L", (_img_width, _img_height), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    mask_np = np.array(mask)
    pixels = _shared_img[mask_np == 255]
    if len(pixels) == 0:
        return (pts, (255, 255, 255))
    lum_weights_arr = np.array(lum_weights_list)
    if grayscale_flag:
        lums = np.dot(pixels, lum_weights_arr)
        if dominant_flag:
            lums_int = np.round(lums).astype(int)
            unique_vals, counts = np.unique(lums_int, return_counts=True)
            dominant_lum = int(unique_vals[np.argmax(counts)])
            return (pts, (dominant_lum, dominant_lum, dominant_lum))
        else:
            mean_lum = int(np.round(np.mean(lums)))
            return (pts, (mean_lum, mean_lum, mean_lum))
    else:
        if dominant_flag:
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_idx = np.argmax(counts)
            color = tuple(int(c) for c in unique_colors[dominant_idx])
            return (pts, color)
        else:
            mean_color = np.mean(pixels, axis=0).astype(int)
            color = tuple(int(c) for c in mean_color)
            return (pts, color)


def triangle_pixelate(
    input_path,
    output_path,
    triangle_size=20,
    use_dominant_color=False,
    grayscale=False,
    workers=None,
):
    """
    Transforms the image into a mosaic of equilateral triangles.

    Args:
        input_path: Input image path
        output_path: Output image path
        triangle_size: Size of the triangles
        use_dominant_color: If True, use the most frequent color instead of the mean
        grayscale: If True, output will be in grayscale (triangles filled with gray values)
    """
    img = Image.open(input_path).convert("RGB")
    width, height = img.size
    triangles = triangle_grid((width, height), triangle_size)
    img_np = np.array(img)
    out_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(out_img)

    # Weights for luminance calculation (Rec. 601 / ITU-R BT.601)
    lum_weights = np.array([0.299, 0.587, 0.114])

    # Run sequentially when only one worker is requested.
    if workers is not None and int(workers) <= 1:
        for pts in triangles:
            mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask).polygon(pts, fill=255)
            mask_np = np.array(mask)
            pixels = img_np[mask_np == 255]
            if len(pixels) == 0:
                color = (255, 255, 255)
            else:
                if grayscale:
                    lums = np.dot(pixels, lum_weights)
                    if use_dominant_color:
                        lums_int = np.round(lums).astype(int)
                        unique_vals, counts = np.unique(lums_int, return_counts=True)
                        dominant_lum = int(unique_vals[np.argmax(counts)])
                        color = (dominant_lum, dominant_lum, dominant_lum)
                    else:
                        mean_lum = int(np.round(np.mean(lums)))
                        color = (mean_lum, mean_lum, mean_lum)
                else:
                    if use_dominant_color:
                        unique_colors, counts = np.unique(
                            pixels, axis=0, return_counts=True
                        )
                        dominant_idx = np.argmax(counts)
                        color = tuple(int(c) for c in unique_colors[dominant_idx])
                    else:
                        mean_color = np.mean(pixels, axis=0).astype(int)
                        color = tuple(int(c) for c in mean_color)
            draw.polygon(pts, fill=color)
        out_img.save(output_path)
        return

    # Create shared memory for the image so workers don't copy the whole array
    dtype = img_np.dtype
    shape = img_np.shape
    shm = shared_memory.SharedMemory(create=True, size=img_np.nbytes)
    try:
        shm_img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        shm_img[:] = img_np[:]

        # Use top-level _init_worker and _compute_color defined in this module

        # Dispatch tasks to worker pool
        if workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            num_workers = max(1, int(workers))
        # Allow override via environment variable or attribute? Keep simple for now
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(shm.name, shape, str(dtype), width, height)) as exec:
            futures = [exec.submit(_compute_color, tuple(pts), grayscale, use_dominant_color, lum_weights.tolist()) for pts in triangles]
            for fut in as_completed(futures):
                pts_res, color = fut.result()
                draw.polygon(pts_res, fill=color)

        out_img.save(output_path)
    finally:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python triangle_pixelate.py input_image output_image [triangle_size] [--dominant] [--grayscale] [--workers=N]"
        )
        print("  --dominant: Use the most frequent color instead of the mean")
        print("  --grayscale: Output in grayscale (triangles filled with gray values)")
        print("  --workers=N: Number of worker processes to use (e.g. --workers=4)")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        triangle_size = 20
        use_dominant_color = False
        grayscale = False
        workers = None

        # Parse args: support numeric triangle_size, flags, and --workers=N
        i = 3
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--dominant":
                use_dominant_color = True
                i += 1
            elif arg == "--grayscale":
                grayscale = True
                i += 1
            elif arg.startswith("--workers="):
                try:
                    workers = int(arg.split("=", 1)[1])
                except Exception:
                    workers = None
                i += 1
            else:
                # Try parsing as triangle size
                try:
                    triangle_size = int(arg)
                except ValueError:
                    pass
                i += 1

        triangle_pixelate(
            input_path, output_path, triangle_size, use_dominant_color, grayscale, workers=workers
        )
