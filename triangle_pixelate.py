import numpy as np
from PIL import Image, ImageDraw


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


def triangle_pixelate(
    input_path, output_path, triangle_size=20, use_dominant_color=False, grayscale=False
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

    for pts in triangles:
        # Create a mask for the triangle
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        mask_np = np.array(mask)
        # Extract the pixels under the triangle
        pixels = img_np[mask_np == 255]
        if len(pixels) == 0:
            color = (255, 255, 255)
        else:
            if grayscale:
                # Compute luminance for each pixel
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
                    # Find the most frequent color
                    unique_colors, counts = np.unique(
                        pixels, axis=0, return_counts=True
                    )
                    dominant_idx = np.argmax(counts)
                    color = tuple(int(c) for c in unique_colors[dominant_idx])
                else:
                    # Use the mean color
                    mean_color = np.mean(pixels, axis=0).astype(int)
                    color = tuple(int(c) for c in mean_color)
        draw.polygon(pts, fill=color)

    out_img.save(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python triangle_pixelate.py input_image output_image [triangle_size] [--dominant] [--grayscale]"
        )
        print("  --dominant: Use the most frequent color instead of the mean")
        print("  --grayscale: Output in grayscale (triangles filled with gray values)")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        triangle_size = 20
        use_dominant_color = False
        grayscale = False

        for arg in sys.argv[3:]:
            if arg == "--dominant":
                use_dominant_color = True
            elif arg == "--grayscale":
                grayscale = True
            else:
                try:
                    triangle_size = int(arg)
                except ValueError:
                    pass

        triangle_pixelate(
            input_path, output_path, triangle_size, use_dominant_color, grayscale
        )
