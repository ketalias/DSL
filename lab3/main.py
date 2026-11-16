
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = Path(__file__).parent / "images" / "Porcine-circovirus.png"

img_bgr = cv2.imread(str(IMAGE_PATH))
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# Histogram Equalization

equalized = cv2.equalizeHist(gray)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
equalized_clahe = clahe.apply(gray)


# Original, Global Equalization and CLAHE in one window

images = [img_rgb, equalized, equalized_clahe]
hist_data = [gray, equalized, equalized_clahe]
titles = ["Original Image", "Global Equalization", "CLAHE"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.canvas.manager.set_window_title("Histogram Equalization")

for i in range(3):
    if i == 0:
        axes[0, i].imshow(images[i])
    else:
        axes[0, i].imshow(images[i], cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(titles[i])
    axes[0, i].axis("off")

    axes[1, i].hist(hist_data[i].ravel(), bins=64, range=(0, 256),
                    color="steelblue", edgecolor="black")
    axes[1, i].set_title("Intensity Histogram")
    axes[1, i].set_xlabel("Intensity (0-255)")
    axes[1, i].set_ylabel("Number of Pixels")
    axes[1, i].set_xlim(0, 255)

ZOOM_STEP = 1.25


def enable_linked_zoom(fig, image_axes):
    """Link the given image axes and let the mouse wheel zoom them together
    around the cursor. The toolbar Home button restores the full view."""
    image_axes = list(image_axes)
    for ax in image_axes[1:]:
        ax.sharex(image_axes[0])
        ax.sharey(image_axes[0])
    axes_set = set(image_axes)

    def on_scroll(event):
        ax = event.inaxes
        if ax not in axes_set or event.xdata is None:
            return
        toolbar = fig.canvas.toolbar
        if toolbar is not None and toolbar._nav_stack() is None:
            toolbar.push_current()  # remember the full view for Home

        scale = 1 / ZOOM_STEP if event.button == "up" else ZOOM_STEP
        x, y = event.xdata, event.ydata
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)


enable_linked_zoom(fig, axes[0])
fig.suptitle("Histogram Equalization  (scroll over a photo to zoom, Home to reset)",
             fontsize=14, fontweight="bold")

fig.tight_layout()


# 3. Unsharp Masking

k = 2.5

blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
sharpened = cv2.addWeighted(gray, 1 + k, blurred, -k, 0)

# Edge detail removed by blur / boosted by sharpening (amplified for display)
detail = cv2.absdiff(sharpened, gray)
detail_vis = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)

images = [gray, blurred, sharpened, detail_vis]
titles = [
    "Original",
    "Gaussian Blur",
    f"Unsharp Masking (k={k})",
    "Added Detail"
]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.canvas.manager.set_window_title("Unsharp Masking")

for ax, im, title in zip(axes, images, titles):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

enable_linked_zoom(fig, axes)
fig.suptitle("Unsharp Masking  (scroll over any image to zoom all, Home to reset)",
             fontsize=14, fontweight="bold")
fig.tight_layout()


# Console summary

def laplacian_sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


print(f"{'Image':<22}{'Mean':>8}{'Std':>8}{'Sharpness':>12}")
for name, im in [("Original", gray), ("Global Equalization", equalized),
                 ("CLAHE", equalized_clahe), ("Gaussian Blur", blurred),
                 ("Unsharp Masking", sharpened)]:
    print(f"{name:<22}{im.mean():>8.1f}{im.std():>8.1f}{laplacian_sharpness(im):>12.1f}")



# 4. Edge Detection: Sobel and Canny

# Sobel
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)

# Canny
canny = cv2.Canny(gray, 50, 150)

# Visualization
images = [gray, sobel, canny]
titles = ["Original", "Sobel", "Canny (50, 150)"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.canvas.manager.set_window_title("Edge Detection")

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

enable_linked_zoom(fig, axes)
fig.tight_layout()


# 5. Motion Blur

size = 15

horizontal_kernel = np.zeros((size, size))
horizontal_kernel[size // 2, :] = 1 / size

vertical_kernel = np.zeros((size, size))
vertical_kernel[:, size // 2] = 1 / size

diagonal_kernel = np.eye(size) / size

horizontal = cv2.filter2D(gray, -1, horizontal_kernel)
vertical = cv2.filter2D(gray, -1, vertical_kernel)
diagonal = cv2.filter2D(gray, -1, diagonal_kernel)

images = [gray, horizontal, vertical, diagonal]
titles = ["Original", "Horizontal Blur",
          "Vertical Blur", "Diagonal Blur"]

fig, axes = plt.subplots(1, 4, figsize=(17, 5))
fig.canvas.manager.set_window_title("Motion Blur")

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

enable_linked_zoom(fig, axes)
fig.tight_layout()

plt.show()
