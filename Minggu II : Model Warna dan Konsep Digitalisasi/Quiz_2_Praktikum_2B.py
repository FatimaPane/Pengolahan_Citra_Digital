import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def load_image():
    """
    Open file dialog to choose image manually
    """
    Tk().withdraw()  # Hide root window
    file_path = askopenfilename(
        title="kakatua",
        filetypes=[("kakatua", "*.jpg *.png *.jpeg *.bmp")]
    )

    if not file_path:
        print("Tidak ada gambar yang dipilih!")
        exit()

    image = cv2.imread(file_path)

    if image is None:
        print("Gagal membaca gambar!")
        exit()

    return image


def simulate_image_aliasing(image, downsampling_factors):
    """
    Simulate aliasing using slicing (manual sampling)
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = {}

    for factor in downsampling_factors:

        # =============================
        # 1️⃣ Downsampling manual (tanpa anti-aliasing)
        # =============================
        downsampled = image_rgb[::factor, ::factor]

        # Kembalikan ke ukuran asli
        upsampled = cv2.resize(
            downsampled,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # =============================
        # 2️⃣ Downsampling dengan Blur (anti-aliasing)
        # =============================
        blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)

        downsampled_blur = blurred[::factor, ::factor]

        upsampled_blur = cv2.resize(
            downsampled_blur,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        results[factor] = {
            "aliasing": upsampled,
            "anti_aliasing": upsampled_blur,
            "analysis": f"""
Factor {factor}:
- Sampling langsung menyebabkan aliasing (bergerigi & detail hilang).
- Blur sebelum sampling mengurangi aliasing.
- Semakin besar factor → efek aliasing semakin kuat.
"""
        }

    return results


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    image = load_image()

    factors = [2, 4, 8]

    results = simulate_image_aliasing(image, factors)

    plt.figure(figsize=(15, 8))

    # Original
    plt.subplot(2, len(factors) + 1, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    for i, factor in enumerate(factors):

        # Aliasing
        plt.subplot(2, len(factors) + 1, i + 2)
        plt.imshow(results[factor]["aliasing"])
        plt.title(f"Aliasing x{factor}")
        plt.axis("off")

        # Anti Aliasing
        plt.subplot(2, len(factors) + 1, i + 2 + len(factors) + 1)
        plt.imshow(results[factor]["anti_aliasing"])
        plt.title(f"Anti-Aliasing x{factor}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Print analisis
    for factor in factors:
        print(results[factor]["analysis"])
