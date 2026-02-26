# ============================================
# PRAKTIKUM 2: MODEL WARNA DAN DIGITALISASI
# ============================================

import cv2
import numpy as npfigure
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== PRAKTIKUM 2: MODEL WARNA DAN DIGITALISASI ===")
print("Materi: Color Models, Sampling, Quantization, Color Conversion\n")

# ==========================================================
# 1. PERBANDINGAN MODEL WARNA
# ==========================================================

def create_color_patches():
    patches = []
    colors = [
        ('Red', [0, 0, 255]),
        ('Green', [0, 255, 0]),
        ('Blue', [255, 0, 0]),
        ('Yellow', [0, 255, 255]),
        ('Magenta', [255, 0, 255]),
        ('Cyan', [255, 255, 0]),
        ('White', [255, 255, 255]),
        ('Black', [0, 0, 0])
    ]
    
    for name, color in colors:
        patch = np.zeros((100, 100, 3), dtype=np.uint8)
        patch[:, :] = color
        patches.append((name, patch))
    return patches

print("1. PERBANDINGAN MODEL WARNA")

color_patches = create_color_patches()
models = ['RGB', 'HSV', 'LAB', 'GRAY']

fig, axes = plt.subplots(4, 8, figsize=(20, 10))

for row, model in enumerate(models):
    for col, (name, patch) in enumerate(color_patches):
        
        if model == 'RGB':
            display_img = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(display_img)
        
        elif model == 'GRAY':
            gray_img = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            axes[row, col].imshow(gray_img, cmap='gray')
        
        elif model == 'HSV':
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            display_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            axes[row, col].imshow(display_img)
        
        elif model == 'LAB':
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            lab_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            axes[row, col].imshow(lab_rgb)

        if row == 0:
            axes[row, col].set_title(name)
        axes[row, col].axis('off')

    axes[row, 0].text(-0.5, 0.5, model,
                      transform=axes[row, 0].transAxes,
                      fontsize=12, fontweight='bold',
                      va='center', ha='right')

plt.suptitle("Perbandingan Model Warna")
plt.tight_layout()
plt.show()


# ==========================================================
# 2. ANALISIS KOMPONEN WARNA PADA CITRA NYATA
# ==========================================================

print("\n2. ANALISIS KOMPONEN WARNA")

img = cv2.imread('kakatua.jpg')

if img is None:
    print("Menggunakan citra sintetik")
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.circle(img, (250, 100), 50, (0, 255, 0), -1)
    cv2.ellipse(img, (300, 200), (80, 40), 30, 0, 360, (0, 0, 255), -1)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0,0].imshow(rgb)
axes[0,0].set_title("RGB")
axes[0,0].axis('off')

axes[0,1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
axes[0,1].set_title("HSV")
axes[0,1].axis('off')

axes[0,2].imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
axes[0,2].set_title("LAB")
axes[0,2].axis('off')

axes[1,0].imshow(gray, cmap='gray')
axes[1,0].set_title("Grayscale")
axes[1,0].axis('off')

axes[1,1].hist(gray.ravel(), 256, [0,256])
axes[1,1].set_title("Histogram Grayscale")

for i, color in enumerate(['b','g','r']):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    axes[1,2].plot(hist, color=color)

axes[1,2].set_title("Histogram RGB")

plt.tight_layout()
plt.show()


# ==========================================================
# 3. DEMONSTRASI TEOREMA SAMPLING DAN ALIASING
# ==========================================================

print("\n3. DEMONSTRASI SAMPLING & ALIASING")

t = np.linspace(0, 1, 1000)
f_high = 50
signal_original = np.sin(2*np.pi*f_high*t)

sampling_rates = [20, 50, 100, 200]
nyquist = 2 * f_high

fig, axes = plt.subplots(2,2, figsize=(12,8))
axes = axes.ravel()

for i, fs in enumerate(sampling_rates):
    t_sample = np.arange(0,1,1/fs)
    idx = (t_sample * 1000).astype(int)
    sampled = signal_original[idx]
    
    axes[i].plot(t, signal_original, alpha=0.5)
    axes[i].stem(t_sample, sampled, basefmt=" ")
    
    if fs < nyquist:
        axes[i].set_title(f"ALIASING (fs={fs}Hz)", color='red')
    else:
        axes[i].set_title(f"TANPA ALIASING (fs={fs}Hz)", color='green')

plt.tight_layout()
plt.show()

print(f"Nyquist Rate = {nyquist} Hz")


# ==========================================================
# 4. KUANTISASI UNIFORM DAN NON-UNIFORM
# ==========================================================

print("\n4. DEMONSTRASI KUANTISASI")

test_img = gray
levels_list = [256, 64, 16, 4, 2]

fig, axes = plt.subplots(2,5, figsize=(15,6))

for idx, levels in enumerate(levels_list):

    # Uniform
    step = 256 // levels
    uniform_q = (test_img // step) * step
    mse_u = np.mean((test_img - uniform_q)**2)
    psnr_u = 10*np.log10(255**2/mse_u) if mse_u > 0 else float('inf')

    axes[0,idx].imshow(uniform_q, cmap='gray', vmin=0, vmax=255)
    axes[0,idx].set_title(f"Uniform {levels}")
    axes[0,idx].axis('off')

    # Non-Uniform berbasis CDF
    hist = cv2.calcHist([test_img],[0],None,[256],[0,256]).flatten()
    cdf = hist.cumsum()
    cdf_norm = cdf / cdf.max()

    mapping = np.floor(cdf_norm * (levels - 1)) * (255 // (levels - 1))
    mapping = mapping.astype(np.uint8)

    nonuniform_q = mapping[test_img]
    mse_n = np.mean((test_img - nonuniform_q)**2)
    psnr_n = 10*np.log10(255**2/mse_n) if mse_n > 0 else float('inf')

    axes[1,idx].imshow(nonuniform_q, cmap='gray', vmin=0, vmax=255)
    axes[1,idx].set_title(f"NonUniform {levels}")
    axes[1,idx].axis('off')

plt.tight_layout()
plt.show()


# ==========================================================
# 5. ANALISIS SQNR TEORITIS
# ==========================================================

print("\nAnalisis SQNR Teoritis:")
for levels in levels_list:
    bits = int(np.log2(levels))
    sqnr = 6.02*bits + 1.76
    print(f"{bits}-bit -> SQNR ≈ {sqnr:.2f} dB")


# ==========================================================
# 6. APLIKASI KONVERSI WARNA (HSV & LAB)
# ==========================================================

print("\n5. APLIKASI PRAKTIS")

# Skin detection (HSV)
lower_skin = np.array([0,20,70])
upper_skin = np.array([20,255,255])
mask = cv2.inRange(hsv, lower_skin, upper_skin)
skin = cv2.bitwise_and(img, img, mask=mask)

# CLAHE pada LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)
clahe = cv2.createCLAHE(3.0,(8,8))
L2 = clahe.apply(L)
lab2 = cv2.merge([L2,A,B])
enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

fig, axes = plt.subplots(1,3, figsize=(15,5))

axes[0].imshow(mask, cmap='gray')
axes[0].set_title("Skin Mask (HSV)")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
axes[1].set_title("Skin Detection")
axes[1].axis('off')

axes[2].imshow(enhanced)
axes[2].set_title("Enhancement (LAB + CLAHE)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("\n=== PRAKTIKUM SELESAI ===")
