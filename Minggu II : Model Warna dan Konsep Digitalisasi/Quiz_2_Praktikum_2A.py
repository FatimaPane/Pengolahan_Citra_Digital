import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# =======================================================
# 1. FUNGSI ANALISIS APLIKASI SPESIFIK (TUGAS BARU)
# =======================================================
def analyze_color_model_suitability(image, application):
    """
    Menganalisis model warna terbaik berdasarkan kasus penggunaan.
    """
    res_img = image.copy()
    
    if application == 'skin_detection':
        best_model = 'HSV'
        analysis = "HSV memisahkan warna (Hue) dari cahaya. Kulit manusia konsisten di rentang Hue rendah."
        # Proses: Masking warna kulit sederhana
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        res_img = cv2.inRange(hsv, lower, upper)
        
    elif application == 'shadow_removal':
        best_model = 'LAB'
        analysis = "LAB memisahkan Lightness (L). Kita bisa mencerahkan bayangan tanpa merusak warna."
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        res_img = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2RGB)
        
    elif application == 'text_extraction':
        best_model = 'GRAYSCALE'
        analysis = "Grayscale fokus pada kontras intensitas, memudahkan pemisahan teks dari background."
        res_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    elif application == 'object_detection':
        best_model = 'HSV'
        analysis = "HSV lebih robust terhadap perubahan bayangan saat mendeteksi objek berwarna."
        res_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    return best_model, analysis, res_img

# =======================================================
# 2. MAIN PROGRAM PRAKTIKUM
# =======================================================

# --- Persiapan Gambar ---
sample_img = cv2.imread('kakatua.jpg')
if sample_img is None:
    # Buat gambar sintetis jika file tidak ada
    sample_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(sample_img, (50, 50), (150, 150), (255, 0, 0), -1)   # Blue
    cv2.circle(sample_img, (250, 100), 50, (0, 255, 0), -1)          # Green
    cv2.putText(sample_img, 'TEXT', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

# --- A. Visualisasi Aplikasi Spesifik ---
apps = ['skin_detection', 'shadow_removal', 'text_extraction', 'object_detection']
plt.figure(figsize=(15, 10))
plt.suptitle("Analisis Model Warna untuk Aplikasi Spesifik", fontsize=16, fontweight='bold')

for i, app in enumerate(apps):
    model, desc, processed = analyze_color_model_suitability(sample_img, app)
    
    plt.subplot(2, 2, i+1)
    if len(processed.shape) == 2: # Jika grayscale/mask
        plt.imshow(processed, cmap='gray')
    else:
        # Jika RGB/BGR (LAB di fungsi sudah diconvert ke RGB untuk display)
        if app == 'object_detection': # HSV display logic
             plt.imshow(processed)
        else:
             plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
             
    plt.title(f"Aplikasi: {app}\nBest Model: {model}", fontsize=12, color='blue')
    plt.xlabel(f"Analisis: {desc}", fontsize=9)
    plt.xticks([]); plt.yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- B. Demonstrasi Sampling & Aliasing ---
print("Menjalankan simulasi Sampling Nyquist...")
t = np.linspace(0, 1, 1000)
f_max = 50 # Frekuensi sinyal 50Hz
signal_pure = np.sin(2 * np.pi * f_max * t)

rates = [20, 50, 100, 200] # Nyquist rate adalah 2 * 50 = 100Hz
plt.figure(figsize=(12, 6))
for i, fs in enumerate(rates):
    plt.subplot(2, 2, i+1)
    t_samp = np.arange(0, 1, 1/fs)
    s_samp = np.sin(2 * np.pi * f_max * t_samp)
    plt.plot(t, signal_pure, alpha=0.3, label='Original')
    plt.stem(t_samp, s_samp, 'r', markerfmt='ro', label=f'fs={fs}Hz')
    status = "AMAN" if fs >= 100 else "ALIASING!"
    plt.title(f"Sampling {fs}Hz ({status})")
    plt.legend(loc='upper right', fontsize='small')

plt.suptitle("Teorema Sampling Nyquist (f_max = 50Hz)", fontsize=14)
plt.tight_layout()
plt.show()

# --- C. Demonstrasi Kuantisasi ---
print("Menjalankan simulasi Kuantisasi...")
gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
levels = [256, 16, 4, 2] # 8-bit, 4-bit, 2-bit, 1-bit

plt.figure(figsize=(15, 4))
for i, L in enumerate(levels):
    step = 256 // L
    quantized = (gray // step) * step
    mse = np.mean((gray - quantized)**2)
    
    plt.subplot(1, 4, i+1)
    plt.imshow(quantized, cmap='gray')
    plt.title(f"Levels: {L}\nMSE: {mse:.2f}")
    plt.axis('off')

plt.suptitle("Efek Kuantisasi terhadap Kualitas Citra", fontsize=14)
plt.show()

print("\n[PROSES SELESAI]")