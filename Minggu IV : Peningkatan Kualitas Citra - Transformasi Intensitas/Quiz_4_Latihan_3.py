import cv2
import numpy as np
import time

class RealTimeEnhancement:
    
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []
        self.buffer_size = 5   # untuk menjaga konsistensi antar frame
        
    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame with real-time constraints
        
        Parameters:
        frame: Input video frame
        enhancement_type: Type of enhancement
        
        Returns:
        Enhanced frame
        """

        # Convert ke grayscale untuk proses cepat
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if enhancement_type == 'adaptive':
            # CLAHE (lebih stabil untuk real-time)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

        elif enhancement_type == 'histogram':
            # Histogram Equalization
            enhanced = cv2.equalizeHist(gray)

        elif enhancement_type == 'bright':
            # Brightness adjustment
            enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=30)

        else:
            enhanced = gray

        # ===============================
        # TEMPORAL CONSISTENCY
        # ===============================
        self.history_buffer.append(enhanced)

        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)

        # rata-rata frame untuk stabilisasi
        avg_frame = np.mean(self.history_buffer, axis=0).astype(np.uint8)

        return avg_frame


# ======================================
# REAL TIME VIDEO PROCESSING
# ======================================

cap = cv2.VideoCapture(0)   # webcam

enhancer = RealTimeEnhancement(target_fps=30)

while True:

    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    enhanced_frame = enhancer.enhance_frame(frame, 'adaptive')

    cv2.imshow("Original Video", frame)
    cv2.imshow("Enhanced Video", enhanced_frame)

    # hitung delay agar FPS stabil
    process_time = time.time() - start
    delay = max(int((1/enhancer.target_fps - process_time) * 1000), 1)

    if cv2.waitKey(delay) & 0xFF == 27:  # tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()