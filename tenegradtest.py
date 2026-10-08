from pathlib import Path
import cv2
import numpy as np

# Path menunjuk ke FOLDER atau FILE gambar Anda
TARGET_PATH = Path(r"E:\（辛远川）清华大学深圳国际研究生院\Resources\Code\DIPA\sample_img\tenegrad")

def read_image_grayscale_unicode(img_path: Path):
    """Membaca gambar secara grayscale dengan aman pada path berkarakter Unicode (Mandarin/Windows)."""
    img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    return img

def get_tenengrad_score(img_path: Path) -> float:
    """
    Menghitung skor Tenengrad Focus Measure.
    Menggunakan operator Sobel untuk mendeteksi gradien tepi.
    """
    gray = read_image_grayscale_unicode(img_path)
    if gray is None:
        raise ValueError("Gambar tidak dapat dibaca.")
        
    # Hitung gradien Sobel pada sumbu X dan Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitudo kuadrat gradien G_x^2 + G_y^2
    grad_sq = sobel_x**2 + sobel_y**2
    
    # Menghitung rata-rata magnitudo kuadrat (invariant terhadap ukuran piksel gambar)
    score = np.mean(grad_sq)
    return float(score)

def evaluate_tenengrad(target_path: Path):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    if not target_path.exists():
        print(f"Error: Path tidak ditemukan -> {target_path}")
        return

    # Filter input folder atau file tunggal
    if target_path.is_file():
        image_files = [target_path]
    elif target_path.is_dir():
        image_files = [f for f in target_path.iterdir() if f.suffix.lower() in valid_extensions]
    else:
        print("Path tidak valid.")
        return
    
    if not image_files:
        print(f"Tidak ada file gambar yang ditemukan di: {target_path}")
        return

    print(f"Memproses {len(image_files)} gambar menggunakan Tenengrad Focus Measure...\n")
    print(f"{'Nama File':<40} | {'Skor Tenengrad':<15} | {'Keterangan'}")
    print("-" * 70)

    # Hitung skor untuk setiap gambar
    results = []
    for img_path in sorted(image_files):
        try:
            score = get_tenengrad_score(img_path)
            results.append((img_path.name, score))
        except Exception as e:
            print(f"{img_path.name:<40} | Error: {e}")

    # Tampilkan hasil
    if results:
        max_score = max(r[1] for r in results)
        for name, score in results:
            ket = "Good Image" if len(results) > 1 and score == max_score else "Bad Image"
            print(f"{name:<40} | {score:<15.2f} | {ket}")

if __name__ == "__main__":
    evaluate_tenengrad(TARGET_PATH)