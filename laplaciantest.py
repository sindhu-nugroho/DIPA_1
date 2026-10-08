from pathlib import Path
import cv2
import numpy as np

# Path menunjuk ke FOLDER atau FILE gambar Anda
TARGET_PATH = Path(r"E:\（辛远川）清华大学深圳国际研究生院\Resources\Code\DIPA\sample_img\tenegrad")

def read_image_grayscale_unicode(img_path: Path) -> np.ndarray:
    """Membaca gambar secara grayscale dengan aman dari path Unicode/Windows."""
    img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    return img

def get_laplacian_variance(img_path: Path) -> float:
    """
    Menghitung skor Variance of Laplacian.
    Semakin tinggi skor -> Gambar semakin tajam / fokus.
    Semakin rendah skor -> Gambar semakin blur / kabur.
    """
    gray = read_image_grayscale_unicode(img_path)
    if gray is None:
        raise ValueError("Gambar tidak dapat dibaca atau file corrupt.")
    
    # 1. Terapkan operator Laplacian (turunan kedua)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 2. Hitung varians dari hasil matriks Laplacian
    score = laplacian.var()
    return float(score)

def evaluate_laplacian_variance(target_path: Path, threshold: float = 100.0):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    if not target_path.exists():
        print(f"Error: Path tidak ditemukan -> {target_path}")
        return

    # Penanganan file tunggal atau folder
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

    print(f"Memproses {len(image_files)} gambar menggunakan Variance of Laplacian...\n")
    print(f"{'Nama File':<40} | {'Skor Varians':<15} | {'Status'}")
    print("-" * 75)

    for img_path in sorted(image_files):
        try:
            score = get_laplacian_variance(img_path)
            
            # Pengkategorian berdasarkan nilai threshold
            status = "Fokus / Sharp" if score >= threshold else "Blur"
            
            print(f"{img_path.name:<40} | {score:<15.2f} | {status}")
        except Exception as e:
            print(f"{img_path.name:<40} | Error: {e}")

if __name__ == "__main__":
    # Threshold 100.0 adalah batas standar umum (dapat disesuaikan)
    evaluate_laplacian_variance(TARGET_PATH, threshold=100.0)