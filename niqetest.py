from pathlib import Path
import torch
import pyiqa

# Path menunjuk ke FOLDER atau FILE gambar Anda
TARGET_PATH = Path(r"E:\（辛远川）清华大学深圳国际研究生院\Resources\Code\DIPA\sample_img\niqe")

def evaluate_niqe(target_path: Path):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    if not target_path.exists():
        print(f"Error: Path tidak ditemukan -> {target_path}")
        return

    # Filter gambar jika input berupa folder atau file tunggal
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

    print("Memuat model NIQE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inisialisasi metrik NIQE dari pyiqa
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    print(f"Memproses {len(image_files)} gambar menggunakan NIQE...\n")
    print(f"{'Nama File':<40} | {'Skor NIQE':<15} | {'Kategori'}")
    print("-" * 70)

    for img_path in sorted(image_files):
        try:
            # pyiqa dapat menerima path file secara langsung
            score = niqe_metric(str(img_path)).item()

            # Pengkategorian skor NIQE (Makin rendah = Makin bagus)
            if score < 4.0:
                kategori = "Very Good"
            elif score < 6.0:
                kategori = "Good"
            elif score < 8.0:
                kategori = "Normal"
            else:
                kategori = "Not Good"

            print(f"{img_path.name:<40} | {score:<15.2f} | {kategori}")
        except Exception as e:
            print(f"{img_path.name:<40} | Error: {e}")

if __name__ == "__main__":
    evaluate_niqe(TARGET_PATH)