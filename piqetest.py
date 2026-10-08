from pathlib import Path
import torch
import pyiqa

# Path menunjuk ke FOLDER atau FILE gambar Anda
TARGET_PATH = Path(r"E:\（辛远川）清华大学深圳国际研究生院\Resources\Code\DIPA\sample_img\piqe")

def evaluate_piqe(target_path: Path):
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

    print("Memuat model PIQE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inisialisasi metrik PIQE dari pyiqa
    piqe_metric = pyiqa.create_metric('piqe', device=device)

    print(f"Memproses {len(image_files)} gambar menggunakan PIQE...\n")
    print(f"{'Nama File':<40} | {'Skor PIQE':<15} | {'Kategori'}")
    print("-" * 70)

    for img_path in sorted(image_files):
        try:
            # Hitung skor PIQE
            score = piqe_metric(str(img_path)).item()

            # Pengkategorian skor PIQE (Skala 0 - 100, Makin rendah = Makin bagus)
            if score < 20:
                kategori = "Very Good"
            elif score < 50:
                kategori = "Good"
            elif score < 80:
                kategori = "Enough"
            else:
                kategori = "Not Good"

            print(f"{img_path.name:<40} | {score:<15.2f} | {kategori}")
        except Exception as e:
            print(f"{img_path.name:<40} | Error: {e}")

if __name__ == "__main__":
    evaluate_piqe(TARGET_PATH)