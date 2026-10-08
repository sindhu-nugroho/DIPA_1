from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as F
from piq import brisque

# Path folder gambar Anda (menggunakan raw string 'r' agar karakter \ dan Mandarin aman)
FOLDER_PATH = Path(r"E:\（辛远川）清华大学深圳国际研究生院\Resources\Code\DIPA\sample_img")

def evaluate_brisque_folder(folder_path: Path):
    # Ekstensi gambar yang didukung
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    if not folder_path.exists():
        print(f"Error: Path folder tidak ditemukan -> {folder_path}")
        return

    # Filter hanya file gambar
    image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"Tidak ada file gambar yang ditemukan di: {folder_path}")
        return

    print(f"Memproses {len(image_files)} gambar dari folder...\n")
    print(f"{'Nama File':<40} | {'Skor BRISQUE':<15}")
    print("-" * 60)

    for img_path in image_files:
        try:
            # PIL handling Unicode path di Windows
            img = Image.open(img_path).convert('RGB')
            img_tensor = F.to_tensor(img).unsqueeze(0)
            
            with torch.no_grad():
                score = brisque(img_tensor, data_range=1.0).item()
                
            print(f"{img_path.name:<40} | {score:.2f}")
        except Exception as e:
            print(f"{img_path.name:<40} | Error: {e}")

if __name__ == "__main__":
    evaluate_brisque_folder(FOLDER_PATH)