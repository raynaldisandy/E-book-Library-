r"""Script Interaktif Sederhana untuk Mencoba Model AI e-Library BPJS Kesehatan.

Cara Menjalankan:
    .\.venv\Scripts\python coba_model.py
"""
import sys
import time
from pathlib import Path

# Setup Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import BookRecommender

print("=" * 75)
print("  🤖 MEMUAT MODEL PRODUKSI AI e-LIBRARY BPJS KESEHATAN...")
print("=" * 75)

t0 = time.perf_counter()
model = BookRecommender.load("models/best_model", allow_fallback=True)
load_time = (time.perf_counter() - t0) * 1000.0

print(f"✅ Model Berhasil Dimuat dalam {load_time:.1f} ms!")
print(f"   • Model Name     : {model.actual_model_name} ({model.model_display_name})")
print(f"   • Koleksi Buku   : {len(model.catalog)} buku katalog BPJS")
print(f"   • Profil Kamus   : {len(model.profiles)} profil kompetensi kerja")
print("=" * 75)

def menu():
    while True:
        print("\nPilih Fitur yang Ingin Anda Coba:")
        print("  [1] Input Buku Baru Bebas (Book -> Multiple Competencies)")
        print("  [2] Cari Buku untuk Competency Gap Karyawan (Gap -> Books)")
        print("  [3] Jalankan Contoh Cepat Otomatis (Preset Demo)")
        print("  [0] Keluar")
        
        pilihan = input("\nMasukkan pilihan (1/2/3/0): ").strip()
        
        if pilihan == "1":
            print("\n" + "-" * 70)
            print("📝 FITUR 1: PEMETAAN BUKU BARU (KATALOGISASI OTOMATIS)")
            print("-" * 70)
            judul = input("Ketik / Paste Judul Buku: ").strip()
            if not judul:
                print("⚠️ Judul tidak boleh kosong.")
                continue
            abstrak = input("Ketik / Paste Abstrak Buku: ").strip()
            if not abstrak:
                abstrak = judul
                
            top_k_str = input("Berapa rekomendasi teratas yang diinginkan? (default: 3): ").strip()
            top_k = int(top_k_str) if top_k_str.isdigit() else 3
            
            print("\n⏳ Menganalisis makna semantik teks...")
            t_inf = time.perf_counter()
            hasil = model.map_book(judul, abstrak, top_k=top_k)
            dt = (time.perf_counter() - t_inf) * 1000.0
            
            print(f"\n🎯 HASIL REKOMENDASI KOMPETENSI (Selesai dalam {dt:.2f} ms):")
            for h in hasil:
                print(f"   Rank {h['rank']}: [{h['competency_id']}] {h['competency_name']}")
                print(f"           Skor Kecocokan: {h['score']:.4f} | Kategori: {h['category']}")
                
        elif pilihan == "2":
            print("\n" + "-" * 70)
            print("📚 FITUR 2: REKOMENDASI BUKU PENUTUP COMPETENCY GAP KARYAWAN")
            print("-" * 70)
            print("Contoh kode kompetensi BPJS:")
            print("  • C023: Risk Management (Manajemen Risiko)")
            print("  • C025: Auditing Skill (Pemeriksaan / Audit Internal)")
            print("  • C009: Fraud Management (Pencegahan Kecurangan Klaim)")
            print("  • C061: IT Security (Keamanan Informasi & Jaringan)")
            print("  • C010: Corporate Budgeting (Manajemen Anggaran)")
            
            cid = input("\nMasukkan Kode Kompetensi (contoh: C023): ").strip().upper()
            if not cid:
                cid = "C023"
                
            print(f"\n⏳ Mencari koleksi buku perpustakaan untuk gap [{cid}]...")
            t_inf = time.perf_counter()
            buku_list = model.recommend_books_for_competency(cid, top_k=3)
            dt = (time.perf_counter() - t_inf) * 1000.0
            
            print(f"\n📖 HASIL REKOMENDASI BUKU KATALOG (Selesai dalam {dt:.2f} ms ⚡):")
            for b in buku_list:
                print(f"   Rank {b['rank']}: \"{b['judul']}\"")
                print(f"           ID: {b['book_id']} | Tahun: {b['tahun']} | Skor Relevansi: {b['score']:.4f}")
                
        elif pilihan == "3":
            print("\n" + "-" * 70)
            print("⚡ MENJALANKAN CONTOH OTOMATIS...")
            print("-" * 70)
            
            # Demo 1
            j_demo = "Tata Kelola Manajemen Risiko Operasional dan Audit Internal Faskes"
            a_demo = "Panduan mitigasi risiko kepatuhan faskes, audit operasional rumah sakit, dan pengawasan internal."
            print(f"📖 Contoh 1 (Buku Baru): \"{j_demo}\"")
            res_demo = model.map_book(j_demo, a_demo, top_k=2)
            for h in res_demo:
                print(f"   • Rank {h['rank']}: [{h['competency_id']}] {h['competency_name']} (Skor: {h['score']:.4f})")
                
            # Demo 2
            print(f"\n👤 Contoh 2 (Pegawai Gap C023 - Risk Management):")
            b_demo = model.recommend_books_for_competency("C023", top_k=2)
            for b in b_demo:
                print(f"   • Rank {b['rank']}: \"{b['judul']}\" (Skor: {b['score']:.4f})")
                
        elif pilihan == "0":
            print("\nTerima kasih! Sampai jumpa.")
            break
        else:
            print("Pilihan tidak valid, silakan ketik 1, 2, 3, atau 0.")

if __name__ == "__main__":
    menu()
