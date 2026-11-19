# BPJS Book Recommendation

Dasbor e-library untuk mencari buku berdasarkan kompetensi kerja dan memetakan buku baru dari judul serta abstraknya. Dataset proyek berisi 973 buku dan 67 profil kompetensi.

## Menjalankan proyek

Gunakan Python 3.11 atau lebih baru dan Node.js 22.12 atau lebih baru. Dari PowerShell di folder proyek, siapkan lingkungan Python pada instalasi pertama.

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\run_api.bat
```

`run_api.bat` memasang dependensi dan membangun frontend jika hasil build belum ada. Di Linux atau macOS, jalankan `bash run_api.sh` setelah dependensi Python terpasang.

Buka [dasbor](http://localhost:8000/), [Swagger](http://localhost:8000/docs), atau [ReDoc](http://localhost:8000/redoc). Untuk mengedit frontend dengan pemuatan ulang otomatis, jalankan `npm.cmd --prefix frontend run dev` saat FastAPI aktif di port 8000. Bangun ulang versi yang disajikan FastAPI dengan `npm.cmd --prefix frontend run build`.

Rekomendasi katalog menggunakan vektor BGE-M3 yang sudah dihitung. Pemetaan buku baru menggunakan BGE-M3 jika bobot encoder tersedia secara lokal, dengan TF-IDF sebagai model cadangan. Dependensi model semantik tercantum di [requirements-semantic.txt](requirements-semantic.txt). Status model tersedia di `/health`.

## Struktur folder

| Lokasi | Isi |
| --- | --- |
| [`frontend/`](frontend/) | Dasbor React, TypeScript, dan Tailwind CSS |
| [`api.py`](api.py), [`src/`](src/) | Endpoint FastAPI, pemrosesan teks, dan inferensi |
| [`data/`](data/), [`models/`](models/) | Dataset, profil kompetensi, dan artefak model |
| [`notebooks/`](notebooks/), [`outputs/`](outputs/), [`figures/`](figures/) | Eksperimen, hasil evaluasi, dan grafik |
| [`reports/`](reports/), [`deploy/`](deploy/) | Laporan, metodologi, dan konfigurasi deployment |
| [`scripts/`](scripts/), [`tests/`](tests/) | Alat bantu dan pengujian |

## Pengujian

```powershell
.\.venv\Scripts\python -m pip install pytest httpx beautifulsoup4
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python -m pytest tests -q
```

Untuk mencoba kedua alur model lewat terminal, jalankan `coba_model.bat`.

## Rujukan

- [Panduan deployment dan integrasi](deploy/PANDUAN_DEPLOYMENT_API.md)
- [Laporan pemilihan model](reports/LAPORAN_PEMILIHAN_MODEL.md)
- [Rubrik dan sumber data evaluasi](reports/RUBRIK_DAN_PROVENANCE_BENCHMARK.md)
- [Metrik evaluasi](outputs/evaluasi_metrik_perbandingan.csv)
