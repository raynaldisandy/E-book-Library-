"""REST API Produksi: BPJS Kesehatan AI e-Library Recommender Service.

Layanan backend siap deploy (FastAPI) untuk integrasi langsung ke portal
AI e-Library dan Human Capital Management (HCM) BPJS Kesehatan.

Fitur Utama:
1. Pemetaan Buku Baru ke Multikompetensi (Katalogisasi Otomatis e-Library)
2. Rekomendasi Buku Katalog untuk Menutup Competency Gap Karyawan
3. Penjelajahan Kamus 67 Kompetensi Kerja BPJS Kesehatan
4. Pencarian Koleksi 973 Buku Katalog e-Library
5. Dokumentasi Interaktif Swagger UI (/docs) & ReDoc (/redoc)
6. Dasbor Pengujian Web Interaktif di Root (/)
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Pastikan workspace root ada di path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import BookRecommender

# Inisialisasi Model Produksi
MODEL_DIR = ROOT / "models" / "best_model"
FALLBACK_MODEL_DIR = ROOT / "models" / "candidates" / "tfidf"
DASHBOARD_DIR = ROOT / "frontend" / "dist"

try:
    # Muat model terbaik (BGE-M3 dengan transparansi fallback jika resource terbatas)
    recommender = BookRecommender.load(MODEL_DIR, allow_fallback=True)
    print(f"[*] AI e-Library Engine Loaded: {recommender.model_name} (Aktual: {recommender.actual_model_name})")
except Exception as e:
    print(f"[!] Warning: Gagal memuat best_model ({e}), beralih ke TF-IDF candidate...")
    recommender = BookRecommender.load(FALLBACK_MODEL_DIR)

app = FastAPI(
    title="BPJS Kesehatan AI e-Library Recommendation API",
    description=(
        "API Resmi Layanan Rekomendasi Buku & Pemetaan Kompetensi Kerja Karyawan BPJS Kesehatan. "
        "Dirancang untuk integrasi sistem Human Capital Management (HCM), Corporate University, "
        "dan portal perpustakaan digital (AI e-Library) BPJS Kesehatan."
    ),
    version="2.0.0",
    contact={
        "name": "Tim NLP & AI Kediklatan BPJS Kesehatan",
        "email": "learning@bpjs-kesehatan.go.id"
    }
)

# Aktifkan CORS agar frontend (React, Vue, Flutter, Next.js) dapat langsung memanggil API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/dashboard-assets",
    StaticFiles(directory=DASHBOARD_DIR / "dashboard-assets", check_dir=False),
    name="dashboard-assets",
)


# =============================================================================
# PYDANTIC SCHEMAS (REQUEST & RESPONSE MODELS)
# =============================================================================

class CompetencyItem(BaseModel):
    rank: int = Field(..., description="Urutan peringkat relevansi (1 = teratas)")
    competency_id: str = Field(..., description="Kode unik kompetensi BPJS (e.g. C023)")
    competency_name: str = Field(..., description="Nama resmi kompetensi kerja")
    score: float = Field(..., description="Skor kecocokan semantik (0.0 - 1.0)")
    confidence: str = Field(..., description="Tingkat keyakinan: SANGAT_TINGGI, TINGGI, SEDANG, INDIKATIF")
    model: str = Field(..., description="Nama model inferensi yang mengeksekusi")
    model_type: str = Field(..., description="Tipe arsitektur model (dense_biencoder / lexical)")
    is_fallback: bool = Field(..., description="Apakah inferensi menggunakan mode fallback")
    latency_breakdown_ms: dict[str, float] = Field(..., description="Rincian waktu inferensi per tahap (ms)")


class BookToCompetenciesRequest(BaseModel):
    title: str = Field(..., min_length=2, example="Healthcare Risk Mitigation Handbook", description="Judul buku (Bahasa Indonesia atau Inggris)")
    abstract: str = Field(..., min_length=5, example="This handbook discusses operational risk identification, clinical safety assessment, and compliance in hospital facilities.", description="Abstrak atau sinopsis buku")
    top_k: int = Field(5, ge=1, le=10, description="Jumlah maksimal rekomendasi kompetensi")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ambang batas skor minimum (opsional, default adaptif)")


class BookToCompetenciesResponse(BaseModel):
    status: str = "success"
    query_title: str
    total_recommendations: int
    execution_time_ms: float
    recommendations: list[CompetencyItem]


class BookItem(BaseModel):
    rank: int
    book_id: str
    judul: str
    author: str
    tahun: str
    kategori: str
    score: float
    skor_model: float
    model: str
    is_fallback: bool
    query_latency_ms: float


class CompetencyToBooksRequest(BaseModel):
    competency_id: str = Field(..., example="C023", description="ID Kompetensi BPJS (e.g. C023 untuk Risk Management)")
    top_k: int = Field(5, ge=1, le=20, description="Jumlah rekomendasi buku yang diinginkan")
    categories: list[str] | None = Field(None, description="Nama kategori katalog yang disertakan sebelum menentukan peringkat")


class CompetencyToBooksResponse(BaseModel):
    status: str = "success"
    competency_id: str
    competency_name: str
    total_books: int
    query_time_ms: float
    books: list[BookItem]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    active_model: str
    model_type: str
    is_fallback: bool
    catalog_books_count: int
    competencies_count: int
    uptime_seconds: float
    embedding_dimension: int | None = None
    catalog_exported_at: str | None = None


START_TIME = time.time()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def home_dashboard():
    """Dasbor visual interaktif Pusat Wawasan Pembelajaran AI e-Library BPJS Kesehatan."""
    index = DASHBOARD_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(
            "<h1>Dasbor belum dibangun</h1><p>Jalankan npm ci dan npm run build di folder frontend.</p>",
            status_code=503,
        )
    return FileResponse(index, media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["Sistem"])
def health_check():
    """Pengecekan status kesehatan layanan backend dan model inferensi aktif."""
    model_type_str = (
        "dense_biencoder" if (recommender.model_name in ("bge_m3", "e5") and not recommender.is_fallback_active)
        else "lexical_tfidf_fallback" if recommender.is_fallback_active
        else "lexical_tfidf" if recommender.model_name == "tfidf"
        else "lexical_bm25"
    )
    return {
        "status": "healthy",
        "service": "BPJS Kesehatan AI e-Library Recommendation API",
        "version": "2.0.0",
        "active_model": recommender.actual_model_name,
        "model_type": model_type_str,
        "is_fallback": recommender.is_fallback_active,
        "catalog_books_count": len(recommender.catalog) if recommender.catalog is not None else 0,
        "competencies_count": len(recommender.profiles),
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "embedding_dimension": recommender.manifest.get("embedding_dimension") if model_type_str == "dense_biencoder" else None,
        "catalog_exported_at": recommender.manifest.get("export_timestamp"),
    }


@app.post("/api/v1/recommend-competencies", response_model=BookToCompetenciesResponse, tags=["Rekomendasi e-Library"])
def map_book_to_competencies(req: BookToCompetenciesRequest):
    """
    Fungsi 1: Pemetaan Buku Baru ke Multikompetensi (Katalogisasi Otomatis AI e-Library).
    
    Menerima judul dan abstrak buku baru (Bahasa Indonesia atau Inggris), lalu
    menghasilkan 1 hingga 5 kompetensi kerja BPJS Kesehatan yang paling relevan
    menggunakan kalibrasi ambang dinamis (Dynamic Score Margin).
    """
    t0 = time.perf_counter()
    try:
        results = recommender.map_book(
            title=req.title,
            abstract=req.abstract,
            top_k=req.top_k,
            min_score=req.min_score
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "success",
        "query_title": req.title,
        "total_recommendations": len(results),
        "execution_time_ms": round(elapsed_ms, 2),
        "recommendations": results
    }


@app.post("/api/v1/recommend-books", response_model=CompetencyToBooksResponse, tags=["Rekomendasi e-Library"])
def recommend_books_for_competency(req: CompetencyToBooksRequest):
    """
    Fungsi 2: Rekomendasi Buku Katalog untuk Menutup Competency Gap Pegawai.
    
    Menerima ID Kompetensi BPJS (misal C023 - Risk Management), lalu mengambil
    daftar buku katalog terbaik yang paling sesuai untuk dipelajari pegawai.
    Beroperasi secara super cepat (~3.8 ms) menggunakan matriks vektor terkomputasi.
    """
    t0 = time.perf_counter()
    try:
        books = recommender.recommend_books_for_competency(
            competency_id=req.competency_id,
            top_k=req.top_k,
            categories=req.categories,
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Cari nama kompetensi
    comp_row = recommender.profiles[recommender.profiles["competency_id"] == req.competency_id]
    comp_name = comp_row.iloc[0]["competency_name"] if len(comp_row) > 0 else req.competency_id

    return {
        "status": "success",
        "competency_id": req.competency_id,
        "competency_name": comp_name,
        "total_books": len(books),
        "query_time_ms": round(elapsed_ms, 2),
        "books": books
    }


@app.get("/api/v1/competencies", tags=["Data Referensi"])
def list_competencies():
    """Mengambil daftar lengkap 67 profil kompetensi kerja BPJS Kesehatan."""
    comps = recommender.profiles[["competency_id", "competency_name"]].to_dict(orient="records")
    return {
        "status": "success",
        "total": len(comps),
        "competencies": comps
    }


@app.get("/api/v1/catalog", tags=["Data Referensi"])
def list_catalog_books(
    q: Optional[str] = Query(None, description="Kata kunci pencarian judul atau penulis"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    category: Optional[list[str]] = Query(None, description="Kategori buku yang dicari"),
    sort: Optional[Literal["newest", "oldest", "title"]] = Query(None, description="Urutan tahun terbit atau judul"),
):
    """Mencari atau menampilkan daftar koleksi katalog buku e-Library internal BPJS."""
    if recommender.catalog is None:
        raise HTTPException(status_code=503, detail="Katalog buku belum dimuat")

    df = recommender.catalog
    if q:
        q_lower = q.lower()
        matching_titles = df["judul"].str.lower().str.contains(q_lower, regex=False, na=False)
        matching_authors = df["author"].str.lower().str.contains(q_lower, regex=False, na=False)
        df = df[matching_titles | matching_authors]
    if category:
        df = df[df["kategori_buku"].isin(category)]
    if sort in ("newest", "oldest"):
        df = df.sort_values("tahun", ascending=sort == "oldest", key=lambda years: pd.to_numeric(years, errors="coerce"), kind="stable")
    elif sort == "title":
        df = df.sort_values("judul", key=lambda titles: titles.str.strip().str.casefold(), kind="stable")

    total_matched = len(df)
    subset = df.iloc[offset:offset+limit]
    cols = [c for c in ["book_id", "judul", "author", "tahun", "kategori_buku"] if c in subset.columns]
    books = subset[cols].to_dict(orient="records")

    return {
        "status": "success",
        "total_matched": total_matched,
        "limit": limit,
        "offset": offset,
        "books": books
    }


class CatalogBookDetail(BaseModel):
    book_id: str
    abstrak: str


@app.get("/api/v1/catalog/{book_id}", response_model=CatalogBookDetail, tags=["Data Referensi"])
def get_catalog_book(book_id: str):
    """Mengambil ringkasan buku dari data katalog."""
    catalog = recommender.catalog
    if catalog is None:
        raise HTTPException(status_code=503, detail="Katalog buku belum dimuat")

    matches = catalog.loc[catalog["book_id"] == book_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Buku tidak ditemukan")

    return {"book_id": book_id, "abstrak": str(matches.iloc[0].get("abstrak", ""))}


# =============================================================================
# RUNNER STANDALONE
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n{'='*70}")
    print(f"🚀 Memulai Server BPJS Kesehatan AI e-Library API di http://localhost:{port}")
    print(f"📚 Dokumentasi Interaktif Swagger UI : http://localhost:{port}/docs")
    print(f"📘 Dokumentasi ReDoc                 : http://localhost:{port}/redoc")
    print(f"{'='*70}\n")
    uvicorn.run("api:app", host=host, port=port, reload=False)
