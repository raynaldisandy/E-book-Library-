"""Protokol Evaluasi Independen & Metrik Multi-Label untuk 67 Kompetensi BPJS.

Mengevaluasi seluruh algoritma/pendekatan NLP berdasarkan ground-truth independen
dengan graded relevance (0 = Tidak Relevan, 1 = Marjinal, 2 = Relevan, 3 = Sangat Relevan/Utama)
pada 30 buku uji representatif (bilingual Indonesia-Inggris, multi-domain).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd


def get_ground_truth_benchmark(csv_path: str | Path | None = None) -> list[dict]:
    """Mengembalikan 30 buku uji (20 katalog riil + 10 buku baru) dari dataset ground truth.
    
    Memuat data anotasi terstruktur dari data/benchmark_ground_truth_30_books.csv
    yang dilengkapi derajat relevansi (0-3), rubrik penilaian, domain, dan alasan kurasi.
    """
    if csv_path is None:
        default_p = Path(__file__).resolve().parents[1] / "data" / "benchmark_ground_truth_30_books.csv"
        csv_path = default_p if default_p.exists() else Path("data/benchmark_ground_truth_30_books.csv")
    else:
        csv_path = Path(csv_path)

    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"competency_id": str, "relevance_grade": int})
        items = []
        for tid, group in df.groupby("test_id", sort=False):
            first = group.iloc[0]
            grades = dict(zip(group["competency_id"], group["relevance_grade"]))
            rationales = dict(zip(group["competency_id"], group["annotation_rationale"]))
            items.append({
                "test_id": str(tid),
                "judul": str(first["judul"]),
                "abstrak": str(first["abstrak"]),
                "bahasa": str(first.get("bahasa", "id")),
                "domain": str(first.get("domain", "General")),
                "is_catalog": bool(first["is_catalog"]),
                "annotation_source": str(first.get("annotation_source", "")),
                "relevant_grades": grades,
                "rationales": rationales
            })
        return items

    return _DEFAULT_HARDCODED_BENCHMARK


_DEFAULT_HARDCODED_BENCHMARK = [
        # --- 20 BUKU KATALOG RIIL ---
        {
            "test_id": "BK00001",
            "judul": "Business Growth Strategy",
            "abstrak": "A guide to help entrepreneurs learn how to grow their business and use the best strategies. Business strategy is the planned schemes a company pursues to create value for the corporation and its shareholders and gain a modest benefit in the market.",
            "bahasa": "en",
            "domain": "Strategy & Business",
            "is_catalog": True,
            "relevant_grades": {"C001": 3, "C011": 2, "C019": 2, "C002": 1, "C010": 1}
        },
        {
            "test_id": "BK00002",
            "judul": "Mastering Business Growth Strategies",
            "abstrak": "Gaining the Upper Hand in Competitive Markets. Provides readers with practical insights into effective strategies that propel businesses forward in highly competitive markets through market analysis and customer segmentation.",
            "bahasa": "en",
            "domain": "Strategy & Business",
            "is_catalog": True,
            "relevant_grades": {"C001": 3, "C011": 3, "C002": 2, "C017": 1, "C016": 1}
        },
        {
            "test_id": "BK00003",
            "judul": "AI-Powered Business Growth Strategies",
            "abstrak": "A practical guide for business owners and marketers who want to use artificial intelligence to increase revenue, automate customer acquisition, demand forecasting, and build scalable growth systems.",
            "bahasa": "en",
            "domain": "Business & AI",
            "is_catalog": True,
            "relevant_grades": {"C001": 3, "C019": 3, "C066": 2, "C002": 2, "C011": 1}
        },
        {
            "test_id": "BK00006",
            "judul": "Strategic Internet Business Management - an Assessment of Internet Business Growth Strategies",
            "abstrak": "Investigates internet business growth models, digital commerce adoption, and competitive advantage for small and medium enterprises.",
            "bahasa": "en",
            "domain": "Digital Strategy",
            "is_catalog": True,
            "relevant_grades": {"C001": 3, "C011": 2, "C055": 2, "C019": 1, "C002": 1}
        },
        {
            "test_id": "BK00025",
            "judul": "Handbook of Marketing Strategy",
            "abstrak": "Authoritative volume on marketing strategy, competitor analysis, customer management, resource allocation, branding, advertising, and digital marketing dynamics.",
            "bahasa": "en",
            "domain": "Marketing",
            "is_catalog": True,
            "relevant_grades": {"C002": 3, "C001": 2, "C038": 2, "C011": 1, "C017": 1}
        },
        {
            "test_id": "BK00028",
            "judul": "Mastering Business Strategy",
            "abstrak": "Frameworks and real-world case studies to craft winning corporate strategies, resource allocation, change management, and sustainable market leadership.",
            "bahasa": "en",
            "domain": "Strategy Management",
            "is_catalog": True,
            "relevant_grades": {"C011": 3, "C001": 2, "C051": 2, "C012": 1, "C010": 1}
        },
        {
            "test_id": "BK00042",
            "judul": "Search Engine Marketing",
            "abstrak": "Explains web analytics, key performance indicators (KPIs), search engine optimization (SEO), digital advertising campaigns, and performance marketing.",
            "bahasa": "en",
            "domain": "Marketing & Analytics",
            "is_catalog": True,
            "relevant_grades": {"C002": 3, "C066": 2, "C038": 2, "C001": 1, "C067": 1}
        },
        {
            "test_id": "BK00046",
            "judul": "Customer Care Excellence: Meningkatkan Kinerja Perusahaan Melalui Pelayanan Pelanggan",
            "abstrak": "Buku ini membahas konsep kepuasan pelanggan, strategi pelayanan prima, penanganan komplain nasabah, standar mutu layanan, dan pengukuran kepuasan pelanggan pada unit operasional.",
            "bahasa": "id",
            "domain": "Customer Service",
            "is_catalog": True,
            "relevant_grades": {"C003": 3, "C015": 2, "C038": 2, "C002": 1, "C041": 1}
        },
        {
            "test_id": "BK00047",
            "judul": "Customer Service Excellence",
            "abstrak": "Presents strategies and practical frameworks for achieving customer service excellence, handling consumer complaints, and building customer loyalty.",
            "bahasa": "en",
            "domain": "Customer Service",
            "is_catalog": True,
            "relevant_grades": {"C003": 3, "C015": 2, "C038": 1, "C002": 1, "C004": 1}
        },
        {
            "test_id": "BK00067",
            "judul": "Customer Relationship Management Essentials",
            "abstrak": "Explores CRM strategies, customer lifetime value, database marketing, loyalty programs, and digital customer relationship workflows.",
            "bahasa": "en",
            "domain": "CRM & Service",
            "is_catalog": True,
            "relevant_grades": {"C003": 3, "C002": 3, "C065": 2, "C004": 1, "C017": 1}
        },
        {
            "test_id": "BK00077",
            "judul": "The Procurement Game Plan",
            "abstrak": "Offers winning strategies and tactical execution for purchasing and supply management professionals, covering vendor selection, RFP negotiation, cost reduction, and supplier contracts.",
            "bahasa": "en",
            "domain": "Procurement",
            "is_catalog": True,
            "relevant_grades": {"C053": 3, "C005": 3, "C037": 2, "C054": 1, "C010": 1}
        },
        {
            "test_id": "BK00094",
            "judul": "Handbook of Hospital Administration E-Book",
            "abstrak": "Covers healthcare institution management, hospital bed utilization, clinical provider coordination, medical audit, patient service administration, and healthcare quality assurance.",
            "bahasa": "en",
            "domain": "Hospital & Health",
            "is_catalog": True,
            "relevant_grades": {"C006": 3, "C008": 3, "C015": 2, "C007": 1, "C003": 1}
        },
        {
            "test_id": "BK00096",
            "judul": "Managed Care",
            "abstrak": "Comprehensive overview of managed care organizations, health insurance provider networks, capitation arrangements, utilization review, and medical cost containment strategies.",
            "bahasa": "en",
            "domain": "Managed Care",
            "is_catalog": True,
            "relevant_grades": {"C008": 3, "C006": 3, "C007": 2, "C005": 2, "C021": 1}
        },
        {
            "test_id": "BK00121",
            "judul": "Forensic Accounting and Fraud Examination",
            "abstrak": "Grounded in forensic accounting principles and fraud examination techniques, covering embezzlement, financial statement fraud, evidence collection, and internal control auditing.",
            "bahasa": "en",
            "domain": "Fraud & Audit",
            "is_catalog": True,
            "relevant_grades": {"C009": 3, "C025": 3, "C024": 2, "C023": 1, "C034": 1}
        },
        {
            "test_id": "BK00122",
            "judul": "Principles of Fraud Examination",
            "abstrak": "Focuses on fraud schemes, prevention, deterrence, red flags detection, interview techniques, and legal elements of fraud in commercial and public institutions.",
            "bahasa": "en",
            "domain": "Fraud Examination",
            "is_catalog": True,
            "relevant_grades": {"C009": 3, "C025": 2, "C024": 2, "C029": 1, "C023": 1}
        },
        {
            "test_id": "BK00164",
            "judul": "Emerging Business Opportunities in Africa: Market Entry, Competitive Strategy, and the Promotion of Foreign Direct Investments",
            "abstrak": "Presents basic business modeling for developing market entry strategies, exploiting trade opportunities, and crafting international competitive positioning.",
            "bahasa": "en",
            "domain": "Business Development",
            "is_catalog": True,
            "relevant_grades": {"C001": 3, "C011": 2, "C037": 1, "C019": 1, "C002": 1}
        },
        {
            "test_id": "BK00205",
            "judul": "The People CMM: A Framework for Human Capital Management",
            "abstrak": "Describes best practices for managing and developing human capital, workforce competence planning, skills assessment, training programs, and performance culture.",
            "bahasa": "en",
            "domain": "HR & Competence",
            "is_catalog": True,
            "relevant_grades": {"C013": 3, "C046": 3, "C045": 2, "C050": 2, "C043": 1}
        },
        {
            "test_id": "BK00246",
            "judul": "The Art of Creative Product Strategy",
            "abstrak": "Step-by-step guide for product managers to shape, launch, and scale digital products, product lifecycle roadmaps, opportunity spaces, and user-centric features.",
            "bahasa": "en",
            "domain": "Product Management",
            "is_catalog": True,
            "relevant_grades": {"C017": 3, "C019": 2, "C001": 2, "C011": 1, "C057": 1}
        },
        {
            "test_id": "BK00400",
            "judul": "The Operational Audit Blueprint - Definitions, Internal Audit Programs and Checklists for Success",
            "abstrak": "Comprehensive guide for operational internal auditing across finance, HR, manufacturing, supply chain, and IT general controls, risk assessments, and compliance checklists.",
            "bahasa": "en",
            "domain": "Audit & Compliance",
            "is_catalog": True,
            "relevant_grades": {"C025": 3, "C024": 2, "C023": 2, "C015": 1, "C009": 1}
        },
        {
            "test_id": "BK00427",
            "judul": "Legal Drafting: Teori dan Praktik Penyusunan Peraturan",
            "abstrak": "Membahas teknik perancangan naskah hukum, perumusan klausul perjanjian kerja sama, harmonisasi peraturan perundang-undangan, dan analisis dampak regulasi.",
            "bahasa": "id",
            "domain": "Legal & Regulation",
            "is_catalog": True,
            "relevant_grades": {"C028": 3, "C029": 3, "C027": 2, "C022": 1, "C024": 1}
        },

        # --- 10 BUKU BARU (NOVEL INPUTS: 5 INDONESIA, 5 INGGRIS) ---
        {
            "test_id": "NEW_001",
            "judul": "Manajemen Risiko Klinis dan Keselamatan Pasien Rumah Sakit",
            "abstrak": "Buku ini membahas identifikasi risiko operasional dan klinis di fasilitas kesehatan, pemetaan potensi bahaya, sistem pelaporan insiden keselamatan pasien, audit mitigasi kerugian, dan standar akreditasi rumah sakit.",
            "bahasa": "id",
            "domain": "Risk & Quality",
            "is_catalog": False,
            "relevant_grades": {"C023": 3, "C006": 2, "C015": 2, "C024": 1, "C008": 1}
        },
        {
            "test_id": "NEW_002",
            "judul": "Investigasi Kecurangan Klaim dan Audit Forensik Asuransi Kesehatan",
            "abstrak": "Teknik mendeteksi fraud klaim fiktif, verifikasi tagihan rumah sakit, audit medis tarif INA-CBG, investigasi potensi penyalahgunaan kartu kepesertaan, serta kepatuhan fasilitas kesehatan mitra BPJS Kesehatan.",
            "bahasa": "id",
            "domain": "Fraud & Claims",
            "is_catalog": False,
            "relevant_grades": {"C009": 3, "C007": 3, "C025": 2, "C024": 1, "C023": 1}
        },
        {
            "test_id": "NEW_003",
            "judul": "Penerapan Big Data Analytics dan Machine Learning pada Layanan Publik",
            "abstrak": "Panduan praktis pemanfaatan analitika data besar, visualisasi dashboard eksekutif, predictive modeling, pemodelan perilaku peserta, dan pengolahan pipeline data terstruktur dan tidak terstruktur.",
            "bahasa": "id",
            "domain": "Data & Analytics",
            "is_catalog": False,
            "relevant_grades": {"C066": 3, "C065": 2, "C067": 2, "C057": 1, "C016": 1}
        },
        {
            "test_id": "NEW_004",
            "judul": "Tata Kelola Perusahaan (GCG) dan Kepatuhan Regulasi Lembaga Publik",
            "abstrak": "Penerapan prinsip keterbukaan, akuntabilitas dewan pengawas, manajemen kepatuhan hukum internal (compliance), sistem pelaporan pelanggaran whistleblowing, dan audit etika organisasi jaminan sosial.",
            "bahasa": "id",
            "domain": "Governance & Compliance",
            "is_catalog": False,
            "relevant_grades": {"C022": 3, "C024": 3, "C029": 2, "C027": 1, "C025": 1}
        },
        {
            "test_id": "NEW_005",
            "judul": "Manajemen Anggaran Korporasi dan Pengendalian Kas Operasional",
            "abstrak": "Penyusunan anggaran tahunan perusahaan, efisiensi belanja modal dan operasional, peramalan arus kas, tata kelola kas dan perbendaharaan, serta akuntansi manajemen keuangan.",
            "bahasa": "id",
            "domain": "Finance & Budgeting",
            "is_catalog": False,
            "relevant_grades": {"C010": 3, "C033": 3, "C035": 2, "C034": 2, "C021": 1}
        },
        {
            "test_id": "NEW_006",
            "judul": "Applied Health Actuary and Solvency Risk Modeling",
            "abstrak": "Actuarial techniques for calculating health insurance premiums, morbidity tables, solvency margin risk requirements, reserve adequacy, and long-term financial sustainability for universal health coverage.",
            "bahasa": "en",
            "domain": "Actuary & Finance",
            "is_catalog": False,
            "relevant_grades": {"C020": 3, "C030": 2, "C021": 2, "C023": 1, "C010": 1}
        },
        {
            "test_id": "NEW_007",
            "judul": "Cybersecurity Defense, Network Hardening, and Threat Intelligence",
            "abstrak": "Covers enterprise network defense, security operations center (SOC) workflows, intrusion prevention, Zero Trust architecture, vulnerability scanning, and cybersecurity compliance.",
            "bahasa": "en",
            "domain": "IT Security & Network",
            "is_catalog": False,
            "relevant_grades": {"C061": 3, "C060": 3, "C059": 2, "C057": 1, "C024": 1}
        },
        {
            "test_id": "NEW_008",
            "judul": "Relational Database Design, Optimization, and Administration",
            "abstrak": "Hands-on guide to database architecture, SQL optimization, schema design, high availability clusters, backup and disaster recovery, and enterprise data storage administration.",
            "bahasa": "en",
            "domain": "Database & Data",
            "is_catalog": False,
            "relevant_grades": {"C056": 3, "C062": 3, "C065": 2, "C057": 1, "C059": 1}
        },
        {
            "test_id": "NEW_009",
            "judul": "Corporate Learning Strategy, Instructional Design, and Competence Development",
            "abstrak": "Methods for designing corporate training curriculums, employee competency frameworks, instructional design for digital academies, and measuring learning ROI in large enterprises.",
            "bahasa": "en",
            "domain": "Learning & Competence",
            "is_catalog": False,
            "relevant_grades": {"C050": 3, "C049": 3, "C013": 2, "C046": 2, "C018": 1}
        },
        {
            "test_id": "NEW_010",
            "judul": "Healthcare Provider Strategic Purchasing and Contract Negotiation",
            "abstrak": "Guidelines for strategic purchasing of health services, negotiating fee schedules with private and public hospitals, monitoring provider quality indicators, and managing supplier agreements.",
            "bahasa": "en",
            "domain": "Purchasing & Providers",
            "is_catalog": False,
            "relevant_grades": {"C005": 3, "C006": 3, "C037": 2, "C053": 1, "C015": 1}
        }
    ]


def calculate_ndcg_at_k(ranked_competency_ids: list[str], ground_truth: dict[str, int], k: int = 5) -> float:
    """Menghitung NDCG@K untuk graded relevance 0..3."""
    top_k_ranks = ranked_competency_ids[:k]
    gains = [float(ground_truth.get(cid, 0)) for cid in top_k_ranks]
    dcg = sum((2**g - 1.0) / np.log2(i + 2) for i, g in enumerate(gains))

    all_positive = sorted([float(v) for v in ground_truth.values() if v > 0], reverse=True)[:k]
    if not all_positive:
        return 0.0
    idcg = sum((2**g - 1.0) / np.log2(i + 2) for i, g in enumerate(all_positive))
    return float(dcg / idcg) if idcg > 0 else 0.0


def calculate_precision_at_k(ranked_competency_ids: list[str], ground_truth: dict[str, int], k: int = 5, min_grade: int = 2) -> float:
    """Menghitung Precision@K (proporsi rekomendasi dengan grade >= min_grade)."""
    top_k_ranks = ranked_competency_ids[:k]
    if not top_k_ranks:
        return 0.0
    hits = sum(1 for cid in top_k_ranks if ground_truth.get(cid, 0) >= min_grade)
    return float(hits / len(top_k_ranks))


def calculate_recall_at_k(ranked_competency_ids: list[str], ground_truth: dict[str, int], k: int = 5, min_grade: int = 2) -> float:
    """Menghitung Recall@K (proporsi target relevan yang berhasil ditemukan di Top-K)."""
    total_relevant = sum(1 for g in ground_truth.values() if g >= min_grade)
    if total_relevant == 0:
        return 0.0
    top_k_ranks = ranked_competency_ids[:k]
    hits = sum(1 for cid in top_k_ranks if ground_truth.get(cid, 0) >= min_grade)
    return float(hits / total_relevant)


def calculate_map_at_k(ranked_competency_ids: list[str], ground_truth: dict[str, int], k: int = 5, min_grade: int = 2) -> float:
    """Menghitung Average Precision (AP@K) untuk satu query."""
    total_relevant = sum(1 for g in ground_truth.values() if g >= min_grade)
    if total_relevant == 0:
        return 0.0
    top_k_ranks = ranked_competency_ids[:k]
    hits = 0
    precisions = []
    for i, cid in enumerate(top_k_ranks, start=1):
        if ground_truth.get(cid, 0) >= min_grade:
            hits += 1
            precisions.append(hits / i)
    return float(sum(precisions) / min(total_relevant, k)) if precisions else 0.0


def evaluate_ranked_predictions(
    model_name: str,
    predictions: list[list[str]],
    benchmark: list[dict],
    latencies: list[float]
) -> dict[str, float]:
    """Mengevaluasi daftar prediksi terurut untuk seluruh 30 item benchmark."""
    ndcg3_list, ndcg5_list = [], []
    p1_list, p3_list, p5_list = [], [], []
    recall5_list, hitrate3_list, hitrate5_list = [], [], []
    ap5_list = []
    unique_recommended = set()

    for item, pred_ids in zip(benchmark, predictions):
        gt = item["relevant_grades"]
        for cid in pred_ids[:5]:
            unique_recommended.add(cid)

        ndcg3_list.append(calculate_ndcg_at_k(pred_ids, gt, k=3))
        ndcg5_list.append(calculate_ndcg_at_k(pred_ids, gt, k=5))
        p1_list.append(calculate_precision_at_k(pred_ids, gt, k=1, min_grade=2))
        p3_list.append(calculate_precision_at_k(pred_ids, gt, k=3, min_grade=2))
        p5_list.append(calculate_precision_at_k(pred_ids, gt, k=5, min_grade=2))
        recall5_list.append(calculate_recall_at_k(pred_ids, gt, k=5, min_grade=2))
        hitrate3_list.append(1.0 if any(gt.get(c, 0) >= 2 for c in pred_ids[:3]) else 0.0)
        hitrate5_list.append(1.0 if any(gt.get(c, 0) >= 2 for c in pred_ids[:5]) else 0.0)
        ap5_list.append(calculate_map_at_k(pred_ids, gt, k=5, min_grade=2))

    coverage_pct = (len(unique_recommended) / 67.0) * 100.0

    return {
        "model": model_name,
        "ndcg@3": round(float(np.mean(ndcg3_list)), 4),
        "ndcg@5": round(float(np.mean(ndcg5_list)), 4),
        "precision@1": round(float(np.mean(p1_list)), 4),
        "precision@3": round(float(np.mean(p3_list)), 4),
        "precision@5": round(float(np.mean(p5_list)), 4),
        "recall@5": round(float(np.mean(recall5_list)), 4),
        "hitrate@3": round(float(np.mean(hitrate3_list)), 4),
        "hitrate@5": round(float(np.mean(hitrate5_list)), 4),
        "map@5": round(float(np.mean(ap5_list)), 4),
        "catalog_coverage_pct": round(float(coverage_pct), 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 2)
    }
