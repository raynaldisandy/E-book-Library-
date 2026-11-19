import { BookOpen, Library, Network, RefreshCw, Sparkles } from "lucide-react";
import type { Health } from "../api";
import { Heading } from "../components/PageHeading";
import { similarityDescription } from "../data/similarity";

export function SettingsPage({ health, healthError, checkedAt, checking, refresh }: {
  health: Health | null;
  healthError: boolean;
  checkedAt: number | null;
  checking: boolean;
  refresh: () => void;
}) {
  const online = health?.status === "healthy" && !healthError;
  return (
    <>
      <Heading eyebrow="INFORMASI LAYANAN" title="Pengaturan">
        Lihat status katalog dan layanan rekomendasi perpustakaan.
      </Heading>
      <div className="settings-layout">
        <section className="paper status-panel">
          <div className="panel-top">
            <span className="step-label">STATUS SISTEM</span>
          </div>
          <h2>Transparan sejak awal.</h2>
          <p className="muted">
            Status diperbarui secara berkala dari layanan perpustakaan.
          </p>
          <dl className="status-ledger">
            <div>
              <dt>
                <Sparkles size={18} />
                Layanan rekomendasi
              </dt>
              <dd>
                <span className={online ? "service-status online" : "service-status"}>
                  {checking ? "Memeriksa" : healthError ? "Tidak terhubung" : online ? "Terhubung" : "Belum siap"}
                </span>
                {online && health?.is_fallback && <p>Pencocokan berbasis kata sedang aktif</p>}
              </dd>
            </div>
            <div>
              <dt>
                <Library size={18} />
                Kompetensi acuan
              </dt>
              <dd>
                <strong>{health?.competencies_count.toLocaleString("id-ID") ?? "–"} kompetensi</strong>
                <p>Tersedia untuk pencarian dan pemetaan</p>
              </dd>
            </div>
            <div>
              <dt>
                <BookOpen size={18} />
                Jumlah buku
              </dt>
              <dd>
                <strong>{health?.catalog_books_count.toLocaleString("id-ID") ?? "–"} buku</strong>
                <p>Total katalog</p>
              </dd>
            </div>
            <div>
              <dt>Waktu pemeriksaan terakhir</dt>
              <dd>
                <strong>
                  {checkedAt
                    ? new Date(checkedAt).toLocaleString("id-ID", {
                        dateStyle: "medium",
                        timeStyle: "medium",
                      })
                    : "Belum diperiksa pada sesi ini"}
                </strong>
              </dd>
            </div>
          </dl>
          <button
            className="btn btn-outline"
            disabled={checking}
            onClick={refresh}
          >
            <RefreshCw className={checking ? "spinning" : ""} size={16} />
            {checking ? "Memeriksa…" : "Periksa status layanan"}
          </button>
        </section>
        <aside className="model-note">
          <div className="eyebrow">TENTANG PENCOCOKAN</div>
          <Network size={39} strokeWidth={1.2} />
          <h2>
            Memahami makna,
            <br />
            lewat isi buku.
          </h2>
          <p>
            Rekomendasi mengurutkan buku berdasarkan kemiripan antara isi buku dan deskripsi kompetensi.
          </p>
          <div className="fine-rule" />
          <p>
            {similarityDescription}
          </p>
        </aside>
      </div>
    </>
  );
}
