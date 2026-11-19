import { ArrowDownUp, ArrowRight, ChevronLeft, ChevronRight, CircleHelp, RefreshCw, Search, X } from "lucide-react";
import { useEffect, useState } from "react";
import { catalogPageSize, fromCatalog, searchCatalog, type CatalogSort } from "../api";
import { BookCard } from "../components/BookCard";
import { GroupFilters } from "../components/GroupFilters";
import { Empty, Skeleton } from "../components/LibraryStates";
import { Heading } from "../components/PageHeading";
import type { CatalogGroup } from "../data/categories";
import type { OpenBook } from "../types";

type Result = Awaited<ReturnType<typeof searchCatalog>>;
const size = catalogPageSize;

export function Catalog({ open, query, setQuery, total }: {
  open: OpenBook;
  query: string;
  setQuery: (value: string) => void;
  total?: number;
}) {
  const [group, setGroup] = useState<CatalogGroup | "Semua">("Semua");
  const [sort, setSort] = useState<CatalogSort>("newest");
  const [page, setPage] = useState(1);
  const [attempt, setAttempt] = useState(0);
  const key = JSON.stringify([query.trim(), group, sort, page, attempt]);
  const [outcome, setOutcome] = useState<{ key: string; result?: Result; error?: string } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      searchCatalog(query.trim(), group, (page - 1) * size, sort, controller.signal)
        .then((result) => { if (!controller.signal.aborted) setOutcome({ key, result }); })
        .catch((error: Error) => { if (!controller.signal.aborted) setOutcome({ key, error: error.message }); });
    }, query ? 180 : 0);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query, group, sort, page, key]);

  function updateQuery(value: string) {
    setPage(1);
    setQuery(value);
  }

  const result = outcome?.key === key ? outcome.result : undefined;
  const issue = outcome?.key === key ? outcome.error : undefined;
  const pages = Math.max(1, Math.ceil((result?.total_matched ?? 0) / size));
  const currentPage = Math.min(page, pages);
  const firstButton = Math.max(1, Math.min(currentPage - 1, pages - 2));
  const visiblePages = Array.from({ length: Math.min(3, pages) }, (_, index) => firstButton + index);
  return (
    <>
      <div className="catalog-heading">
        <Heading eyebrow="JELAJAHI RAK PENGETAHUAN" title="Katalog Buku">
          Koleksi bacaan untuk membuka wawasan dan memperkaya kompetensi.
        </Heading>
        <div className="catalog-total">
          <strong>
            {total?.toLocaleString("id-ID") ?? "–"}<span> buku</span>
          </strong>
          <p>Total katalog</p>
        </div>
      </div>
      <section className="catalog-tools">
        <div className="catalog-search-row">
          <div className="input-with-icon">
            <Search size={18} />
            <input
              aria-label="Cari judul atau penulis di katalog"
              placeholder="Cari judul atau penulis…"
              value={query}
              onChange={(e) => updateQuery(e.target.value)}
            />
            {query && (
              <button onClick={() => updateQuery("")} aria-label="Hapus pencarian katalog">
                <X size={16} />
              </button>
            )}
          </div>
          <div className="sort-control">
            <ArrowDownUp size={15} />
            <select
              value={sort}
              onChange={(e) => { setSort(e.target.value as CatalogSort); setPage(1); }}
              aria-label="Urutkan buku"
            >
              <option value="newest">Terbaru</option>
              <option value="oldest">Tahun terlama</option>
              <option value="title">Judul A–Z</option>
            </select>
          </div>
        </div>
        <GroupFilters value={group} onChange={(value) => { setGroup(value); setPage(1); }} />
      </section>
      <div className="catalog-count">
        <p>
          <strong>{result?.total_matched.toLocaleString("id-ID") ?? "–"}</strong> buku{query && <> untuk “{query}”</>}
        </p>
      </div>
      {issue ? (
        <Empty title="Katalog belum dapat dimuat">
          <span role="alert">{issue}</span>
          <button className="text-link" onClick={() => setAttempt((value) => value + 1)}>
            <RefreshCw size={15} /> Coba lagi
          </button>
        </Empty>
      ) : !result ? (
        <Skeleton count={4} />
      ) : result.books.length ? (
        <div className="book-grid catalog-grid">
          {result.books.map(fromCatalog).map((book) => (
            <BookCard key={book.id} book={book} open={open} />
          ))}
        </div>
      ) : (
        <Empty title="Buku belum ditemukan">
          Coba kata kunci yang lebih singkat atau jelajahi kategori lain.
          <button
            className="text-link"
            onClick={() => {
              updateQuery("");
              setGroup("Semua");
            }}
          >
            Atur ulang pencarian <ArrowRight size={15} />
          </button>
        </Empty>
      )}
      {result && <div className="pagination">
        <p>
          {result.total_matched
            ? `Menampilkan ${(currentPage - 1) * size + 1}–${Math.min(currentPage * size, result.total_matched)} dari ${result.total_matched.toLocaleString("id-ID")} buku`
            : "Tidak ada buku ditampilkan"}
        </p>
        <div>
          <button
            aria-label="Halaman sebelumnya"
            disabled={currentPage === 1}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft size={16} />
          </button>
          {visiblePages.map((i) => (
            <button
              key={i}
              className={currentPage === i ? "active" : ""}
              aria-label={`Halaman ${i}`}
              aria-current={currentPage === i ? "page" : undefined}
              onClick={() => setPage(i)}
            >
              {i}
            </button>
          ))}
          <button
            aria-label="Halaman berikutnya"
            disabled={currentPage === pages}
            onClick={() => setPage((p) => p + 1)}
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>}
      <div className="library-note">
        <CircleHelp size={15} />
        <p>
          Sampul buku diilustrasikan untuk membantu penelusuran katalog.
        </p>
      </div>
    </>
  );
}
