import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Book, Page } from "./types";
import { BookModal } from "./components/BookModal";
import { Topbar } from "./components/Topbar";
import { Sidebar } from "./components/Sidebar";
import { Dashboard } from "./pages/Dashboard";
import { Recommendations } from "./pages/Recommendations";
import { Mapping } from "./pages/Mapping";
import { Catalog } from "./pages/Catalog";
import { SettingsPage } from "./pages/SettingsPage";
import { findCompetency, getCompetencies, getHealth, type Competency, type Health } from "./api";

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const mobileDialog = useRef<HTMLDialogElement>(null);
  const menuButton = useRef<HTMLButtonElement>(null);
  const [selected, setSelected] = useState<Book | null>(null);
  const [catalogQuery, setCatalogQuery] = useState("");
  const [catalogVersion, setCatalogVersion] = useState(0);
  const [recommendationSearch, setRecommendationSearch] = useState<{ competency: Competency | null; version: number }>({ competency: null, version: 0 });
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [healthCheckedAt, setHealthCheckedAt] = useState<number | null>(null);
  const [healthRefresh, setHealthRefresh] = useState(0);
  const [healthChecking, setHealthChecking] = useState(true);
  const [competencies, setCompetencies] = useState<Competency[]>([]);
  const [competenciesError, setCompetenciesError] = useState(false);
  const [competenciesAttempt, setCompetenciesAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function refresh() {
      setHealthChecking(true);
      try {
        const status = await getHealth(controller.signal);
        if (controller.signal.aborted) return;
        setHealth(status);
        setHealthError(false);
      } catch {
        if (controller.signal.aborted) return;
        setHealthError(true);
      }
      setHealthChecking(false);
      setHealthCheckedAt(Date.now());
      timer = window.setTimeout(refresh, 30_000);
    }
    void refresh();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [healthRefresh]);

  useEffect(() => {
    const controller = new AbortController();
    getCompetencies(controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setCompetencies(data.competencies);
        setCompetenciesError(false);
      })
      .catch(() => { if (!controller.signal.aborted) setCompetenciesError(true); });
    return () => controller.abort();
  }, [competenciesAttempt]);

  useEffect(() => {
    try {
      localStorage.removeItem("bpjs-elibrary.activity.v1");
    } catch {
      // Storage access may be disabled in the browser.
    }
  }, []);

  function navigate(next: Page) {
    mobileDialog.current?.close();
    if (next !== "recommendations") {
      setRecommendationSearch((current) => current.competency ? { ...current, competency: null } : current);
    }
    setPage(next);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function search(query: string) {
    const competency = findCompetency(competencies, query);
    if (competency) {
      setRecommendationSearch(({ version }) => ({ competency, version: version + 1 }));
      navigate("recommendations");
      return;
    }
    setCatalogQuery(query);
    setCatalogVersion((version) => version + 1);
    navigate("catalog");
  }

  function refreshHealth() {
    setHealthRefresh((value) => value + 1);
  }

  return (
    <div className={`app ${collapsed ? "is-collapsed" : ""}`}>
      <aside className="sidebar">
        <Sidebar page={page} go={navigate} compact={collapsed} toggle={() => setCollapsed(!collapsed)} />
      </aside>
      <dialog
        ref={mobileDialog}
        className="mobile-dialog"
        aria-label="Menu navigasi"
        onClose={() => menuButton.current?.focus()}
        onClick={(event) => { if (event.target === mobileDialog.current) mobileDialog.current.close(); }}
      >
        <div className="mobile-sidebar">
          <button className="mobile-close" onClick={() => mobileDialog.current?.close()} aria-label="Tutup menu">
            <X size={21} />
          </button>
          <Sidebar page={page} go={navigate} />
        </div>
      </dialog>
      <div className="main-shell">
        <Topbar page={page} menuButton={menuButton} onMenu={() => mobileDialog.current?.showModal()} onSearch={search} />
        <main id="main-content">
          <div className="page-content" key={page}>
            {page === "dashboard" && <Dashboard go={navigate} open={setSelected} health={health} />}
            {page === "recommendations" && (
              <Recommendations
                key={recommendationSearch.version}
                open={setSelected}
                competencies={competencies}
                initialCompetency={recommendationSearch.competency}
                competenciesError={competenciesError}
                retryCompetencies={() => setCompetenciesAttempt((value) => value + 1)}
              />
            )}
            {page === "mapping" && <Mapping onMapped={refreshHealth} />}
            {page === "catalog" && (
              <Catalog key={catalogVersion} open={setSelected} query={catalogQuery} setQuery={setCatalogQuery} total={health?.catalog_books_count} />
            )}
            {page === "settings" && (
              <SettingsPage health={health} healthError={healthError} checkedAt={healthCheckedAt} checking={healthChecking} refresh={refreshHealth} />
            )}
          </div>
          <footer className="app-footer">
            <span>e-Library <span className="footer-dot">·</span> BPJS Kesehatan</span>
            <span>Pengetahuan membuka kemungkinan.</span>
          </footer>
        </main>
      </div>
      {selected && <BookModal key={selected.id} book={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
