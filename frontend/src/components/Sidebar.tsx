import { PanelLeftClose, PanelLeftOpen, Settings } from "lucide-react";
import { nav } from "../data/navigation";
import type { Page } from "../types";
import { Brand } from "./Brand";

export function Sidebar({ page, go, compact = false, toggle }: {
  page: Page;
  go: (page: Page) => void;
  compact?: boolean;
  toggle?: () => void;
}) {
  return (
    <>
      <Brand compact={compact} />
      <div className="nav-divider" />
      {!compact && <span className="nav-caption">PERPUSTAKAAN</span>}
      <nav aria-label="Navigasi utama">
        {nav
          .filter((n) => n.id !== "settings")
          .map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              title={compact ? label : undefined}
              aria-label={compact ? label : undefined}
              aria-current={page === id ? "page" : undefined}
              className={`nav-item ${page === id ? "active" : ""}`}
              onClick={() => go(id)}
            >
              <Icon size={19} strokeWidth={1.65} />
              {!compact && <span>{label}</span>}
              {!compact && page === id && <span className="nav-active-dot" />}
            </button>
          ))}
      </nav>
      <div className="sidebar-bottom">
        {!compact && (
          <div className="sidebar-quote">
            <span>“</span>
            <p>
              Satu buku,
              <br />
              perspektif baru.
            </p>
            <div className="quote-line" />
            <small>RUANG UNTUK TERUS BELAJAR</small>
          </div>
        )}
        <button
          className={`nav-item ${page === "settings" ? "active" : ""}`}
          onClick={() => go("settings")}
          title={compact ? "Pengaturan" : undefined}
          aria-label={compact ? "Pengaturan" : undefined}
          aria-current={page === "settings" ? "page" : undefined}
        >
          <Settings size={19} strokeWidth={1.65} />
          {!compact && <span>Pengaturan</span>}
        </button>
        {toggle && <button
          className="collapse-button"
          onClick={toggle}
          aria-label={compact ? "Perluas menu" : "Ringkas menu"}
        >
          {compact ? (
            <PanelLeftOpen size={18} />
          ) : (
            <>
              <PanelLeftClose size={18} />
              <span>Ringkas menu</span>
            </>
          )}
        </button>}
      </div>
    </>
  );
}
