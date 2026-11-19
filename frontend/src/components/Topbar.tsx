import { ArrowRight, ChevronRight, Menu, Search } from "lucide-react";
import { useState, type Ref } from "react";
import { nav } from "../data/navigation";
import type { Page } from "../types";

export function Topbar({ page, menuButton, onMenu, onSearch }: {
  page: Page;
  menuButton: Ref<HTMLButtonElement>;
  onMenu: () => void;
  onSearch: (query: string) => void;
}) {
  const [query, setQuery] = useState("");
  return (
    <header className="topbar">
      <button ref={menuButton} className="mobile-menu icon-button" onClick={onMenu} aria-label="Buka menu navigasi">
        <Menu size={22} />
      </button>
      <div className="breadcrumb">
        <span>Perpustakaan</span>
        <ChevronRight size={13} />
        <strong>{nav.find((item) => item.id === page)?.label}</strong>
      </div>
      <form className="global-search" role="search" onSubmit={(event) => {
        event.preventDefault();
        onSearch(query.trim());
      }}>
        <Search size={16} />
        <input
          aria-label="Cari buku, penulis, atau kode kompetensi"
          placeholder="Cari buku atau penulis…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button aria-label="Cari di katalog"><ArrowRight size={16} /></button>
      </form>
      <span className="topbar-edition">PUSTAKA DIGITAL</span>
    </header>
  );
}
