import { categoryGroups, type CatalogGroup } from "../data/categories";

export function GroupFilters({
  value,
  onChange,
  all = "Semua",
}: {
  value: CatalogGroup | "Semua";
  onChange: (g: CatalogGroup | "Semua") => void;
  all?: string;
}) {
  return (
    <div className="group-filters" aria-label="Kategori buku">
      {categoryGroups.map((c) => (
        <button
          type="button"
          aria-pressed={c === value}
          className={c === value ? "active" : ""}
          onClick={() => onChange(c)}
          key={c}
        >
          {c === "Semua" ? all : c}
        </button>
      ))}
    </div>
  );
}
