import { BookOpenText, Network, Search } from "lucide-react";
import type { ReactNode } from "react";

export function Empty({
  icon = "search",
  title,
  children,
  action,
}: {
  icon?: "search" | "book" | "map";
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  const Icon = icon === "book" ? BookOpenText : icon === "map" ? Network : Search;
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon size={30} strokeWidth={1.3} />
      </div>
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Skeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="skeletons" role="status" aria-label="Memuat hasil">
      <span className="sr-only">Sedang memuat hasil…</span>
      {Array.from({ length: count }, (_, i) => (
        <div className="skeleton-row" key={i}>
          <div />
          <section>
            <span />
            <span />
            <span />
          </section>
        </div>
      ))}
    </div>
  );
}
