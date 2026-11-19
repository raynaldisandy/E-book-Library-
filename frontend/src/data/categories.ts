export const catalogGroups = {
  Pelayanan: ["Medical", "Political Science", "Social Science"],
  Kepemimpinan: ["Business & Economics", "Education", "Psychology"],
  Teknologi: ["Computers", "Technology & Engineering"],
} as const;

export type CatalogGroup = keyof typeof catalogGroups;
export const categoryGroups: readonly (CatalogGroup | "Semua")[] = ["Semua", "Pelayanan", "Kepemimpinan", "Teknologi"];
