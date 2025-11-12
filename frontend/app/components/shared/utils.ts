/** Shared utility functions */

export const numberFromForm = (value: FormDataEntryValue | null) => {
  if (value === null || value === undefined || value === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

export const rangeFromForm = (
  form: FormData,
  minKey: string,
  maxKey: string
): [number, number] | undefined => {
  const min = numberFromForm(form.get(minKey));
  const max = numberFromForm(form.get(maxKey));
  if (min === undefined || max === undefined) return undefined;
  return [min, max];
};

export const formatDateStamp = () => {
  const date = new Date();
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}-${String(
    date.getHours()
  ).padStart(2, "0")}${String(date.getMinutes()).padStart(2, "0")}`;
};

export const supportsFilePicker = () =>
  typeof window !== "undefined" && typeof (window as Window & { showSaveFilePicker?: () => unknown }).showSaveFilePicker === "function";

