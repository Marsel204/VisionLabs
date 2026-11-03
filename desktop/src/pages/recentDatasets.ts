/** Dataset folders opened in this app, newest first, kept in local storage. */
const KEY = 'anotara.recentDatasets';
const LIMIT = 8;

export function recentDatasets(): string[] {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) || '[]');
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
  } catch {
    return [];
  }
}

export function rememberDataset(directory: string): void {
  if (!directory) return;
  try {
    const next = [directory, ...recentDatasets().filter(item => item !== directory)].slice(0, LIMIT);
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* Storage can be unavailable (private mode); the list is only a convenience. */
  }
}

export function forgetDataset(directory: string): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(recentDatasets().filter(item => item !== directory)));
  } catch {
    /* See rememberDataset. */
  }
}
