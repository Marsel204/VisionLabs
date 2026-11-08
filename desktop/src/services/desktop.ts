/** Bridge to the Electron shell. Absent when the UI runs in a plain browser (`npm run dev`). */

export type UpdateStatus =
  | 'disabled'
  | 'idle'
  | 'checking'
  | 'up-to-date'
  | 'available'
  | 'downloading'
  | 'downloaded'
  | 'error';

export interface UpdateState {
  status: UpdateStatus;
  currentVersion: string;
  availableVersion: string | null;
  releaseNotes: string | null;
  percent: number;
  checkedAt: number | null;
  error: string | null;
  disabledReason: string | null;
}

export interface AppInfo {
  version: string;
  platform: string;
  arch: string;
  packaged: boolean;
  backendUrl: string;
  logsPath: string;
}

export interface AnotaraBridge {
  isDesktop: true;
  getAppInfo(): Promise<AppInfo>;
  openLogs(): Promise<string>;
  retryStart(): Promise<void>;
  update: {
    getState(): Promise<UpdateState>;
    check(): Promise<UpdateState>;
    download(): Promise<UpdateState>;
    install(): Promise<UpdateState>;
    onState(listener: (state: UpdateState) => void): () => void;
  };
}

declare global {
  interface Window {
    anotara?: AnotaraBridge;
  }
}

export function desktopBridge(): AnotaraBridge | undefined {
  return typeof window === 'undefined' ? undefined : window.anotara;
}
