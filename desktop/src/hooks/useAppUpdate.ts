import { useCallback, useEffect, useState } from 'react';
import { desktopBridge, type UpdateState } from '../services/desktop';

/** Live update state from the desktop shell; `null` when running in a browser. */
export function useAppUpdate() {
  const [state, setState] = useState<UpdateState | null>(null);

  useEffect(() => {
    const bridge = desktopBridge();
    if (!bridge) return undefined;
    let active = true;
    bridge.update.getState().then(next => { if (active) setState(next); }).catch(() => undefined);
    const stop = bridge.update.onState(next => setState(next));
    return () => { active = false; stop(); };
  }, []);

  const run = useCallback((action: 'check' | 'download' | 'install') => {
    desktopBridge()?.update[action]().then(setState).catch(() => undefined);
  }, []);

  return {
    state,
    check: useCallback(() => run('check'), [run]),
    download: useCallback(() => run('download'), [run]),
    install: useCallback(() => run('install'), [run]),
  };
}
