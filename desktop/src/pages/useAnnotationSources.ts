import { useEffect, useState } from 'react';
import { fetchAnnotationSources, type AnnotationSource } from '../services/api';

export interface AnnotationSourcesState {
  /** `null` while the folder is being searched. */
  sources: AnnotationSource[] | null;
  error: string | null;
}

/** COCO annotation files found next to the active dataset's images. */
export function useAnnotationSources(datasetKey: string | undefined, reloadToken = 0): AnnotationSourcesState {
  const [state, setState] = useState<AnnotationSourcesState>({ sources: null, error: null });
  useEffect(() => {
    if (!datasetKey) return;
    let active = true;
    setState({ sources: null, error: null });
    fetchAnnotationSources()
      .then(result => { if (active) setState({ sources: result.sources, error: null }); })
      .catch(err => { if (active) setState({ sources: [], error: String(err) }); });
    return () => { active = false; };
  }, [datasetKey, reloadToken]);
  return state;
}
