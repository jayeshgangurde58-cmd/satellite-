// ===================================================================
// ChronoTerra — Analyses (History) Page
// ===================================================================

import { icons } from '../icons';
import { getState, setState, type AnalysisRecord } from '../state';
import { esc, Badge, EmptyState, formatDate } from '../components';

export function renderAnalyses(): string {
  const state = getState();
  const history = state.analysisHistory;

  return `
    <div class="page-container">
      <div class="page-header">
        <div>
          <h1 class="page-title">Analyses</h1>
          <p class="page-subtitle">Analysis history from this session. History is stored locally and will be lost when the browser is closed.</p>
        </div>
        <button class="btn btn--primary" id="analyses-new">${icons.plus} New Analysis</button>
      </div>

      <div class="error-banner error-banner--info" style="margin-bottom: var(--sp-4);">
        <span class="error-banner__icon">${icons.info}</span>
        <div class="error-banner__content">
          <div class="error-banner__title">Session-only storage</div>
          <div class="error-banner__text">
            Analysis records are held in browser memory for this session only.
            Persistent database storage is not implemented. Exported files saved to your local machine will persist.
          </div>
        </div>
      </div>

      ${history.length === 0 ? EmptyState(
        icons.list,
        'No analyses in history',
        'Completed and in-progress analyses will appear here. Start a new analysis to begin.',
        { label: 'Start New Analysis', id: 'analyses-empty-new' }
      ) : renderAnalysesTable(history)}
    </div>
  `;
}

function renderAnalysesTable(items: AnalysisRecord[]): string {
  const taskLabels: Record<string, string> = { general: 'General Change', urban: 'Urban Screening', disaster: 'Disaster Assessment', urban_change: 'Urban Change Detection', flood_extent: 'Flood Extent Mapping', building_damage: 'Building Damage Assessment' };
  const statusMap: Record<string, { badge: 'success' | 'warning' | 'error' | 'info' | 'neutral'; label: string }> = {
    complete: { badge: 'success', label: 'Complete' },
    running: { badge: 'info', label: 'Running' },
    failed: { badge: 'error', label: 'Failed' },
    draft: { badge: 'neutral', label: 'Draft' },
    cancelled: { badge: 'neutral', label: 'Cancelled' },
    configured: { badge: 'info', label: 'Configured' },
    'quality-check': { badge: 'warning', label: 'Quality Check' },
  };

  return `
    <div class="data-table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Task</th>
            <th>Status</th>
            <th>Created</th>
            <th>Warnings</th>
            <th>Class pixels</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${items.map(item => {
            const s = statusMap[item.status] || { badge: 'neutral' as const, label: item.status };
            return `
              <tr data-open-analysis="${item.id}" style="cursor: pointer;">
                <td style="font-weight: 600;">${esc(item.name)}</td>
                <td>${esc(taskLabels[item.task] || item.task)}</td>
                <td>${Badge(s.badge, s.label)}</td>
                <td class="cell-mono">${formatDate(item.createdAt)}</td>
                <td>${item.warningCount > 0 ? `<span style="color: var(--warning);">⚠ ${item.warningCount}</span>` : '—'}</td>
                <td class="cell-mono">${item.resultCount !== undefined ? item.resultCount : '—'}</td>
                <td>
                  ${item.status === 'complete' ? `
                    <button class="btn btn--ghost btn--sm" data-view-results="${item.id}">
                      ${icons.eye} View
                    </button>
                  ` : '—'}
                </td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

export function bindAnalysesEvents(): void {
  const startNew = () => setState({ currentPage: 'new-analysis', wizardStep: 0, beforeFile: null, afterFile: null, aoiFile: null, referenceBoundaryFile: null, boundarySourceName: '', boundaryEffectiveDate: '', boundaryProximityM: 10, beforeSclFile: null, afterSclFile: null, assumeBandOrder: false, beforeDate: '', afterDate: '', locationName: '', locationPlaceId: '', locationLat: undefined, locationLng: undefined, analysisResult: null, analysisPreflight: null, analysisError: null, changeRegions: [], analysisStatus: 'draft', selectedTask: 'urban', analysisModelTask: 'urban_change', selectedRegionId: null, detailsPanelOpen: false, resultViewMode: 'split' });
  document.getElementById('analyses-new')?.addEventListener('click', startNew);
  document.getElementById('analyses-empty-new')?.addEventListener('click', startNew);
  const openRecord = (id: string) => {
    const record = getState().analysisHistory.find(item => item.id === id);
    if (!record?.result) return;
    const task = ['urban_change','flood_extent','building_damage'].includes(record.task) ? record.task as 'urban_change'|'flood_extent'|'building_damage' : 'urban_change';
    setState({ currentPage: 'results', analysisResult: record.result, analysisStatus: 'complete', analysisError: null, analysisModelTask: task, selectedTask: task === 'urban_change' ? 'urban' : 'disaster', selectedRegionId: null, detailsPanelOpen: false, changeRegions: [] });
  };
  document.querySelectorAll<HTMLElement>('[data-view-results]').forEach(el => el.addEventListener('click', e => { e.stopPropagation(); openRecord(el.dataset.viewResults!); }));
  document.querySelectorAll<HTMLElement>('[data-open-analysis]').forEach(el => el.addEventListener('click', () => openRecord(el.dataset.openAnalysis!)));
}
