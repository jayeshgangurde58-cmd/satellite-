// ===================================================================
// ChronoTerra — App Shell (Top Navigation + Main Content)
// ===================================================================

import { icons } from './icons';
import { getState, setState } from './state';

export function renderShell(): string {
  const state = getState();
  
  return `
    <header class="top-nav" role="banner">
      <div class="brand" data-nav="overview" style="cursor: pointer;">
        <div class="brand__logo">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#009688" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
            <path d="M2 12h20"/>
          </svg>
        </div>
        <div class="brand__text">
          <span class="brand__name">ChronoTerra</span>
          <span class="brand__sub">SEE THE CHANGE. MAP THE IMPACT.</span>
        </div>
      </div>
      
      <nav class="nav-links" role="navigation">
        <button class="nav-link ${state.currentPage === 'overview' ? 'nav-link--active' : ''}" data-nav="overview">Home</button>
        <button class="nav-link ${state.currentPage === 'new-analysis' ? 'nav-link--active' : ''}" data-nav="new-analysis">Upload & Analyze</button>
        <button class="nav-link ${state.currentPage === 'results' ? 'nav-link--active' : ''}" data-nav="results">Map Results</button>
        <button class="nav-link ${state.currentPage === 'analyses' ? 'nav-link--active' : ''}" data-nav="analyses">History</button>
        <button class="nav-link ${state.currentPage === 'datasets' ? 'nav-link--active' : ''}" data-nav="datasets">Models & Data</button>
      </nav>

      <div class="nav-actions">
        <div class="status-badge" data-tooltip="Backend API status; model readiness is shown under Models & Data">
          <div class="status-badge__dot ${state.backendConnected ? 'status-badge__dot--online' : 'status-badge__dot--offline'}"></div>
          ${state.backendConnected ? 'API Online' : 'Backend Offline'}
        </div>
        <button class="btn btn--primary btn--sm" id="nav-new-analysis">
          ${icons.plus} New Analysis
        </button>
      </div>
    </header>

    <div class="app-shell">
      <main class="main-content" id="main-content" role="main">
        <!-- Page content rendered here -->
      </main>
    </div>
  `;
}

export function bindShellEvents(): void {
  // Navigation links
  document.querySelectorAll<HTMLElement>('[data-nav]').forEach(btn => {
    btn.addEventListener('click', () => {
      const page = btn.dataset.nav!;
      setState({ currentPage: page });
    });
  });

  // CTA button
  document.getElementById('nav-new-analysis')?.addEventListener('click', () => {
    setState({ currentPage: 'new-analysis', wizardStep: 0, beforeFile: null, afterFile: null, aoiFile: null, referenceBoundaryFile: null, boundarySourceName: '', boundaryEffectiveDate: '', boundaryProximityM: 10, beforeSclFile: null, afterSclFile: null, assumeBandOrder: false, beforeDate: '', afterDate: '', locationName: '', locationPlaceId: '', locationLat: undefined, locationLng: undefined, analysisResult: null, analysisPreflight: null, analysisError: null, analysisStatus: 'draft', changeRegions: [], selectedTask: 'urban', analysisModelTask: 'urban_change', overlapValidation: null, confirmedArea: false, selectedRegionId: null, detailsPanelOpen: false, resultViewMode: 'split' });
  });
}
