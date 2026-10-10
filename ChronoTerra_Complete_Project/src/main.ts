import './style.css';
import { getState, subscribe } from './state';
import { renderShell, bindShellEvents } from './shell';
import { renderOverview, bindOverviewEvents } from './pages/overview';
import { renderNewAnalysis, bindNewAnalysisEvents } from './pages/new-analysis';
import { renderResults, bindResultsEvents, destroyResultsMap } from './pages/results';
import { renderAnalyses, bindAnalysesEvents } from './pages/analyses';
import { renderDatasets, bindDatasetsEvents } from './pages/datasets';
import { icons } from './icons';

function renderPage(pageId: string): string {
  switch (pageId) {
    case 'overview': return renderOverview();
    case 'new-analysis': return renderNewAnalysis();
    case 'results': return renderResults();
    case 'analyses': return renderAnalyses();
    case 'datasets': return renderDatasets();
    default: return `
      <div class="page-container">
        <div class="empty-state">
          <div class="empty-state__icon">${icons.alertTriangle}</div>
          <div class="empty-state__title">Page unavailable</div>
          <div class="empty-state__desc">This view is not available in the current build. Use the navigation above to return to a supported page.</div>
        </div>
      </div>
    `;
  }
}

function bindPageEvents(pageId: string): void {
  switch (pageId) {
    case 'overview': bindOverviewEvents(); break;
    case 'new-analysis': bindNewAnalysisEvents(); break;
    case 'results': bindResultsEvents(); break;
    case 'analyses': bindAnalysesEvents(); break;
    case 'datasets': bindDatasetsEvents(); break;
  }
}

function updateDOM(): void {
  const app = document.getElementById('app');
  if (!app) return;

  const state = getState();
  // This app currently rerenders the root on state changes; clean Leaflet before detaching its DOM.
  destroyResultsMap();
  
  // Render top nav shell and bind nav links
  app.innerHTML = renderShell();
  bindShellEvents();

  // Render main content page and bind page-specific events
  const mainContent = document.getElementById('main-content');
  if (mainContent) {
    mainContent.innerHTML = renderPage(state.currentPage);
    bindPageEvents(state.currentPage);
  }
}

// Subscribe to state changes
subscribe(updateDOM);

// Initial render
updateDOM();

// Perform backend health check on startup
import('./api').then(({ checkBackendHealth }) => {
  checkBackendHealth().then((isOnline) => {
    import('./state').then(({ setState }) => {
      setState({ backendConnected: isOnline });
    });
  });
});

