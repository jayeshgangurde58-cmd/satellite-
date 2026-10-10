// ===================================================================
// ChronoTerra — Results & Maps Workspace
// ===================================================================

import { icons } from '../icons';
import { getState, setState } from '../state';
import { esc, Badge, formatArea } from '../components';

let mapLifecycleGeneration = 0;
let activeMapResultId: string | null = null;
let savedMapView: { resultId: string; center: [number, number]; zoom: number } | null = null;

/** Tear down Leaflet and invalidate pending image/fetch callbacks before the DOM is replaced. */
export function destroyResultsMap(): void {
  mapLifecycleGeneration += 1;
  const runtime = window as any;
  const map = runtime.mapInstance;
  try {
    if (map && activeMapResultId && typeof map.getCenter === 'function' && typeof map.getZoom === 'function') {
      const center = map.getCenter();
      savedMapView = { resultId: activeMapResultId, center: [Number(center.lat), Number(center.lng)], zoom: Number(map.getZoom()) };
    }
    map?.remove?.();
  } catch { console.warn('Results map cleanup failed safely.'); }
  delete runtime.mapInstance;
  delete runtime.mapOverlays;
}

export function renderResults(): string {
  const state = getState();
  
  if (state.analysisStatus !== 'complete' || !state.analysisResult) {
    const failed = state.analysisStatus === 'failed';
    return `
      <div class="page-container" style="display:flex;align-items:center;justify-content:center;min-height:70vh;">
        <div class="card" style="text-align:center;padding:var(--sp-12);max-width:620px;border-style:dashed;">
          <div style="font-size:48px;color:var(--text-tertiary);margin-bottom:var(--sp-4);">${icons.map}</div>
          <h2 style="font-size:20px;color:#fff;margin-bottom:var(--sp-2);">${failed ? 'No prediction available' : 'No analysis results yet'}</h2>
          <p style="color:var(--text-secondary);margin-bottom:var(--sp-6);">${esc(state.analysisError || 'Run an analysis with a compatible input and an installed, evaluated model checkpoint. No synthetic results are displayed.')}</p>
          ${state.analysisPreflight?.report_url ? `<p style="margin-bottom:14px;"><a href="${esc(state.analysisPreflight.report_url)}" target="_blank" rel="noreferrer">Open preflight QA report</a></p>` : ''}
          <button class="btn btn--primary" id="results-start-analysis">${icons.satellite} Return to Analysis</button>
        </div>
      </div>
    `;
  }

  const selected = state.selectedRegionId ? (state.analysisResult?.regions || []).find((r: any) => r.id === state.selectedRegionId) : null;
  const isImageSpace = state.analysisResult?.regions_coordinate_space !== 'geographic' && !state.analysisResult?.geographic_overlay_verified;

  return `
    <div class="map-layout">
      ${renderMapArea(isImageSpace)}
      ${state.detailsPanelOpen && selected ? renderModelRegionDetails(selected) : renderRegionsPanel(state.analysisResult?.regions || [], isImageSpace)}
    </div>
  `;
}

function renderMapArea(isImageSpace: boolean): string {
  const state = getState();
  return `
    <div class="map-container" id="results-map-container">
      ${isImageSpace ? '<div style="position:absolute;top:var(--sp-4);left:50%;transform:translateX(-50%);z-index:1000;background:rgba(0,0,0,.8);color:white;padding:6px 12px;border-radius:4px;font-size:12px;">Image-space viewer · geographic mapping unavailable</div>' : '<div style="position:absolute;top:var(--sp-4);left:50%;transform:translateX(-50%);z-index:1000;background:rgba(0,0,0,.78);color:white;padding:6px 12px;border-radius:4px;font-size:12px;">Georeferenced display · check registration and QA warnings before interpreting changes</div>'}
      <div id="results-leaflet-map" style="width:100%;height:100%; z-index: 0; background: ${isImageSpace ? '#0f172a' : '#1e1e1e'};"></div>
      ${state.resultViewMode === 'split' && state.analysisResult?.task !== 'flood_extent' ? '<div id="result-swipe-divider" role="slider" aria-label="Before and after swipe divider" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" tabindex="0" style="position:absolute;top:0;bottom:0;left:50%;width:3px;transform:translateX(-50%);background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.5);z-index:1200;cursor:ew-resize;touch-action:none;"><div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:34px;height:34px;border-radius:50%;background:#fff;color:#0f172a;display:flex;align-items:center;justify-content:center;font-weight:900;box-shadow:0 2px 10px rgba(0,0,0,.35);">↔</div></div>' : ''}

      <!-- Map Toolbar -->
      <div class="map-toolbar">
        <button class="map-toolbar__btn" data-tooltip="Dashboard" id="map-back-btn" style="background: rgba(8, 13, 24, 0.9); border-color: var(--accent-primary); color: var(--accent-primary); margin-bottom: var(--sp-4);">${icons.arrowLeft}</button>
        <button class="map-toolbar__btn" data-tooltip="Zoom In" id="map-zoom-in">${icons.zoomIn}</button>
        <button class="map-toolbar__btn" data-tooltip="Zoom Out" id="map-zoom-out">${icons.zoomOut}</button>
        <button class="map-toolbar__btn" data-tooltip="Fit Extent" id="map-fit">${icons.crosshair}</button>
        <button class="map-toolbar__btn ${state.layerVisibility.predictionMask ? 'map-toolbar__btn--active' : ''}"
                data-tooltip="Toggle Prediction Mask" id="map-toggle-change" style="${state.layerVisibility.predictionMask ? 'background: var(--accent-primary); color: var(--bg-main); border-color: var(--accent-primary);' : ''}">${icons.layers}</button>
      </div>

      <!-- Layer Control -->
      <div style="position: absolute; top: var(--sp-4); right: var(--sp-4); background: rgba(18, 28, 47, 0.85); backdrop-filter: blur(10px); border: 1px solid var(--border-light); border-radius: var(--radius-md); padding: var(--sp-4); width: 220px; z-index: 1000; box-shadow: var(--shadow-md);">
        <div style="font-size: 11px; font-weight: 700; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: var(--sp-3);">Layer Control</div>
        ${renderLayerItems()}
      </div>

      ${renderMapLegend()}

      <!-- View mode controls are task-aware: flood extent is a single-image task. -->
      <div style="position:absolute;bottom:var(--sp-4);left:50%;transform:translateX(-50%);background:var(--surface-card);border:1px solid var(--border-light);border-radius:var(--radius-full);padding:4px;display:flex;gap:4px;z-index:1000;box-shadow:var(--shadow-lg);">
        ${state.analysisResult?.task === 'flood_extent' ? `<button class="btn btn--sm" id="view-t2" style="border-radius:var(--radius-full);background:var(--accent-primary);color:var(--bg-main);">Flood Input</button>` : `
          <button class="btn btn--sm" id="view-t1" style="border-radius:var(--radius-full);${state.resultViewMode === 'before' ? 'background:var(--accent-primary);color:var(--bg-main);' : 'color:var(--text-secondary);'}">T1 Earlier</button>
          <button class="btn btn--sm" id="view-split" style="border-radius:var(--radius-full);${state.resultViewMode === 'split' ? 'background:var(--accent-primary);color:var(--bg-main);' : 'color:var(--text-secondary);'}">Split Swipe</button>
          <button class="btn btn--sm" id="view-t2" style="border-radius:var(--radius-full);${state.resultViewMode === 'after' ? 'background:var(--accent-primary);color:var(--bg-main);' : 'color:var(--text-secondary);'}">T2 Recent</button>
        `}
      </div>
    </div>
  `;
}

function renderMapLegend(): string {
  const task = getState().analysisResult?.task;
  const items = task === 'flood_extent' ? [['#00d2ff','Predicted water']] : task === 'building_damage' ? [['#32cd50','No damage'],['#ffd700','Minor damage'],['#ff8719','Major damage'],['#eb232d','Destroyed']] : [['#ef4444','Candidate change']];
  return `<div style="position:absolute;bottom:var(--sp-8);left:var(--sp-4);background:rgba(18,28,47,.9);border:1px solid var(--border-light);border-radius:var(--radius-md);padding:var(--sp-3) var(--sp-4);z-index:1000;font-size:12px;">${items.map(([color,label])=>`<div style="display:flex;align-items:center;gap:var(--sp-3);margin-bottom:3px;"><div style="width:12px;height:12px;background:${color};border-radius:2px;"></div><span style="color:#fff;">${label}</span></div>`).join('')}<div style="font-size:10px;color:var(--text-secondary);margin-top:4px;">Transparent areas: background or invalid pixels; see QA report.</div></div>`;
}

function renderLayerItems(): string {
  const state = getState();
  const result = state.analysisResult;
  const layers = [
    { id: 'before', label: 'T1 Observation', color: '#4A90D9', enabled: Boolean(result?.before_preview_url || state.beforeFile?.previewUrl || state.beforeFile?.file) && result?.task !== 'flood_extent' },
    { id: 'after', label: result?.task === 'flood_extent' ? 'Flood Observation' : 'T2 Observation', color: '#7B68EE', enabled: Boolean(result?.input_preview_url || result?.after_preview_url || state.afterFile?.previewUrl || state.afterFile?.file || (result?.task === 'flood_extent' && state.beforeFile?.file)) },
    { id: 'predictionMask', label: 'Model Prediction Mask', color: '#EF4444', enabled: Boolean(result?.mask_preview_url) },
    { id: 'changePolygons', label: 'Mask-derived regions', color: '#F59E0B', enabled: Boolean(result?.regions_url && result?.region_count) },
    { id: 'boundary', label: 'Reference zone (user-supplied)', color: '#A78BFA', enabled: Boolean(result?.boundary_overlay_url) },
  ];
  return layers.map(layer => {
    if (!layer.enabled) {
      return `<div title="Layer is unavailable for this result" style="display:flex;align-items:center;gap:var(--sp-2);margin-bottom:var(--sp-2);cursor:not-allowed;opacity:.3;"><div style="width:14px;height:14px;border-radius:2px;border:1px solid var(--border-strong);"></div><span style="color:#fff;font-size:13px;">${layer.label} (Unavailable)</span></div>`;
    }
    const visible = Boolean(state.layerVisibility[layer.id]);
    return `<div data-toggle-layer="${layer.id}" role="button" tabindex="0" style="display:flex;align-items:center;gap:var(--sp-2);margin-bottom:var(--sp-2);cursor:pointer;opacity:${visible ? '1' : '.5'};"><div style="width:14px;height:14px;border-radius:2px;border:1px solid ${visible ? layer.color : 'var(--border-strong)'};background:${visible ? layer.color : 'transparent'};display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;">${visible ? icons.check : ''}</div><span style="color:#fff;font-size:13px;">${layer.label}</span></div>`;
  }).join('');
}

function renderPredictionSummary(): string {
  const result = getState().analysisResult;
  if (!result) return '';
  const labels: Record<string,string> = { urban_change:'Urban change detection', flood_extent:'Flood extent mapping', building_damage:'Building damage assessment', urban_change_classical:'Urban change (Classical Baseline)' };
  const counts = result.predicted_class_pixel_counts || {};
  const countRows = Object.entries(counts).map(([name, value]) => `<div class="flex-between" style="font-size:12px;padding:3px 0;"><span>${esc(name.replaceAll('_',' '))}</span><strong>${Number(value).toLocaleString()}</strong></div>`).join('');
  const warnings: string[] = Array.isArray(result.warnings) ? [...result.warnings] : [];
  if (result.regions_truncated) warnings.push('Only the largest 500 mask-derived regions are shown; inspect the complete mask for all predicted pixels.');
  const locationCheck = result.location_footprint_check;
  const boundary = result.boundary_analysis;
  const temporal = boundary?.temporal_compatibility;
  const temporalNote = temporal?.message ? `<div style="font-size:11px;color:var(--warning);margin-top:5px;">Temporal compatibility: ${esc(temporal.status || 'NOT_VERIFIED')} — ${esc(temporal.message)}</div>` : '';
  const boundarySummary = boundary ? `<div style="margin-top:10px;padding:10px;border:1px solid #8b5cf666;border-radius:8px;background:#8b5cf60a;"><strong style="font-size:12px;color:#c4b5fd;">Reference-zone screening</strong>${boundary.status === 'COMPLETED' ? `<div style="font-size:12px;color:var(--text-secondary);margin-top:5px;">${Number(boundary.boundary_overlap_candidate_count || 0)} boundary-overlap candidate(s) · ${Number(boundary.near_boundary_candidate_count || 0)} near-boundary candidate(s) · ${Number(boundary.candidate_regions_tested || 0)} stored region(s) tested</div>` : `<div style="font-size:12px;color:var(--warning);margin-top:5px;">Comparison not run: ${esc(boundary.status || 'not available')}.</div>`}${temporalNote}<div style="font-size:11px;color:var(--warning);margin-top:5px;">${esc(boundary.authority_status || 'USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED')}. This is spatial screening, not proof of ownership, violation or illegal encroachment.</div></div>` : '';
  return `<div class="card" style="padding:12px;margin-bottom:14px;"><div style="font-weight:800;font-size:14px;margin-bottom:5px;">${esc(labels[result.task] || result.task || 'Prediction result')}</div><div style="font-size:12px;color:var(--text-secondary);margin-bottom:8px;">Status: ${esc(result.status || 'completed')} · Valid coverage: ${typeof result.valid_fraction === 'number' ? (result.valid_fraction*100).toFixed(1)+'%' : 'Not reported'}</div>${countRows || '<div style="font-size:12px;color:var(--text-secondary);">Class counts not reported.</div>'}${locationCheck ? `<div style="font-size:11px;margin-top:8px;color:${locationCheck.status === 'PASS' ? 'var(--success)' : 'var(--warning)'};">Selected place vs raster bounds: ${esc(locationCheck.status)} — ${esc(locationCheck.note || '')}</div>` : ''}${boundarySummary}<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;">${result.mask_url ? `<a class="btn btn--secondary btn--sm" href="${esc(result.mask_url)}" target="_blank" rel="noreferrer">Download mask</a>` : ''}${result.scores_url ? `<a class="btn btn--secondary btn--sm" href="${esc(result.scores_url)}" target="_blank" rel="noreferrer">Scores</a>` : ''}${result.uncertainty_url ? `<a class="btn btn--secondary btn--sm" href="${esc(result.uncertainty_url)}" target="_blank" rel="noreferrer">Uncertainty</a>` : ''}${result.report_url ? `<a class="btn btn--secondary btn--sm" href="${esc(result.report_url)}" target="_blank" rel="noreferrer">QA report</a>` : ''}</div>${warnings.length ? `<details style="margin-top:10px;"><summary style="cursor:pointer;font-size:12px;color:var(--warning);">${warnings.length} model/data warning(s)</summary><ul style="font-size:11px;color:var(--text-secondary);padding-left:18px;">${warnings.map((w:string)=>`<li>${esc(w)}</li>`).join('')}</ul></details>` : ''}</div>`;
}

function renderRegionsPanel(regions: any[], isImageSpace: boolean): string {
  const sortedRegions = [...regions].sort((a, b) => (Number(b.area_m2 ?? b.area_pixels) || 0) - (Number(a.area_m2 ?? a.area_pixels) || 0));
  return `
    <div class="details-panel" style="padding:var(--sp-4);display:flex;flex-direction:column;">
      ${renderPredictionSummary()}
      <div class="flex-between" style="margin-bottom:var(--sp-4);">
        <h3 style="font-size:16px;font-weight:600;color:#fff;">Mask-derived regions (${regions.length})</h3>
        <button class="btn-icon" id="toggle-table-view" title="Switch list/grid view" style="color:var(--text-secondary);">${icons.list}</button>
      </div>
      <div style="position:relative;margin-bottom:var(--sp-4);">
        <input type="text" class="form-input" id="region-search" placeholder="Search by region or class..." style="width:100%;padding-left:36px;background:rgba(0,0,0,0.2);" />
        <span style="position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--text-tertiary);">${icons.search}</span>
      </div>
      <div id="regions-list" style="flex:1;overflow-y:auto;padding-right:4px;margin-right:-4px;">
        ${regions.length === 0 ? `<div style="text-align:center;color:var(--text-secondary);padding:var(--sp-8) 0;"><div style="margin-bottom:var(--sp-2);">${icons.info}</div><div>No non-background regions were found in the model mask. Check the prediction mask and QA report; this is not a claim that no real-world change occurred.</div></div>` : sortedRegions.map((r:any) => `
          <div data-select-region="${esc(r.id)}" style="background:rgba(255,255,255,.02);border:1px solid var(--border-light);border-radius:var(--radius-md);padding:var(--sp-3);margin-bottom:var(--sp-3);cursor:pointer;">
            <div class="flex-between" style="margin-bottom:var(--sp-1);"><div style="font-size:13px;font-weight:700;color:#fff;">${esc(r.id)} · ${esc(String(r.class_name || 'predicted class').replaceAll('_',' '))}</div>${Badge('neutral', `Class ${r.class_id}`)}</div>
            <div style="display:flex;gap:var(--sp-4);font-size:12px;color:var(--text-secondary);font-family:var(--font-mono);">
              <div>${r.coordinate_space === 'geographic' && typeof r.area_m2 === 'number' ? formatArea(r.area_m2) : `${Number(r.area_pixels || 0).toFixed(1)} px²`}</div>
              <div>${r.coordinate_space === 'geographic' ? `${Number(r.centroid_lat).toFixed(5)}, ${Number(r.centroid_lng).toFixed(5)}` : `pixel ${Number(r.centroid_x).toFixed(1)}, ${Number(r.centroid_y).toFixed(1)}`}</div>
            </div>
          </div>`).join('')}
      </div>
      ${isImageSpace ? '<div style="font-size:11px;color:var(--warning);margin-top:8px;">Coordinates and polygon areas are in image pixels, not geographic units.</div>' : ''}
    </div>
  `;
}

function renderModelRegionDetails(region: any): string {
  const boundaryRows = region.boundary_relation ? `<div class="flex-between" style="margin-top:8px;font-size:13px;"><span style="color:var(--text-secondary);">Reference-zone relation</span><strong>${esc(String(region.boundary_relation).replaceAll('_',' '))}</strong></div><div class="flex-between" style="margin-top:8px;font-size:13px;"><span style="color:var(--text-secondary);">Boundary overlap</span><span>${Number.isFinite(Number(region.boundary_overlap_m2)) ? `${Number(region.boundary_overlap_m2).toFixed(1)} m²` : 'Not calculated'}</span></div><div class="flex-between" style="margin-top:8px;font-size:13px;"><span style="color:var(--text-secondary);">Distance to reference zone</span><span>${Number.isFinite(Number(region.distance_to_reference_zone_m)) ? `${Number(region.distance_to_reference_zone_m).toFixed(1)} m` : 'Not calculated'}</span></div>${region.boundary_temporal_note ? `<p style="font-size:11px;color:var(--warning);margin-top:10px;">Temporal compatibility (${esc(region.boundary_temporal_status || 'NOT_VERIFIED')}): ${esc(region.boundary_temporal_note)}</p>` : ''}<p style="font-size:11px;color:var(--warning);margin-top:10px;">${esc(region.boundary_authority_status || 'Boundary data not supplied or not independently verified.')}</p>` : '';
  return `<div class="details-panel" style="padding:var(--sp-5);overflow-y:auto;"><div class="flex-between" style="margin-bottom:var(--sp-6);"><h3 style="font-size:18px;font-weight:700;color:#fff;">Region ${esc(region.id)}</h3><button class="btn-icon" id="close-details" aria-label="Close region details" style="color:var(--text-secondary);background:rgba(255,255,255,.05);border-radius:50%;width:32px;height:32px;">${icons.x}</button></div><div style="margin-bottom:var(--sp-4);">${Badge('neutral', String(region.class_name || 'predicted class').replaceAll('_',' '))}</div><div style="background:rgba(0,0,0,.2);border:1px solid var(--border-light);border-radius:var(--radius-md);padding:var(--sp-4);"><div class="flex-between" style="margin-bottom:var(--sp-2);font-size:13px;"><span style="color:var(--text-secondary);">Class ID</span><span>${Number(region.class_id)}</span></div><div class="flex-between" style="margin-bottom:var(--sp-2);font-size:13px;"><span style="color:var(--text-secondary);">Area</span><span>${region.coordinate_space === 'geographic' ? formatArea(Number(region.area_m2 || 0)) : `${Number(region.area_pixels || 0).toFixed(1)} px²`}</span></div><div class="flex-between" style="font-size:13px;"><span style="color:var(--text-secondary);">Coordinate space</span><span>${esc(region.coordinate_space || 'unknown')}</span></div>${boundaryRows}</div><p style="font-size:12px;color:var(--text-secondary);margin-top:16px;">This polygon is derived from the prediction mask. It is a candidate result, not independently verified ground truth or a legal determination.</p></div>`;
}

// ── Events ───────────────────────────────────────────────────────────

export function bindResultsEvents(): void {
  // Back button
  document.getElementById('map-back-btn')?.addEventListener('click', () => {
    setState({ currentPage: 'overview' });
  });

  // Start analysis from empty state
  document.getElementById('results-start-analysis')?.addEventListener('click', () => {
    setState({ currentPage: 'new-analysis', wizardStep: 0, beforeFile: null, afterFile: null, aoiFile: null, referenceBoundaryFile: null, boundarySourceName: '', boundaryEffectiveDate: '', boundaryProximityM: 10, beforeSclFile: null, afterSclFile: null, assumeBandOrder: false, analysisResult: null, analysisPreflight: null, analysisError: null, analysisStatus: 'draft', changeRegions: [], selectedTask: 'urban', analysisModelTask: 'urban_change', overlapValidation: null, confirmedArea: false, selectedRegionId: null, detailsPanelOpen: false, resultViewMode: 'split' });
  });

  // Region selection
  document.querySelectorAll<HTMLElement>('[data-select-region]').forEach(el => {
    el.addEventListener('click', () => {
      const id = el.dataset.selectRegion!;
      setState({ selectedRegionId: id, detailsPanelOpen: true });
    });
  });

  // Close details
  document.getElementById('close-details')?.addEventListener('click', () => {
    setState({ selectedRegionId: null, detailsPanelOpen: false, resultViewMode: 'split' });
  });

  // Layer toggles
  document.querySelectorAll<HTMLElement>('[data-toggle-layer]').forEach(el => {
    const toggle = () => {
      const layerId = el.dataset.toggleLayer!;
      const vis = { ...getState().layerVisibility, [layerId]: !getState().layerVisibility[layerId] };
      setState({ layerVisibility: vis });
      const overlay = (window as any).mapOverlays?.[layerId];
      const map = (window as any).mapInstance;
      if (overlay) {
        if (typeof overlay.setOpacity === 'function') overlay.setOpacity(vis[layerId] ? 1 : 0);
        else if (map) { if (vis[layerId]) map.addLayer(overlay); else map.removeLayer(overlay); }
      }
    };
    el.addEventListener('click', toggle);
    el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  });
  document.getElementById('map-toggle-change')?.addEventListener('click', () => {
    const vis = { ...getState().layerVisibility, predictionMask: !getState().layerVisibility.predictionMask };
    setState({ layerVisibility: vis });
    const overlay = (window as any).mapOverlays?.predictionMask;
    if (overlay) overlay.setOpacity(vis.predictionMask ? 1 : 0);
  });
  document.getElementById('region-search')?.addEventListener('input', (event) => {
    const query = (event.target as HTMLInputElement).value.trim().toLowerCase();
    document.querySelectorAll<HTMLElement>('[data-select-region]').forEach(card => {
      card.style.display = card.textContent?.toLowerCase().includes(query) ? '' : 'none';
    });
  });
  document.getElementById('toggle-table-view')?.addEventListener('click', () => {
    const list = document.getElementById('regions-list');
    if (!list) return;
    const tableMode = list.dataset.tableMode !== 'true';
    list.dataset.tableMode = String(tableMode);
    list.style.display = tableMode ? 'grid' : 'block';
    list.style.gridTemplateColumns = tableMode ? 'repeat(auto-fit,minmax(220px,1fr))' : '';
    const button = document.getElementById('toggle-table-view');
    if (button) button.title = tableMode ? 'Switch to list view' : 'Switch to grid view';
  });

  // Swipe View Toggle
  const viewT1 = document.getElementById('view-t1');
  const viewSplit = document.getElementById('view-split');
  const viewT2 = document.getElementById('view-t2');

  const updateViewBtns = (activeId: string) => {
    [viewT1, viewSplit, viewT2].forEach(btn => {
      if (!btn) return;
      if (btn.id === activeId) {
        btn.style.background = 'var(--accent-primary)';
        btn.style.color = 'var(--bg-main)';
      } else {
        btn.style.background = 'transparent';
        btn.style.color = 'var(--text-secondary)';
      }
    });
  };

  const applyVisibility = () => {
    const vis = getState().layerVisibility;
    ['before', 'after'].forEach(layerId => {
      const overlay = (window as any).mapOverlays?.[layerId];
      if (overlay) {
        overlay.setOpacity(vis[layerId] ? 1 : 0);
        const el = overlay.getElement();
        if (el) el.style.clipPath = 'none'; // reset clip path
      }
    });
  };

  viewT1?.addEventListener('click', () => {
    updateViewBtns('view-t1');
    setState({ resultViewMode: 'before', layerVisibility: { ...getState().layerVisibility, before: true, after: false } });
    applyVisibility();
  });
  viewT2?.addEventListener('click', () => {
    updateViewBtns('view-t2');
    setState({ resultViewMode: 'after', layerVisibility: { ...getState().layerVisibility, before: false, after: true } });
    applyVisibility();
  });
  viewSplit?.addEventListener('click', () => {
    updateViewBtns('view-split');
    setState({ resultViewMode: 'split', layerVisibility: { ...getState().layerVisibility, before: true, after: true } });
    applyVisibility();
    
    // Handle split swipe logic (CSS clip path on after layer)
    const overlay = (window as any).mapOverlays?.after;
    if (overlay) {
        const el = overlay.getElement();
        if (el) el.style.clipPath = 'polygon(50% 0, 100% 0, 100% 100%, 50% 100%)';
    }
  });

  // Review actions
  document.querySelectorAll<HTMLElement>('[data-review-action]').forEach(el => {
    el.addEventListener('click', () => {
      const action = el.dataset.reviewAction as any;
      const regionId = el.dataset.regionId!;
      const regions = getState().changeRegions.map(r =>
        r.id === regionId ? { ...r, reviewStatus: action } : r
      );
      setState({ changeRegions: regions });
    });
  });

  // Initialize Leaflet map
  initResultsMap();
}

function initResultsMap(): void {
  const mapEl = document.getElementById('results-leaflet-map');
  if (!mapEl) return;
  const L = (window as any).L;
  if (!L) {
    const fallback = document.createElement('div');
    fallback.setAttribute('role', 'status');
    fallback.style.cssText = 'position:absolute;inset:16px;display:flex;align-items:center;justify-content:center;padding:24px;text-align:center;color:#cbd5e1;background:#0f172a;border:1px solid #475569;border-radius:10px;';
    fallback.textContent = 'Interactive map library is unavailable. Prediction summaries and downloads remain available; check network access to the configured map assets and reload this page.';
    mapEl.appendChild(fallback);
    return;
  }
  const state = getState();
  const result = state.analysisResult;
  if (!result) return;
  const resultId = String(result.session_id || result.id || result.task || 'current-result');
  if (activeMapResultId !== resultId) savedMapView = null;
  activeMapResultId = resultId;
  const isImageSpace = result.regions_coordinate_space !== 'geographic' && !result.geographic_overlay_verified;
  const generation = ++mapLifecycleGeneration;
  const isCurrentMap = () => generation === mapLifecycleGeneration && mapEl.isConnected;
  const map = isImageSpace
    ? L.map(mapEl, { crs: L.CRS.Simple, minZoom: -4, zoomControl: false })
    : L.map(mapEl, { center: [0,0], zoom: 2, zoomControl: false });
  if (!isImageSpace) {
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }).addTo(map);
  }
  (window as any).mapInstance = map;
  (window as any).mapOverlays = { before: null, after: null, predictionMask: null, changePolygons: null, boundary: null };
  let boundsToFit: any = null;
  const fitOrRestoreView = (bounds: any) => {
    if (boundsToFit) return;
    boundsToFit = bounds;
    if (savedMapView && savedMapView.resultId === resultId) {
      const previous = savedMapView;
      savedMapView = null;
      map.setView(previous.center, previous.zoom);
    } else {
      map.fitBounds(bounds);
    }
  };
  const getSource = (role: 'before'|'after') => {
    if (result.task === 'flood_extent') {
      return role === 'after' ? (result.input_preview_url || state.beforeFile?.previewUrl || null) : null;
    }
    if (role === 'before') return result.before_preview_url || state.beforeFile?.previewUrl || null;
    return result.after_preview_url || state.afterFile?.previewUrl || null;
  };
  const getBounds = (role: 'before'|'after', image: HTMLImageElement) => {
    if (isImageSpace) return [[0,0],[image.height,image.width]];
    const specific = role === 'before' ? result.before_bounds : result.after_bounds;
    const b = specific || result.bounds;
    if (Array.isArray(b) && b.length === 2) return b;
    const old = state.overlapValidation?.bounds;
    if (old) return [[old.bottom,old.left],[old.top,old.right]];
    return null;
  };
  const addImageLayer = (url: string, layerId: string, role: 'before'|'after', zIndex: number) => {
    const img = new Image();
    img.onload = () => {
      if (!isCurrentMap()) return;
      const bounds = getBounds(role, img);
      if (!bounds) return;
      const overlay = L.imageOverlay(url, bounds, { zIndex, opacity: state.layerVisibility[layerId] ? 1 : 0, interactive: false }).addTo(map);
      (window as any).mapOverlays[layerId] = overlay;
      if (layerId === 'after' && state.resultViewMode === 'split') {
        const element = overlay.getElement();
        if (element) element.style.clipPath = 'polygon(50% 0,100% 0,100% 100%,50% 100%)';
      }
      fitOrRestoreView(bounds);
    };
    img.onerror = () => console.warn(`Could not load ${layerId} preview.`);
    img.src = url;
  };
  const beforeUrl = getSource('before');
  const afterUrl = getSource('after');
  if (beforeUrl) addImageLayer(beforeUrl, 'before', 'before', 1);
  if (afterUrl) addImageLayer(afterUrl, 'after', 'after', 2);
  if (result.mask_preview_url) {
    const maskImg = new Image();
    maskImg.onload = () => {
      if (!isCurrentMap()) return;
      const bounds = isImageSpace ? [[0,0],[maskImg.height,maskImg.width]] : (result.bounds || getBounds(result.task === 'flood_extent' ? 'after' : 'before', maskImg));
      if (!bounds) return;
      const overlay = L.imageOverlay(result.mask_preview_url, bounds, { zIndex: 5, opacity: state.layerVisibility.predictionMask ? 1 : 0, interactive: false }).addTo(map);
      (window as any).mapOverlays.predictionMask = overlay;
      fitOrRestoreView(bounds);
    };
    maskImg.onerror = () => console.warn('Could not load prediction mask preview.');
    maskImg.src = result.mask_preview_url;
  }
  if (result.regions_url && Number(result.region_count) > 0) {
    fetch(result.regions_url).then(response => { if (!response.ok) throw new Error(`Region request failed (${response.status})`); return response.json(); }).then(collection => {
      if (!isCurrentMap()) return;
      const regionsLayer = L.geoJSON(collection, {
        style: (feature: any) => {
          const props = feature?.properties || {};
          const cls = Number(props.class_id);
          const relation = String(props.boundary_relation || '');
          const color = relation === 'BOUNDARY_OVERLAP_CANDIDATE' ? '#dc2626' : relation === 'NEAR_BOUNDARY_CANDIDATE' ? '#f59e0b' : result.task === 'flood_extent' ? '#00d2ff' : result.task === 'building_damage' ? ({1:'#32cd50',2:'#ffd700',3:'#ff8719',4:'#eb232d'} as any)[cls] || '#94a3b8' : '#ef4444';
          return { color, weight: relation === 'BOUNDARY_OVERLAP_CANDIDATE' ? 2.5 : 1.5, fillColor: color, fillOpacity: 0.28 };
        },
        onEachFeature: (feature: any, layer: any) => {
          const props = feature.properties || {};
          const relation = props.boundary_relation ? `<br>Boundary screening: ${esc(String(props.boundary_relation).replaceAll('_',' '))}` : '';
          const overlap = Number.isFinite(Number(props.boundary_overlap_m2)) ? `<br>Overlap: ${Number(props.boundary_overlap_m2).toFixed(1)} m²` : '';
          layer.bindPopup(`<strong>${esc(String(props.class_name || 'region').replaceAll('_',' '))}</strong><br>Class: ${Number(props.class_id)}<br>${props.area_m2 != null ? `${Number(props.area_m2).toFixed(1)} m²` : `${Number(props.area_pixels || 0).toFixed(1)} px²`}${relation}${overlap}`);
          layer.on('click', () => setState({ selectedRegionId: props.id || null, detailsPanelOpen: true }));
        }
      }).addTo(map);
      (window as any).mapOverlays.changePolygons = regionsLayer;
      if (!state.layerVisibility.changePolygons) map.removeLayer(regionsLayer);
    }).catch(error => console.warn('Could not load mask-derived regions:', error));
  }
  if (result.boundary_overlay_url && !isImageSpace) {
    fetch(result.boundary_overlay_url).then(response => { if (!response.ok) throw new Error(`Reference-zone request failed (${response.status})`); return response.json(); }).then(collection => {
      if (!isCurrentMap()) return;
      const layer = L.geoJSON(collection, { style: { color: '#A78BFA', weight: 2, fill: false, opacity: 0.95 } }).addTo(map);
      (window as any).mapOverlays.boundary = layer;
      if (!state.layerVisibility.boundary) map.removeLayer(layer);
      if (!boundsToFit) { const bounds = layer.getBounds(); if (bounds.isValid()) fitOrRestoreView(bounds); }
    }).catch(error => console.warn('Could not load reference zone overlay:', error));
  }
  const swipeDivider = document.getElementById('result-swipe-divider');
  if (swipeDivider) {
    let dragging = false;
    const setSwipe = (clientX: number) => {
      const rect = mapEl.getBoundingClientRect();
      const percent = Math.max(0, Math.min(100, ((clientX - rect.left) / Math.max(rect.width, 1)) * 100));
      swipeDivider.style.left = `${percent}%`;
      swipeDivider.setAttribute('aria-valuenow', String(Math.round(percent)));
      const overlay = (window as any).mapOverlays?.after;
      const image = overlay?.getElement();
      if (image) image.style.clipPath = `polygon(${percent}% 0,100% 0,100% 100%,${percent}% 100%)`;
    };
    swipeDivider.addEventListener('pointerdown', (event) => { dragging = true; swipeDivider.setPointerCapture(event.pointerId); setSwipe(event.clientX); });
    swipeDivider.addEventListener('pointermove', (event) => { if (dragging) setSwipe(event.clientX); });
    swipeDivider.addEventListener('pointerup', () => { dragging = false; });
    swipeDivider.addEventListener('pointercancel', () => { dragging = false; });
    swipeDivider.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault(); const rect = mapEl.getBoundingClientRect();
        const current = Number(swipeDivider.getAttribute('aria-valuenow') || 50);
        const next = Math.max(0, Math.min(100, current + (event.key === 'ArrowRight' ? 2 : -2)));
        setSwipe(rect.left + rect.width * next / 100);
      }
    });
  }
  document.getElementById('map-zoom-in')?.addEventListener('click', () => map.zoomIn());
  document.getElementById('map-zoom-out')?.addEventListener('click', () => map.zoomOut());
  document.getElementById('map-fit')?.addEventListener('click', () => { if (boundsToFit) map.fitBounds(boundsToFit); });
  // Reinitialize safely when the browser resizes the result view.
  window.setTimeout(() => { if (isCurrentMap()) map.invalidateSize(); }, 120);
}
