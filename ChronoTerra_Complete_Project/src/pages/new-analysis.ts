// ===================================================================
// ChronoTerra — Image Comparison & Upload Page
// ===================================================================

import { getState, setState, generateId, type FileUpload, type FileMetadata, type ProcessingStep } from '../state';
import { esc, formatBytes } from '../components';
import { inspectImageApi, preflightPredictionApi, runPredictionApi, getModelStatusApi } from '../api';

export function renderNewAnalysis(): string {
  const state = getState();

  // If currently running analysis, render the processing pipeline step
  if (state.wizardStep === 4) {
    return renderRunAnalysis();
  }
  
  // If analysis complete, render the completion review step
  if (state.wizardStep === 5) {
    return renderReviewExport();
  }

  // Otherwise, render the Compare Before and After upload page
  return renderComparePage();
}

function renderComparePage(): string {
  const state = getState();
  const beforeFile = state.beforeFile;
  const afterFile = state.afterFile;
  const selectedTask = state.selectedTask || 'urban';
  const isDisaster = selectedTask === 'disaster';
  const modelTask = state.analysisModelTask || (isDisaster ? 'flood_extent' : 'urban_change');
  const isFlood = modelTask === 'flood_extent';
  const bothUploaded = isFlood ? beforeFile?.status === 'ready' : beforeFile?.status === 'ready' && afterFile?.status === 'ready';
  const categoryTag = isDisaster ? 'DISASTER IMPACT MANAGEMENT' : 'URBAN CHANGE SCREENING';
  const mainTitle = isDisaster 
    ? (isFlood ? 'Disaster Impact Management — Flood Extent' : 'Disaster Impact Management — Compare Before and After') 
    : 'Urban Change Screening — Compare Before and After';
  const subtitleText = isFlood
    ? 'Upload one georeferenced Sentinel-1 GeoTIFF with VV and VH bands. This model estimates water extent from a single observation; it does not by itself compare before and after dates to prove flood-caused change.'
    : isDisaster
      ? 'Upload compatible pre- and post-disaster RGB images. The building-damage model requires a matching image pair.'
      : 'Upload compatible 13-band Sentinel-2 GeoTIFFs. RGB JPG/PNG images are not compatible with this multispectral model.';

  return `
    <div class="compare-page-container">
      
      <!-- Page Title Header -->
      <div class="compare-header">
        <div class="compare-category-tag">${categoryTag}</div>
        <h1 class="compare-title">${mainTitle}</h1>
        <p class="compare-subtitle">
          ${subtitleText}
        </p>
      </div>

      <!-- Upload cards retain the existing visual design but reflect the selected model contract. -->
      <div class="upload-cards-grid" style="${isFlood ? 'grid-template-columns:minmax(0,680px);justify-content:center;' : ''}">
        ${isFlood ? `
          <div class="upload-card">
            <div class="card-tag">SENTINEL-1 INPUT</div>
            <h2 class="card-main-title">Flood Observation</h2>
            <p class="card-sub-desc">Upload a single georeferenced two-band VV/VH SAR GeoTIFF.</p>
            ${renderImageDropzone('before', beforeFile, 'Georeferenced two-band VV/VH GeoTIFF required', '.tif,.tiff')}
          </div>
        ` : `
          <div class="upload-card">
            <div class="card-tag">IMAGE 1</div>
            <h2 class="card-main-title">Before the Event</h2>
            <p class="card-sub-desc">${isDisaster ? 'Pre-disaster RGB image.' : 'Earlier 13-band Sentinel-2 observation.'}</p>
            ${renderImageDropzone('before', beforeFile)}
          </div>
          <div class="upload-card">
            <div class="card-tag">IMAGE 2</div>
            <h2 class="card-main-title">After the Event</h2>
            <p class="card-sub-desc">${isDisaster ? 'Post-disaster RGB image.' : 'Later 13-band Sentinel-2 observation.'}</p>
            ${renderImageDropzone('after', afterFile)}
          </div>
        `}
      </div>

      <!-- Middle Card: Metadata Details & Confirmation -->
      <div class="details-form-card">
        <div class="details-inputs-grid">
          <div>
            <label class="form-field-label">Prediction task</label>
            <select class="form-text-input" id="analysis-model-task" ${!isDisaster ? 'disabled' : ''}>
              ${isDisaster ? `
                <option value="flood_extent" ${modelTask === 'flood_extent' ? 'selected' : ''}>Flood extent — Sentinel-1 VV/VH</option>
                <option value="building_damage" ${modelTask === 'building_damage' ? 'selected' : ''}>Building damage — paired RGB</option>
              ` : `
                <option value="urban_change" ${modelTask === 'urban_change' ? 'selected' : ''}>Urban change (ML Model) — Sentinel-2 13-band</option>
                <option value="urban_change_classical" ${modelTask === 'urban_change_classical' ? 'selected' : ''}>Urban change (Classical Baseline) — Sentinel-2 13-band</option>
              `}
            </select>
          </div>
          <div>
            <label class="form-field-label">Location name (optional)</label>
            <input type="text" class="form-text-input" id="input-location" placeholder="Search a place (Google Places if configured)" value="${esc(state.locationName || '')}" autocomplete="off" />
            <small id="location-selection-status" style="display:block;margin-top:5px;color:#64748b;">${state.locationPlaceId ? `Selected place · ${Number(state.locationLat).toFixed(5)}, ${Number(state.locationLng).toFixed(5)}` : 'A place name is not proof of image geolocation. Configure VITE_GOOGLE_MAPS_API_KEY to enable place autocomplete.'}</small>
          </div>
          ${!isFlood ? `
            <div>
              <label class="form-field-label">Before-image date (optional)</label>
              <input type="date" class="form-text-input" id="input-before-date" value="${esc(state.beforeDate || '')}" />
            </div>
            <div>
              <label class="form-field-label">After-image date (optional)</label>
              <input type="date" class="form-text-input" id="input-after-date" value="${esc(state.afterDate || '')}" />
            </div>
          ` : ''}
          ${modelTask === 'urban_change' ? `
            <div style="grid-column:1/-1;padding:10px;border:1px solid #cbd5e1;border-radius:8px;">
              <label style="display:flex;gap:8px;align-items:flex-start;font-size:12px;color:#334155;">
                <input type="checkbox" id="chk-band-order" ${state.assumeBandOrder ? 'checked' : ''} />
                <span><strong>I verified the 13-band OSCD order.</strong> Use only if authoritative source metadata confirms B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12 in that order. This is an assertion, not automatic verification.</span>
              </label>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-top:10px;">
                <label style="font-size:12px;color:#334155;">Before SCL quality mask (optional GeoTIFF)<input id="before-scl-file" type="file" accept=".tif,.tiff" style="display:block;margin-top:5px;max-width:100%;" />${state.beforeSclFile ? `<small>${esc(state.beforeSclFile.name)} <button type="button" id="remove-before-scl" class="btn btn--ghost btn--sm">Remove</button></small>` : ''}</label>
                <label style="font-size:12px;color:#334155;">After SCL quality mask (optional GeoTIFF)<input id="after-scl-file" type="file" accept=".tif,.tiff" style="display:block;margin-top:5px;max-width:100%;" />${state.afterSclFile ? `<small>${esc(state.afterSclFile.name)} <button type="button" id="remove-after-scl" class="btn btn--ghost btn--sm">Remove</button></small>` : ''}</label>
              </div>
              <small style="display:block;margin-top:5px;color:#64748b;">Supply both SCL masks or neither. They must correspond to the matching Sentinel-2 observations.</small>
            </div>
          ` : modelTask === 'flood_extent' ? `
            <div style="grid-column:1/-1;padding:10px;border:1px solid #cbd5e1;border-radius:8px;">
              <label style="display:flex;gap:8px;align-items:flex-start;font-size:12px;color:#334155;"><input type="checkbox" id="chk-band-order" ${state.assumeBandOrder ? 'checked' : ''} /><span><strong>I verified the two-band VV/VH order</strong> from the Sentinel-1 product metadata. This does not validate radiometric scaling or preprocessing.</span></label>
            </div>
          ` : ''}
          ${modelTask !== 'urban_change_classical' ? `
          <div style="grid-column:1/-1;">
            <label class="form-field-label">Area of Interest boundary (optional GeoJSON)</label>
            <input type="file" id="aoi-geojson-file" accept=".geojson,.json,application/geo+json,application/json" />
            <small style="display:block;color:#64748b;margin-top:4px;">A selected place is not an AOI polygon. Upload a valid GeoJSON geometry to restrict the output area.</small>
            ${state.aoiFile ? `<small>Selected: ${esc(state.aoiFile.name)} <button type="button" id="remove-aoi" class="btn btn--ghost btn--sm">Remove</button></small>` : ''}
          </div>
          ` : ''}
          ${modelTask === 'urban_change' ? `
            <div style="grid-column:1/-1;padding:12px;border:1px solid #7c3aed66;border-radius:8px;background:#7c3aed0a;">
              <strong style="display:block;font-size:13px;color:#ddd6fe;margin-bottom:4px;">Reference-zone comparison (optional; separate from AOI)</strong>
              <p style="font-size:12px;color:#94a3b8;margin:0 0 10px;">Upload a GeoJSON Polygon/MultiPolygon for a target zone such as a road reserve or protected area. This overlay is not the AOI clipping geometry and does not independently verify legal authority, ownership or illegality.</p>
              <label class="form-field-label">Reference-zone polygons (WGS84 RFC 7946 GeoJSON)</label>
              <input type="file" id="reference-boundary-geojson-file" accept=".geojson,.json,application/geo+json,application/json" />
              ${state.referenceBoundaryFile ? `<small>Selected: ${esc(state.referenceBoundaryFile.name)} <button type="button" id="remove-reference-boundary" class="btn btn--ghost btn--sm">Remove</button></small>` : ''}
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-top:10px;">
                <label class="form-field-label">Source / dataset name<input type="text" class="form-text-input" id="boundary-source-name" value="${esc(state.boundarySourceName || '')}" placeholder="e.g. municipal road reserve layer" /></label>
                <label class="form-field-label">Boundary effective date (if known)<input type="date" class="form-text-input" id="boundary-effective-date" value="${esc(state.boundaryEffectiveDate || '')}" /></label>
                <label class="form-field-label">Near-boundary review distance (metres)<input type="number" class="form-text-input" id="boundary-proximity-distance" min="0" max="1000" step="1" value="${Number.isFinite(state.boundaryProximityM) ? state.boundaryProximityM : 10}" /></label>
              </div>
              <small style="display:block;color:#fbbf24;margin-top:8px;">Outputs say “boundary-overlap candidate” or “near-boundary candidate”—never “illegal encroachment confirmed”.</small>
            </div>
          ` : ''}
        </div>

        <!-- Confirmation Box -->
        ${!isFlood ? `
          <div class="confirm-box">
            <label class="confirm-checkbox-label">
              <input type="checkbox" id="chk-confirm-area" ${state.confirmedArea ? 'checked' : ''} />
              <span class="confirm-checkbox-text"><strong>I confirm these images show approximately the same geographic area.</strong></span>
            </label>
            <p class="confirm-sub-text">This is user-provided information, not automatic geographic verification.</p>
          </div>
        ` : ''}
      </div>

      ${state.analysisError ? `<div class="error-banner error-banner--info" style="margin-top:12px;"><div class="error-banner__text">${esc(state.analysisError)}</div></div>` : ''}

      ${state.previewImagesActive ? `
        <div class="preview-images-container" style="margin-bottom: 20px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #f8fafc;">
          <div style="margin-bottom: 12px; color: #334155; font-size: 14px; text-align: center;">
            <strong>Note:</strong> These preview images are illustrative RGB JPEGs and are not compatible with the required ${isFlood ? 'Sentinel-1 SAR' : modelTask === 'building_damage' ? 'paired RGB' : '13-band Urban Change'} input profile. ${modelTask === 'building_damage' ? 'Real analysis requires properly paired RGB image inputs.' : 'Real analysis requires georeferenced Sentinel GeoTIFF inputs.'}
          </div>
          <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
            ${isFlood ? `
            <div style="flex: 1; max-width: 500px;">
               <img src="/satellite-after.jpg" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);" alt="SAR preview" />
            </div>
            ` : `
            <div style="flex: 1; max-width: 500px;">
               <p style="font-size: 12px; font-weight: bold; margin-bottom: 4px; text-align: center;">Before</p>
               <img src="/demo-before.jpg" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);" alt="Before preview" />
            </div>
            <div style="flex: 1; max-width: 500px;">
               <p style="font-size: 12px; font-weight: bold; margin-bottom: 4px; text-align: center;">After</p>
               <img src="/demo-after.jpg" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);" alt="After preview" />
            </div>
            `}
          </div>
        </div>
      ` : ''}

      <!-- Footer Action Bar -->
      <div class="footer-action-bar">
        <div style="display: flex; align-items: center; gap: 12px;">
          <button class="btn btn--secondary btn--sm" id="btn-load-sample-images" style="border-radius: 99px;">
            ${state.previewImagesActive ? 'Hide Preview Images' : '📷 View Preview Images'}
          </button>
          <span class="footer-hint-text">
            ${bothUploaded ? (isFlood ? 'Input selected. Model compatibility will be checked.' : 'Both images selected. Model compatibility will be checked.') : (isFlood ? 'Select one Sentinel-1 GeoTIFF input.' : 'Select both images to continue.')}
          </span>
        </div>

        <button class="btn-analyze" id="btn-run-analysis" ${!bothUploaded ? 'disabled' : ''}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          Analyze Changes
        </button>
      </div>

    </div>
  `;
}

function renderOverlapValidationBanner(state: any): string {
  const overlap = state.overlapValidation;
  if (!overlap || overlap.status === 'pending') {
    return '';
  }

  if (overlap.status === 'verified') {
    return `
      <div class="warning-banner-box" style="background-color: #ecfdf5; border-color: #10b981;">
        <div class="warning-banner-icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
          </svg>
        </div>
        <div>
          <h4 class="warning-banner-title" style="color: #065f46;">Geographic bounds available.</h4>
          <p class="warning-banner-desc" style="color: #047857;">
            ${esc(overlap.message || (typeof overlap.percentage === 'number' ? `Measured footprint overlap: ${overlap.percentage}%.` : 'Map coordinates are available from raster metadata. Content registration and prediction accuracy remain separate checks.'))}
          </p>
        </div>
      </div>
    `;
  }

  if (overlap.status === 'insufficient') {
    return `
      <div class="warning-banner-box" style="background-color: #fef2f2; border-color: #ef4444;">
        <div class="warning-banner-icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
        </div>
        <div>
          <h4 class="warning-banner-title" style="color: #991b1b;">Insufficient Geographic Overlap</h4>
          <p class="warning-banner-desc" style="color: #b91c1c;">
            The uploaded images have a measured geographic overlap of ${overlap.percentage}%, which does not meet the configured 80% minimum.
          </p>
        </div>
      </div>
    `;
  }

  return `
    <div class="warning-banner-box">
      <div class="warning-banner-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      </div>
      <div>
        <h4 class="warning-banner-title">Unable to Verify Geographic Overlap</h4>
        <p class="warning-banner-desc">
          The selected images do not provide sufficient supported georeferencing information to verify their geographic bounds. Matching dimensions and readable images do not prove geographic overlap. The 80% minimum cannot be enforced until geographic overlap can be measured reliably. Upload georeferenced GeoTIFFs or provide supported georeferencing metadata.
        </p>
      </div>
    </div>
  `;
}

function renderImageDropzone(role: string, file: FileUpload | null, formatText: string = 'JPG, PNG or GeoTIFF when supported', acceptAttr: string = 'image/*,.tif,.tiff'): string {
  if (file && file.status === 'ready') {
    return `
      <div class="dropzone-dashed dropzone--ready" data-dropzone="${role}">
        <div style="position: absolute; top: 12px; right: 12px; z-index: 10;">
          <button data-remove-file="${role}" style="background: #ffffff; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; color: #ef4444; border: 1px solid #fca5a5; cursor: pointer;">
            ✕
          </button>
        </div>
        <div class="upload-circle-icon" style="background: #ecfdf5; color: #10b981;">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        </div>
        <div style="font-size: 15px; font-weight: 700; color: #0f2942; margin-bottom: 2px;">${esc(file.name)}</div>
        <div style="font-size:13px;color:#10b981;font-weight:600;">Inspected • Ready for model-specific checks</div>
      </div>
    `;
  }

  if (file && file.status === 'error') {
    return `
      <div class="dropzone-dashed" data-dropzone="${role}" tabindex="0" role="button" style="border-color:#ef4444;">
        <div style="font-size:14px;font-weight:700;color:#b91c1c;">Inspection failed</div>
        <div style="font-size:12px;color:#7f1d1d;margin:6px 0;">${esc(file.name)}</div>
        <div style="font-size:12px;color:#7f1d1d;">${esc(file.error || 'This file is not ready for analysis.')}</div>
        <div style="font-size:12px;color:#64748b;margin-top:8px;">Click to choose another file</div>
        <input type="file" data-file-input="${role}" style="display:none;" accept="${esc(acceptAttr)}" />
      </div>
    `;
  }

  if (file && file.status === 'uploading') {
    return `
      <div class="dropzone-dashed" data-dropzone="${role}">
        <div style="width: 32px; height: 32px; border: 3px solid #e2e8f0; border-top-color: #009688; border-radius: 50%; animation: spin 0.8s linear infinite; margin-bottom: 12px;"></div>
        <div style="font-size: 14px; font-weight: 700; color: #0f2942;">Uploading ${esc(file.name)}...</div>
        <div style="width: 60%; height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: 10px; overflow: hidden;">
          <div style="width: ${file.progress}%; height: 100%; background: #009688; transition: width 0.2s;"></div>
        </div>
      </div>
    `;
  }

  return `
    <div class="dropzone-dashed" data-dropzone="${role}" tabindex="0" role="button">
      <div class="upload-circle-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div style="font-size: 15px; font-weight: 700; color: #0f2942; margin-bottom: 2px;">Drop an image here</div>
      <div style="font-size: 13px; color: #64748b; margin-bottom: 6px;">or click to browse your device</div>
      <div style="font-size: 12px; color: #94a3b8;">${esc(formatText)}</div>
      <input type="file" data-file-input="${role}" style="display:none;" accept="${esc(acceptAttr)}" />
    </div>
  `;
}

// ── Step 4: Pipeline Execution Render ────────────────────────────────

function renderRunAnalysis(): string {
  const state = getState();
  const steps = state.processingSteps;
  const status = state.analysisStatus;

  const doneCount = steps.filter(s => s.status === 'done').length;
  const totalCount = steps.length;
  const pct = totalCount > 0 ? Math.round((doneCount / totalCount) * 100) : 0;
  const allDone = status === 'complete';
  const failed = status === 'failed';

  return `
    <div class="compare-page-container">
      <div class="compare-header">
        <div class="compare-category-tag">PROCESSING PIPELINE</div>
        <h1 class="compare-title">Executing Analysis Pipeline</h1>
        <p class="compare-subtitle">
          ${allDone ? 'Model inference completed. Review the actual mask and QA report.' : failed ? 'Analysis stopped; no prediction was fabricated.' : 'Validating inputs and running the selected model when an evaluated checkpoint is available...'}
        </p>
      </div>

      <div class="details-form-card" style="max-width: 680px; margin: 0 auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h3 style="font-size: 16px; font-weight: 700; color: #0f2942;">Analysis Pipeline Status</h3>
          <span style="font-size: 13px; font-weight: 700; color: #009688;">${pct}% Complete</span>
        </div>

        <div style="width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; margin-bottom: 24px;">
          <div style="width: ${pct}%; height: 100%; background: #009688; transition: width 0.3s ease;"></div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 12px;">
          ${steps.map(s => `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #f8fafc; border-radius: 8px; border: 1px solid #f1f5f9;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <span style="width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; ${s.status === 'done' ? 'background: #ecfdf5; color: #10b981;' : s.status === 'running' ? 'background: #e0f2fe; color: #0284c7;' : 'background: #e2e8f0; color: #94a3b8;'}">
                  ${s.status === 'done' ? '✓' : s.status === 'running' ? '●' : '○'}
                </span>
                <span style="font-size: 14px; font-weight: 600; color: ${s.status === 'pending' ? '#94a3b8' : '#0f2942'};">${esc(s.label)}</span>
              </div>
              ${s.duration ? `<span style="font-size: 12px; color: #64748b; font-family: var(--font-mono);">${esc(s.duration)}</span>` : ''}
            </div>
          `).join('')}
        </div>

        ${failed && state.analysisError ? `<div class="error-banner error-banner--info" style="margin-top:18px;"><div class="error-banner__title">No prediction produced</div><div class="error-banner__text">${esc(state.analysisError)}</div></div>` : ''}
        ${state.analysisPreflight ? `<div class="card" style="margin-top:14px;padding:14px;"><div style="font-weight:700;font-size:13px;">Input quality-gate result: ${esc(String(state.analysisPreflight.status || 'Not reported'))}</div><p style="font-size:12px;color:var(--text-secondary);margin:6px 0;">Preflight is separate from model readiness and does not establish prediction accuracy or verify the legal status of a reference boundary.</p>${Array.isArray(state.analysisPreflight.hard_errors) && state.analysisPreflight.hard_errors.length ? `<ul style="font-size:12px;color:var(--warning);padding-left:18px;">${state.analysisPreflight.hard_errors.map((message:string)=>`<li>${esc(message)}</li>`).join('')}</ul>` : ''}${Array.isArray(state.analysisPreflight.warnings) && state.analysisPreflight.warnings.length ? `<details><summary style="font-size:12px;cursor:pointer;">${state.analysisPreflight.warnings.length} preflight warning(s)</summary><ul style="font-size:12px;color:var(--text-secondary);padding-left:18px;">${state.analysisPreflight.warnings.map((message:string)=>`<li>${esc(message)}</li>`).join('')}</ul></details>` : ''}${state.analysisPreflight.report_url ? `<a class="btn btn--secondary btn--sm" href="${esc(state.analysisPreflight.report_url)}" target="_blank" rel="noopener noreferrer">Open QA report</a>` : ''}</div>` : ''}
        <div style="margin-top: 32px; display:flex; justify-content:space-between; gap:12px;">
          ${failed ? '<button class="btn btn--secondary" id="wizard-back-inputs">Back to inputs</button>' : '<span></span>'}
          ${allDone ? '<button class="btn-analyze" id="wizard-next-4">Explore Map Results &rarr;</button>' : ''}
        </div>
      </div>
    </div>
  `;
}

function renderReviewExport(): string {
  const state = getState();
  return `
    <div class="compare-page-container">
      <div class="details-form-card" style="max-width: 680px; margin: 40px auto; padding: 48px 32px;">
        <h2 style="font-size: 26px; font-weight: 800; color: #0f2942; margin-bottom: 24px; text-align: center;">Prediction Pipeline Complete</h2>
        
        ${renderOverlapValidationBanner(state)}
        
        <div style="margin-top: 32px;">
          <button class="btn-analyze" id="open-results-map" style="width: 100%; justify-content: center; font-size: 16px;">
            Continue to Map Viewer
          </button>
        </div>
      </div>
    </div>
  `;
}

// ── Event Handlers ───────────────────────────────────────────────────

export function bindNewAnalysisEvents(): void {
  // Dropzones
  document.querySelectorAll<HTMLElement>('[data-dropzone]').forEach(el => {
    const role = el.dataset.dropzone!;
    const input = document.querySelector<HTMLInputElement>(`[data-file-input="${role}"]`);
    if (!input) return;

    el.addEventListener('click', (e) => {
      if (e.target === input) return;
      if ((e.target as HTMLElement).closest('[data-remove-file]')) return;
      input.click();
    });
    el.addEventListener('keydown', (e) => {
      if ((e as KeyboardEvent).key === 'Enter' || (e as KeyboardEvent).key === ' ') {
        e.preventDefault(); input.click();
      }
    });

    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (file) simulateFileUpload(role, file);
    });

    el.addEventListener('dragover', (e) => { e.preventDefault(); });
    el.addEventListener('drop', (e) => {
      e.preventDefault();
      const file = (e as DragEvent).dataTransfer?.files[0];
      if (file) simulateFileUpload(role, file);
    });
  });

  // Remove File
  document.querySelectorAll<HTMLElement>('[data-remove-file]').forEach(el => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const role = el.dataset.removeFile as string;
      setState({ [role + 'File']: null } as any);
    });
  });

  // Sample images are for UI preview only. They are not georeferenced model inputs.
  document.getElementById('btn-load-sample-images')?.addEventListener('click', () => {
    setState({ previewImagesActive: !getState().previewImagesActive });
  });

  // Form inputs. Use change (not input) for the app-wide rerender so typing does not lose focus.
  document.getElementById('analysis-model-task')?.addEventListener('change', (e) => {
    const value = (e.target as HTMLSelectElement).value as 'urban_change' | 'flood_extent' | 'building_damage';
    setState({ analysisModelTask: value, assumeBandOrder: false, beforeFile: null, afterFile: null, beforeSclFile: null, afterSclFile: null, referenceBoundaryFile: null, boundarySourceName: '', boundaryEffectiveDate: '', analysisResult: null, analysisPreflight: null, analysisError: null, analysisStatus: 'draft', wizardStep: 0, beforeDate: '', afterDate: '', selectedRegionId: null, detailsPanelOpen: false });
  });
  document.getElementById('chk-band-order')?.addEventListener('change', (e) => {
    setState({ assumeBandOrder: (e.target as HTMLInputElement).checked });
  });
  document.getElementById('before-scl-file')?.addEventListener('change', (e) => {
    setState({ beforeSclFile: (e.target as HTMLInputElement).files?.[0] || null });
  });
  document.getElementById('after-scl-file')?.addEventListener('change', (e) => {
    setState({ afterSclFile: (e.target as HTMLInputElement).files?.[0] || null });
  });
  document.getElementById('aoi-geojson-file')?.addEventListener('change', (e) => {
    setState({ aoiFile: (e.target as HTMLInputElement).files?.[0] || null });
  });
  document.getElementById('reference-boundary-geojson-file')?.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0] || null;
    if (file && file.size > 10 * 1024 * 1024) {
      setState({ analysisError: 'Reference-zone GeoJSON must be 10 MiB or smaller.', referenceBoundaryFile: null });
      return;
    }
    setState({ referenceBoundaryFile: file, analysisError: null });
  });
  document.getElementById('boundary-source-name')?.addEventListener('change', (e) => setState({ boundarySourceName: (e.target as HTMLInputElement).value.slice(0, 160) }));
  document.getElementById('boundary-effective-date')?.addEventListener('change', (e) => setState({ boundaryEffectiveDate: (e.target as HTMLInputElement).value }));
  document.getElementById('boundary-proximity-distance')?.addEventListener('change', (e) => {
    const value = Number((e.target as HTMLInputElement).value);
    if (!Number.isFinite(value) || value < 0 || value > 1000) {
      setState({ analysisError: 'Boundary proximity distance must be between 0 and 1000 metres.' });
      return;
    }
    setState({ boundaryProximityM: value, analysisError: null });
  });
  document.getElementById('remove-reference-boundary')?.addEventListener('click', () => setState({ referenceBoundaryFile: null }));
  document.getElementById('remove-before-scl')?.addEventListener('click', () => setState({ beforeSclFile: null }));
  document.getElementById('remove-after-scl')?.addEventListener('click', () => setState({ afterSclFile: null }));
  document.getElementById('remove-aoi')?.addEventListener('click', () => setState({ aoiFile: null }));
  const locationInput = document.getElementById('input-location') as HTMLInputElement | null;
  locationInput?.addEventListener('change', () => {
    const current = getState();
    if (locationInput.value !== current.locationName) setState({ locationName: locationInput.value, locationPlaceId: '', locationLat: undefined, locationLng: undefined });
  });
  if (locationInput) attachGooglePlaces(locationInput);
  document.getElementById('input-before-date')?.addEventListener('change', (e) => {
    setState({ beforeDate: (e.target as HTMLInputElement).value });
  });
  document.getElementById('input-after-date')?.addEventListener('change', (e) => {
    setState({ afterDate: (e.target as HTMLInputElement).value });
  });
  document.getElementById('chk-confirm-area')?.addEventListener('change', (e) => {
    setState({ confirmedArea: (e.target as HTMLInputElement).checked });
  });

  // Action button: Analyze Changes
  document.getElementById('btn-run-analysis')?.addEventListener('click', () => {
    setState({ wizardStep: 4, analysisStatus: 'running', analysisError: null, analysisResult: null, analysisPreflight: null });
    void startProcessing();
  });

  document.getElementById('wizard-next-4')?.addEventListener('click', () => {
    setState({ currentPage: 'results' });
  });

  document.getElementById('open-results-map')?.addEventListener('click', () => {
    setState({ currentPage: 'results' });
  });
  document.getElementById('wizard-back-inputs')?.addEventListener('click', () => {
    setState({ wizardStep: 0, analysisStatus: 'draft' });
  });
}

async function simulateFileUpload(role: string, file: File): Promise<void> {
  const upload: FileUpload = {
    id: generateId(), name: file.name, size: file.size, type: file.type || 'application/octet-stream',
    status: 'uploading', progress: 30, file, previewUrl: URL.createObjectURL(file),
  };
  const key = role + 'File';
  setState({ [key]: upload, analysisError: null } as any);

  const apiReport = await inspectImageApi(file);

  const currentState = getState() as any;
  if (currentState[key]?.id !== upload.id) {
    return; // File was removed or replaced during inspection
  }

  if (!apiReport) {
    setState({ [key]: { ...upload, status: 'error', progress: 100, error: 'Backend inspection failed or is unavailable. Start the backend and upload again.' } } as any);
    return;
  }
  if (String((apiReport as any).status || '').toUpperCase() === 'REJECT') {
    setState({ [key]: { ...upload, status: 'error', progress: 100, error: (apiReport as any).checks?.find((c: any) => c.status === 'REJECT')?.message || 'Image inspection rejected this file.' } } as any);
    return;
  }
  const dimensions = apiReport.dimensions ? `${apiReport.dimensions.width} × ${apiReport.dimensions.height}` : 'Unknown';
  const metadata = (apiReport as any).metadata || {};
  const pixelSize = apiReport.pixel_size || metadata.resolution;
  const meta: FileMetadata = {
    filename: file.name, dimensions, crs: apiReport.crs || metadata.crs || 'Not available',
    resolution: Array.isArray(pixelSize) ? pixelSize.join(' × ') : (pixelSize ? String(pixelSize) : 'Not available'),
    bands: apiReport.bands_count ? `${apiReport.bands_count} band(s) (${apiReport.format || 'detected image'})` : 'Not available',
    nodata: metadata.nodata !== undefined && metadata.nodata !== null ? String(metadata.nodata) : 'Not available',
    dateAcquired: metadata.acquisition_date || 'Not available', fileSize: formatBytes(file.size),
  };
  setState({ [key]: { ...upload, status: 'ready', progress: 100, metadata: meta, file } } as any);
}

function attachGooglePlaces(input: HTMLInputElement): void {
  const apiKey = String((import.meta as any).env?.VITE_GOOGLE_MAPS_API_KEY || '').trim();
  if (!apiKey || input.dataset.placesAttached === 'true') return;
  const win = window as any;
  const attach = () => {
    const PlaceAutocompleteElement = win.google?.maps?.places?.PlaceAutocompleteElement;
    if (!PlaceAutocompleteElement || input.dataset.placesAttached === 'true') return;
    input.dataset.placesAttached = 'true';
    const widget = new PlaceAutocompleteElement();
    widget.setAttribute('aria-label', 'Search Google Maps places');
    widget.setAttribute('placeholder', 'Search a city, address, village or landmark');
    widget.style.width = '100%';
    widget.style.display = 'block';
    widget.style.minHeight = '42px';
    widget.style.marginTop = '4px';
    try { widget.includedRegionCodes = ['in']; } catch { /* country restriction is a UX hint, not a validation gate */ }
    input.style.display = 'none';
    input.parentElement?.insertBefore(widget, input);
    widget.addEventListener('gmp-select', async (event: any) => {
      try {
        const prediction = event.placePrediction;
        if (!prediction) throw new Error('No place prediction was returned.');
        const place = prediction.toPlace();
        await place.fetchFields({ fields: ['id', 'displayName', 'formattedAddress', 'location', 'viewport'] });
        const location = place.location;
        const lat = typeof location?.lat === 'function' ? location.lat() : location?.lat;
        const lng = typeof location?.lng === 'function' ? location.lng() : location?.lng;
        if (typeof lat !== 'number' || typeof lng !== 'number') throw new Error('Selected place has no usable coordinates. Choose a more specific suggestion.');
        const name = place.formattedAddress || place.displayName?.text || place.displayName || input.value;
        setState({ locationName: String(name), locationPlaceId: place.id || '', locationLat: lat, locationLng: lng, analysisError: null });
      } catch (error) {
        setState({ analysisError: error instanceof Error ? error.message : 'Google Places could not resolve the selected location.' });
      }
    });
  };
  if (win.google?.maps?.places?.PlaceAutocompleteElement) { attach(); return; }
  if (!win.__chronoPlacesLoader) {
    win.__chronoPlacesLoader = new Promise<void>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&libraries=places&loading=async&v=weekly`;
      script.async = true; script.defer = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Google Maps JavaScript API failed to load'));
      document.head.appendChild(script);
    });
  }
  Promise.resolve(win.__chronoPlacesLoader).then(attach).catch(() => {
    const status = document.getElementById('location-selection-status');
    if (status) status.textContent = 'Google Places could not load; manual text is available but remains unverified.';
  });
}

async function startProcessing(): Promise<void> {
  const initial = getState();
  const task = initial.analysisModelTask;
  const steps: ProcessingStep[] = [
    { id: 'preflight', label: 'Inspecting image(s) and checking task compatibility', status: 'running' },
    { id: 'inference', label: 'Running the matching trained ML checkpoint', status: 'pending' },
    { id: 'outputs', label: 'Preparing prediction mask and QA report', status: 'pending' },
  ];
  setState({ processingSteps: steps, analysisStatus: 'running', analysisError: null, analysisResult: null, changeRegions: [] });

  const before = initial.beforeFile?.file;
  const after = initial.afterFile?.file;
  if (task === 'flood_extent' && !before) {
    setState({ analysisStatus: 'failed', analysisError: 'Upload one Sentinel-1 VV/VH GeoTIFF for flood extent mapping.', processingSteps: [{ ...steps[0], status: 'error' }] });
    return;
  }
  if (task !== 'flood_extent' && (!before || !after)) {
    setState({ analysisStatus: 'failed', analysisError: 'Upload both required images before starting prediction.', processingSteps: [{ ...steps[0], status: 'error' }] });
    return;
  }
  if (task === 'urban_change' && Boolean(initial.beforeSclFile) !== Boolean(initial.afterSclFile)) {
    setState({ analysisStatus: 'failed', analysisError: 'Supply both before and after SCL masks, or remove both.', processingSteps: [{ ...steps[0], status: 'error' }] });
    return;
  }
  if (task !== 'flood_extent' && initial.beforeDate && initial.afterDate && initial.afterDate <= initial.beforeDate) {
    setState({ analysisStatus: 'failed', analysisError: 'Chronology is invalid: the after-image date must be later than the before-image date.', processingSteps: [{ ...steps[0], status: 'error' }] });
    return;
  }
  const req: any = task === 'flood_extent'
    ? { task, inputFile: before, inputName: initial.beforeFile?.name, aoiFile: initial.aoiFile, assumeBandOrder: initial.assumeBandOrder }
    : { task, beforeFile: before, afterFile: after, beforeName: initial.beforeFile?.name, afterName: initial.afterFile?.name, aoiFile: initial.aoiFile, beforeSclFile: initial.beforeSclFile, afterSclFile: initial.afterSclFile, beforeDate: initial.beforeDate, afterDate: initial.afterDate, locationName: initial.locationName, locationPlaceId: initial.locationPlaceId, locationLat: initial.locationLat, locationLng: initial.locationLng, assumeBandOrder: initial.assumeBandOrder,
        referenceBoundaryFile: task === 'urban_change' ? initial.referenceBoundaryFile : null,
        boundarySourceName: initial.boundarySourceName, boundaryEffectiveDate: initial.boundaryEffectiveDate,
        boundaryProximityM: initial.boundaryProximityM };

  try {
    const preflight = await preflightPredictionApi(req);
    setState({ analysisPreflight: preflight });
    if (preflight?.status === 'REJECT') {
      const reasons = (preflight.hard_errors || []).join(' ');
      throw new Error(reasons || 'The quality gate rejected the input. Review the preflight report before trying again.');
    }
    steps[0] = { ...steps[0], status: 'done', duration: 'Passed with review' };
    setState({ processingSteps: [...steps] });

    if (task !== 'urban_change_classical') {
      const modelStatusResponse = await getModelStatusApi();
      const modelStatus = modelStatusResponse?.models?.[task];
      if (modelStatus?.status !== 'APPROVED_FOR_INFERENCE') {
        const statusLabel = typeof modelStatus?.status === 'string' ? modelStatus.status : 'MODEL_STATUS_UNAVAILABLE';
        steps[1] = { ...steps[1], status: 'error', duration: 'Inference not started' };
        setState({
          processingSteps: [...steps], analysisStatus: 'failed', analysisError: `Input preflight completed (${preflight.status || 'status not reported'}), but the ${task} model is not approved for inference (${statusLabel}). No prediction was produced. Open Models & Data for the model-evidence status; install a hash-bound held-out evaluation report and explicit review approval before running inference.`,
          analysisPreflight: preflight, wizardStep: 4, analysisResult: null, changeRegions: [],
        });
        return;
      }
    }

    steps[1] = { ...steps[1], status: 'running' };
    setState({ processingSteps: [...steps] });
    const result = await runPredictionApi(req);
    steps[1] = { ...steps[1], status: 'done', duration: 'Inference completed' };
    steps[2] = { ...steps[2], status: 'done', duration: 'Artifacts saved' };
    const boundsArray = result.bounds;
    const bounds = Array.isArray(boundsArray) && boundsArray.length === 2
      ? { left: boundsArray[0][1], bottom: boundsArray[0][0], right: boundsArray[1][1], top: boundsArray[1][0] }
      : undefined;
    const overlapValidation = result.geographic_overlay_verified && bounds
      ? { status: 'verified' as const, percentage: undefined, bounds, message: 'Georeferenced overlay bounds are available; content registration still depends on the model preflight report.' }
      : { status: 'unverifiable' as const, message: 'Geographic placement is not verified. Results will be shown in image space.' };
    // A Google Places point is compared with actual raster bounds when available. It is a
    // point-in-footprint check, not proof of the exact acquisition location or a drawn AOI.
    if (typeof initial.locationLat === 'number' && typeof initial.locationLng === 'number') {
      const b = result.bounds;
      if (Array.isArray(b) && b.length === 2) {
        const [southWest, northEast] = b;
        const inside = initial.locationLat >= southWest[0] && initial.locationLat <= northEast[0] && initial.locationLng >= southWest[1] && initial.locationLng <= northEast[1];
        result.location_footprint_check = { status: inside ? 'PASS' : 'WARNING', place_id: initial.locationPlaceId || null, latitude: initial.locationLat, longitude: initial.locationLng, inside_raster_bounds: inside, note: 'Point-in-bounding-box only; it does not verify image identity or AOI polygon coverage.' };
        if (!inside) result.warnings = [...(result.warnings || []), 'The selected Google Places point falls outside the model input raster bounds. Confirm the location and uploaded imagery.'];
      } else {
        result.location_footprint_check = { status: 'NOT_VERIFIED', place_id: initial.locationPlaceId || null, latitude: initial.locationLat, longitude: initial.locationLng, note: 'Uploaded imagery has no verified geographic bounds.' };
        result.warnings = [...(result.warnings || []), 'The selected location could not be checked against georeferenced raster bounds.'];
      }
    }
    const countValues = Object.values(result.predicted_class_pixel_counts || {}).map((v: any) => Number(v) || 0);
    const historyRecord = {
      id: result.session_id || generateId(),
      name: `${task.replaceAll('_', ' ')} · ${initial.locationName || 'Unspecified location'}`,
      task, status: 'complete' as const, createdAt: new Date().toISOString(),
      region: initial.locationName || undefined, warningCount: Array.isArray(result.warnings) ? result.warnings.length : 0,
      resultCount: countValues.reduce((sum, value) => sum + value, 0), result,
    };
    const history = [historyRecord, ...getState().analysisHistory.filter((r: any) => r.id !== historyRecord.id)].slice(0, 50);
    setState({
      processingSteps: steps, analysisStatus: 'complete', analysisResult: result, analysisError: null,
      overlapValidation, changeRegions: [], wizardStep: 5, analysisHistory: history,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const last = steps.findIndex(s => s.status === 'running');
    if (last >= 0) steps[last] = { ...steps[last], status: 'error' };
    setState({ processingSteps: steps, analysisStatus: 'failed', analysisError: message, analysisResult: null, wizardStep: 4, changeRegions: [] });
  }
}
