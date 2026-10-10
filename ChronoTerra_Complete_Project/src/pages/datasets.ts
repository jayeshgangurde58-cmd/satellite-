// ===================================================================
// ChronoTerra — Datasets & Evaluation Page
// ===================================================================

import { icons } from '../icons';
import { getModelStatusApi } from '../api';
import { esc, Badge, Banner, EmptyState } from '../components';

export function renderDatasets(): string {
  return `
    <div class="page-container">
      <div class="page-header">
        <div>
          <h1 class="page-title">Datasets & Evaluation</h1>
          <p class="page-subtitle">Manage labelled datasets and review benchmark evaluation results.</p>
        </div>
      </div>

      ${Banner('info', 'Evaluation context',
        'This page is for technical evaluation using labelled change-detection datasets (e.g., OSCD). Keep generic urban-change results separate from disaster-specific results.'
      )}

      <div style="margin-top: var(--sp-6);">
        <div class="card">
          <div class="card__header">
            <div class="card__title">Available Datasets</div>
          </div>
          <div class="card__body" style="padding: 0;">
            ${renderDatasetRow('OSCD (Onera Satellite Change Detection)', 'Candidate public benchmark; local dataset inventory not scanned', 'reference-only', 'Not verified', 'Not verified', 'Not run')}
            ${renderDatasetRow('Sen1Floods11 (Sentinel-1 flood water)', 'Candidate flood-water benchmark; verify label subset, licence and local files before training', 'reference-only', 'Not verified', 'Not verified', 'Not run')}
            ${renderDatasetRow('xBD (xView2 Building Damage)', 'Candidate damage benchmark; confirm access and licence before use', 'reference-only', 'Not verified', 'Not verified', 'Not run')}
            ${renderDatasetRow('S2Looking', 'Candidate optical change benchmark; local dataset inventory not scanned', 'reference-only', 'Not verified', 'Not verified', 'Not run')}
          </div>
        </div>
      </div>

      <div style="margin-top: var(--sp-6);">
        <div class="card">
          <div class="card__header"><div class="card__title">Installed Model Checkpoints</div></div>
          <div class="card__body" id="model-checkpoint-status"><p style="color:var(--text-secondary);">Checking backend checkpoint availability…</p></div>
        </div>
      </div>

      <div style="margin-top: var(--sp-6);">
        <div class="card">
          <div class="card__header">
            <div class="card__title">Evaluation Results</div>
            ${Badge('neutral', 'Only manifest-backed metrics')}
          </div>
          <div class="card__body" id="model-evaluation-reports">
            ${EmptyState(
              icons.barChart,
              'No validated evaluation manifest',
              'Run evaluation on labelled held-out scenes and add a checkpoint-hash-bound deployment manifest. This panel remains empty until verified metrics are exposed by the backend.'
            )}
          </div>
        </div>
      </div>



      <div style="margin-top: var(--sp-6);">
        <div class="card">
          <div class="card__header">
            <div class="card__title">Evaluation Metrics (when available)</div>
          </div>
          <div class="card__body">
            <p style="font-size: var(--text-sm); color: var(--text-secondary); margin-bottom: var(--sp-4);">
              When an evaluation is completed, the following metrics will be reported:
            </p>
            <div class="data-table-wrapper">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Description</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td>Precision</td><td>Fraction of predicted change pixels that are true positives</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                  <tr><td>Recall</td><td>Fraction of true change pixels that are correctly detected</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                  <tr><td>F1 Score</td><td>Harmonic mean of precision and recall</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                  <tr><td>IoU</td><td>Intersection over Union of change regions</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                  <tr><td>FPR</td><td>Conventional false-positive rate</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                  <tr><td>FP Fraction</td><td>False-positive fraction among valid pixels</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                  <tr><td>Valid Coverage</td><td>Fraction of evaluation area with valid observations</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                  <tr><td>Runtime</td><td>Total processing time for the evaluation</td><td class="cell-mono" style="color: var(--text-tertiary);">Not evaluated</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderDatasetRow(name: string, desc: string, status: string, pairs: string, split: string, lastRun: string): string {
  const visibleStatus = status === 'reference-only' ? 'Candidate reference' : status === 'not-acquired' ? 'Local data not verified' : status;
  return `
    <div class="qc-row">
      <div class="qc-row__icon qc-row__icon--pending" style="font-size: 14px;">${icons.database}</div>
      <div class="qc-row__content">
        <div class="qc-row__title">${esc(name)}</div>
        <div class="qc-row__detail">${esc(desc)} · Pairs: ${pairs} · Split: ${split} · Last run: ${lastRun}</div>
      </div>
      <div>
        ${Badge('neutral', visibleStatus === 'ready' ? 'Ready' : visibleStatus)}
      </div>
    </div>
  `;
}

export function bindDatasetsEvents(): void {
  const target = document.getElementById('model-checkpoint-status');
  if (!target) return;
  void getModelStatusApi().then((data) => {
    const models = data?.models || {};
    const labels: Record<string,string> = { urban_change:'Urban change · 13-band Sentinel-2', flood_extent:'Flood extent · Sentinel-1 VV/VH', building_damage:'Building damage · paired RGB' };
    target.innerHTML = Object.keys(labels).map(key => {
      const item = models[key] || {};
      const available = Boolean(item.checkpoint_exists);
      const approved = item.status === 'APPROVED_FOR_INFERENCE';
      const status = approved ? 'Approved for inference; check report and scene coverage' : !available ? 'Checkpoint missing — inference disabled' : item.status === 'NEEDS_EVIDENCE_REVIEW' ? 'Checkpoint present, but held-out evidence/review gate is not approved' : 'Evidence not verified — inference disabled';
      const color = approved ? 'var(--success)' : available ? 'var(--warning)' : 'var(--error)';
      const reference = item.checkpoint_reference ? esc(item.checkpoint_reference) : 'Expected checkpoint reference not available';
      return `<div class="qc-row"><div class="qc-row__icon qc-row__icon--pending">${icons.database}</div><div class="qc-row__content"><div class="qc-row__title">${labels[key]}</div><div class="qc-row__detail">${esc(status)}</div><div class="qc-row__detail">Expected relative reference: <code>${reference}</code></div><div class="qc-row__detail">Held-out metrics file: ${item.held_out_test_metrics_exists ? 'present (still requires validated manifest and review)' : 'not available'}</div></div><div style="font-size:12px;color:${color};font-weight:700;">${approved ? 'APPROVED' : available ? 'BLOCKED' : 'MISSING'}</div></div>`;
    }).join('');
    target.insertAdjacentHTML('beforeend', '<p style="font-size:11px;color:var(--text-secondary);margin-top:8px;">A checkpoint file is not proof of genuine training or acceptable held-out performance. Only a hash-matched manifest with scene-level held-out metrics and explicit reviewer approval enables inference. Dataset rows above are candidate references; this page does not scan your local filesystem for datasets.</p>');
    const evaluationTarget = document.getElementById('model-evaluation-reports');
    if (evaluationTarget) {
      const reports = Object.entries(models).filter((entry: any) => entry[1]?.evaluation_summary?.metrics);
      if (reports.length) {
        evaluationTarget.innerHTML = reports.map(([key, item]: any) => {
          const summary = item.evaluation_summary;
          const metricLabels: Record<string,string> = { precision: 'Precision', recall: 'Recall', f1: 'F1', iou: 'IoU', false_positive_rate: 'False-positive rate' };
          const metricRows = Object.entries(metricLabels).map(([metric, label]) => {
            const value = summary.metrics[metric];
            return `<div class="flex-between" style="font-size:12px;padding:5px 0;border-bottom:1px solid var(--border-light);"><span>${label}</span><strong>${typeof value === 'number' && Number.isFinite(value) ? (value * 100).toFixed(2) + '%' : 'Not measured'}</strong></div>`;
          }).join('');
          return `<div style="border:1px solid var(--border-light);border-radius:8px;padding:12px;margin-bottom:10px;"><strong>${esc(key.replaceAll('_',' '))}</strong><div style="font-size:12px;color:var(--text-secondary);margin:5px 0;">Dataset: ${esc(summary.dataset_name)} · Held-out scenes: ${Number(summary.held_out_scene_count)} · Split: ${esc(summary.split_type)} · Date: ${esc(summary.evaluation_date || 'Not supplied')}</div>${metricRows}<div style="font-size:11px;color:var(--warning);margin-top:7px;">Manifest-backed summary. Inspect the full evaluation report and failure cases before interpreting model quality.</div></div>`;
        }).join('');
      }
    }
  }).catch(() => { target.innerHTML = '<p style="color:var(--warning);">Backend checkpoint status is unavailable. Start the FastAPI server and reload this page.</p>'; });
}
