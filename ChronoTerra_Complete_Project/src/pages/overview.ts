// ===================================================================
// ChronoTerra — Front Page / Overview
// ===================================================================

import { setState } from '../state';

export function renderOverview(): string {
  return `
    <div class="landing-page-container">
      
      <!-- Top Hero Section: Text Left + Image Slider Right -->
      <section class="hero-grid">
        
        <!-- Left Content -->
        <div class="hero-content">
          <div class="cyan-accent-bar"></div>
          <h1 class="hero-main-heading">
            See the <span class="highlight-teal">Change.</span><br>
            Map the <span class="highlight-blue">Impact.</span>
          </h1>
          <p class="hero-subtitle">
            Screen satellite observations for candidate urban change, flood-water extent, or building damage using separate task-specific workflows.
          </p>

        </div>

        <!-- Right Side: Interactive Satellite Image Comparison Slider -->
        <div class="hero-preview-card">
          <div class="image-slider-container" id="before-after-slider" role="slider" aria-label="Illustrative before and after preview divider" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" tabindex="0">
            
            <!-- Base Image: Before -->
            <img src="/satellite-before.jpg" class="slider-img slider-img-before" alt="Illustrative satellite image preview, not an analysis input" />
            
            <!-- Clipped Overlay: After -->
            <div class="slider-after-wrapper" id="slider-after-wrapper">
              <img src="/satellite-after.jpg" class="slider-img slider-img-after" id="slider-img-after-el" alt="Illustrative satellite image preview, not a prediction result" />
            </div>

            <!-- Badges -->
            <span class="slider-badge badge-before">Before</span>
            <span class="slider-badge badge-after">After</span>

            <!-- Draggable Divider Line & Button Handle -->
            <div class="slider-handle" id="slider-handle">
              <div class="handle-button">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="15 18 9 12 15 6"/>
                  <polyline points="9 18 15 12 9 6" transform="rotate(180 12 12)"/>
                </svg>
              </div>
            </div>

          </div>
          <p class="preview-disclaimer">Illustrative preview only — these images are not a registered source pair and do not show ChronoTerra model output.</p>
        </div>

      </section>

      <!-- Bottom Task Action Cards (Two Side-by-Side Cards) -->
      <section class="task-cards-grid">
        
        <!-- Teal Card: Urban Change Screening -->
        <div class="task-card task-card--urban" id="btn-urban-task">
          <div class="card-bg-graphic card-bg-graphic--urban"></div>
          <div class="card-icon card-icon--urban">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="4" y="2" width="16" height="20" rx="1"/>
              <line x1="8" y1="6" x2="8.01" y2="6"/>
              <line x1="12" y1="6" x2="12.01" y2="6"/>
              <line x1="16" y1="6" x2="16.01" y2="6"/>
              <line x1="8" y1="10" x2="8.01" y2="10"/>
              <line x1="12" y1="10" x2="12.01" y2="10"/>
              <line x1="16" y1="10" x2="16.01" y2="10"/>
              <line x1="8" y1="14" x2="8.01" y2="14"/>
              <line x1="12" y1="14" x2="12.01" y2="14"/>
              <line x1="16" y1="14" x2="16.01" y2="14"/>
              <path d="M10 22v-4h4v4"/>
            </svg>
          </div>
          <div class="card-content">
            <h3 class="card-title">Urban Change Screening</h3>
            <p class="card-desc">Flag candidate surface changes. Boundary overlap can support review only when a real, appropriately dated boundary layer is supplied; the satellite model cannot confirm illegality.</p>
            <button class="pill-btn pill-btn--teal" id="btn-urban-pill">
              Explore Urban Changes &rarr;
            </button>
          </div>
        </div>

        <!-- Orange Card: Disaster Impact Management -->
        <div class="task-card task-card--disaster" id="btn-disaster-task">
          <div class="card-bg-graphic card-bg-graphic--disaster"></div>
          <div class="card-icon card-icon--disaster">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M4 14.89 9 11l5 4 6-5.11"/>
              <path d="m2 20 6-7 4 3 6-5 4 4"/>
              <path d="M12 2v4"/>
              <path d="m4.93 4.93 2.83 2.83"/>
              <path d="m16.24 7.76 2.83-2.83"/>
            </svg>
          </div>
          <div class="card-content">
            <h3 class="card-title">Disaster Impact Management</h3>
            <p class="card-desc">Choose one separate task: flood-water extent from a Sentinel-1 VV/VH scene, or building-damage classes from a compatible paired RGB dataset.</p>
            <button class="pill-btn pill-btn--orange" id="btn-disaster-pill">
              Analyze Disaster Impact &rarr;
            </button>
          </div>
        </div>

      </section>

      <footer class="footer-light">
        <p>© ChronoTerra Geospatial AI Platform 2026 • See the Change. Map the Impact.</p>
      </footer>

    </div>
  `;
}

export function bindOverviewEvents(): void {
  const navigateToTask = (task: 'urban' | 'disaster') => {
    // Selecting a workflow must not create placeholder uploads or claim they were validated.
    setState({
      currentPage: 'new-analysis',
      selectedTask: task,
      analysisModelTask: task === 'urban' ? 'urban_change' : 'flood_extent',
      wizardStep: 0,
      beforeFile: null,
      afterFile: null,
      aoiFile: null, referenceBoundaryFile: null, boundarySourceName: '', boundaryEffectiveDate: '', boundaryProximityM: 10,
      beforeSclFile: null, afterSclFile: null, assumeBandOrder: false, beforeDate: '', afterDate: '',
      confirmedArea: false,
      overlapValidation: null,
      analysisResult: null,
      analysisPreflight: null,
      analysisError: null,
      locationName: '',
      locationPlaceId: '',
      locationLat: undefined,
      locationLng: undefined,
      analysisStatus: 'draft',
      changeRegions: [], selectedRegionId: null, detailsPanelOpen: false,
      resultViewMode: 'split',
    });
  };

  const urbanCard = document.getElementById('btn-urban-task');
  const urbanPill = document.getElementById('btn-urban-pill');
  const disasterCard = document.getElementById('btn-disaster-task');
  const disasterPill = document.getElementById('btn-disaster-pill');

  urbanCard?.addEventListener('click', () => navigateToTask('urban'));
  urbanPill?.addEventListener('click', (e) => { e.stopPropagation(); navigateToTask('urban'); });

  disasterCard?.addEventListener('click', () => navigateToTask('disaster'));
  disasterPill?.addEventListener('click', (e) => { e.stopPropagation(); navigateToTask('disaster'); });

  // Interactive illustrative comparison slider. Preserve a shared image scale from first paint.
  const container = document.getElementById('before-after-slider');
  const wrapper = document.getElementById('slider-after-wrapper');
  const handle = document.getElementById('slider-handle');
  const afterImg = document.getElementById('slider-img-after-el') as HTMLImageElement | null;

  if (container && wrapper && handle) {
    let isDragging = false;
    let positionPct = 50;

    const applyPosition = (percent: number) => {
      positionPct = Math.max(0, Math.min(100, percent));
      wrapper.style.width = `${positionPct}%`;
      handle.style.left = `${positionPct}%`;
      container.setAttribute('aria-valuenow', String(Math.round(positionPct)));
      const width = container.getBoundingClientRect().width;
      if (afterImg && width > 0) {
        // The overlaid image must remain the full container width, not the clipped wrapper width.
        afterImg.style.width = `${width}px`;
        afterImg.style.maxWidth = 'none';
      }
    };

    const setFromClientX = (clientX: number) => {
      const rect = container.getBoundingClientRect();
      if (rect.width <= 0) return;
      applyPosition(((clientX - rect.left) / rect.width) * 100);
    };

    applyPosition(50);
    container.addEventListener('pointerdown', (event) => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      isDragging = true;
      try { container.setPointerCapture(event.pointerId); } catch { /* capture is best effort */ }
      setFromClientX(event.clientX);
    });
    container.addEventListener('pointermove', (event) => {
      if (isDragging) setFromClientX(event.clientX);
    });
    const stopDragging = () => { isDragging = false; };
    container.addEventListener('pointerup', stopDragging);
    container.addEventListener('pointercancel', stopDragging);
    container.addEventListener('lostpointercapture', stopDragging);
    container.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'Home' || event.key === 'End') {
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? 100 : positionPct + (event.key === 'ArrowRight' ? 2 : -2);
        applyPosition(next);
      }
    });
    window.addEventListener('resize', () => applyPosition(positionPct), { passive: true });
  }
}
