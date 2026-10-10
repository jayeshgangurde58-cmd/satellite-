// ===================================================================
// ChronoTerra — Application State
// ===================================================================

export type TaskType = 'general' | 'urban' | 'disaster';
export type AnalysisModelTask = 'urban_change' | 'flood_extent' | 'building_damage' | 'urban_change_classical';
export type AnalysisStatus = 'draft' | 'quality-check' | 'configured' | 'running' | 'complete' | 'failed' | 'cancelled';
export type QCStatus = 'pass' | 'warn' | 'fail' | 'pending' | 'skipped';
export type ReviewStatus = 'unreviewed' | 'confirmed' | 'rejected' | 'flagged';

export interface FileUpload {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'uploading' | 'reading' | 'ready' | 'error';
  progress: number;
  error?: string;
  metadata?: FileMetadata;
  file?: File;
  previewUrl?: string;
}

export interface FileMetadata {
  filename: string;
  dimensions: string;
  crs: string;
  resolution: string;
  bands: string;
  nodata: string;
  dateAcquired: string;
  fileSize: string;
}

export interface QualityCheck {
  id: string;
  label: string;
  status: QCStatus;
  value?: string;
  detail: string;
}

export interface AnalysisConfig {
  method: string;
  histogramMatching: boolean;
  thresholdMethod: string;
  thresholdValue: number;
  minRegionSize: number;
  morphFiltering: boolean;
  bandSelection: string;
  maskHandling: string;
}

export interface ChangeRegion {
  id: string;
  area: number;
  centroidLat: number;
  centroidLng: number;
  changeMagnitude: number;
  validFraction: number;
  beforeDate: string;
  afterDate: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  reviewStatus: ReviewStatus;
  warnings: string[];
  boundaryOverlap?: { area: number; pct: number; layerName: string };
}

export interface ProcessingStep {
  id: string;
  label: string;
  status: 'pending' | 'running' | 'done' | 'error';
  duration?: string;
}

export interface AnalysisRecord {
  id: string;
  name: string;
  task: TaskType | AnalysisModelTask;
  status: AnalysisStatus;
  createdAt: string;
  region?: string;
  warningCount: number;
  resultCount?: number;
  result?: any;
}

export interface OverlapValidation {
  status: 'pending' | 'verified' | 'insufficient' | 'unverifiable';
  percentage?: number;
  message?: string;
  bounds?: { left: number; bottom: number; right: number; top: number };
}

export interface AppState {
  currentPage: string;
  wizardStep: number;
  selectedTask: TaskType | null;
  beforeFile: FileUpload | null;
  afterFile: FileUpload | null;
  aoiFile: File | null;
  referenceBoundaryFile: File | null;
  boundarySourceName: string;
  boundaryEffectiveDate: string;
  boundaryProximityM: number;
  beforeSclFile: File | null;
  afterSclFile: File | null;
  assumeBandOrder: boolean;
  maskFile: FileUpload | null;
  boundaryFile: FileUpload | null;
  locationName?: string;
  locationPlaceId?: string;
  locationLat?: number;
  locationLng?: number;
  analysisModelTask: AnalysisModelTask;
  analysisResult: any | null;
  analysisPreflight: any | null;
  analysisError: string | null;
  resultViewMode: 'before' | 'split' | 'after';
  beforeDate?: string;
  afterDate?: string;
  confirmedArea?: boolean;
  qualityChecks: QualityCheck[];
  qualityOutcome: 'ready' | 'warnings' | 'cannot' | null;
  overlapValidation: OverlapValidation | null;
  analysisConfig: AnalysisConfig;
  processingSteps: ProcessingStep[];
  analysisStatus: AnalysisStatus;
  changeRegions: ChangeRegion[];
  selectedRegionId: string | null;
  analysisHistory: AnalysisRecord[];
  layerVisibility: Record<string, boolean>;
  detailsPanelOpen: boolean;
  sidebarOpen: boolean;
  backendConnected: boolean;
  previewImagesActive: boolean;
}

export function createInitialState(): AppState {
  return {
    currentPage: 'overview',
    wizardStep: 0,
    selectedTask: null,
    beforeFile: null,
    afterFile: null,
    aoiFile: null,
    referenceBoundaryFile: null,
    boundarySourceName: '',
    boundaryEffectiveDate: '',
    boundaryProximityM: 10,
    beforeSclFile: null,
    afterSclFile: null,
    assumeBandOrder: false,
    maskFile: null,
    boundaryFile: null,
    locationName: '',
    locationPlaceId: '',
    locationLat: undefined,
    locationLng: undefined,
    analysisModelTask: 'urban_change',
    analysisResult: null,
    analysisPreflight: null,
    analysisError: null,
    resultViewMode: 'split',
    beforeDate: '',
    afterDate: '',
    confirmedArea: false,
    qualityChecks: [],
    qualityOutcome: null,
    overlapValidation: null,
    analysisConfig: {
      method: 'classical-multiband',
      histogramMatching: false,
      thresholdMethod: 'otsu',
      thresholdValue: 0.35,
      minRegionSize: 500,
      morphFiltering: true,
      bandSelection: 'all',
      maskHandling: 'exclude',
    },
    processingSteps: [],
    analysisStatus: 'draft',
    changeRegions: [],
    selectedRegionId: null,
    analysisHistory: [],
    layerVisibility: {
      before: true,
      after: true,
      changeMask: true,
      predictionMask: true,
      changePolygons: true,
      validMask: false,
      cloudMask: false,
      boundary: false,
    },
    detailsPanelOpen: false,
    sidebarOpen: true,
    backendConnected: false,
    previewImagesActive: false,
  };
}

// Singleton state
let _state = createInitialState();
let _listeners: Array<() => void> = [];

export function getState(): AppState {
  return _state;
}

export function setState(partial: Partial<AppState>): void {
  // File preview URLs stay alive while their FileUpload record remains in state,
  // and are revoked as soon as the file is replaced or cleared.
  for (const key of ['beforeFile', 'afterFile', 'maskFile', 'boundaryFile'] as const) {
    if (!(key in partial)) continue;
    const previous = _state[key];
    const next = partial[key];
    const previousUrl = previous && 'previewUrl' in previous ? previous.previewUrl : undefined;
    const nextUrl = next && 'previewUrl' in next ? next.previewUrl : undefined;
    if (previousUrl && previousUrl !== nextUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(previousUrl);
    }
  }
  _state = { ..._state, ...partial };
  _listeners.forEach(fn => fn());
}

export function subscribe(fn: () => void): () => void {
  _listeners.push(fn);
  return () => { _listeners = _listeners.filter(l => l !== fn); };
}

export function generateId(): string {
  // These are local UI/history identifiers, not authentication tokens.
  // Prefer collision-resistant browser IDs when available.
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}
