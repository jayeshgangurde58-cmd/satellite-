// ===================================================================
// ChronoTerra — Backend API Integration Service
// ===================================================================

const API_BASE_URL = String((import.meta as any).env?.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api').replace(/\/$/, '');

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  max_upload_mb: number;
}

export interface InspectReport {
  request_id: string;
  filename: string;
  file_size_bytes: number;
  format: string;
  dimensions?: { width: number; height: number };
  bands_count?: number;
  crs?: string;
  pixel_size?: number[];
  valid_geospatial?: boolean;
  metadata?: any;
}

export interface PairValidationResult {
  request_id: string;
  status: string;
  analysis_mode: string;
  before_image: InspectReport;
  after_image: InspectReport;
  pair: {
    geographic_overlap: {
      status: string;
      footprint_intersection_over_union: number | null;
      method: string;
    };
    same_crs: boolean;
    same_dimensions: boolean;
    same_grid: boolean;
    checks: any[];
    warnings: string[];
    blockers: string[];
  };
}

/**
  Check backend connection status
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (res.ok) {
      const data: HealthResponse = await res.json();
      return data.status === 'ok';
    }
  } catch (err) {
    console.warn('Backend server ping failed:', err);
  }
  return false;
}

/**
  Inspect uploaded image via FastAPI backend
 */
export async function inspectImageApi(file: File): Promise<InspectReport | null> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE_URL}/inspect-image`, {
      method: 'POST',
      body: formData,
    });

    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('Failed to inspect image via API:', err);
  }
  return null;
}

/**
  Validate image pair via FastAPI backend
 */
export async function validatePairApi(
  beforeFile: File | Blob,
  afterFile: File | Blob,
  mode: 'urban' | 'disaster',
  locationName?: string,
  beforeDate?: string,
  afterDate?: string,
  sameAreaConfirmed: boolean = true
): Promise<PairValidationResult | null> {
  try {
    const formData = new FormData();
    formData.append('before_image', beforeFile, 'before.jpg');
    formData.append('after_image', afterFile, 'after.jpg');
    formData.append('analysis_mode', mode);
    if (locationName) formData.append('location_name', locationName);
    if (beforeDate) formData.append('before_date', beforeDate);
    if (afterDate) formData.append('after_date', afterDate);
    formData.append('same_area_confirmed', String(sameAreaConfirmed));

    const res = await fetch(`${API_BASE_URL}/validate-pair`, {
      method: 'POST',
      body: formData,
    });

    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('Failed to validate pair via API:', err);
  }
  return null;
}

// The prediction sub-application is mounted by FastAPI at /ml. URLs returned by
// its artifact endpoints are rooted at /outputs, so use mlAssetUrl() for them.
export const API_ORIGIN = new URL(API_BASE_URL).origin;
export const ML_API_BASE_URL = `${API_ORIGIN}/ml/api`;

export function mlAssetUrl(path?: string | null): string | null {
  if (!path) return null;
  if (/^https?:\/\//i.test(path)) return path;
  if (path.startsWith('/ml/')) return `${API_ORIGIN}${path}`;
  if (path.startsWith('/outputs/')) return `${API_ORIGIN}/ml${path}`;
  return `${API_ORIGIN}${path.startsWith('/') ? path : `/${path}`}`;
}

export interface PredictionRequest {
  task: 'urban_change' | 'flood_extent' | 'building_damage' | 'urban_change_classical';
  beforeFile?: File | Blob;
  afterFile?: File | Blob;
  inputFile?: File | Blob;
  beforeName?: string;
  afterName?: string;
  inputName?: string;
  assumeBandOrder?: boolean;
  aoiFile?: File | null;
  referenceBoundaryFile?: File | null;
  boundarySourceName?: string;
  boundaryEffectiveDate?: string;
  boundaryProximityM?: number;
  beforeSclFile?: File | null;
  afterSclFile?: File | null;
  beforeDate?: string;
  afterDate?: string;
  locationName?: string;
  locationPlaceId?: string;
  locationLat?: number;
  locationLng?: number;
}

async function parseApiError(res: Response): Promise<string> {
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    const d = body?.detail;
    detail = typeof d === 'string' ? d : (d?.message || d?.reason || JSON.stringify(d || body));
  } catch { /* keep HTTP fallback */ }
  return detail;
}

function appendPredictionMetadata(form: FormData, req: PredictionRequest): void {
  if (req.assumeBandOrder) form.append('assume_band_order', 'true');
  if (req.aoiFile) form.append('aoi_geojson_file', req.aoiFile, req.aoiFile.name || 'aoi.geojson');
  if (req.task === 'urban_change' && req.referenceBoundaryFile) {
    form.append('reference_boundary_geojson_file', req.referenceBoundaryFile, req.referenceBoundaryFile.name || 'reference-zone.geojson');
    if (req.boundarySourceName?.trim()) form.append('boundary_source_name', req.boundarySourceName.trim());
    if (req.boundaryEffectiveDate) form.append('boundary_effective_date', req.boundaryEffectiveDate);
    if (typeof req.boundaryProximityM === 'number') form.append('boundary_proximity_m', String(req.boundaryProximityM));
  }
  if (req.task === 'urban_change' && (req.beforeSclFile || req.afterSclFile)) {
    if (!req.beforeSclFile || !req.afterSclFile) throw new Error('Supply both before and after SCL GeoTIFF masks, or leave both empty.');
    form.append('before_scl_file', req.beforeSclFile, req.beforeSclFile.name || 'before_scl.tif');
    form.append('after_scl_file', req.afterSclFile, req.afterSclFile.name || 'after_scl.tif');
  }
  if (req.task !== 'flood_extent' && (req.beforeDate || req.afterDate || req.locationName || req.locationPlaceId)) {
    const metadata = {
      source: 'user_supplied_form_metadata_not_sensor_verified',
      before: { acquisition_date: req.beforeDate || null },
      after: { acquisition_date: req.afterDate || null },
      location: { name: req.locationName || null, place_id: req.locationPlaceId || null, latitude: req.locationLat ?? null, longitude: req.locationLng ?? null },
    };
    form.append('pair_metadata_file', new Blob([JSON.stringify(metadata)], { type: 'application/json' }), 'pair_metadata.json');
  }
}

export async function preflightPredictionApi(req: PredictionRequest): Promise<any> {
  const form = new FormData();
  form.append('task', req.task === 'urban_change_classical' ? 'urban_change' : req.task);
  if (req.task === 'flood_extent') {
    if (!req.inputFile) throw new Error('Flood extent requires a single Sentinel-1 VV/VH GeoTIFF.');
    form.append('input_file', req.inputFile, req.inputName || (req.inputFile as File).name || 'flood_input.tif');
  } else {
    if (!req.beforeFile || !req.afterFile) throw new Error('This prediction task requires both before and after images.');
    form.append('before_file', req.beforeFile, req.beforeName || (req.beforeFile as File).name || 'before.tif');
    form.append('after_file', req.afterFile, req.afterName || (req.afterFile as File).name || 'after.tif');
  }
  appendPredictionMetadata(form, req);
  const res = await fetch(`${ML_API_BASE_URL}/preflight`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await parseApiError(res));
  const report = await res.json();
  if (report.report_url) report.report_url = mlAssetUrl(report.report_url);
  return report;
}

export async function runPredictionApi(req: PredictionRequest): Promise<any> {
  const form = new FormData();
  
  if (req.task === 'urban_change_classical') {
    if (!req.beforeFile || !req.afterFile) throw new Error('Classical baseline requires before and after images.');
    form.append('before_file', req.beforeFile, req.beforeName || (req.beforeFile as File).name || 'before.tif');
    form.append('after_file', req.afterFile, req.afterName || (req.afterFile as File).name || 'after.tif');
    form.append('histogram_match', 'true');
    form.append('min_object_size', '8');
    form.append('opening_radius', '1');
    
    const root = API_BASE_URL.replace('/api', '/ml');
    const res = await fetch(`${root}/api/analyze`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(await parseApiError(res));
    const result = await res.json();
    for (const key of ['mask_url', 'geojson_url', 'before_image_url', 'after_image_url']) {
      if (result[key]) result[key] = mlAssetUrl(result[key]);
    }
    result.before_preview_url = result.before_image_url;
    result.after_preview_url = result.after_image_url;
    result.regions_url = result.geojson_url;
    result.mask_preview_url = result.mask_url;
    result.task = 'urban_change_classical';
    result.predicted_class_pixel_counts = {
      'Candidate change pixels': result.metrics?.predicted_change_pixels || 0
    };
    return result;
  }

  form.append('task', req.task);
  if (req.task === 'flood_extent') {
    if (!req.inputFile) throw new Error('Flood extent requires one Sentinel-1 VV/VH GeoTIFF input.');
    form.append('input_file', req.inputFile, req.inputName || (req.inputFile as File).name || 'flood_input.tif');
  } else {
    if (!req.beforeFile || !req.afterFile) throw new Error('This prediction task requires before and after images.');
    form.append('before_file', req.beforeFile, req.beforeName || (req.beforeFile as File).name || 'before.tif');
    form.append('after_file', req.afterFile, req.afterName || (req.afterFile as File).name || 'after.tif');
  }
  appendPredictionMetadata(form, req);
  const res = await fetch(`${ML_API_BASE_URL}/predict`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await parseApiError(res));
  const result = await res.json();
  for (const key of ['mask_url', 'scores_url', 'uncertainty_url', 'overlay_url', 'report_url', 'mask_preview_url', 'before_preview_url', 'after_preview_url', 'input_preview_url', 'regions_url', 'boundary_overlay_url']) {
    if (result[key]) result[key] = mlAssetUrl(result[key]);
  }
  return result;
}

export async function getModelStatusApi(): Promise<any> {
  const res = await fetch(`${ML_API_BASE_URL}/models/status`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}
