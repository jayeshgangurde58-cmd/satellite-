// ===================================================================
// ChronoTerra — Shared UI Components
// ===================================================================

import { icons } from './icons';

/** Safely escape HTML */
export function esc(str: string): string {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

/** Create element from HTML string */
export function html(template: string): HTMLElement {
  const d = document.createElement('div');
  d.innerHTML = template.trim();
  return d.firstElementChild as HTMLElement;
}

/** Empty-state placeholder */
export function EmptyState(icon: string, title: string, desc: string, action?: { label: string; id: string }): string {
  return `
    <div class="empty-state">
      <div class="empty-state__icon">${icon}</div>
      <div class="empty-state__title">${esc(title)}</div>
      <div class="empty-state__desc">${esc(desc)}</div>
      ${action ? `<button class="btn btn--primary btn--lg" id="${action.id}">${esc(action.label)}</button>` : ''}
    </div>
  `;
}

/** Error/warning/info banner */
export function Banner(type: 'error' | 'warning' | 'info' | 'success', title: string, text: string): string {
  const iconMap: Record<string, string> = {
    error: '✕',
    warning: icons.alertTriangle,
    info: icons.info,
    success: icons.check,
  };
  return `
    <div class="error-banner error-banner--${type}">
      <span class="error-banner__icon">${iconMap[type]}</span>
      <div class="error-banner__content">
        <div class="error-banner__title">${esc(title)}</div>
        <div class="error-banner__text">${esc(text)}</div>
      </div>
    </div>
  `;
}

/** Warning banner (shorthand) */
export function WarningBanner(text: string): string {
  return `
    <div class="warning-banner">
      <span class="warning-banner__icon">${icons.alertTriangle}</span>
      <span class="warning-banner__text">${text}</span>
    </div>
  `;
}

/** Badge */
export function Badge(type: 'success' | 'warning' | 'error' | 'info' | 'neutral', label: string): string {
  return `<span class="badge badge--${type}">${esc(label)}</span>`;
}

/** Status indicator */
export function StatusIndicator(status: 'pass' | 'warn' | 'fail' | 'pending', label: string): string {
  return `
    <span class="status-indicator status-indicator--${status}">
      <span class="status-indicator__dot"></span>
      ${esc(label)}
    </span>
  `;
}

/** Metric card */
export function MetricCard(label: string, value: string, sub?: string): string {
  return `
    <div class="metric-card">
      <div class="metric-card__label">${esc(label)}</div>
      <div class="metric-card__value">${esc(value)}</div>
      ${sub ? `<div class="metric-card__sub">${esc(sub)}</div>` : ''}
    </div>
  `;
}

/** Tabs component */
export function Tabs(tabs: { id: string; label: string; active?: boolean }[]): string {
  return `
    <div class="tabs">
      ${tabs.map(t => `<button class="tab ${t.active ? 'tab--active' : ''}" data-tab="${t.id}">${esc(t.label)}</button>`).join('')}
    </div>
  `;
}

/** Card wrapper */
export function Card(title: string, body: string, footer?: string, headerRight?: string): string {
  return `
    <div class="card">
      <div class="card__header">
        <div class="card__title">${title}</div>
        ${headerRight || ''}
      </div>
      <div class="card__body">${body}</div>
      ${footer ? `<div class="card__footer">${footer}</div>` : ''}
    </div>
  `;
}

/** Collapsible section */
export function Collapsible(title: string, body: string, open = false): string {
  const id = 'coll-' + (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`);
  return `
    <div class="collapsible" data-collapsible="${id}">
      <button class="collapsible__trigger ${open ? 'collapsible__trigger--open' : ''}" data-collapsible-trigger="${id}">
        <span class="collapsible__arrow">${icons.chevronRight}</span>
        ${esc(title)}
      </button>
      <div class="collapsible__body ${open ? '' : 'collapsible__body--hidden'}" data-collapsible-body="${id}">
        ${body}
      </div>
    </div>
  `;
}

/** Bind collapsible toggle behavior */
export function bindCollapsibles(container: HTMLElement): void {
  container.querySelectorAll<HTMLElement>('[data-collapsible-trigger]').forEach(trigger => {
    trigger.addEventListener('click', () => {
      const id = trigger.dataset.collapsibleTrigger!;
      const body = container.querySelector(`[data-collapsible-body="${id}"]`) as HTMLElement;
      if (!body) return;
      const open = trigger.classList.toggle('collapsible__trigger--open');
      body.classList.toggle('collapsible__body--hidden', !open);
    });
  });
}

/** Format file size */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

/** Format area */
export function formatArea(sqm: number): string {
  if (sqm < 10000) return sqm.toLocaleString() + ' m²';
  return (sqm / 10000).toFixed(2) + ' ha';
}

/** Format date */
export function formatDate(iso: string): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  } catch {
    return iso;
  }
}
