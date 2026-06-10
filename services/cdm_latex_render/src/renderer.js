const { DEFAULT_TIMEOUT_MS } = require('./constants');
const { tokenizeLatex } = require('./latex');

let dependencyProbe;
let rendererDeps;
let browserLaunchConfirmed = false;
const injectedReadinessProbeCache = new WeakMap();
const renderCache = new Map();
const MAX_CACHE_ENTRIES = 512;

function lazyLoadRendererDeps() {
  if (rendererDeps) {
    return rendererDeps;
  }

  let katex = null;
  let playwright = null;
  try {
    // Lazy require so local tests and Python reward clients can run without network installs.
    // eslint-disable-next-line global-require, import/no-extraneous-dependencies
    katex = require('katex');
  } catch (_error) {
    katex = null;
  }
  try {
    // eslint-disable-next-line global-require, import/no-extraneous-dependencies
    playwright = require('playwright');
  } catch (_error) {
    try {
      // eslint-disable-next-line global-require, import/no-extraneous-dependencies
      playwright = require('@playwright/test');
    } catch (_innerError) {
      playwright = null;
    }
  }

  rendererDeps = {
    katex,
    playwright,
    launchBrowser: async (launchOptions = {}) => playwright.chromium.launch(launchOptions),
  };
  return rendererDeps;
}

function resolveRendererDeps(overrides = {}) {
  if (overrides.rendererDeps) {
    return overrides.rendererDeps;
  }
  return lazyLoadRendererDeps();
}

function hasRenderableDeps(deps) {
  return Boolean(
    deps &&
      deps.katex &&
      typeof deps.katex.renderToString === 'function' &&
      typeof deps.launchBrowser === 'function'
  );
}

function probeDependencies(options = {}) {
  const deps = resolveRendererDeps(options);
  const katexReady = Boolean(deps && deps.katex && typeof deps.katex.renderToString === 'function');
  const playwrightReady = Boolean(
    deps &&
      (typeof deps.launchBrowser === 'function' ||
        (deps.playwright && deps.playwright.chromium && typeof deps.playwright.chromium.launch === 'function'))
  );
  const browserReady = Boolean(katexReady && playwrightReady && (browserLaunchConfirmed || deps.browserReady === true));
  const probe = {
    katexReady,
    playwrightReady,
    browserReady,
    renderer: browserReady ? 'katex-chromium' : 'katex-fallback-pseudolayout',
  };

  if (!options.rendererDeps) {
    dependencyProbe = probe;
  }
  return probe;
}

async function probeBrowserReady(options = {}) {
  const deps = resolveRendererDeps(options);
  if (options.rendererDeps) {
    const cached = injectedReadinessProbeCache.get(deps);
    if (cached) {
      return cached;
    }
  }

  const probe = probeDependencies(options);
  if (!probe.katexReady || !probe.playwrightReady || probe.browserReady) {
    if (options.rendererDeps) {
      injectedReadinessProbeCache.set(deps, probe);
    }
    return probe;
  }

  let browser;
  let readinessProbe = probe;
  try {
    browser = await deps.launchBrowser({ headless: true, timeout: Number(options.timeoutMs || DEFAULT_TIMEOUT_MS) });
    if (!options.rendererDeps) {
      browserLaunchConfirmed = true;
      dependencyProbe = null;
    }
    readinessProbe = probeDependencies({ ...options, rendererDeps: options.rendererDeps ? { ...deps, browserReady: true } : undefined });
  } catch (_error) {
    readinessProbe = probe;
  } finally {
    if (browser && typeof browser.close === 'function') {
      await browser.close();
    }
  }

  if (options.rendererDeps) {
    injectedReadinessProbeCache.set(deps, readinessProbe);
  }
  return readinessProbe;
}

function cachedProbeDependencies(options = {}) {
  if (!options.rendererDeps && dependencyProbe) {
    return dependencyProbe;
  }
  return probeDependencies(options);
}

function putCache(key, value) {
  if (renderCache.size >= MAX_CACHE_ENTRIES) {
    const firstKey = renderCache.keys().next().value;
    renderCache.delete(firstKey);
  }
  renderCache.set(key, value);
}

function pseudoBox(token, index, total) {
  const commandWidth = token.value.startsWith('\\') ? 14 + token.value.length : 10;
  const structuralOffset = token.value === '^' ? -7 : token.value === '_' ? 7 : 0;
  return {
    x: index * 12,
    y: 20 + structuralOffset,
    width: commandWidth,
    height: 16,
    centerX: index * 12 + commandWidth / 2,
    centerY: 28 + structuralOffset,
    order: total <= 1 ? 0 : index / (total - 1),
  };
}

function htmlEscape(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function buildKaTeXHtml(normalizedLatex, tokens, katex) {
  const tokenSpans = tokens
    .map((token, index) => {
      const renderedToken = katex.renderToString(token.value, {
        throwOnError: false,
        strict: 'ignore',
        displayMode: false,
      });
      return `<span class="katex-token" data-token-index="${index}">${renderedToken}</span>`;
    })
    .join('');

  return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin: 0; padding: 16px; font-size: 24px; }
  .katex-render { display: inline-block; }
  .katex-token { display: inline-block; margin-right: 2px; vertical-align: middle; }
</style>
</head>
<body>
  <div class="katex-render" data-normalized-latex="${htmlEscape(normalizedLatex)}">${tokenSpans}</div>
</body>
</html>`;
}

function boxesFromDomScript() {
  return `Array.from(document.querySelectorAll('[data-token-index]')).map((element) => {
    const rect = element.getBoundingClientRect();
    return {
      index: Number(element.getAttribute('data-token-index')),
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    };
  })`;
}

function boxWithDerivedFields(rawBox, index, total) {
  const x = Number(rawBox.x || 0);
  const y = Number(rawBox.y || 0);
  const width = Number(rawBox.width || 0);
  const height = Number(rawBox.height || 0);
  return {
    x,
    y,
    width,
    height,
    centerX: x + width / 2,
    centerY: y + height / 2,
    order: total <= 1 ? 0 : index / (total - 1),
  };
}

function timeoutError(timeoutMs) {
  const error = new Error(`render timed out after ${timeoutMs}ms`);
  error.code = 'RENDER_TIMEOUT';
  return error;
}

function ensureWithinTimeout(started, timeoutMs) {
  if (Date.now() - started > timeoutMs) {
    throw timeoutError(timeoutMs);
  }
}

function remainingTimeoutMs(started, timeoutMs) {
  const remaining = timeoutMs - (Date.now() - started);
  if (remaining <= 0) {
    throw timeoutError(timeoutMs);
  }
  return remaining;
}

async function withTimeoutBudget(operationPromise, started, timeoutMs) {
  const remaining = remainingTimeoutMs(started, timeoutMs);
  let timer;
  try {
    return await Promise.race([
      operationPromise,
      new Promise((_resolve, reject) => {
        timer = setTimeout(() => reject(timeoutError(timeoutMs)), remaining);
      }),
    ]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

async function renderWithChromium(parsed, deps, timeoutMs, started) {
  const browser = await deps.launchBrowser({ headless: true, timeout: timeoutMs });
  let page;
  try {
    ensureWithinTimeout(started, timeoutMs);
    page = await browser.newPage();
    ensureWithinTimeout(started, timeoutMs);
    const remainingForContent = Math.max(1, timeoutMs - (Date.now() - started));
    await page.setContent(buildKaTeXHtml(parsed.normalized, parsed.tokens, deps.katex), {
      waitUntil: 'load',
      timeout: remainingForContent,
    });
    const rawBoxes = await withTimeoutBudget(page.evaluate(boxesFromDomScript()), started, timeoutMs);
    const boxesByIndex = new Map((rawBoxes || []).map((box) => [Number(box.index), box]));
    const total = parsed.tokens.length;
    return parsed.tokens.map((token, index) => ({
      ...token,
      box: boxWithDerivedFields(boxesByIndex.get(index) || pseudoBox(token, index, total), index, total),
    }));
  } finally {
    if (page && typeof page.close === 'function') {
      await page.close();
    }
    if (!deps.browser && browser && typeof browser.close === 'function') {
      await browser.close();
    }
  }
}

async function renderLatex(input, options = {}) {
  const timeoutMs = Number(options.timeoutMs || DEFAULT_TIMEOUT_MS);
  const cacheKey = options.rendererDeps ? null : String(input ?? '');
  const cached = cacheKey ? renderCache.get(cacheKey) : null;
  if (cached) {
    return { ...cached, cacheHit: true };
  }

  const started = Date.now();
  const parsed = tokenizeLatex(input);
  ensureWithinTimeout(started, timeoutMs);
  const deps = resolveRendererDeps(options);
  const probe = options.rendererDeps ? probeDependencies(options) : cachedProbeDependencies();

  let tokens;
  let fallbackUsed = true;
  let renderer = 'katex-fallback-pseudolayout';
  if (hasRenderableDeps(deps)) {
    try {
      tokens = await renderWithChromium(parsed, deps, timeoutMs, started);
      fallbackUsed = false;
      renderer = 'katex-chromium';
      if (!options.rendererDeps) {
        browserLaunchConfirmed = true;
        dependencyProbe = null;
        cachedProbeDependencies();
      }
    } catch (error) {
      if (error && error.code === 'RENDER_TIMEOUT') {
        throw error;
      }
      tokens = null;
    }
  }

  if (!tokens) {
    const total = parsed.tokens.length;
    tokens = parsed.tokens.map((token, index) => ({
      ...token,
      box: pseudoBox(token, index, total),
    }));
  }

  const rendered = {
    normalized: parsed.normalized,
    tokens,
    renderer,
    fallbackUsed,
    cacheHit: false,
  };
  if (cacheKey) {
    putCache(cacheKey, rendered);
  }
  return rendered;
}

function cacheStats() {
  return {
    size: renderCache.size,
    maxEntries: MAX_CACHE_ENTRIES,
  };
}

module.exports = {
  probeDependencies: cachedProbeDependencies,
  probeBrowserReady,
  renderLatex,
  cacheStats,
  buildKaTeXHtml,
};
