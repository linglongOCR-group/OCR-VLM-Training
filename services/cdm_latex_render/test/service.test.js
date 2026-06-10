const assert = require('node:assert/strict');
const test = require('node:test');
const { createServer } = require('../src/server');
const { scoreLatexPair } = require('../src/scoring');
const { buildKaTeXHtml, probeDependencies, renderLatex } = require('../src/renderer');

async function withServer(t) {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.listen(0, '127.0.0.1', resolve);
    server.once('error', reject);
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const { port } = server.address();
  return `http://127.0.0.1:${port}`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  return { response, payload };
}

async function postJson(url, body) {
  return fetchJson(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

test('renderLatex uses injected KaTeX/Chromium renderer to extract token boxes', async () => {
  const calls = [];
  const page = {
    setContent: async (html, options) => {
      calls.push(['setContent', html, options]);
    },
    evaluate: async (script) => {
      calls.push(['evaluate', script]);
      return [
        { index: 0, x: 11, y: 13, width: 17, height: 19 },
        { index: 1, x: 31, y: 37, width: 41, height: 43 },
      ];
    },
    close: async () => {
      calls.push(['page.close']);
    },
  };
  const browser = {
    newPage: async () => {
      calls.push(['newPage']);
      return page;
    },
  };

  const rendered = await renderLatex('x + y', {
    timeoutMs: 250,
    rendererDeps: {
      katex: {
        renderToString: (latex, options) => {
          calls.push(['renderToString', latex, options]);
          return '<span class="katex">mock-katex</span>';
        },
      },
      launchBrowser: async () => {
        calls.push(['launchBrowser']);
        return browser;
      },
      browserReady: true,
    },
  });

  assert.equal(rendered.fallbackUsed, false);
  assert.equal(rendered.renderer, 'katex-chromium');
  assert.equal(rendered.tokens.length, 3);
  assert.deepEqual(rendered.tokens[0].box, {
    x: 11,
    y: 13,
    width: 17,
    height: 19,
    centerX: 19.5,
    centerY: 22.5,
    order: 0,
  });
  assert.deepEqual(rendered.tokens[1].box, {
    x: 31,
    y: 37,
    width: 41,
    height: 43,
    centerX: 51.5,
    centerY: 58.5,
    order: 0.5,
  });
  assert.equal(calls.some((call) => call[0] === 'renderToString'), true);
  const setContentCall = calls.find((call) => call[0] === 'setContent');
  assert.match(setContentCall[1], /data-token-index="0"/);
  assert.match(setContentCall[1], /class="katex-render"/);
  assert.equal(calls.some((call) => call[0] === 'evaluate'), true);
});

test('buildKaTeXHtml wraps each token index around KaTeX-rendered HTML', () => {
  const html = buildKaTeXHtml('x + y', [
    { value: 'x' },
    { value: '+' },
    { value: 'y' },
  ], {
    renderToString: (latex) => `<span class="katex"><span class="mord">${latex}</span></span>`,
  });

  assert.doesNotMatch(html, /class="token-probe"/);
  assert.doesNotMatch(html, /class="token-probes"/);
  for (const index of [0, 1, 2]) {
    assert.match(
      html,
      new RegExp(`<span class="katex-token" data-token-index="${index}">\\s*<span class="katex">`)
    );
  }
});

test('renderLatex times out slow DOM evaluation and closes resources', async () => {
  const calls = [];
  const page = {
    setContent: async () => {
      calls.push(['setContent']);
    },
    evaluate: async () => {
      calls.push(['evaluate:start']);
      await new Promise((resolve) => setTimeout(resolve, 80));
      calls.push(['evaluate:end']);
      return [];
    },
    close: async () => {
      calls.push(['page.close']);
    },
  };
  const browser = {
    newPage: async () => {
      calls.push(['newPage']);
      return page;
    },
    close: async () => {
      calls.push(['browser.close']);
    },
  };

  await assert.rejects(
    renderLatex('x', {
      timeoutMs: 10,
      rendererDeps: {
        katex: { renderToString: () => '<span class="katex">x</span>' },
        launchBrowser: async () => {
          calls.push(['launchBrowser']);
          return browser;
        },
        browserReady: true,
      },
    }),
    (error) => error && error.code === 'RENDER_TIMEOUT'
  );

  assert.equal(calls.some((call) => call[0] === 'evaluate:end'), false);
  assert.equal(calls.some((call) => call[0] === 'page.close'), true);
  assert.equal(calls.some((call) => call[0] === 'browser.close'), true);
});

test('probeDependencies does not claim browserReady from package resolution alone', () => {
  const probe = probeDependencies({
    rendererDeps: {
      katex: { renderToString: () => '<span class="katex">x</span>' },
      launchBrowser: async () => ({ newPage: async () => ({}) }),
    },
  });

  assert.equal(probe.katexReady, true);
  assert.equal(probe.playwrightReady, true);
  assert.equal(probe.browserReady, false);
  assert.equal(probe.renderer, 'katex-fallback-pseudolayout');
});

test('GET /health returns service metadata and browser readiness', async (t) => {
  const baseUrl = await withServer(t);

  const { response, payload } = await fetchJson(`${baseUrl}/health`);

  assert.equal(response.status, 200);
  assert.equal(payload.service, 'cdm-latex-render');
  assert.equal(payload.version, 'cdm_katex_v1');
  assert.match(payload.renderer, /katex|fallback/);
  assert.equal(typeof payload.browser_ready, 'boolean');
  assert.equal(payload.ok, true);
});

test('identical formulas score 1.0 with successful diagnostics', async () => {
  const result = await scoreLatexPair({ prediction: String.raw`\frac{1}{2}`, reference: String.raw`\frac{1}{2}` });

  assert.equal(result.score, 1.0);
  assert.equal(result.diagnostics.parse_status, 'ok');
  assert.equal(result.diagnostics.render_status, 'ok');
  assert.equal(result.diagnostics.timeout, false);
  assert.equal(typeof result.diagnostics.fallback_used, 'boolean');
  assert.ok(result.diagnostics.matched_tokens > 0);
});

test('scoreLatexPair validates invalid prediction before rendering reference', async () => {
  const uniqueReference = `cache_probe_${Date.now()}_${Math.random()}`;

  const result = await scoreLatexPair({
    prediction: String.raw`\frac{1}{`,
    reference: uniqueReference,
  });
  const subsequent = await scoreLatexPair({
    prediction: uniqueReference,
    reference: uniqueReference,
  });

  assert.equal(result.score, 0.0);
  assert.equal(result.diagnostics.parse_status, 'failed');
  assert.equal(result.diagnostics.render_status, 'error');
  assert.equal(subsequent.diagnostics.cache_hit, false);
});

test('different formulas score lower than identical formulas', async () => {
  const identical = await scoreLatexPair({ prediction: 'x + y', reference: 'x + y' });
  const different = await scoreLatexPair({ prediction: 'x + z', reference: 'x + y' });

  assert.equal(identical.score, 1.0);
  assert.ok(different.score >= 0.0);
  assert.ok(different.score < identical.score);
  assert.equal(different.diagnostics.parse_status, 'ok');
});

test('POST /score returns structured zero-score failure for invalid LaTeX', async (t) => {
  const baseUrl = await withServer(t);

  const { response, payload } = await postJson(`${baseUrl}/score`, {
    prediction: String.raw`\frac{1}{`,
    reference: 'x',
  });

  assert.equal(response.status, 200);
  assert.equal(payload.score, 0.0);
  assert.equal(payload.diagnostics.parse_status, 'failed');
  assert.equal(payload.diagnostics.render_status, 'error');
  assert.equal(payload.diagnostics.timeout, false);
  assert.equal(payload.diagnostics.fallback_used, false);
  assert.match(payload.diagnostics.message, /invalid latex/i);
});

test('POST /score scores valid pairs and includes compact diagnostics', async (t) => {
  const baseUrl = await withServer(t);

  const { response, payload } = await postJson(`${baseUrl}/score`, {
    prediction: 'x + 1',
    reference: 'x + 1',
  });

  assert.equal(response.status, 200);
  assert.equal(payload.score, 1.0);
  assert.equal(payload.diagnostics.render_status, 'ok');
  assert.equal(payload.diagnostics.parse_status, 'ok');
  assert.equal(payload.diagnostics.timeout, false);
  assert.ok(payload.diagnostics.token_count_prediction > 0);
  assert.ok(payload.diagnostics.cache_hit === true || payload.diagnostics.cache_hit === false);
});

test('POST /score_batch preserves item identifiers and returns one result per input', async (t) => {
  const baseUrl = await withServer(t);

  const { response, payload } = await postJson(`${baseUrl}/score_batch`, {
    items: [
      { id: 'same', prediction: 'a+b', reference: 'a+b' },
      { id: 'diff', prediction: 'a+c', reference: 'a+b' },
      { id: 42, prediction: String.raw`\sqrt{`, reference: 'a' },
    ],
  });

  assert.equal(response.status, 200);
  assert.deepEqual(payload.results.map((item) => item.id), ['same', 'diff', 42]);
  assert.equal(payload.results.length, 3);
  assert.equal(payload.results[0].score, 1.0);
  assert.ok(payload.results[1].score < payload.results[0].score);
  assert.equal(payload.results[2].score, 0.0);
  assert.equal(payload.results[2].diagnostics.parse_status, 'failed');
});
