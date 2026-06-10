const http = require('node:http');
const { SERVICE_NAME, SERVICE_VERSION, DEFAULT_PORT } = require('./constants');
const { probeBrowserReady, cacheStats } = require('./renderer');
const { scoreLatexPair } = require('./scoring');

function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(statusCode, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  response.end(body);
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        reject(new Error('request body too large'));
        request.destroy();
      }
    });
    request.on('end', () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (_error) {
        reject(new Error('request body must be valid JSON'));
      }
    });
    request.on('error', reject);
  });
}

async function healthPayload(options = {}) {
  const deps = await probeBrowserReady(options);
  return {
    ok: true,
    healthy: true,
    service: SERVICE_NAME,
    version: SERVICE_VERSION,
    renderer: deps.renderer,
    browser_ready: deps.browserReady,
    cache: cacheStats(),
  };
}

function validateScoreBody(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new Error('request body must be an object');
  }
  if (typeof body.prediction !== 'string' || typeof body.reference !== 'string') {
    throw new Error('prediction and reference must be strings');
  }
}

async function handleScore(body) {
  validateScoreBody(body);
  return scoreLatexPair({
    prediction: body.prediction,
    reference: body.reference,
    timeoutMs: body.timeout_ms,
  });
}

async function route(request, response) {
  const url = new URL(request.url, 'http://127.0.0.1');
  if (request.method === 'GET' && url.pathname === '/health') {
    sendJson(response, 200, await healthPayload());
    return;
  }

  if (request.method === 'POST' && url.pathname === '/score') {
    try {
      const body = await readJson(request);
      const result = await handleScore(body);
      sendJson(response, 200, result);
    } catch (error) {
      sendJson(response, 400, {
        score: 0.0,
        diagnostics: {
          render_status: 'error',
          parse_status: 'failed',
          timeout: false,
          fallback_used: false,
          message: String(error.message || 'invalid request').slice(0, 160),
        },
      });
    }
    return;
  }

  if (request.method === 'POST' && (url.pathname === '/score_batch' || url.pathname === '/score/batch')) {
    try {
      const body = await readJson(request);
      if (!body || !Array.isArray(body.items)) {
        throw new Error('items must be an array');
      }
      const results = [];
      for (const item of body.items) {
        const result = await handleScore(item);
        results.push({ id: item.id, ...result });
      }
      sendJson(response, 200, { results });
    } catch (error) {
      sendJson(response, 400, {
        results: [],
        diagnostics: {
          render_status: 'error',
          parse_status: 'failed',
          timeout: false,
          fallback_used: false,
          message: String(error.message || 'invalid request').slice(0, 160),
        },
      });
    }
    return;
  }

  sendJson(response, 404, { error: 'not found' });
}

function createServer() {
  return http.createServer((request, response) => {
    route(request, response).catch((error) => {
      sendJson(response, 500, {
        error: 'internal_error',
        message: String(error.message || error).slice(0, 160),
      });
    });
  });
}

if (require.main === module) {
  const server = createServer();
  server.listen(DEFAULT_PORT, '127.0.0.1', async () => {
    const deps = await probeBrowserReady();
    console.log(`${SERVICE_NAME} ${SERVICE_VERSION} listening on 127.0.0.1:${DEFAULT_PORT} (${deps.renderer})`);
  });
}

module.exports = {
  createServer,
  healthPayload,
};
