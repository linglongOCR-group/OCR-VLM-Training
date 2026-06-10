const { DEFAULT_TIMEOUT_MS } = require('./constants');
const { tokenizeLatex } = require('./latex');
const { renderLatex } = require('./renderer');

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0.0;
  }
  return Math.max(0.0, Math.min(1.0, value));
}

function tokenCost(predToken, refToken, predTotal, refTotal) {
  if (predToken.value !== refToken.value) {
    return Infinity;
  }
  const orderCost = Math.abs(predToken.box.order - refToken.box.order);
  const positionScale = Math.max(predTotal, refTotal, 1) * 12;
  const dx = Math.abs(predToken.box.centerX - refToken.box.centerX) / positionScale;
  const dy = Math.abs(predToken.box.centerY - refToken.box.centerY) / 32;
  return orderCost * 0.6 + dx * 0.3 + dy * 0.1;
}

function matchTokens(predTokens, refTokens) {
  const candidates = [];
  for (const pred of predTokens) {
    for (const ref of refTokens) {
      const cost = tokenCost(pred, ref, predTokens.length, refTokens.length);
      if (Number.isFinite(cost)) {
        candidates.push({ pred, ref, cost });
      }
    }
  }
  candidates.sort((a, b) => a.cost - b.cost || a.pred.index - b.pred.index || a.ref.index - b.ref.index);

  const usedPred = new Set();
  const usedRef = new Set();
  const matches = [];
  for (const candidate of candidates) {
    if (candidate.cost > 0.65) {
      continue;
    }
    if (usedPred.has(candidate.pred.index) || usedRef.has(candidate.ref.index)) {
      continue;
    }
    usedPred.add(candidate.pred.index);
    usedRef.add(candidate.ref.index);
    matches.push(candidate);
  }

  if (matches.length <= 2) {
    return matches;
  }
  const costs = matches.map((match) => match.cost).sort((a, b) => a - b);
  const median = costs[Math.floor(costs.length / 2)];
  const threshold = Math.max(0.35, median * 3 + 0.05);
  return matches.filter((match) => match.cost <= threshold);
}

function f1Score(matchCount, predCount, refCount) {
  if (predCount === 0 && refCount === 0) {
    return 1.0;
  }
  if (predCount === 0 || refCount === 0 || matchCount === 0) {
    return 0.0;
  }
  const precision = matchCount / predCount;
  const recall = matchCount / refCount;
  return clamp01((2 * precision * recall) / (precision + recall));
}

function failureResult(message, overrides = {}) {
  return {
    score: 0.0,
    diagnostics: {
      render_status: overrides.render_status || 'error',
      parse_status: overrides.parse_status || 'failed',
      timeout: Boolean(overrides.timeout),
      fallback_used: Boolean(overrides.fallback_used),
      message: String(message || 'CDM scoring failed').slice(0, 160),
    },
  };
}

async function scoreLatexPair({ prediction, reference, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  try {
    tokenizeLatex(prediction);
    tokenizeLatex(reference);

    const started = Date.now();
    const [predRendered, refRendered] = await Promise.all([
      renderLatex(prediction, { timeoutMs }),
      renderLatex(reference, { timeoutMs }),
    ]);
    if (Date.now() - started > timeoutMs) {
      return failureResult(`CDM scoring timed out after ${timeoutMs}ms`, {
        parse_status: 'ok',
        timeout: true,
        fallback_used: predRendered.fallbackUsed || refRendered.fallbackUsed,
      });
    }

    const matches = matchTokens(predRendered.tokens, refRendered.tokens);
    const score = f1Score(matches.length, predRendered.tokens.length, refRendered.tokens.length);
    return {
      score,
      diagnostics: {
        render_status: 'ok',
        parse_status: 'ok',
        timeout: false,
        fallback_used: predRendered.fallbackUsed || refRendered.fallbackUsed,
        renderer: predRendered.renderer,
        token_count_prediction: predRendered.tokens.length,
        token_count_reference: refRendered.tokens.length,
        matched_tokens: matches.length,
        filtered_matches: Math.max(0, Math.min(predRendered.tokens.length, refRendered.tokens.length) - matches.length),
        cache_hit: Boolean(predRendered.cacheHit && refRendered.cacheHit),
      },
    };
  } catch (error) {
    if (error && error.code === 'RENDER_TIMEOUT') {
      return failureResult(error.message, { parse_status: 'ok', timeout: true });
    }
    return failureResult(error && error.message ? error.message : 'Invalid LaTeX');
  }
}

module.exports = {
  scoreLatexPair,
  matchTokens,
  f1Score,
};
