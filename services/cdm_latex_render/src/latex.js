const COMMAND_RE = /^\\[a-zA-Z]+/;

function normalizeLatex(input) {
  return String(input ?? '')
    .replace(/\s+/g, ' ')
    .trim();
}

function validateLatex(latex) {
  const stack = [];
  const pairs = { '{': '}', '[': ']', '(': ')' };
  const closers = new Set(Object.values(pairs));
  for (let i = 0; i < latex.length; i += 1) {
    const ch = latex[i];
    if (pairs[ch]) {
      stack.push({ opener: ch, index: i });
      continue;
    }
    if (closers.has(ch)) {
      const last = stack.pop();
      if (!last || pairs[last.opener] !== ch) {
        return { ok: false, message: `Invalid LaTeX: unmatched '${ch}' at position ${i}` };
      }
    }
  }
  if (stack.length > 0) {
    const last = stack[stack.length - 1];
    return { ok: false, message: `Invalid LaTeX: unmatched '${last.opener}' at position ${last.index}` };
  }
  if (/\\(?:frac|sqrt|overline|underline|text|mathrm|mathbf|mathit)\s*(?:$|[^\s{[])/.test(latex)) {
    return { ok: false, message: 'Invalid LaTeX: command requires an argument group' };
  }
  if (/\\$/.test(latex)) {
    return { ok: false, message: 'Invalid LaTeX: trailing escape' };
  }
  return { ok: true };
}

function tokenizeLatex(input) {
  const normalized = normalizeLatex(input);
  const validation = validateLatex(normalized);
  if (!validation.ok) {
    const error = new Error(validation.message);
    error.code = 'INVALID_LATEX';
    throw error;
  }

  const tokens = [];
  let i = 0;
  while (i < normalized.length) {
    const ch = normalized[i];
    if (/\s/.test(ch)) {
      i += 1;
      continue;
    }
    if (ch === '\\') {
      const rest = normalized.slice(i);
      const match = rest.match(COMMAND_RE);
      if (match) {
        tokens.push(match[0]);
        i += match[0].length;
        continue;
      }
      if (i + 1 < normalized.length) {
        tokens.push(normalized.slice(i, i + 2));
        i += 2;
        continue;
      }
    }
    if (/[a-zA-Z0-9]/.test(ch)) {
      tokens.push(ch);
      i += 1;
      continue;
    }
    if ('{}[]()'.includes(ch)) {
      i += 1;
      continue;
    }
    tokens.push(ch);
    i += 1;
  }

  return {
    normalized,
    tokens: tokens.map((value, index) => ({ id: `t${index}`, value, index })),
  };
}

module.exports = {
  normalizeLatex,
  tokenizeLatex,
};
