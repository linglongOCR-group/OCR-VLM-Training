const SERVICE_NAME = 'cdm-latex-render';
const SERVICE_VERSION = 'cdm_katex_v1';
const DEFAULT_PORT = Number(process.env.PORT || process.env.CDM_LATEX_RENDER_PORT || 8765);
const DEFAULT_TIMEOUT_MS = Number(process.env.CDM_LATEX_RENDER_TIMEOUT_MS || 1000);

module.exports = {
  SERVICE_NAME,
  SERVICE_VERSION,
  DEFAULT_PORT,
  DEFAULT_TIMEOUT_MS,
};
