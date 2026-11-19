// ------------------------------------------------------------------ settings

export const ASSISTANT_NAME = "rag-tuda";
export const API_PATH = "/api/chat";
export const PROFILE_PATH = "/api/profile";
export const STORAGE_KEY = "assistant-chat-v1";
export const CHIPS_KEY = "assistant-chips-v1";
export const TOKEN_KEY = "assistant-token-v1";
export const OLD_AFTER_YEARS = 3;
export const VISIBLE_SOURCES = 3;
export const RESUME_THRESHOLD_PX = 8;
export const NEAR_END_PX = 48;
// Page-side watchdog before the meta frame has named the request's limits (it comes first): the longest silence and the
// longest request the page waits for. The meta frame's deadline_s and idle_s replace them, plus WATCHDOG_GRACE_MS.
export const SILENCE_LIMIT_MS = 90000;
export const TOTAL_LIMIT_MS = 150000;
export const WATCHDOG_GRACE_MS = 5000;
// Question length: the backend answers longer questions only with a "too long" reply (guard.max_question_chars in the
// profile, keep in step; a contract test checks it); the counter appears from QUESTION_WARN_AT (80 %).
export const QUESTION_LIMIT = 2000;
export const QUESTION_WARN_AT = 1600;
export const FRAME_MARK = "data: ";
export const FORM_TYPE = "Formular";
export const YEAR_PATTERN = /(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)/g;
export const CITATION_PATTERN = /\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g;
// An e-mail address, or a Darmstadt telephone number (area code 06151, or +49 6151, then up to four groups of digits).
export const CONTACT_PATTERN = new RegExp(
    "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}|(?<![\\d+])(?:\\+49[ \\u00a0]?6151|06151)(?:[ \\u00a0/-]?\\d+){1,4}", "g");

// Response formats v1: the page tells the backend it renders them; the shape names match the backend's.
export const FORMATS_VERSION = "v1";
export const SHAPE_NAMES = ["text", "table", "checklist", "steps", "mindmap"];
export { VARIANT_LABELS } from "./i18n.js";
// The "⋯" menu: ids and the keys of their texts (TEXT).
export const MENU_GROUPS = [
    { label: "menuForm", kind: "model", items: [["table", "asTable"], ["checklist", "asChecklist"], ["steps", "asSteps"], ["mindmap", "asMindmap"]] },
    { label: "menuLength", kind: "model", items: [["shorter", "shorter"], ["longer", "longer"]] },
    { label: "menuLanguage", kind: "model", items: [["simple", "simple"], ["english", "english"]] },
    { label: "menuExport", kind: "export", items: [["mail", "asMail"], ["print", "asPrint"]] },
];
export const MAILTO_LIMIT = 1800;
// Pinned, integrity-checked, loaded only when the first mind map or number table appears.
export const LIBRARIES = {
    d3: { src: "vendor/d3-7.9.0.min.js",
        integrity: "sha384-CjloA8y00+1SDAUkjs099PVfnY2KmDC2BZnws9kh8D/lX1s46w6EPhpXdqMfjK6i" },
    markmap: { src: "vendor/markmap-view-0.18.12.js",
        integrity: "sha384-p+gyhsDIg0RmvIKRr9BBGSyJ9NDDkiFsbilRwdZd20Q8mUL2v7e8+orY4pvTV52w" },
    chart: { src: "vendor/chart-4.5.1.umd.min.js",
        integrity: "sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ" },
};

// The degree names and the page's texts in the language in use.
export { DEGREE_NAMES, TEXT } from "./i18n.js";
