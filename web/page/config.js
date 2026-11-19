// ------------------------------------------------------------------ settings

export const ASSISTANT_NAME = "Assistant";
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
export const CONTACT_PATTERN = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;

// Response formats v1: the page tells the backend it renders them; the shape names match the backend's.
export const FORMATS_VERSION = "v1";
export const SHAPE_NAMES = ["text", "table", "checklist", "steps", "mindmap"];
export const VARIANT_LABELS = { original: "Original", table: "Tabelle", checklist: "Checkliste", steps: "Ablauf",
    mindmap: "Mindmap", shorter: "Kürzer", longer: "Ausführlicher", simple: "Einfache Sprache", english: "English" };
export const MENU_GROUPS = [
    { label: "Form", kind: "model", items: [["table", "Als Tabelle"], ["checklist", "Als Checkliste"], ["steps", "Als Ablauf"], ["mindmap", "Als Mindmap"]] },
    { label: "Länge", kind: "model", items: [["shorter", "Kürzer"], ["longer", "Ausführlicher"]] },
    { label: "Sprache", kind: "model", items: [["simple", "Einfache Sprache"], ["english", "Auf Englisch"]] },
    { label: "Export", kind: "export", items: [["mail", "Als E-Mail"], ["print", "Drucken / PDF"]] },
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

// The degree ids roles.yaml uses, as the chips and the filter line show them.
export const DEGREE_NAMES = { bachelor: "Bachelor", master: "Master", promotion: "Promotion" };

export const TEXT = {
    pending: "Frage wird bearbeitet …",
    pendingDone: "Frage bearbeitet",
    analysing: "Frage wird analysiert …",
    analysingDone: "Frage analysiert",
    searching: "Suche in den Dokumenten …",
    searchingDone: "In den Dokumenten gesucht",
    writing: "Antwort wird formuliert …",
    writingDone: "Antwort formuliert",
    failed: "Die Antwort konnte nicht geladen werden. Bitte versuchen Sie es noch einmal.",
    unauthorized: "Der Zugang wurde nicht akzeptiert. Bitte öffnen Sie den Link aus „rag up“ noch einmal.",
    chipsRefused: "Diese Auswahl aus Abschluss, Studiengang und PO ist nicht möglich. Bitte ändern Sie die Auswahl über dem Eingabefeld.",
    chipsLabel: "Auswahl für Ihre Frage",
    chipDegree: "Abschluss",
    chipProgramme: "Studiengang",
    chipPo: "PO-Version",
    filterApplied: "Filter: {filter}",
    stopped: "Antwort angehalten.",
    interrupted: "Die Antwort wurde unterbrochen.",
    reference: "Bitte nennen Sie bei Rückfragen die Referenz {ref}.",
    sourcesNote: "KI-generiert – verbindlich ist das Originaldokument.",
    copy: "Kopieren",
    copied: "Kopiert",
    retry: "Nochmal versuchen",
    edit: "Bearbeiten",
    editing: "Frage {n} wird bearbeitet – alle späteren Nachrichten werden entfernt.",
    cancelEdit: "Abbrechen",
    nearLimit: "Ihre Frage wird lang. Tipp: Teilen Sie umfangreiche Anliegen in mehrere Fragen auf – der Assistent berücksichtigt dabei die letzten Antworten.",
    overLimit: "Zu lang zum Absenden (höchstens {limit} Zeichen). Bitte kürzen Sie die Frage auf das Wesentliche oder stellen Sie mehrere kürzere Fragen nacheinander.",
    overLimitTitle: "Die Frage ist zu lang zum Absenden",
    send: "Absenden",
    stop: "Antwort anhalten",
    oldDoc: "älteres Dokument",
    openForm: "Formular öffnen",
    newTab: "öffnet in neuem Tab",
    less: "Weniger anzeigen",
    more: "Weitere Aktionen",
    versions: "Versionen dieser Antwort",
    tick: "Erledigt",
    mindmapWait: "Mindmap wird erstellt …",
    mindmapFull: "Vollbild",
    mindmapFullExit: "Vollbild beenden",
    menuNote: "Neue Fassung aus der Original-Antwort",
    mindmapSvg: "Als Bild (SVG)",
    mindmapMm: "Für Freeplane (.mm)",
    showChart: "Diagramm",
    showTable: "Tabelle",
    chartLabel: "Diagramm aus der Tabelle",
    variantFailed: "Die Umwandlung ist leider nicht gelungen. Die bisherige Antwort bleibt.",
    variantStopped: "Umwandlung angehalten. Die bisherige Antwort bleibt.",
    selectFailed: "Die Version konnte nicht gewechselt werden. Bitte versuchen Sie es noch einmal.",
    mailPaste: "Text liegt in der Zwischenablage – bitte einfügen.",
};
