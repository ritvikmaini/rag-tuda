// ---- page language: the strings of both languages, the one in use, and the switch between them.
// German is the default and the fallback: a key missing in English shows its German text.

export const LANGS = ["de", "en"];
export const LANG_KEY = "assistant-lang-v1";

export const STRINGS = {
    de: {
        // the page's own markup (index.html, data-i18n)
        headerSub: "für den Fachbereich Informatik der TU Darmstadt",
        newChat: "Neues Gespräch",
        newChatTitle: "Neues Gespräch beginnen",
        greeting: "Hallo, ich bin {name}, ein KI-Assistent für Fragen rund ums Informatikstudium an der TU Darmstadt.",
        greetingMore: "Ich beantworte gerne Ihre Fragen zu den Dokumenten.",
        intro: "Meine Antworten stützen sich auf die hinterlegten Dokumente; die Quellen stehen jeweils dabei. Bitte geben Sie keine persönlichen Daten ein.",
        jump: "Zur neuesten Antwort",
        questionLabel: "Ihre Frage",
        placeholder: "Was möchten Sie wissen?",
        footer: "Studierendenprojekt, kein offizieller Dienst der TU Darmstadt. Keine Rechtsberatung: Alle Antworten werden KI-generiert und sind ohne Gewähr.",
        providerLine: "Antworten von {model} über {provider}.",
        switchTo: "EN",
        switchLabel: "Switch to English",
        // thinking steps
        pending: "Frage wird bearbeitet …",
        pendingDone: "Frage bearbeitet",
        analysing: "Frage wird analysiert …",
        analysingDone: "Frage analysiert",
        understood: "Verstanden als: „{question}“",
        searching: "Suche in den Dokumenten …",
        searchingFor: "Suche nach {queries} …",
        searchingDone: "In den Dokumenten gesucht",
        searchedFor: "Gesucht nach: {queries}",
        searchingMore: "Suche nach weiteren Dokumenten …",
        searchingMoreDone: "Weitere Dokumente gesucht",
        noDocuments: "Keine passenden Dokumente gefunden",
        documentOne: "{n} Dokument",
        documentMany: "{n} Dokumente",
        foundOne: "{n} Dokument gefunden",
        foundMany: "{n} Dokumente gefunden",
        noDocumentsShort: "keine Dokumente",
        queryOne: "{n} Suchbegriff",
        queryMany: "{n} Suchbegriffe",
        writing: "Antwort wird formuliert …",
        writingDone: "Antwort formuliert",
        decimal: ",",
        // notices
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
        // sources
        sources: "Quellen",
        sourcesNote: "KI-generiert – verbindlich ist das Originaldokument.",
        sourcesCopy: "Quellen:",
        sourceLabel: "Quelle {n}: {title}",
        pageAbbr: "S.",
        moreSources: "+ {n} weitere",
        document: "Dokument",
        oldDoc: "älteres Dokument",
        openForm: "Formular öffnen",
        official: "Amtlich veröffentlicht",
        poVersion: "PO {years}",
        newTab: "öffnet in neuem Tab",
        less: "Weniger anzeigen",
        // composer and actions
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
        more: "Weitere Aktionen",
        versions: "Versionen dieser Antwort",
        tick: "Erledigt",
        mindmapWait: "Mindmap wird erstellt …",
        mindmapFull: "Vollbild",
        mindmapFullExit: "Vollbild beenden",
        mindmapRoot: "Überblick",
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
        mailSubject: "Auskunft: {question}",
        printLocale: "de-DE",
        // the "⋯" menu
        menuForm: "Form",
        menuLength: "Länge",
        menuLanguage: "Sprache",
        menuExport: "Export",
        asTable: "Als Tabelle",
        asChecklist: "Als Checkliste",
        asSteps: "Als Ablauf",
        asMindmap: "Als Mindmap",
        shorter: "Kürzer",
        longer: "Ausführlicher",
        simple: "Einfache Sprache",
        english: "Auf Englisch",
        asMail: "Als E-Mail",
        asPrint: "Drucken / PDF",
    },
    en: {
        headerSub: "for the Department of Computer Science at TU Darmstadt",
        newChat: "New conversation",
        newChatTitle: "Start a new conversation",
        greeting: "Hello, I am {name}, an AI assistant for questions about studying computer science at TU Darmstadt.",
        greetingMore: "I am happy to answer your questions about the documents.",
        intro: "My answers rest on the stored documents; the sources are listed with each answer. Please do not enter personal data.",
        jump: "Jump to the latest answer",
        questionLabel: "Your question",
        placeholder: "What would you like to know?",
        footer: "Student project, not an official service of TU Darmstadt. Not legal advice: all answers are AI-generated and come without guarantee.",
        providerLine: "Answers by {model} via {provider}.",
        switchTo: "DE",
        switchLabel: "Auf Deutsch wechseln",
        pending: "Processing the question …",
        pendingDone: "Question processed",
        analysing: "Analysing the question …",
        analysingDone: "Question analysed",
        understood: "Understood as: “{question}”",
        searching: "Searching the documents …",
        searchingFor: "Searching for {queries} …",
        searchingDone: "Searched the documents",
        searchedFor: "Searched for: {queries}",
        searchingMore: "Searching for further documents …",
        searchingMoreDone: "Searched for further documents",
        noDocuments: "No matching documents found",
        documentOne: "{n} document",
        documentMany: "{n} documents",
        foundOne: "{n} document found",
        foundMany: "{n} documents found",
        noDocumentsShort: "no documents",
        queryOne: "{n} search term",
        queryMany: "{n} search terms",
        writing: "Writing the answer …",
        writingDone: "Answer written",
        decimal: ".",
        failed: "The answer could not be loaded. Please try again.",
        unauthorized: "Access was not accepted. Please open the link from “rag up” again.",
        chipsRefused: "This combination of degree, programme and PO is not possible. Please change the selection above the input field.",
        chipsLabel: "Selection for your question",
        chipDegree: "Degree",
        chipProgramme: "Programme",
        chipPo: "PO version",
        filterApplied: "Filter: {filter}",
        stopped: "Answer stopped.",
        interrupted: "The answer was interrupted.",
        reference: "Please quote the reference {ref} if you ask about this.",
        sources: "Sources",
        sourcesNote: "AI-generated – the original document is binding.",
        sourcesCopy: "Sources:",
        sourceLabel: "Source {n}: {title}",
        pageAbbr: "p.",
        moreSources: "+ {n} more",
        document: "Document",
        oldDoc: "older document",
        openForm: "Open form",
        official: "Officially published",
        poVersion: "PO {years}",
        newTab: "opens in a new tab",
        less: "Show less",
        copy: "Copy",
        copied: "Copied",
        retry: "Try again",
        edit: "Edit",
        editing: "Editing question {n} – all later messages will be removed.",
        cancelEdit: "Cancel",
        nearLimit: "Your question is getting long. Tip: split a large request into several questions – the assistant takes the latest answers into account.",
        overLimit: "Too long to send (at most {limit} characters). Please shorten the question to the essentials or ask several shorter questions in turn.",
        overLimitTitle: "The question is too long to send",
        send: "Send",
        stop: "Stop the answer",
        more: "More actions",
        versions: "Versions of this answer",
        tick: "Done",
        mindmapWait: "Creating the mind map …",
        mindmapFull: "Full screen",
        mindmapFullExit: "Exit full screen",
        mindmapRoot: "Overview",
        menuNote: "New version made from the original answer",
        mindmapSvg: "As image (SVG)",
        mindmapMm: "For Freeplane (.mm)",
        showChart: "Chart",
        showTable: "Table",
        chartLabel: "Chart from the table",
        variantFailed: "The conversion did not work. The previous answer stays.",
        variantStopped: "Conversion stopped. The previous answer stays.",
        selectFailed: "The version could not be switched. Please try again.",
        mailPaste: "The text is on the clipboard – please paste it.",
        mailSubject: "Answer: {question}",
        printLocale: "en-GB",
        menuForm: "Form",
        menuLength: "Length",
        menuLanguage: "Language",
        menuExport: "Export",
        asTable: "As table",
        asChecklist: "As checklist",
        asSteps: "As steps",
        asMindmap: "As mind map",
        shorter: "Shorter",
        longer: "More detailed",
        simple: "Plain language",
        english: "In English",
        asMail: "As e-mail",
        asPrint: "Print / PDF",
    },
};

const DEGREES = {
    de: { bachelor: "Bachelor", master: "Master", promotion: "Promotion" },
    en: { bachelor: "Bachelor", master: "Master", promotion: "Doctorate" },
};

// The tab labels of the versions of an answer (the keys are the backend's variant names).
const VARIANTS = {
    de: { original: "Original", table: "Tabelle", checklist: "Checkliste", steps: "Ablauf", mindmap: "Mindmap",
        shorter: "Kürzer", longer: "Ausführlicher", simple: "Einfache Sprache", english: "English" },
    en: { original: "Original", table: "Table", checklist: "Checklist", steps: "Steps", mindmap: "Mind map",
        shorter: "Shorter", longer: "More detailed", simple: "Plain language", english: "English" },
};

let current = "de";

export function getLang() {
    return current;
}

export function setLang(lang) {
    current = LANGS.includes(lang) ? lang : "de";
}

// The language to start in: the stored choice, else the browser's, else German.
export function detectLang(stored, browser) {
    if (LANGS.includes(stored)) return stored;
    return String(browser || "").toLowerCase().startsWith("en") ? "en" : "de";
}

export function text(key) {
    const own = STRINGS[current][key];
    return own === undefined ? STRINGS.de[key] : own;
}

// `values` fills the {name} marks of a string.
export function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, (mark, name) => (name in values ? String(values[name]) : mark));
}

// Reads like a plain object of the strings in use: TEXT.failed, TEXT.chipPo.
function table(pick) {
    return new Proxy({}, {
        get: (target, key) => (typeof key === "string" ? pick(key) : undefined),
        has: (target, key) => typeof key === "string" && pick(key) !== undefined,
    });
}

export const TEXT = table(text);
export const DEGREE_NAMES = table(key => DEGREES[current][key]);
export const VARIANT_LABELS = table(key => VARIANTS[current][key]);

// The line under the composer about the model that answers; "" while the answer stays on this machine.
export function providerLine(answer) {
    if (!answer || answer.remote !== true) return "";
    return fill(TEXT.providerLine, { model: answer.model, provider: answer.provider });
}

// Puts the strings in use into the page's markup: data-i18n="key" sets the text, data-i18n-attrs="aria-label=key,title=key"
// sets attributes. Strings are set as text, never as markup.
export function applyStatic(root, values) {
    root.querySelectorAll("[data-i18n]").forEach(el => {
        el.textContent = fill(text(el.getAttribute("data-i18n")), values || {});
    });
    root.querySelectorAll("[data-i18n-attrs]").forEach(el => {
        el.getAttribute("data-i18n-attrs").split(",").forEach(pair => {
            const [attr, key] = pair.split("=");
            el.setAttribute(attr.trim(), fill(text(key.trim()), values || {}));
        });
    });
}
