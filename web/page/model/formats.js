import { CITATION_PATTERN, MAILTO_LIMIT, MENU_GROUPS, SHAPE_NAMES, TEXT } from "../config.js";
import { fill } from "../i18n.js";
import { copyText, escapeHtml } from "./text.js";

// ---- response formats v1: pure helpers

export function shapeName(value) {
    return SHAPE_NAMES.indexOf(value) >= 0 ? value : "text";
}

// The "⋯" menu: shapes, length and language only on the last finished answer that had sources (a citations message) and
// an id; the version on screen is disabled. Export on every answer with text.
export function menuEntries(turn, isLast) {
    if (!turn || !String(turn.answer || "").trim()) return [];
    const canVary = !!isLast && turn.state === "done" && !!turn.id && Array.isArray(turn.citations) && turn.citations.length > 0
        && typeof turn.shape === "string";
    const shown = turn.active || "original";
    // neither the version on screen nor the form the original already has is offered (a table answer offers no "Als
    // Tabelle"; a table that was asked for but not written still is). Every version is made from the original.
    const has = detectShape(originalAnswer(turn));
    const same = name => name === shown || name === has;
    return MENU_GROUPS.filter(group => group.kind === "export" || canVary).map(group => ({
        label: TEXT[group.label],
        kind: group.kind,
        items: group.items.map(([name, label]) => ({ name, label: TEXT[label], enabled: group.kind === "export" || !same(name) })),
    }));
}

// The original answer of a turn: its stored "original" version, else the answer itself (no versions yet).
export function originalAnswer(turn) {
    const original = (turn.versions || []).find(v => v && v.name === "original");
    return original ? original.answer : turn.answer;
}

// The line at the top of the menu while another version than the original is on screen: new versions are always made
// from the original answer, not from the version shown.
export function menuNote(turn, isLast) {
    const varies = menuEntries(turn, isLast).some(group => group.kind !== "export");
    return varies && (turn.active || "original") !== "original" ? TEXT.menuNote : "";
}

// What an answer contains, as the backend's shape detection: a mind-map fence, a Markdown table with a data
// row, at least 2 checklist items, at least 2 numbered lines, else text.
export const SHAPE_ROW = /^[ \t]*\|.*\|[ \t]*$/;
export const SHAPE_SEPARATOR = /^[ \t]*\|[ \t:|-]*-[ \t:|-]*\|[ \t]*$/;
export const SHAPE_TASK = /^[ \t]*[-*][ \t]+\[[ xX]\][ \t]+\S/;
export const SHAPE_STEP = /^[ \t]*\d{1,2}[.)][ \t]+\S/;

export function detectShape(answer) {
    const text = String(answer || "");
    if (/```mindmap[ \t]*\n/.test(text)) return "mindmap";
    const lines = text.split(/\r?\n/);
    let inTable = false;
    for (const line of lines) {
        if (SHAPE_SEPARATOR.test(line)) inTable = true;
        else if (inTable && SHAPE_ROW.test(line)) return "table";
        else if (!SHAPE_ROW.test(line)) inTable = false;
    }
    if (lines.filter(line => SHAPE_TASK.test(line)).length >= 2) return "checklist";
    if (lines.filter(line => SHAPE_STEP.test(line)).length >= 2) return "steps";
    return "text";
}

// "[ ] text" or "[x] text" at the start of a list item.
export function parseTask(text) {
    const m = /^\s*\[([ xX])\]\s+([\s\S]*)$/.exec(String(text || ""));
    return m ? { checked: m[1] !== " ", rest: m[2] } : null;
}

export function tickKey(active, index) {
    return (active || "original") + ":" + index;
}

// A table cell as a number: German thousands and decimals, %, €, Tsd./Mio./Mrd., [n] removed; anything else is null.
export function parseGermanNumber(cell) {
    let s = String(cell === null || cell === undefined ? "" : cell).replace(CITATION_PATTERN, "").replace(/ /g, " ").trim();
    if (!s) return null;
    let factor = 1;
    if (/\bMrd\b/i.test(s)) factor = 1e9;
    else if (/\bMio\b/i.test(s)) factor = 1e6;
    else if (/\bTsd\b/i.test(s)) factor = 1e3;
    s = s.replace(/\b(Mrd|Mio|Tsd)\.?/gi, "").replace(/€|%|\bEUR\b|\bEuro\b/gi, "").replace(/\s+/g, " ").trim();
    // spaces only as thousands separators ("58 316"); a leading zero is a phone number or an id, not a value
    if (!/^[-+−]?(?:\d{1,3}(?:\.\d{3})+|\d{1,3}(?: \d{3})+|\d+)(?:,\d+)?$/.test(s) || /^[-+−]?0\d/.test(s)) return null;
    const value = Number(s.replace("−", "-").replace(/[. ]/g, "").replace(",", "."));
    return Number.isFinite(value) ? value * factor : null;
}

// Chart data from a table: the first column gives the labels, up to three columns whose every cell is a number give the
// series; at least three rows; labels that are all years give a line, anything else bars. null when nothing fits.
export function chartSpec(header, rows) {
    if (!Array.isArray(header) || header.length < 2 || !Array.isArray(rows) || rows.length < 3) return null;
    const clean = text => String(text || "").replace(CITATION_PATTERN, "").trim();
    let series = [];
    for (let c = 1; c < header.length; c++) {
        // the source column (tables end with one) is never a series, also when it holds bare numbers
        if (/^quellen?$/i.test(clean(header[c]))) continue;
        const values = rows.map(row => parseGermanNumber(Array.isArray(row) ? row[c] : null));
        if (values.every(v => v !== null)) {
            series.push({ name: clean(header[c]), values, years: values.every(v => Number.isInteger(v) && v >= 1900 && v <= 2099) });
        }
    }
    // a column of years is a label, not a value, when any other column qualifies
    if (series.some(s => !s.years)) series = series.filter(s => !s.years);
    series = series.slice(0, 3).map(s => ({ name: s.name, values: s.values }));
    if (!series.length) return null;
    const labels = rows.map(row => clean(Array.isArray(row) ? row[0] : ""));
    return { kind: labels.every(label => /^(19|20)\d\d$/.test(label)) ? "line" : "bar", labels, series };
}

export function chartConfig(spec) {
    return {
        type: spec.kind,
        data: { labels: spec.labels, datasets: spec.series.map(s => ({ label: s.name, data: s.values })) },
        // German number format on the axes and in the tooltips (66.500), whatever the browser's language
        options: { animation: false, responsive: true, maintainAspectRatio: false, locale: "de-DE",
            plugins: { legend: { display: spec.series.length > 1 } } },
    };
}

// A mind map from the ```mindmap block: "# Thema", "## Bereich", "- Punkt"; deeper items join their area. The node text
// is escaped here, because markmap inserts it as HTML.
export function mindmapTree(src) {
    let root = null;
    let area = null;
    const node = text => ({ content: escapeHtml(text.trim()), children: [] });
    String(src || "").split(/\r?\n/).forEach(line => {
        let m;
        if (!root && (m = /^#\s+(.+)$/.exec(line))) {
            root = node(m[1]);
        } else if ((m = /^##\s+(.+)$/.exec(line))) {
            if (!root) root = node(TEXT.mindmapRoot);
            area = node(m[1]);
            root.children.push(area);
        } else if ((m = /^\s*[-*]\s+(.+)$/.exec(line))) {
            if (!root) root = node(TEXT.mindmapRoot);
            (area || root).children.push(node(m[1]));
        }
    });
    return root && root.children.length ? root : null;
}

export function freeplaneXml(tree) {
    const text = s => escapeHtml(String(s).replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&"));
    const walk = n => '<node TEXT="' + text(n.content) + '">' + n.children.map(walk).join("") + "</node>";
    return '<map version="1.0.1">' + walk(tree) + "</map>";
}

// The answer as plain text for an e-mail: table rows as "cell – cell", checklist items with a box, Markdown marks removed.
export function plainAnswer(markdown) {
    const out = [];
    String(markdown || "").split("\n").forEach(line => {
        if (/^\s*\|[\s:|-]+\|\s*$/.test(line) || /^\s*```/.test(line)) return;
        let m;
        if ((m = /^\s*\|(.*)\|\s*$/.exec(line))) out.push(m[1].split("|").map(cell => cell.trim()).join(" – "));
        else if ((m = /^(\s*)[-*]\s+\[[ xX]\]\s+(.*)$/.exec(line))) out.push(m[1] + "☐ " + m[2]);
        else out.push(line.replace(/^#{1,6}\s+/, "").replace(/\*\*(.+?)\*\*/g, "$1").replace(/__(.+?)__/g, "$1"));
    });
    return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

// mailto: with subject and body; a body too long for a mailto link goes to the clipboard instead.
export function mailPlan(question, answerMarkdown, cards) {
    const q = String(question || "").trim();
    const subject = fill(TEXT.mailSubject, { question: q.length > 60 ? q.slice(0, 59) + "…" : q });
    const body = copyText(plainAnswer(answerMarkdown), cards);
    const href = "mailto:?subject=" + encodeURIComponent(subject) + "&body=" + encodeURIComponent(body);
    if (href.length <= MAILTO_LIMIT) return { href, clipboard: null };
    return { href: "mailto:?subject=" + encodeURIComponent(subject) + "&body=" + encodeURIComponent(TEXT.mailPaste), clipboard: body };
}

// The answer with an odd number of ``` fences closed, so a cut-off mind map block is still drawn (nothing else is healed).
export function closeOpenFence(text) {
    const s = String(text || "");
    return (s.match(/```/g) || []).length % 2 === 1 ? s + (s.endsWith("\n") ? "" : "\n") + "```" : s;
}

// Where the "⋯" menu opens: below its button when it fits above the composer, else above it when it fits under the header,
// else on the roomier side with a scrolling height (a short window, the page embedded in a frame). above and below are the
// free distances in px; the last answer sits right above the composer, so its menu opens upwards.
export const MENU_MIN_HEIGHT_PX = 120;
export const MENU_EDGE_PX = 12;

export function menuPlacement(natural, above, below) {
    if (natural <= below) return { up: false, maxHeight: null };
    if (natural <= above) return { up: true, maxHeight: null };
    const up = above > below;
    return { up, maxHeight: Math.max(MENU_MIN_HEIGHT_PX, Math.floor(up ? above : below)) };
}
