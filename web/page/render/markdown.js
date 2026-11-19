import { CONTACT_PATTERN } from "../config.js";
import { markCharts } from "../features/chart.js";
import { markMindmaps } from "../features/mindmap.js";
import { markTasks } from "../features/tasks.js";
import { citationFacts, citationHref, citationLabel, citationParts, entriesForTurn } from "../model/citations.js";
import { readableSlug } from "../model/sources.js";
import { escapeHtml, keepPlaceholders } from "../model/text.js";
import { h, svg } from "./dom.js";
import { state, store } from "../ui/state.js";

// ---- answer rendering

export function renderMarkdown(markdown) {
    if (state.converter && window.DOMPurify) {
        return window.DOMPurify.sanitize(state.converter.makeHtml(keepPlaceholders(markdown)), {
            ADD_ATTR: ["target"],
            // answers are text: no styles, images, forms or embedded documents (overlays, remote requests)
            FORBID_TAGS: ["style", "img", "form", "input", "button", "textarea", "select", "option", "iframe", "svg", "math"],
            FORBID_ATTR: ["style"],
        });
    }
    return escapeHtml(markdown).replace(/\n/g, "<br>");
}

export function textNodes(root) {
    const out = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            return node.parentElement && node.parentElement.closest("a, code, pre, button") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
        },
    });
    while (walker.nextNode()) out.push(walker.currentNode);
    return out;
}

// The link of an e-mail address (mailto:) or of a 06151 / +49 6151 telephone number (tel:, in international form).
export function contactHref(contact) {
    if (contact.includes("@")) return "mailto:" + contact;
    return "tel:+49" + contact.replace(/\D/g, "").replace(/^(?:49)?0?/, "");
}

// A text cut into plain parts ({ text }) and contact links ({ text, href }).
export function contactParts(text) {
    const parts = [];
    let last = 0;
    let match;
    CONTACT_PATTERN.lastIndex = 0;
    while ((match = CONTACT_PATTERN.exec(text)) !== null) {
        if (match.index > last) parts.push({ text: text.slice(last, match.index) });
        parts.push({ text: match[0], href: contactHref(match[0]) });
        last = match.index + match[0].length;
    }
    if (last < text.length || !parts.length) parts.push({ text: text.slice(last) });
    return parts;
}

export function linkifyContacts(root) {
    textNodes(root).forEach(node => {
        const parts = contactParts(node.nodeValue);
        if (parts.length === 1 && !parts[0].href) return;
        node.replaceWith(...parts.map(part => (part.href ? h("a", { href: part.href, text: part.text }) : part.text)));
    });
}

// [n] in the answer text become markers: links to the cited document and page once the answer's citations are known
// (after the answer), plain numbers otherwise (while streaming, unknown numbers, sources without a link).
export function markCitations(root, entries) {
    const byN = new Map((entries || []).map(entry => [entry.n, entry]));
    textNodes(root).forEach(node => {
        const parts = citationParts(node.nodeValue);
        if (!parts.some(part => part.refs)) return;
        const fragment = document.createDocumentFragment();
        parts.forEach(part => {
            if (part.text !== undefined) fragment.append(part.text);
            else part.refs.forEach(n => fragment.append(citationMarker(n, byN.get(n))));
        });
        node.replaceWith(fragment);
    });
}

export function citationMarker(n, entry) {
    if (!entry || !entry.url) return h("span", { class: "cite", "data-n": n, text: String(n) });
    return h("a", {
        class: "cite",
        "data-n": n,
        href: citationHref(entry.url, entry.page),
        target: "_blank",
        rel: "noopener noreferrer",
        "aria-label": citationLabel(entry),
        "data-tip-title": entry.title || readableSlug(entry.url),
        "data-tip-facts": citationFacts(entry),
        "data-tip-quote": entry.quote,
        text: String(n),
    });
}

export function decorateAnswer(body, turn, final) {
    body.querySelectorAll("a[href]").forEach(a => {
        a.target = "_blank";
        a.rel = "noopener noreferrer";
    });
    body.querySelectorAll("table").forEach(table => {
        if (table.parentElement.classList.contains("table-wrap")) return;
        const wrap = h("div", { class: "table-wrap" });
        table.replaceWith(wrap);
        wrap.appendChild(table);
    });
    markMindmaps(body, turn, final);
    markTasks(body, turn, final);
    if (final) markCharts(body);
    if (final) linkifyContacts(body);
    // markers link to their sources once the answer's citations are known (after the answer); a failure here leaves
    // the answer as rendered
    try {
        markCitations(body, final ? entriesForTurn(store.turns, store.turns.indexOf(turn)) : null);
    } catch (error) {
        console.warn("[Citations] markers not linked:", error);
    }
}
