import { CITATION_PATTERN, FORM_TYPE, TEXT, YEAR_PATTERN } from "../config.js";
import { isOld, readableSlug } from "./sources.js";

export function citationParts(text) {
    const parts = [];
    let last = 0;
    let match;
    CITATION_PATTERN.lastIndex = 0;
    while ((match = CITATION_PATTERN.exec(text)) !== null) {
        if (match.index > last) parts.push({ text: text.slice(last, match.index) });
        parts.push({ refs: match[1].split(",").map(s => Number(s.trim())) });
        last = match.index + match[0].length;
    }
    if (last < text.length) parts.push({ text: text.slice(last) });
    return parts;
}

// ---- citations: each [n] of an answer, linked to the document and page it cites

// The validated entries of a citations message: one per number, in number order; only http(s) links.
export function citationEntries(content) {
    const out = [];
    const seen = new Set();
    (Array.isArray(content) ? content : []).forEach(item => {
        if (!item || typeof item !== "object") return;
        const n = item.n;
        if (!Number.isInteger(n) || n < 1 || seen.has(n)) return;
        seen.add(n);
        const url = typeof item.url === "string" ? item.url.trim() : "";
        const page = item.page;
        out.push({
            n,
            url: /^https?:\/\//i.test(url) ? url : "",
            page: Number.isInteger(page) && page > 0 ? page : null,
            title: typeof item.title === "string" ? item.title : "",
            type: typeof item.type === "string" ? item.type : "",
            origin: typeof item.origin === "string" ? item.origin : "",
            date: typeof item.date === "string" ? item.date : "",
            quote: typeof item.quote === "string" ? item.quote : "",
        });
    });
    return out.sort((a, b) => a.n - b.n);
}

// A PDF opens at the cited page (the viewer's page number); other links open as they are.
export function citationHref(url, page) {
    const u = String(url || "");
    if (!u) return "";
    return Number.isInteger(page) && page > 0 && u.indexOf("#") < 0 ? u + "#page=" + page : u;
}

// The numbers an answer cites as [n] or [n, m], ascending.
export function citedNumbers(text) {
    const found = new Set();
    citationParts(String(text || "")).forEach(part => (part.refs || []).forEach(n => found.add(n)));
    return Array.from(found).sort((a, b) => a - b);
}

export function citationFacts(entry) {
    return [entry.type, entry.date, entry.page ? "S.\u00a0" + entry.page : ""].filter(Boolean).join(" · ");
}

export function citationLabel(entry) {
    const title = entry.title || readableSlug(entry.url);
    return "Quelle " + entry.n + ": " + title + (entry.page ? ", S. " + entry.page : "") + " (" + TEXT.newTab + ")";
}

// The entries an answer's markers use. A searched answer uses only its own. A direct reply without a search (e.g. "kürzer")
// repeats the numbers of the answer it summarises: the last searched answer the backend kept in its history (direct
// replies and answers stopped or failed on the page are skipped); if that answer has no mapping, the markers stay plain.
export function entriesForTurn(turns, index) {
    const turn = turns[index];
    if (!turn) return [];
    if (Array.isArray(turn.citations) && turn.citations.length) return turn.citations;
    if (turn.thinking && turn.thinking.retrieval) return [];
    for (let i = index - 1; i >= 0; i--) {
        const earlier = turns[i];
        if (!earlier || !earlier.thinking || !earlier.thinking.retrieval) continue;
        if (earlier.state === "stopped" || earlier.state === "error") continue;
        return Array.isArray(earlier.citations) ? earlier.citations : [];
    }
    return [];
}

export function yearFromDate(date) {
    const years = String(date || "").match(YEAR_PATTERN);
    return years ? Number(years[years.length - 1]) : null;
}

// Cards in display order: cited documents first (by their lowest cited number), then the others as listed. A card
// shows the numbers the answer cites from it and their pages; its link opens the page of its lowest cited number, else
// its first passage's page. A cited source missing from the list gets a card from its entry; one without a link none.
export function citedCards(cards, entries, cited, nowYear) {
    const citedSet = new Set(cited || []);
    const out = (Array.isArray(cards) ? cards : []).map(card =>
        Object.assign({}, card, { ns: [], pages: [], linkPage: card.page || null }));
    const byUrl = new Map(out.map(card => [card.url, card]));
    (Array.isArray(entries) ? entries : []).forEach(entry => {
        if (!citedSet.has(entry.n) || !entry.url) return;
        let card = byUrl.get(entry.url);
        if (!card) {
            const year = yearFromDate(entry.date);
            card = { url: entry.url, title: entry.title || readableSlug(entry.url), type: entry.type || "Dokument",
                origin: entry.origin, year, page: entry.page, ns: [], pages: [], linkPage: entry.page,
                old: isOld(year, nowYear), isForm: entry.type === FORM_TYPE };
            byUrl.set(entry.url, card);
            out.push(card);
        }
        if (!card.ns.length && entry.page) card.linkPage = entry.page;  // entries arrive in number order
        card.ns.push(entry.n);
        if (entry.page && card.pages.indexOf(entry.page) < 0) card.pages.push(entry.page);
    });
    out.forEach(card => card.pages.sort((a, b) => a - b));
    const first = card => (card.ns.length ? card.ns[0] : Infinity);
    return out.map((card, i) => ({ card, i }))
        .sort((a, b) => first(a.card) - first(b.card) || a.i - b.i)
        .map(item => item.card);
}

// The pages a card shows and the page its link opens (cards stored before citedCards have only "page").
export function cardPages(card) {
    return card.pages && card.pages.length ? card.pages : card.page ? [card.page] : [];
}

export function cardLinkPage(card) {
    return card.linkPage !== undefined ? card.linkPage : card.page;
}
