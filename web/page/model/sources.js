import { FORM_TYPE, OLD_AFTER_YEARS, TEXT, YEAR_PATTERN } from "../config.js";
import { citedCards } from "./citations.js";

export function parseUrl(url) {
    try {
        return new URL(String(url || ""), "https://example.org/");
    } catch (error) {
        return null;
    }
}

export function originFor(url, meta) {
    const raw = String(meta.origin || "").toLowerCase();
    if (raw) return raw.charAt(0).toUpperCase() + raw.slice(1);
    const u = parseUrl(url);
    return u ? u.hostname.toLowerCase().replace(/^www\./, "") : "";
}

// metadata.doc_year, else the last plausible year (1950-2049) in the URL path (as rag_core's extract_year).
export function yearFor(url, meta) {
    const y = Number(meta.doc_year);
    if (Number.isInteger(y) && y >= 1950 && y <= 2049) return y;
    const u = parseUrl(url);
    let path = u ? u.pathname : String(url || "");
    try {
        path = decodeURIComponent(path);
    } catch (error) {
        // keep the raw path
    }
    const years = path.match(YEAR_PATTERN);
    return years ? Number(years[years.length - 1]) : null;
}

export function readableSlug(url) {
    const u = parseUrl(url);
    const parts = (u ? u.pathname : String(url || "")).split("/").filter(Boolean);
    let slug = parts.length ? parts[parts.length - 1] : "";
    try {
        slug = decodeURIComponent(slug);
    } catch (error) {
        // keep the raw slug
    }
    slug = slug.replace(/\.(pdf|docx?|xlsx?)$/i, "").replace(/(.)pdf$/i, "$1").replace(/[-_]+/g, " ").trim();
    return slug ? slug.charAt(0).toUpperCase() + slug.slice(1) : TEXT.document;
}

export function titleFor(url, meta) {
    const keys = ["title", "download_title", "document_name"];
    for (let i = 0; i < keys.length; i++) {
        const value = String(meta[keys[i]] || "").trim();
        if (value) return value;
    }
    return readableSlug(url);
}

export function isOld(year, nowYear) {
    return year !== null && year <= nowYear - OLD_AFTER_YEARS;
}

// One card per document (several parents of one PDF share a card; its page is the first passage's page).
// The numbers come only from the citations message (citedCards).
export function sourceCards(docs, nowYear) {
    const byUrl = new Map();
    const cards = [];
    (Array.isArray(docs) ? docs : []).forEach(doc => {
        const meta = doc && doc.metadata && typeof doc.metadata === "object" ? doc.metadata : null;
        if (!meta) return;
        const url = String(meta.download_url || meta.url || "").trim();
        if (!/^https?:\/\//i.test(url)) return;
        let card = byUrl.get(url);
        if (!card) {
            const type = String(meta.doc_type || TEXT.document);
            const year = yearFor(url, meta);
            const page = Number(meta.page_number);
            card = {
                url,
                title: titleFor(url, meta),
                type,
                origin: originFor(url, meta),
                year,
                page: Number.isInteger(page) && page > 0 ? page : null,
                ns: [],
                old: isOld(year, nowYear),
                isForm: type === FORM_TYPE,
            };
            byUrl.set(url, card);
            cards.push(card);
        }
    });
    return cards;
}
