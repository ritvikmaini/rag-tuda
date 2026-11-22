import { TEXT } from "../config.js";
import { cardLinkPage, cardPages, citationHref } from "./citations.js";

// The answer as plain text plus one line per card: its numbers, title, facts, pages and the link that opens them.
export function copyText(answerText, cards) {
    let out = String(answerText || "").trim();
    if (cards && cards.length) {
        out += "\n\n" + TEXT.sourcesCopy + "\n" + cards.map(card => {
            const facts = [card.type, card.year].filter(Boolean).join(", ");
            const pages = cardPages(card);
            return (card.ns && card.ns.length ? "[" + card.ns.join(", ") + "] " : "- ") + card.title +
                (facts ? " (" + facts + ")" : "") + (pages.length ? ", " + TEXT.pageAbbr + " " + pages.join(", ") : "") +
                ": " + citationHref(card.url, cardLinkPage(card));
        }).join("\n");
    }
    return out;
}

export function escapeHtml(text) {
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Fenced blocks and inline code spans: left as they are.
const CODE_RUN = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/;
// The tags an answer can hold stay (they are also the only ones the sanitiser lets through); an autolink or an e-mail
// address in angle brackets stays; any other <...> run is shown as text (a placeholder such as <Rechnungsnummer> or <datum>).
export const ANSWER_TAGS = ["a", "b", "i", "u", "s", "p", "em", "strong", "br", "ul", "ol", "li", "table", "thead", "tbody", "tr",
    "th", "td", "code", "pre", "sup", "sub", "span", "div", "hr", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "del", "small", "mark"];
const HTML_TAG = new RegExp("^/?(" + ANSWER_TAGS.join("|") + ")(?=[\\s/]|$)");
const AUTOLINK = /^[a-z][a-z0-9+.-]*:|^[^\s@]+@[^\s@]+$/i;

// Escapes placeholders before the markdown converter, which would drop them as unknown elements.
export function keepPlaceholders(markdown) {
    return String(markdown).split(CODE_RUN).map((part, i) => i % 2 ? part : part.replace(/<([^<>\n]+)>/g,
        (run, name) => HTML_TAG.test(name) || AUTOLINK.test(name) ? run : "&lt;" + name + "&gt;")).join("");
}
