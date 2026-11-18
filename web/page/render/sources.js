import { TEXT, VISIBLE_SOURCES } from "../config.js";
import { cardLinkPage, cardPages, citationHref, citedCards, citedNumbers } from "../model/citations.js";
import { h, icon } from "./dom.js";

// ---- sources

export function sourceElement(card) {
    const pages = cardPages(card);
    const href = citationHref(card.url, cardLinkPage(card));
    const facts = [card.origin, card.year, pages.length ? "S. " + pages.join(", ") : null].filter(Boolean).join(" · ");
    const body = h("div", { class: "source-body" },
        h("a", { class: "source-title", href, target: "_blank", rel: "noopener noreferrer", text: card.title }),
        h("div", { class: "source-meta" },
            h("span", { class: "chip", text: card.type }),
            facts ? h("span", { class: "source-facts", text: facts }) : null,
            card.old ? h("span", { class: "chip chip-old", text: TEXT.oldDoc }) : null));
    return h("li", { class: "source", "data-ns": card.ns.join(" ") },
        card.ns.length ? h("span", { class: "source-num", text: card.ns.join(", ") }) : null,
        body,
        card.isForm ? h("a", { class: "source-form", href, target: "_blank", rel: "noopener noreferrer", text: TEXT.openForm }) : null,
        icon("external", "ext"));
}

export function setMoreLabel(view) {
    const extra = Math.max(0, view.cardCount - VISIBLE_SOURCES);
    view.moreBtn.textContent = view.more.classList.contains("open") ? TEXT.less : "+ " + extra + " weitere";
}

export function toggleMore(view) {
    const open = !view.more.classList.contains("open");
    view.more.classList.toggle("open", open);
    view.moreBtn.setAttribute("aria-expanded", String(open));
    setMoreLabel(view);
}

export function renderSources(view, turn) {
    const cards = displayCards(turn);
    view.cardCount = cards.length;
    view.sources.hidden = cards.length === 0;
    view.sourceList.textContent = "";
    view.moreList.textContent = "";
    if (!cards.length) return;
    view.sourcesCount.textContent = "(" + cards.length + ")";
    cards.forEach((card, i) => (i < VISIBLE_SOURCES ? view.sourceList : view.moreList).appendChild(sourceElement(card)));
    view.moreBtn.hidden = cards.length <= VISIBLE_SOURCES;
    setMoreLabel(view);
}

// The cards of a turn in display order, with the numbers and pages its answer cites.
export function displayCards(turn) {
    return citedCards(turn.sources, turn.citations, citedNumbers(turn.answer), new Date().getFullYear());
}
