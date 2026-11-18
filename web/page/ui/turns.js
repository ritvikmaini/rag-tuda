import { TEXT } from "../config.js";
import { copyTurn } from "../features/export.js";
import { closeMenu, menuKey, toggleMenu } from "../features/menu.js";
import { referenceLine, retryVisible } from "../model/turns.js";
import { h, icon, svg } from "../render/dom.js";
import { toggleMore } from "../render/sources.js";
import { toggleThinking } from "../render/thinking.js";
import { editTurn, retryTurn } from "./composer.js";
import { state, store, ui, views } from "./state.js";

// ---- turns

export function renderTurnState(view, turn) {
    const s = turn.state;
    const message = s === "error" ? TEXT.failed : s === "stopped" ? TEXT.stopped : s === "interrupted" ? TEXT.interrupted : "";
    view.notice.hidden = !message;
    view.noticeText.textContent = message;
    view.notice.classList.toggle("is-error", s === "error" || s === "interrupted");
    const reference = referenceLine(turn);
    view.refLine.hidden = !reference;
    view.refLine.textContent = reference;
    view.copyBtn.hidden = !turn.answer.trim();
    view.menuBtn.hidden = view.copyBtn.hidden;
    view.retryBtn.hidden = !retryVisible(store.turns, store.turns.indexOf(turn));
    view.retryBtn.disabled = state.live !== null;
    view.actions.hidden = s === "streaming" || (view.copyBtn.hidden && view.retryBtn.hidden);
    view.card.classList.toggle("is-streaming", s === "streaming");
}

export function renderTurn(turn) {
    const view = { stepEls: [], thinkingToggled: false, cardCount: 0, shown: 0, pumping: false, copyTimer: 0 };
    view.editBtn = h("button", { type: "button", class: "q-edit", hidden: true, onclick: () => editTurn(turn) }, icon("edit"), h("span", { text: TEXT.edit }));
    view.steps = h("ol", { class: "steps" });
    view.toggleLabel = h("span", { class: "thinking-label" });
    view.toggle = h("button", { type: "button", class: "thinking-toggle", "aria-expanded": "false", hidden: true, onclick: () => toggleThinking(view) },
        icon("chevron", "chev"), view.toggleLabel);
    view.collapse = h("div", { class: "collapsible open" }, h("div", { class: "collapsible-inner" }, view.steps));
    view.thinking = h("div", { class: "thinking" }, view.toggle, view.collapse);
    view.body = h("div", { class: "md" });
    view.noticeText = h("span");
    view.notice = h("p", { class: "notice", hidden: true }, icon("info"), view.noticeText);
    view.refLine = h("p", { class: "notice-ref", hidden: true });
    view.sourcesCount = h("span", { class: "sources-count" });
    view.sourceList = h("ul", { class: "source-list" });
    view.moreList = h("ul", { class: "source-list" });
    view.more = h("div", { class: "collapsible" }, h("div", { class: "collapsible-inner" }, view.moreList));
    view.moreBtn = h("button", { type: "button", class: "sources-more", hidden: true, "aria-expanded": "false", onclick: () => toggleMore(view) });
    view.sources = h("section", { class: "sources", hidden: true, "aria-label": "Quellen" },
        h("h2", { class: "sources-title" }, "Quellen", view.sourcesCount),
        view.sourceList, view.more, view.moreBtn,
        h("p", { class: "sources-note", text: TEXT.sourcesNote }));
    view.copyLabel = h("span", { text: TEXT.copy });
    view.copyIcon = icon("copy");
    view.copyBtn = h("button", { type: "button", class: "act", onclick: () => copyTurn(view, turn) }, view.copyIcon, view.copyLabel);
    view.retryBtn = h("button", { type: "button", class: "act", hidden: true, onclick: () => retryTurn(turn) },
        icon("retry"), h("span", { text: TEXT.retry }));
    view.menuBtn = h("button", { type: "button", class: "act act-more", "aria-haspopup": "menu", "aria-expanded": "false",
        "aria-label": TEXT.more, title: TEXT.more, onclick: () => toggleMenu(view, turn),
        onkeydown: event => {
            // Esc on the open menu's button closes the menu only; it must never reach the page's stop-the-answer handler
            if (event.key === "Escape" && !view.menu.hidden) {
                event.preventDefault();
                event.stopPropagation();
                closeMenu(view, true);
            }
        } }, icon("more"));
    view.menu = h("div", { class: "menu", role: "menu", hidden: true, onkeydown: event => menuKey(view, event) });
    view.actions = h("div", { class: "actions", hidden: true }, view.copyBtn, view.retryBtn, view.menuBtn, view.menu);
    view.tabs = h("div", { class: "versions", role: "tablist", "aria-label": TEXT.versions, hidden: true });
    view.variantNoticeText = h("span");
    view.variantNotice = h("p", { class: "notice is-error", hidden: true }, icon("info"), view.variantNoticeText);
    view.card = h("div", { class: "a-card", "aria-busy": "false" },
        view.thinking, view.tabs, view.variantNotice, view.body, view.notice, view.refLine, view.sources, view.actions);
    view.root = h("section", { class: "turn" },
        h("div", { class: "q-row" }, h("div", { class: "q-bubble", text: turn.question }), view.editBtn),
        h("div", { class: "a-row" }, h("div", { class: "avatar", "aria-hidden": "true", html: svg("mark", 18) }), view.card));
    ui.conversation.appendChild(view.root);
    views.push(view);
    return view;
}
