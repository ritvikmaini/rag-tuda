import { STORAGE_KEY } from "../config.js";
import { renderTabs } from "../features/variants.js";
import { citationEntries } from "../model/citations.js";
import { haltThinking } from "../model/thinking.js";
import { newConversationId, restoreVersions, validTurn } from "../model/turns.js";
import { renderAnswer } from "../render/answer.js";
import { prefersReducedMotion } from "../render/dom.js";
import { renderSources } from "../render/sources.js";
import { renderThinking } from "../render/thinking.js";
import { syncChips } from "./chips.js";
import { autosize, cancelEdit, updateSend } from "./composer.js";
import { scrollToEnd } from "./scroll.js";
import { state, store, ui, views } from "./state.js";
import { renderTurn, renderTurnState } from "./turns.js";

export function newConversation() {
    if (state.live !== null) return;
    cancelEdit(false);
    store.id = newConversationId();
    store.turns = [];
    views.splice(0).forEach(view => view.root.remove());
    save();
    syncChips();  // chips belong to a conversation: the new one starts without
    ui.question.value = "";
    autosize();
    updateSend();
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
    ui.question.focus({ preventScroll: true });
}

export function save() {
    try {
        window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ id: store.id, turns: store.turns }));
    } catch (error) {
        // storage full or blocked: the conversation still works, it just does not survive a reload
    }
}

export function restore() {
    let data = null;
    try {
        data = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) || "null");
    } catch (error) {
        data = null;
    }
    store.id = data && typeof data.id === "string" && data.id ? data.id : newConversationId();
    store.turns = data && Array.isArray(data.turns) ? data.turns.filter(validTurn) : [];
    if (store.turns.length) {
        try {
            renderRestored();
        } catch (error) {
            // a stored conversation this page cannot show: start clean instead of leaving the tab unusable
            views.splice(0).forEach(view => view.root.remove());
            store.id = newConversationId();
            store.turns = [];
            document.body.classList.remove("no-anim");
        }
    }
    save();
    syncChips();
}

export function renderRestored() {
    document.body.classList.add("no-anim");
    store.turns.forEach(turn => {
        turn.sources = turn.sources.filter(card => card && typeof card.url === "string" && Array.isArray(card.ns));
        turn.citations = citationEntries(turn.citations);  // older stored turns have none
        restoreVersions(turn);  // a variant cut off by the reload: its last complete version
        if (turn.state === "streaming") {
            turn.state = "interrupted";
            haltThinking(turn.thinking, turn.thinking.startedAt);
        }
        const view = renderTurn(turn);
        renderAnswer(view, turn, true);
        renderThinking(view, turn.thinking);
        renderSources(view, turn);
        renderTurnState(view, turn);
        renderTabs(view, turn);
    });
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => document.body.classList.remove("no-anim")));
    scrollToEnd(false);
}
