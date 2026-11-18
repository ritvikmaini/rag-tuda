import { TEXT } from "../config.js";
import { canEdit, lengthState, replacePlan, retryVisible } from "../model/turns.js";
import { ask, stopLive } from "./ask.js";
import { updateJump } from "./scroll.js";
import { state, store, ui, views } from "./state.js";
import { save } from "./storage.js";

// ---- composer

export function autosize() {
    const q = ui.question;
    q.style.height = "auto";
    q.style.height = q.scrollHeight + "px";
    updateScrollPadding();
    updateJump();       // a taller composer moves the jump button up with it
}

// Keeps scrolled-to elements (citation targets, focused links) clear of the sticky header and composer.
export function updateScrollPadding() {
    const header = document.querySelector(".header");
    const top = header && header.offsetParent !== null ? header.offsetHeight : 0;
    document.documentElement.style.scrollPaddingTop = top + 12 + "px";
    document.documentElement.style.scrollPaddingBottom = ui.composerWrap.offsetHeight + 12 + "px";
}

export function updateSend() {
    const streaming = state.live !== null;
    ui.send.classList.toggle("is-stop", streaming);
    ui.send.setAttribute("aria-label", streaming ? TEXT.stop : TEXT.send);
    const length = renderLimit();
    ui.send.title = streaming ? TEXT.stop + " (Esc)" : length.state === "over" ? TEXT.overLimitTitle : TEXT.send + " (Enter)";
    ui.send.disabled = !streaming && (!ui.question.value.trim() || length.state === "over");
    ui.newChat.disabled = streaming;
    views.forEach((view, i) => {
        const turn = store.turns[i];
        view.editBtn.hidden = !canEdit(turn, streaming);
        view.retryBtn.hidden = !retryVisible(store.turns, i);
        view.retryBtn.disabled = streaming;
        view.actions.hidden = !turn || turn.state === "streaming" || (view.copyBtn.hidden && view.retryBtn.hidden);
    });
}

export function renderLimit() {
    const info = lengthState(ui.question.value);
    const over = info.state === "over";
    ui.limitBar.hidden = info.state === "ok";
    ui.limitBar.classList.toggle("is-over", over);
    ui.form.classList.toggle("is-over", over);
    ui.question.setAttribute("aria-invalid", over ? "true" : "false");
    if (ui.limitText.textContent !== info.hint) ui.limitText.textContent = info.hint;
    if (ui.limitLive.textContent !== info.hint) ui.limitLive.textContent = info.hint;
    ui.limitCount.textContent = info.counter;
    return info;
}

export function onKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    submitQuestion();
}

export function onSubmit(event) {
    event.preventDefault();
    if (state.live !== null) {
        stopLive();
        return;
    }
    submitQuestion();
}

export function submitQuestion() {
    const text = ui.question.value.trim();
    if (!text || state.live !== null || lengthState(text).state === "over") return;
    ui.question.value = "";
    autosize();
    let plan = null;
    if (state.editing) {
        plan = replacePlan(store.turns, state.editing);
        cancelEdit(false);
        if (plan) cutTurns(plan.keep.length);
    }
    ask(text, plan);
}

export function editTurn(turn) {
    if (!canEdit(turn, state.live !== null)) return;
    state.editing = turn;
    ui.editText.textContent = TEXT.editing.replace("{n}", String(store.turns.indexOf(turn) + 1));
    ui.editBar.hidden = false;
    ui.question.value = turn.question;
    autosize();
    updateSend();
    ui.question.focus();
    ui.question.setSelectionRange(ui.question.value.length, ui.question.value.length);
}

export function cancelEdit(clearInput) {
    state.editing = null;
    if (ui.editBar) ui.editBar.hidden = true;
    if (clearInput) {
        ui.question.value = "";
        autosize();
        updateSend();
    }
}

// The page shows what the backend keeps: an edit or a retry removes the turn and every later one before asking.
export function cutTurns(count) {
    store.turns.splice(count);
    views.splice(count).forEach(view => view.root.remove());
    save();
}

export function retryTurn(turn) {
    if (state.live !== null) return;
    const plan = replacePlan(store.turns, turn);
    if (plan) cutTurns(plan.keep.length);
    ask(turn.question, plan);
}
