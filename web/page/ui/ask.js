import { withToken } from "../api/auth.js";
import { createFrameParser } from "../api/frames.js";
import { questionUrl } from "../api/urls.js";
import { renderTabs } from "../features/variants.js";
import { citationEntries } from "../model/citations.js";
import { shapeName } from "../model/formats.js";
import { sourceCards } from "../model/sources.js";
import { watchdogExpired, watchdogLimits } from "../model/streaming.js";
import { applyDocs, applyStatus, finishThinking, haltThinking, newThinking } from "../model/thinking.js";
import { newTurnId, recordStatus, refOf, replacePlan } from "../model/turns.js";
import { pump, renderAnswer } from "../render/answer.js";
import { prefersReducedMotion } from "../render/dom.js";
import { renderSources } from "../render/sources.js";
import { renderThinking, tickTimer } from "../render/thinking.js";
import { currentChips, programmeNames } from "./chips.js";
import { updateSend } from "./composer.js";
import { keepFollowing, scrollToEnd, updateJump } from "./scroll.js";
import { state, store, views } from "./state.js";
import { save } from "./storage.js";
import { renderTurn, renderTurnState } from "./turns.js";

// ---- request

// plan (from replacePlan, or null for a new question): the turn this one replaces and the kept questions before it.
export function ask(question, plan) {
    const turn = { id: newTurnId(), replaceFrom: plan ? plan.replaceFrom : "", kept: plan ? plan.kept : [],
        question, answer: "", sources: [], citations: [],
        thinking: newThinking(Date.now()), state: "streaming", sawError: false };
    store.turns.push(turn);
    const view = renderTurn(turn);
    views.forEach((v, i) => { if (store.turns[i] !== turn) renderTabs(v, store.turns[i]); });  // tabs only on the last turn
    renderThinking(view, turn.thinking);
    renderTurnState(view, turn);
    state.follow = true;
    state.ignoreScrollUntil = Date.now() + 700;
    scrollToEnd(true);
    save();
    stream(turn, view);
}

export async function stream(turn, view) {
    const session = Object.assign({ turn, view, controller: new AbortController(), stopped: false, timedOut: false, timer: 0,
        startedAt: Date.now(), lastData: Date.now() }, watchdogLimits());
    state.live = session;
    recordStatus(turn, 0);  // a retry starts clean: no earlier refusal or rejected token stays
    session.timer = window.setInterval(() => {
        tickTimer(view, turn.thinking);
        if (watchdogExpired(session, Date.now())) {
            session.timedOut = true;
            session.controller.abort();
        }
    }, 500);
    updateSend();
    let ok = false;
    try {
        const response = await fetch(questionUrl(turn.question, store.id, turn.id, turn.replaceFrom, turn.kept, currentChips()),
            withToken({ signal: session.controller.signal, cache: "no-store" }));
        recordStatus(turn, response.status);
        if (!response.ok || !response.body) throw new Error("HTTP " + response.status);
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        const parser = createFrameParser();
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            session.lastData = Date.now();
            parser.push(decoder.decode(value, { stream: true })).forEach(frame => handleFrame(turn, view, frame));
        }
        parser.push(decoder.decode()).concat(parser.end()).forEach(frame => handleFrame(turn, view, frame));
        ok = true;
    } catch (error) {
        ok = false;
    }
    finishTurn(turn, view, ok);
}

export function handleFrame(turn, view, frame) {
    if (!frame || typeof frame !== "object") return;
    if (frame.type === "status") {
        applyStatus(turn.thinking, frame.content, turn.question, programmeNames());
        renderThinking(view, turn.thinking);
        keepFollowing();
    } else if (frame.type === "docs") {
        // the sources are shown once the answer is complete (finishTurn); the step line already counts them
        turn.sources = sourceCards(frame.content, new Date().getFullYear());
        if (!turn.answer) applyDocs(turn.thinking, turn.sources.length);
        renderThinking(view, turn.thinking);
        keepFollowing();
        save();
    } else if (frame.type === "citations") {
        // which document and page each [n] of this answer cites; the markers link once the answer is complete
        turn.citations = citationEntries(frame.content);
        save();
    } else if (frame.type === "shape") {
        // response formats v1: the shape the backend asked for (steps get their layout from it)
        turn.shape = shapeName(frame.content);
        save();
    } else if (frame.type === "chunk") {
        const text = frame.content === null || frame.content === undefined ? "" : String(frame.content);
        if (!text) return;
        if (!turn.answer) {
            finishThinking(turn.thinking, Date.now());
            renderThinking(view, turn.thinking);
        }
        turn.answer += text;
        pump(turn, view);
    } else if (frame.type === "meta") {
        // the turn the backend names; a question's meta id is the page's own, so only a different one is kept
        const ref = refOf(frame.content && frame.content.turn_id);
        if (ref && ref !== refOf(turn.id)) turn.ref = ref;
        // the request's own limits replace the page's fixed ones (a cold model may take longer than they allow)
        if (state.live !== null && state.live.turn === turn) Object.assign(state.live, watchdogLimits(frame.content));
    } else if (frame.type === "error" || (frame.type === undefined && frame.message !== undefined)) {
        turn.sawError = true;
    }
}

export function finishTurn(turn, view, ok) {
    const stopped = state.live !== null && state.live.stopped && !state.live.timedOut;
    if (state.live !== null) window.clearInterval(state.live.timer);
    state.live = null;
    const hasText = turn.answer.trim() !== "";
    // an answer with text but without a clean end (network, proxy or watchdog) is marked as cut off
    turn.state = stopped ? "stopped" : !hasText ? "error" : ok ? "done" : "interrupted";
    if (turn.state === "done") finishThinking(turn.thinking, Date.now());
    else haltThinking(turn.thinking, Date.now());
    view.shown = turn.answer.length;
    renderAnswer(view, turn, true);
    renderThinking(view, turn.thinking);
    renderSources(view, turn);
    renderTurnState(view, turn);
    updateSend();
    save();
    updateJump();
    // a reader who kept following sees the start of the sources (or the actions) with one gentle scroll
    const target = (view.sources.hidden ? null : view.sources.querySelector(".source")) || (view.actions.hidden ? null : view.actions);
    if (state.follow && target) target.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
}

export function stopLive() {
    if (state.live === null || state.live.timedOut) return;
    state.live.stopped = true;
    state.live.controller.abort();
}
