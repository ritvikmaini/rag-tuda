import { withToken } from "../api/auth.js";
import { createFrameParser } from "../api/frames.js";
import { variantUrl } from "../api/urls.js";
import { SILENCE_LIMIT_MS, TEXT, TOTAL_LIMIT_MS, VARIANT_LABELS } from "../config.js";
import { citationEntries } from "../model/citations.js";
import { shapeName } from "../model/formats.js";
import { pump, renderAnswer } from "../render/answer.js";
import { DOTS_HTML, h } from "../render/dom.js";
import { renderSources } from "../render/sources.js";
import { updateSend } from "../ui/composer.js";
import { state, store } from "../ui/state.js";
import { save } from "../ui/storage.js";
import { renderTurnState } from "../ui/turns.js";

// Tabs only on the last turn and only once a second version exists.
export function renderTabs(view, turn) {
    const versions = turn && Array.isArray(turn.versions) ? turn.versions : [];
    view.tabs.hidden = store.turns.indexOf(turn) !== store.turns.length - 1 || versions.length < 2;
    if (view.tabs.hidden) return;
    const active = turn.active || "original";
    view.tabs.replaceChildren(...versions.map(v => h("button", { type: "button", class: "version", role: "tab",
        "aria-selected": v.name === active ? "true" : "false", disabled: state.live !== null && v.name !== active,
        onclick: () => selectVersion(turn, view, v.name) },
        VARIANT_LABELS[v.name] || v.name, v.streaming ? h("span", { class: "version-dots", html: DOTS_HTML }) : null)));
}

export async function startVariant(turn, view, name) {
    if (state.live !== null || view.switching || !turn.id || !Object.prototype.hasOwnProperty.call(VARIANT_LABELS, name) || name === "original") return;
    if (!Array.isArray(turn.versions) || !turn.versions.length) {
        turn.versions = [{ name: "original", answer: turn.answer, citations: turn.citations, shape: shapeName(turn.shape) }];
    }
    const before = { answer: turn.answer, citations: turn.citations, shape: turn.shape, active: turn.active || "original" };
    const version = { name, answer: "", citations: [], shape: shapeName(turn.shape), streaming: true };
    turn.restoreTo = before.active;  // a reload during the variant returns to the version that was on screen
    turn.versions.push(version);
    Object.assign(turn, { active: name, answer: "", citations: [], state: "streaming" });
    view.shown = 0;
    view.variantNotice.hidden = true;
    renderTabs(view, turn);
    renderAnswer(view, turn, false);
    renderTurnState(view, turn);
    save();
    const result = await streamVariant(turn, view, version);
    delete version.streaming;
    delete turn.restoreTo;
    if (result.ok) {
        Object.assign(turn, { answer: version.answer, citations: version.citations, shape: version.shape });
    } else {
        turn.versions = turn.versions.filter(v => v !== version);
        if (turn.versions.length < 2) turn.versions = [];
        Object.assign(turn, before);
        view.variantNoticeText.textContent = result.stopped ? TEXT.variantStopped : TEXT.variantFailed;
        view.variantNotice.hidden = false;
    }
    turn.state = "done";
    view.shown = turn.answer.length;
    renderAnswer(view, turn, true);
    renderTabs(view, turn);
    renderSources(view, turn);
    renderTurnState(view, turn);
    updateSend();
    save();
}

// Streams one variant like a question (same limits and stop button); ok only with the backend's confirmation.
export async function streamVariant(turn, view, version) {
    const session = { turn, view, controller: new AbortController(), stopped: false, timedOut: false, timer: 0,
        startedAt: Date.now(), lastData: Date.now() };
    state.live = session;
    session.timer = window.setInterval(() => {
        const now = Date.now();
        if (now - session.lastData > SILENCE_LIMIT_MS || now - session.startedAt > TOTAL_LIMIT_MS) {
            session.timedOut = true;
            session.controller.abort();
        }
    }, 500);
    updateSend();
    let fetched = false;
    let confirmed = false;
    const take = frame => {
        if (!frame || typeof frame !== "object") return;
        if (frame.type === "chunk" && frame.content) {
            version.answer += String(frame.content);
            turn.answer = version.answer;
            pump(turn, view);
        } else if (frame.type === "citations") {
            version.citations = citationEntries(frame.content);
        } else if (frame.type === "shape") {
            version.shape = shapeName(frame.content);
            turn.shape = version.shape;
        } else if (frame.type === "variant") {
            confirmed = !!(frame.content && frame.content.ok);
        }
    };
    try {
        const response = await fetch(variantUrl(turn.question, store.id, turn.id, version.name, false),
            withToken({ signal: session.controller.signal, cache: "no-store" }));
        if (!response.ok || !response.body) throw new Error("HTTP " + response.status);
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        const parser = createFrameParser();
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            session.lastData = Date.now();
            parser.push(decoder.decode(value, { stream: true })).forEach(take);
        }
        parser.push(decoder.decode()).concat(parser.end()).forEach(take);
        fetched = true;
    } catch (error) {
        fetched = false;
    }
    window.clearInterval(session.timer);
    const stopped = session.stopped && !session.timedOut;
    state.live = null;
    return { ok: fetched && confirmed && version.answer.trim() !== "", stopped };
}

// A tab click: the backend puts that version into the history first; the page switches only when it confirms.
export async function selectVersion(turn, view, name) {
    if (state.live !== null || view.switching || (turn.active || "original") === name) return;
    const version = (turn.versions || []).find(v => v.name === name);
    if (!version) return;
    let ok = false;
    view.switching = true;
    try {
        const response = await fetch(variantUrl(turn.question, store.id, turn.id, name, true), withToken({ cache: "no-store" }));
        const parser = createFrameParser();
        const frames = parser.push(await response.text()).concat(parser.end());
        ok = response.ok && frames.some(frame => frame && frame.type === "variant" && frame.content && frame.content.ok);
    } catch (error) {
        ok = false;
    } finally {
        view.switching = false;
    }
    if (!ok) {
        view.variantNoticeText.textContent = TEXT.selectFailed;
        view.variantNotice.hidden = false;
        return;
    }
    view.variantNotice.hidden = true;
    Object.assign(turn, { active: name, answer: version.answer, citations: version.citations, shape: version.shape });
    view.shown = turn.answer.length;
    renderAnswer(view, turn, true);
    renderTabs(view, turn);
    renderSources(view, turn);
    save();
}
