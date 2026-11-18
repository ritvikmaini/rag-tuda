import { QUESTION_LIMIT, QUESTION_WARN_AT, TEXT } from "../config.js";
import { citationEntries } from "./citations.js";
import { shapeName } from "./formats.js";

export function newConversationId(cryptoApi) {
    const api = cryptoApi || (typeof crypto !== "undefined" ? crypto : null);
    if (api && typeof api.randomUUID === "function") return "p-" + api.randomUUID();
    const bytes = new Uint8Array(16);
    if (api && typeof api.getRandomValues === "function") api.getRandomValues(bytes);
    else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
    return "p-" + Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
}

// Counted like the backend counts (Python len of the sent text): code points of the trimmed question.
export function questionLength(text) {
    return Array.from(String(text || "").trim()).length;
}

export function formatCount(n) {
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

// ok: nothing shown; near: counter and tip; over: counter, the reason and sending blocked.
export function lengthState(text) {
    const count = questionLength(text);
    const state = count > QUESTION_LIMIT ? "over" : count >= QUESTION_WARN_AT ? "near" : "ok";
    const hint = state === "over" ? TEXT.overLimit.replace("{limit}", formatCount(QUESTION_LIMIT)) :
        state === "near" ? TEXT.nearLimit : "";
    return { count, state, counter: formatCount(count) + " / " + formatCount(QUESTION_LIMIT), hint };
}

// A question's id: a canonical UUID the backend stores as the message id, so an edit can name it.
export function newTurnId(cryptoApi) {
    const api = cryptoApi || (typeof crypto !== "undefined" ? crypto : null);
    if (api && typeof api.randomUUID === "function") return api.randomUUID();
    const bytes = new Uint8Array(16);
    if (api && typeof api.getRandomValues === "function") api.getRandomValues(bytes);
    else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
    return hex.slice(0, 8) + "-" + hex.slice(8, 12) + "-" + hex.slice(12, 16) + "-" + hex.slice(16, 20) + "-" + hex.slice(20);
}

// Editing (or retrying) a question keeps every turn before it and replaces it and everything after it. kept names the
// newest five kept questions: the backend places the cut by them when it never stored the replaced one.
export function replacePlan(turns, turn) {
    const index = turns.indexOf(turn);
    if (index < 0 || typeof turn.id !== "string" || !turn.id) return null;
    const keep = turns.slice(0, index);
    const kept = keep.filter(t => typeof t.id === "string" && t.id).map(t => t.id).slice(-5);
    return { keep, replaceFrom: turn.id, kept };
}

export function canEdit(turn, streaming) {
    return !streaming && !!turn && typeof turn.id === "string" && turn.id !== "";
}

// The error reference: the first 8 characters of the turn id, which the backend puts on every log line of the
// request as ref=. "" when the id is no hex id (a restored turn without one).
export function refOf(id) {
    return typeof id === "string" && /^[0-9a-f]{8}/i.test(id) ? id.slice(0, 8).toLowerCase() : "";
}

// The line under a failed, interrupted or error-frame answer that lets the user quote the reference; "" otherwise.
export function referenceLine(turn) {
    if (!turn) return "";
    const shown = turn.state === "error" || turn.state === "interrupted" || (turn.state === "done" && !!turn.sawError);
    const ref = turn.ref || refOf(turn.id);
    return shown && ref ? TEXT.reference.replace("{ref}", ref) : "";
}

export function isFailed(turn) {
    return !!turn && (turn.state === "error" || turn.state === "stopped" || turn.state === "interrupted" || !!turn.sawError);
}

// Retry only on the last turn, only when it failed, and only when it has an id (turns restored from older storage
// have none).
export function retryVisible(turns, index) {
    return index === turns.length - 1 && isFailed(turns[index]) && canEdit(turns[index], false);
}

// A turn stored while a variant was streaming comes back as its last complete version, as a finished answer.
export function restoreVersions(turn) {
    if (!turn || !Array.isArray(turn.versions) || !turn.versions.length) return turn;
    const wasStreaming = turn.versions.some(v => v && v.streaming);
    const kept = turn.versions.filter(v => v && typeof v.name === "string" && typeof v.answer === "string" && !v.streaming);
    const shown = (wasStreaming && kept.find(v => v.name === turn.restoreTo)) || kept.find(v => v.name === turn.active) || kept[0];
    delete turn.restoreTo;
    if (shown) {
        turn.active = shown.name;
        turn.answer = shown.answer;
        turn.citations = citationEntries(shown.citations);
        turn.shape = shapeName(shown.shape);
    }
    turn.versions = kept.length > 1 ? kept : [];
    if (wasStreaming && turn.state === "streaming") turn.state = "done";
    return turn;
}

export function validTurn(turn) {
    return !!turn && typeof turn.question === "string" && typeof turn.answer === "string" && Array.isArray(turn.sources) &&
        !!turn.thinking && Array.isArray(turn.thinking.steps) && Array.isArray(turn.thinking.queries) &&
        turn.thinking.steps.every(step => !!step && typeof step.text === "string" && typeof step.state === "string") &&
        typeof turn.thinking.startedAt === "number" &&
        ["streaming", "done", "error", "stopped", "interrupted"].indexOf(turn.state) >= 0;
}
