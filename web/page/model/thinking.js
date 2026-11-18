import { TEXT } from "../config.js";

export function quoteList(items) {
    return items.map(item => "„" + item + "“").join(", ");
}

export function normalizeQuestion(text) {
    return String(text || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

export function differsFromTyped(typed, understood) {
    const u = normalizeQuestion(understood);
    return u !== "" && u !== normalizeQuestion(typed);
}

export function formatSeconds(ms) {
    const s = Math.max(0, ms) / 1000;
    return (s >= 10 ? String(Math.round(s)) : s.toFixed(1).replace(".", ",")) + " s";
}

export function foundText(count) {
    if (count === 0) return "Keine passenden Dokumente gefunden";
    return count + (count === 1 ? " Dokument" : " Dokumente") + " gefunden";
}

// ---- thinking model: steps of one answer, fed by status frames (or by the docs frame)

export function newThinking(now) {
    return {
        startedAt: now,
        endedAt: null,
        statusSeen: false,
        retrieval: false,
        queries: [],
        documents: null,
        steps: [{ key: "pending", text: TEXT.pending, state: "active" }],
    };
}

export function findStep(t, key) {
    for (let i = 0; i < t.steps.length; i++) if (t.steps[i].key === key) return t.steps[i];
    return null;
}

export function doneText(t, step) {
    if (step.key === "pending") return TEXT.pendingDone;
    if (step.key === "analysing") return TEXT.analysingDone;
    if (step.key === "search") return t.queries.length ? "Gesucht nach: " + quoteList(t.queries) : TEXT.searchingDone;
    if (step.key === "writing") return TEXT.writingDone;
    return step.text;
}

export function completeActive(t) {
    t.steps.forEach(step => {
        if (step.state === "active") {
            step.text = doneText(t, step);
            step.state = "done";
        }
    });
}

export function applyStatus(t, content, typed) {
    const c = content && typeof content === "object" ? content : {};
    t.statusSeen = true;
    if (c.step === "analysing") {
        const pending = findStep(t, "pending");
        if (pending) {
            pending.key = "analysing";
            pending.text = TEXT.analysing;
            pending.state = "active";
        } else if (!findStep(t, "analysing")) {
            t.steps.push({ key: "analysing", text: TEXT.analysing, state: "active" });
        }
    } else if (c.step === "understood") {
        completeActive(t);
        const question = String(c.question || "").trim();
        if (differsFromTyped(typed, question)) {
            t.steps.push({ key: "understood", text: "Verstanden als: „" + question + "“", state: "done" });
        }
    } else if (c.step === "searching") {
        // sent when the document search starts; direct replies never search
        completeActive(t);
        t.retrieval = true;
        t.queries = Array.isArray(c.queries) ? c.queries.map(q => String(q).trim()).filter(Boolean) : [];
        const text = t.queries.length ? "Suche nach " + quoteList(t.queries) + " …" : TEXT.searching;
        t.steps.push({ key: "search", text, state: "active" });
    } else if (c.step === "found") {
        const count = Math.max(0, Math.floor(Number(c.documents) || 0));
        t.retrieval = true;
        t.documents = count;
        completeActive(t);
        t.steps.push({ key: "found", text: foundText(count), state: "done" });
    } else if (c.step === "writing") {
        completeActive(t);
        if (!findStep(t, "writing")) t.steps.push({ key: "writing", text: TEXT.writing, state: "active" });
    }
}

export function applyDocs(t, count) {
    if (t.statusSeen || t.endedAt !== null || t.documents !== null) return;
    t.retrieval = true;
    t.documents = count;
    completeActive(t);
    t.steps.push({ key: "found", text: foundText(count), state: "done" });
    t.steps.push({ key: "writing", text: TEXT.writing, state: "active" });
}

export function finishThinking(t, now) {
    if (t.endedAt !== null) return;
    completeActive(t);
    t.endedAt = now;
}

export function haltThinking(t, now) {
    if (t.endedAt !== null) return;
    t.steps.forEach(step => {
        if (step.state === "active") step.state = "halted";
    });
    t.endedAt = now;
}

export function summaryText(t) {
    const parts = [];
    const q = t.queries.length;
    if (q) parts.push(q + (q === 1 ? " Suchbegriff" : " Suchbegriffe"));
    if (t.documents !== null) {
        parts.push(t.documents === 0 ? "keine Dokumente" : t.documents + (t.documents === 1 ? " Dokument" : " Dokumente"));
    }
    const ms = t.endedAt !== null ? t.endedAt - t.startedAt : 0;
    if (ms > 0) parts.push(formatSeconds(ms));
    return parts.join(" · ");
}
