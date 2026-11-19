// Chip tests (degree / programme / PO selectors; native ES modules). From web/page: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyStatus, newThinking } from "../model/thinking.js";
import { questionUrl } from "../api/urls.js";
import { chipChoices, cleanChips, readChips, writeChips } from "../ui/chips.js";
import { renderTurnState } from "../ui/turns.js";

const plain = value => JSON.parse(JSON.stringify(value));

// what GET /api/profile answers for the example roles
const PROFILE = {
    degrees: ["bachelor", "master", "promotion"],
    programmes: [
        { id: "bsc-informatik", label_de: "B.Sc. Informatik", label_en: "B.Sc. Informatik", degree: "bachelor", po_versions: [2023, 2015] },
        { id: "msc-informatik", label_de: "M.Sc. Informatik", label_en: "M.Sc. Informatik", degree: "master", po_versions: [2023, 2015] },
        { id: "msc-computer-science", label_de: "M.Sc. Computer Science", label_en: "M.Sc. Computer Science", degree: "master", po_versions: [2023] },
    ],
    po_versions: [2023, 2015],
    answer: { model: "m", provider: "p", remote: false },
};
const NONE = { degree: "", programme: "", po: "" };

function fakeStorage(blocked) {
    const kept = new Map();
    return {
        kept,
        getItem: key => { if (blocked) throw new Error("blocked"); return kept.has(key) ? kept.get(key) : null; },
        setItem: (key, value) => { if (blocked) throw new Error("blocked"); kept.set(key, value); },
    };
}

test("chips are sticky per conversation (test_chips_sticky_per_conversation)", () => {
    const storage = fakeStorage();
    const chosen = { degree: "master", programme: "msc-informatik", po: "2015" };
    writeChips(storage, "conv-1", chosen);
    // a reload of the same conversation gets the selection back; another conversation starts without chips
    assert.deepEqual(readChips(storage, "conv-1"), chosen);
    assert.deepEqual(readChips(storage, "conv-2"), NONE);
    writeChips(storage, "conv-2", { degree: "bachelor", programme: "", po: "" });
    assert.deepEqual(readChips(storage, "conv-2"), { degree: "bachelor", programme: "", po: "" });
    assert.deepEqual(readChips(storage, "conv-1"), NONE);
    // storage that is blocked, empty or holds something else never breaks the page
    assert.deepEqual(readChips(fakeStorage(true), "conv-1"), NONE);
    assert.doesNotThrow(() => writeChips(fakeStorage(true), "conv-1", chosen));
    const odd = fakeStorage();
    odd.setItem("assistant-chips-v1", "{not json");
    assert.deepEqual(readChips(odd, "conv-1"), NONE);
    odd.setItem("assistant-chips-v1", JSON.stringify({ id: "conv-1", degree: 7, programme: ["x"], po: null }));
    assert.deepEqual(readChips(odd, "conv-1"), NONE);
});

test("chips are sent as request parameters, a cleared chip as absent (test_chips_sent_as_params)", () => {
    const base = "/api/chat?question=Q&conversation_id=c-1&formats=v1";
    assert.equal(questionUrl("Q", "c-1"), base);
    assert.equal(questionUrl("Q", "c-1", "", "", undefined, NONE), base);
    assert.equal(questionUrl("Q", "c-1", "", "", undefined, { degree: "master", programme: "msc-informatik", po: "2023" }),
        base + "&degree=master&programme=msc-informatik&po=2023");
    // each chip on its own: the other two are left out, not sent empty
    assert.equal(questionUrl("Q", "c-1", "", "", undefined, { degree: "", programme: "msc-informatik", po: "" }),
        base + "&programme=msc-informatik");
    // they come after a replaced turn's parameters
    const url = questionUrl("Q", "c-1", "tid", "rid", ["a"], { degree: "bachelor", programme: "", po: "" });
    assert.ok(url.endsWith("&turn_id=tid&replace_from=rid&kept=a&degree=bachelor"));
});

test("the selectors offer only what roles.yaml lists: programmes of the degree, PO years of the programme", () => {
    assert.deepEqual(plain(chipChoices(PROFILE, NONE)), {
        degrees: ["bachelor", "master", "promotion"],
        programmes: PROFILE.programmes.map(p => ({ id: p.id, label: p.label_de })),
        pos: [2023, 2015],
    });
    const master = chipChoices(PROFILE, { degree: "master", programme: "", po: "" });
    assert.deepEqual(master.programmes.map(p => p.id), ["msc-informatik", "msc-computer-science"]);
    assert.deepEqual(chipChoices(PROFILE, { degree: "promotion", programme: "", po: "" }).programmes, []);
    // the 2015 PO exists for two programmes, not for the third
    assert.deepEqual(chipChoices(PROFILE, { degree: "", programme: "msc-computer-science", po: "" }).pos, [2023]);
    assert.deepEqual(chipChoices(PROFILE, { degree: "", programme: "bsc-informatik", po: "" }).pos, [2023, 2015]);
});

test("a selection that no longer fits is cleaned from the changed chip downwards", () => {
    // a PO the chosen programme lacks goes; the programme and degree stay
    assert.deepEqual(cleanChips(PROFILE, { degree: "master", programme: "msc-computer-science", po: "2015" }),
        { degree: "master", programme: "msc-computer-science", po: "" });
    // a programme of another degree goes, and its PO with it only when that PO is not offered any more
    assert.deepEqual(cleanChips(PROFILE, { degree: "bachelor", programme: "msc-informatik", po: "2015" }),
        { degree: "bachelor", programme: "", po: "2015" });
    // values the profile does not know (stored by an older profile) go
    assert.deepEqual(cleanChips(PROFILE, { degree: "diplom", programme: "x", po: "1999" }), NONE);
    assert.deepEqual(cleanChips(PROFILE, { degree: "master", programme: "msc-informatik", po: "2023" }),
        { degree: "master", programme: "msc-informatik", po: "2023" });
    // without a profile (not loaded, or no roles) nothing can be offered, so nothing is sent
    assert.deepEqual(cleanChips(null, { degree: "master", programme: "", po: "" }), NONE);
});

function analysing() {
    const t = newThinking(0);
    applyStatus(t, { step: "analysing" }, "Q");
    return t;
}

test("the understood step names the applied filter, with the programme's label", () => {
    const labels = { "msc-informatik": "M.Sc. Informatik" };
    const filter = { degree: ["master"], programme: ["msc-informatik"], po_version: [2023], regulation: null, date: null,
        include_archived: false, po_source: "chip", from_chips: ["degree", "programme", "po_version"] };
    const t = analysing();
    applyStatus(t, { step: "understood", question: "Wie melde ich mich an?", intent: "answer", applied_filter: filter }, "Wie melde ich mich an?", labels);
    assert.deepEqual(t.steps.map(s => [s.key, s.text, s.state]), [
        ["analysing", "Frage analysiert", "done"],
        ["filter", "Filter: Master · M.Sc. Informatik · PO 2023", "done"],
    ]);
    // a rewritten question keeps its line, the filter follows it; several values of one field are listed
    const both = analysing();
    applyStatus(both, { step: "understood", question: "Anmeldung zur Prüfung", intent: "answer",
        applied_filter: Object.assign({}, filter, { degree: null, programme: null, po_version: [2023, 2015] }) }, "anmeldung?", labels);
    assert.deepEqual(both.steps.map(s => s.text), ["Frage analysiert", "Verstanden als: „Anmeldung zur Prüfung“", "Filter: PO 2023, 2015"]);
    // no filter, an empty one or a malformed one adds no line
    for (const applied of [undefined, null, "x", { degree: null, programme: null, po_version: null }]) {
        const none = analysing();
        applyStatus(none, { step: "understood", question: "Q", intent: "answer", applied_filter: applied }, "Q", labels);
        assert.ok(!none.steps.some(s => s.key === "filter"));
    }
});

test("a refused chip selection shows its own message, never the generic failure", () => {
    const view = {
        notice: { hidden: true, classList: { toggle: () => {} } }, noticeText: { textContent: "" }, refLine: { hidden: true, textContent: "" },
        copyBtn: { hidden: false }, menuBtn: { hidden: false }, retryBtn: { hidden: true, disabled: false }, actions: { hidden: true },
        card: { classList: { toggle: () => {} } },
    };
    const turn = { id: "t", question: "Q", answer: "", sources: [], state: "error", refused: true };
    renderTurnState(view, turn);
    assert.equal(view.notice.hidden, false);
    assert.match(view.noticeText.textContent, /Auswahl/);
    assert.match(view.noticeText.textContent, /nicht möglich/);
    assert.doesNotMatch(view.noticeText.textContent, /nicht geladen/);
});
