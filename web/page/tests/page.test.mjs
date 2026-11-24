// v2 page helper tests (citations, formats, turns, menu; native ES modules). From web/page: npm test
// Every module of the page folder except the entry is imported; the tests read the names they use from them.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";

const SRC = new URL("../", import.meta.url);
// Windows lists vendor\x.js: normalise, or the vendored libraries would be imported as page modules. The tests folder holds the
// browser suite's Python environment (tests/e2e/.venv) with JavaScript of its own.
const files = readdirSync(SRC, { recursive: true }).map(f => f.replaceAll("\\", "/"))
    .filter(f => f.endsWith(".js") && f !== "app.js" && !f.startsWith("vendor/") && !f.startsWith("tests/")).sort();
const page = Object.assign({}, ...await Promise.all(files.map(f => import(new URL(f, SRC)))));
// kept from the vm loader (values once came from another context): compare as plain JSON
const plain = value => JSON.parse(JSON.stringify(value));

const DOC_A = "https://docs.example.org/handbook/guidelines/ai-guideline-2025-02/?layout=file";
const DOC_B = "https://docs.example.org/handbook/forms/accident-report/?layout=file";
const WEB = "https://www.example.org/office/service";
const MAP = "https://maps.example.org/#m=osm&zoomto=true&uid=4711";

function entry(n, url, page, extra) {
    return Object.assign({ n, url, page, title: "Dienstanweisung KI", type: "Dienstanweisung", origin: "Handbook",
        date: "2025", quote: "Zitat " + n }, extra || {});
}

function doc(url, page, extra) {
    return { page_content: "x", metadata: Object.assign({ download_url: url, page_number: page, title: "Titel " + page }, extra || {}) };
}

test("citation entries keep valid numbers in order and clean every field", () => {
    const got = plain(page.citationEntries([
        entry(2, DOC_A, 4), entry(1, DOC_A, 2), entry(2, DOC_A, 9), { n: 0 }, { n: "3" }, null,
        entry(4, "javascript:alert(1)", 3), entry(5, WEB, -1, { quote: 7 }), entry(6, DOC_A, 1.5),
    ]));
    assert.deepEqual(got.map(e => [e.n, e.url, e.page]),
        [[1, DOC_A, 2], [2, DOC_A, 4], [4, "", 3], [5, WEB, null], [6, DOC_A, null]]);
    assert.equal(got[3].quote, "");
    assert.deepEqual(plain(page.citationEntries("kein Array")), []);
    assert.deepEqual(plain(page.citationEntries(undefined)), []);
});

test("a PDF link opens at the cited page; other links stay as they are", () => {
    assert.equal(page.citationHref(DOC_A, 4), DOC_A + "#page=4");
    assert.equal(page.citationHref(WEB, null), WEB);
    assert.equal(page.citationHref(MAP, 3), MAP);
    assert.equal(page.citationHref("", 4), "");
});

test("cited numbers come from [n] and [n, m] only, not from Markdown links", () => {
    assert.deepEqual(plain(page.citedNumbers("Erlaubt [2]. Siehe [4, 2] und [www.example.org](https://www.example.org) sowie [15].")),
        [2, 4, 15]);
    assert.deepEqual(plain(page.citedNumbers("")), []);
});

test("cards: cited documents first with their numbers and pages, the link on the lowest number's page", () => {
    const cards = page.sourceCards([doc(WEB, -1), doc(DOC_A, 2), doc(DOC_A, 4), doc(DOC_A, 12), doc(DOC_B, 7)], 2026);
    const entries = page.citationEntries([entry(1, DOC_A, 2), entry(2, DOC_A, 4), entry(3, DOC_A, 4), entry(4, DOC_A, 12),
        entry(5, WEB, null), entry(6, DOC_B, 7)]);
    const got = plain(page.citedCards(cards, entries, [2, 3, 4, 15], 2026));
    assert.deepEqual(got.map(c => [c.url, c.ns, c.pages, c.linkPage]), [
        [DOC_A, [2, 3, 4], [4, 12], 4],
        [WEB, [], [], null],
        [DOC_B, [], [], 7],
    ]);
});

test("a cited source missing from the list gets its own card; one without a link gets none", () => {
    const entries = page.citationEntries([
        entry(1, DOC_B, 3, { title: "Unfallanzeige", type: "Formular", date: "12.03.2024" }),
        entry(2, "", 5),
    ]);
    const got = plain(page.citedCards([], entries, [1, 2], 2026));
    assert.equal(got.length, 1);
    assert.deepEqual([got[0].url, got[0].title, got[0].ns, got[0].pages, got[0].linkPage, got[0].year, got[0].isForm, got[0].old],
        [DOC_B, "Unfallanzeige", [1], [3], 3, 2024, true, false]);
});

test("a direct reply links to the last searched answer the backend kept; a searched answer never borrows", () => {
    const answer = (citations, state) => ({ citations, thinking: { retrieval: true }, state: state || "done" });
    const reply = { thinking: { retrieval: false }, state: "done" };
    const own = [entry(1, DOC_A, 2)];
    // its numbers repeat the answer it summarises
    assert.equal(plain(page.entriesForTurn([answer(own), reply], 1))[0].page, 2);
    // over other direct replies ("Danke", an earlier "kürzer")
    assert.equal(plain(page.entriesForTurn([answer(own), reply, reply], 2))[0].n, 1);
    // over answers stopped or failed on the page (not in the backend's history)
    assert.equal(plain(page.entriesForTurn([answer(own), answer([], "stopped"), reply], 2))[0].n, 1);
    assert.equal(plain(page.entriesForTurn([answer(own), answer([], "error"), reply], 2))[0].n, 1);
    // an interrupted answer completed on the backend: its own mapping
    assert.equal(plain(page.entriesForTurn([answer(own), answer([entry(1, DOC_B, 7)], "interrupted"), reply], 2))[0].url, DOC_B);
    // the summarised answer has no mapping: plain markers, never an older answer's documents
    assert.deepEqual(plain(page.entriesForTurn([answer(own), answer([]), reply], 2)), []);
    // a searched answer uses only its own mapping
    assert.deepEqual(plain(page.entriesForTurn([answer(own), answer([])], 1)), []);
    assert.equal(plain(page.entriesForTurn([answer(own)], 0))[0].page, 2);
    assert.deepEqual(plain(page.entriesForTurn([reply], 0)), []);
});

test("tooltip facts and the accessible label name the document and page", () => {
    const e = entry(2, DOC_A, 4);
    // "S." and the number never break apart in the tooltip
    assert.equal(page.citationFacts(e), "Dienstanweisung · 2025 · S.\u00a04");
    assert.equal(page.citationLabel(e), "Quelle 2: Dienstanweisung KI, S. 4 (öffnet in neuem Tab)");
    assert.equal(page.citationFacts(entry(5, WEB, null, { type: "Webseite", date: "" })), "Webseite");
});

test("copy lists each card with its numbers, pages and the link that opens them", () => {
    const cards = page.citedCards(page.sourceCards([doc(DOC_A, 2, { doc_type: "Dienstanweisung/Dienstvereinbarung" }), doc(DOC_B, 7, { doc_type: "Formular" })], 2026),
        page.citationEntries([entry(1, DOC_A, 4), entry(2, DOC_A, 12)]), [1, 2], 2026);
    assert.equal(page.copyText("Antwort [1][2]", cards),
        "Antwort [1][2]\n\nQuellen:\n" +
        "[1, 2] Titel 2 (Dienstanweisung/Dienstvereinbarung, 2025), S. 4, 12: " + DOC_A + "#page=4\n" +
        "- Titel 7 (Formular), S. 7: " + DOC_B + "#page=7");
});

test("a conversation stored before citations existed (no citations, cards with ns) still renders its cards", () => {
    const stored = [{ url: DOC_A, title: "Alt", type: "Dokument", origin: "Handbook", year: 2025, page: 2, ns: [], old: false, isForm: false }];
    const got = plain(page.citedCards(stored, page.citationEntries(undefined), page.citedNumbers("Text [1]"), 2026));
    assert.deepEqual(got.map(c => [c.ns, c.pages, c.linkPage]), [[[], [], 2]]);
});

test("cards take type and origin from the metadata; without it, the type is Dokument and the origin the host", () => {
    const cards = plain(page.sourceCards([doc(DOC_A, 2, { doc_type: "Richtlinie", origin: "web" }), doc(WEB, 1)], 2026));
    assert.deepEqual(cards.map(c => [c.type, c.origin]), [["Richtlinie", "Web"], ["Dokument", "example.org"]]);
});

test("cards take their numbers only from the citations message", () => {
    const cards = plain(page.sourceCards([doc(DOC_A, 2, { citation_number: 3 })], 2026));
    assert.deepEqual(cards[0].ns, []);
});

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

test("turn ids are canonical UUIDs, with and without randomUUID", () => {
    assert.equal(page.newTurnId({ randomUUID: () => "0b0e7c9a-4f5e-4a1b-9c3d-2e1f0a9b8c7d" }), "0b0e7c9a-4f5e-4a1b-9c3d-2e1f0a9b8c7d");
    assert.match(page.newTurnId({ getRandomValues: a => { a.fill(255); return a; } }), UUID_RE);
    assert.match(page.newTurnId({}), UUID_RE);
});

test("the request carries turn_id and replace_from only when given", () => {
    const plainUrl = page.questionUrl("Was & wie?", "p-1");
    assert.equal(plainUrl, "/api/chat?question=Was%20%26%20wie%3F&conversation_id=p-1" +
        "&formats=v1");
    const url = page.questionUrl("Q", "p-1", "tid", "rid");
    assert.ok(url.endsWith("&turn_id=tid&replace_from=rid"));
    assert.ok(!page.questionUrl("Q", "p-1", "tid", "").includes("replace_from"));
});

test("an edit keeps the turns before the edited one and names its id", () => {
    const t1 = { id: "a", question: "1" }, t2 = { id: "b", question: "2" }, t3 = { id: "c", question: "3" };
    const plan = page.replacePlan([t1, t2, t3], t2);
    assert.deepEqual(plain(plan.keep.map(t => t.id)), ["a"]);
    assert.equal(plan.replaceFrom, "b");
    assert.equal(page.replacePlan([t1, t2], { id: "x" }), null);
    assert.equal(page.replacePlan([{ question: "old" }], { question: "old" }), null);
});

test("edit needs an id and no running answer; retry only on the last failed turn", () => {
    assert.equal(page.canEdit({ id: "a" }, false), true);
    assert.equal(page.canEdit({ id: "a" }, true), false);
    assert.equal(page.canEdit({ question: "restored before ids" }, false), false);
    const ok = { id: "a", state: "done", sawError: false }, failed = { id: "b", state: "stopped", sawError: false };
    assert.equal(page.isFailed(failed), true);
    assert.equal(page.retryVisible([ok, failed], 1), true);
    assert.equal(page.retryVisible([failed, ok], 0), false);
    assert.equal(page.retryVisible([ok], 0), false);
});

test("a failed turn restored from before the turn ids shows no retry", () => {
    const restored = { question: "restored before ids", state: "interrupted", sawError: false };
    assert.equal(page.isFailed(restored), true);
    assert.equal(page.retryVisible([restored], 0), false);
});

test("an edit also names the page's last kept questions", () => {
    const turns = [{ question: "restored before ids" }].concat(["a", "b", "c", "d", "e", "f", "g"].map(id => ({ id, question: id })));
    assert.deepEqual(plain(page.replacePlan(turns, turns[7]).kept), ["b", "c", "d", "e", "f"]);  // newest five with an id
    assert.deepEqual(plain(page.replacePlan(turns, turns[1]).kept), []);
    assert.equal(page.questionUrl("Q", "p-1", "tid", "rid", ["k1", "k2"]), "/api/chat?question=Q&conversation_id=p-1" +
        "&formats=v1&turn_id=tid&replace_from=rid&kept=k1%2Ck2");
    assert.ok(page.questionUrl("Q", "p-1", "tid", "rid", []).endsWith("&replace_from=rid&kept="));
    assert.ok(!page.questionUrl("Q", "p-1", "tid", "", ["k1"]).includes("kept"));
});

test("length guidance: nothing below 80 %, a red counter near the limit, sending blocked above it", () => {
    assert.equal(page.QUESTION_LIMIT, 2000);  // = the profile's guard.max_question_chars
    const ok = page.lengthState("x".repeat(1599));
    assert.equal(ok.state, "ok");
    assert.equal(ok.hint, "");
    const near = page.lengthState("x".repeat(1600));
    assert.equal(near.state, "near");
    assert.equal(near.counter, "1.600 / 2.000");
    assert.match(near.hint, /mehrere Fragen/);
    assert.equal(page.lengthState("x".repeat(2000)).state, "near");
    const over = page.lengthState("x".repeat(2001));
    assert.equal(over.state, "over");
    assert.equal(over.counter, "2.001 / 2.000");
    assert.match(over.hint, /höchstens 2\.000 Zeichen/);
    assert.match(over.hint, /mehrere kürzere Fragen/);
    assert.equal(page.lengthState(" ".repeat(2500)).state, "ok");  // only the trimmed text is sent
});

test("the page counts like the backend: code points of the trimmed text", () => {
    assert.equal(page.questionLength("  \u{1F600}\u{1F600}ab\n "), 4);
    assert.equal(page.formatCount(12345), "12.345");
    assert.equal(page.formatCount(999), "999");
});

// ---- response formats v1

test("every request says the page renders formats v1; variants and switches name the last question", () => {
    assert.equal(page.FORMATS_VERSION, "v1");
    assert.ok(page.questionUrl("Q", "p-1").endsWith("&formats=v1"));
    assert.equal(page.variantUrl("Was?", "p-1", "tid", "table", false),
        "/api/chat?question=Was%3F&conversation_id=p-1&formats=v1&variant_of=tid&variant=table");
    assert.ok(page.variantUrl("Q", "p-1", "tid", "original", true).endsWith("&variant_of=tid&variant_select=original"));
});

test("the menu: shapes, length and language only on the last finished answer with sources; export on every answer", () => {
    const turn = { id: "a", answer: "Text [1]", state: "done", citations: [{ n: 1 }], active: "table", shape: "text" };
    const groups = plain(page.menuEntries(turn, true));
    assert.deepEqual(groups.map(g => g.label), ["Form", "Länge", "Sprache", "Export"]);
    assert.deepEqual(groups[0].items.map(i => [i.name, i.enabled]),
        [["table", false], ["checklist", true], ["steps", true], ["mindmap", true]]);
    assert.deepEqual(groups[3].items.map(i => [i.name, i.enabled]), [["mail", true], ["print", true]]);
    const only = t => plain(page.menuEntries(t, true)).map(g => g.label);
    assert.deepEqual(plain(page.menuEntries(turn, false)).map(g => g.label), ["Export"]);
    assert.deepEqual(only(Object.assign({}, turn, { citations: [] })), ["Export"]);
    assert.deepEqual(only(Object.assign({}, turn, { state: "interrupted" })), ["Export"]);
    assert.deepEqual(only(Object.assign({}, turn, { id: undefined })), ["Export"]);
    // no shape frame = a backend without formats v1: only Export
    assert.deepEqual(only(Object.assign({}, turn, { shape: undefined })), ["Export"]);
    assert.deepEqual(plain(page.menuEntries(Object.assign({}, turn, { answer: " " }), true)), []);
    assert.equal(plain(page.menuEntries(Object.assign({}, turn, { active: undefined }), true))[0].items[0].enabled, true);
    // the form the answer on screen already has is not offered again (what it contains, not what was asked for)
    const TABLE = "Kurz [1].\n\n| A | B |\n| --- | --- |\n| x [1] | y [1] |";
    const STEPS = "So geht es [1].\n\n1. Antrag stellen [1]\n2. Genehmigen lassen [1]";
    const formOf = t => plain(page.menuEntries(t, true))[0].items.map(i => [i.name, i.enabled]);
    assert.deepEqual(formOf(Object.assign({}, turn, { active: undefined, shape: "table", answer: TABLE })),
        [["table", false], ["checklist", true], ["steps", true], ["mindmap", true]]);
    assert.deepEqual(formOf(Object.assign({}, turn, { active: "shorter", shape: "text", answer: STEPS })),
        [["table", true], ["checklist", true], ["steps", false], ["mindmap", true]]);
    // a table was asked for but the model wrote text: "Als Tabelle" stays on offer
    assert.deepEqual(formOf(Object.assign({}, turn, { active: undefined, shape: "table" })),
        [["table", true], ["checklist", true], ["steps", true], ["mindmap", true]]);
    assert.deepEqual([TABLE, STEPS, "- [ ] a [1]\n- [x] b [1]", "```mindmap\n# A\n```", "1. nur eins [1]", "| a | b |", ""]
        .map(a => page.detectShape(a)), ["table", "steps", "checklist", "mindmap", "text", "text", "text"]);
});

test("version labels and shape names", () => {
    assert.equal(page.VARIANT_LABELS.original, "Original");
    assert.equal(page.VARIANT_LABELS.simple, "Einfache Sprache");
    assert.equal(page.VARIANT_LABELS.english, "English");
    assert.equal(page.shapeName("steps"), "steps");
    assert.equal(page.shapeName("<b>"), "text");
    assert.equal(page.shapeName(undefined), "text");
});

test("checklist items: the [ ] prefix, ticked or not, and the key a tick is stored under", () => {
    assert.deepEqual(plain(page.parseTask("[ ] Antrag stellen [1]")), { checked: false, rest: "Antrag stellen [1]" });
    assert.deepEqual(plain(page.parseTask(" [x] Erledigt")), { checked: true, rest: "Erledigt" });
    assert.equal(page.parseTask("Antrag stellen"), null);
    assert.equal(page.parseTask("[1] Quelle"), null);
    assert.equal(page.tickKey(undefined, 2), "original:2");
    assert.equal(page.tickKey("checklist", 0), "checklist:0");
});

test("German numbers: thousands, decimals, units, citations; anything else is not a number", () => {
    assert.equal(page.parseGermanNumber("1.234,5"), 1234.5);
    assert.equal(page.parseGermanNumber("58 316 [3]"), 58316);
    assert.equal(page.parseGermanNumber("12 %"), 12);
    assert.equal(page.parseGermanNumber("3,5 Mio. €"), 3500000);
    assert.equal(page.parseGermanNumber("372.914.508 €"), 372914508);
    assert.equal(page.parseGermanNumber("−2,5"), -2.5);
    assert.equal(page.parseGermanNumber("2021"), 2021);
    assert.equal(page.parseGermanNumber("58 316"), 58316);
    assert.equal(page.parseGermanNumber("0,5"), 0.5);
    assert.equal(page.parseGermanNumber("0"), 0);
    for (const bad of ["1 2", "30 40", "0123 456789", "0123456789"]) assert.equal(page.parseGermanNumber(bad), null, bad);
    for (const bad of ["–", "", "30 Tage", "ab 2020", "1,2,3", "n. v."]) assert.equal(page.parseGermanNumber(bad), null, bad);
});

test("a chart needs three rows and fully numeric columns; years give a line (Review Focus 4)", () => {
    const header = ["Jahr", "Studierende [1]", "Bemerkung"];
    const rows = [["2019", "81.352", "x"], ["2020", "82.019", "y"], ["2021", "82.746 [2]", "z"]];
    assert.deepEqual(plain(page.chartSpec(header, rows)),
        { kind: "line", labels: ["2019", "2020", "2021"], series: [{ name: "Studierende", values: [81352, 82019, 82746] }] });
    assert.equal(page.chartSpec(header, rows.slice(0, 2)), null);
    assert.equal(page.chartSpec(["Jahr", "Wert"], [["2019", "81.352"], ["2020", "–"], ["2021", "82.746"]]), null);
    const mixed = plain(page.chartSpec(["Jahr", "A", "B"], [["2019", "1", "–"], ["2020", "2", "5"], ["2021", "3", "6"]]));
    assert.deepEqual(mixed.series.map(s => s.name), ["A"]);
    const yearCol = plain(page.chartSpec(["Haushalt", "Jahr", "Betrag"], [["A", "2024", "5"], ["B", "2025", "7"], ["C", "2026", "9"]]));
    assert.deepEqual(yearCol.series.map(s => s.name), ["Betrag"]);
    assert.equal(plain(page.chartSpec(["Amt", "Gebühr"], [["A", "5 €"], ["B", "7 €"], ["C", "9 €"]])).kind, "bar");
    const config = plain(page.chartConfig(page.chartSpec(header, rows)));
    assert.equal(config.type, "line");
    assert.deepEqual(config.data.labels, ["2019", "2020", "2021"]);
    assert.deepEqual(config.data.datasets, [{ label: "Studierende", data: [81352, 82019, 82746] }]);
    assert.equal(config.options.plugins.legend.display, false);
    // axis and tooltip numbers in German (66.500), whatever the browser's language
    assert.equal(config.options.locale, "de-DE");
    // a source column is never a series, even when the model writes bare numbers into it
    const sourced = plain(page.chartSpec(["Jahr", "Studierende", "Quelle"], [["2019", "81.352", "1"], ["2020", "82.019", "2"],
        ["2021", "82.746", "2"]]));
    assert.deepEqual(sourced.series.map(s => s.name), ["Studierende"]);
});

test("a mind map from # / ## / - lines, escaped, deeper levels flattened; the Freeplane export", () => {
    const tree = plain(page.mindmapTree("# Arbeitszeit\n## Gleitzeit\n- Kernzeit 9–15 Uhr [1]\n  - tiefer <b>\n## Mobile Arbeit\n- DV [2]"));
    assert.equal(tree.content, "Arbeitszeit");
    assert.deepEqual(tree.children.map(c => c.content), ["Gleitzeit", "Mobile Arbeit"]);
    assert.deepEqual(tree.children[0].children.map(c => c.content), ["Kernzeit 9–15 Uhr [1]", "tiefer &lt;b&gt;"]);
    const crlf = plain(page.mindmapTree("# Arbeitszeit\r\n## Gleitzeit\r\n- Kernzeit 9–15 Uhr [1]\r\n  - tiefer <b>\r\n## Mobile Arbeit\r\n- DV [2]"));
    assert.deepEqual(crlf, tree);
    assert.equal(page.mindmapTree("nur Text"), null);
    assert.equal(plain(page.mindmapTree("## Nur Bereich\n- Punkt")).content, "Überblick");
    assert.equal(page.freeplaneXml(page.mindmapTree("# A & B\n## C\n- \"D\" [1]")),
        '<map version="1.0.1"><node TEXT="A &amp; B"><node TEXT="C"><node TEXT="&quot;D&quot; [1]"></node></node></node></map>');
});

test("plain text for e-mail: tables as lines, checklist boxes, no Markdown marks", () => {
    const md = "## Überblick\n\nEs gilt **Folgendes** [1]:\n\n| Gruppe | Tage |\n| --- | --- |\n| Beamte | 30 [1] |\n\n- [ ] Antrag [2]\n- Punkt";
    assert.equal(page.plainAnswer(md),
        "Überblick\n\nEs gilt Folgendes [1]:\n\nGruppe – Tage\nBeamte – 30 [1]\n\n☐ Antrag [2]\n- Punkt");
});

test("mail: subject from the question, answer and sources in the body; long bodies go to the clipboard", () => {
    const cards = page.citedCards(page.sourceCards([doc(DOC_A, 2)], 2026), page.citationEntries([entry(1, DOC_A, 4)]), [1], 2026);
    const short = page.mailPlan("Wie beantrage ich Urlaub?", "Mit dem Antrag [1].", cards);
    assert.equal(short.clipboard, null);
    assert.ok(short.href.startsWith("mailto:?subject=Auskunft%3A%20Wie%20beantrage%20ich%20Urlaub%3F&body="));
    assert.ok(decodeURIComponent(short.href).includes("Mit dem Antrag [1].\n\nQuellen:\n[1] Titel 2"));
    const long = page.mailPlan("x".repeat(80), "y ".repeat(1500), cards);
    assert.ok(long.clipboard.startsWith("y y"));
    assert.ok(decodeURIComponent(long.href).includes("Auskunft: " + "x".repeat(59) + "…"));
    assert.ok(decodeURIComponent(long.href).endsWith("Text liegt in der Zwischenablage – bitte einfügen."));
    assert.ok(long.href.length <= 1800);
});

test("an odd number of code fences is closed before the final render", () => {
    assert.equal(page.closeOpenFence("Text\n```mindmap\n# A\n## B"), "Text\n```mindmap\n# A\n## B\n```");
    assert.equal(page.closeOpenFence("Text\n```mindmap\n# A\n```"), "Text\n```mindmap\n# A\n```");
    assert.equal(page.closeOpenFence("| a |\n| b |"), "| a |\n| b |");
});

test("a reload during a variant returns to the version that was on screen (restoreTo)", () => {
    const original = { name: "original", answer: "Text [1]", citations: [], shape: "text" };
    const table = { name: "table", answer: "| A |\n| --- |", citations: [], shape: "table" };
    const turn = { state: "streaming", active: "english", restoreTo: "table", answer: "Hal", citations: [],
        versions: [original, table, { name: "english", answer: "Hal", citations: [], shape: "table", streaming: true }] };
    const got = plain(page.restoreVersions(turn));
    assert.equal(got.active, "table");
    assert.equal(got.answer, table.answer);
    assert.equal(got.shape, "table");
    assert.equal(got.versions.length, 2);
    assert.equal(got.state, "done");
    assert.ok(!("restoreTo" in got));
});

test("a turn stored while a variant streamed comes back as its last complete version (Review Focus 2)", () => {
    const original = { name: "original", answer: "Text [1]", citations: [entry(1, DOC_A, 2)], shape: "text" };
    const turn = { state: "streaming", active: "table", answer: "| halb", citations: [],
        versions: [original, { name: "table", answer: "| halb", citations: [], shape: "table", streaming: true }] };
    const got = plain(page.restoreVersions(turn));
    assert.equal(got.state, "done");
    assert.equal(got.answer, "Text [1]");
    assert.equal(got.active, "original");
    assert.deepEqual(got.versions, []);
    assert.equal(got.citations[0].page, 2);
    const two = { state: "done", active: "table", answer: "T", citations: [],
        versions: [original, { name: "table", answer: "T", citations: [], shape: "table" }] };
    assert.equal(plain(page.restoreVersions(two)).versions.length, 2);
    assert.equal(plain(page.restoreVersions(two)).shape, "table");
    assert.equal(plain(page.restoreVersions({ state: "done", answer: "x" })).answer, "x");
});

test("the menu opens below when it fits, else above, else on the roomier side with a scrolling height", () => {
    // an older answer high on the page: room below
    assert.deepEqual(plain(page.menuPlacement(470, 100, 600)), { up: false, maxHeight: null });
    // the last answer just above the composer: no room below, room above
    assert.deepEqual(plain(page.menuPlacement(470, 520, 20)), { up: true, maxHeight: null });
    // a short window (the page embedded in a frame): neither fits, the roomier side scrolls
    assert.deepEqual(plain(page.menuPlacement(470, 300, 90)), { up: true, maxHeight: 300 });
    assert.deepEqual(plain(page.menuPlacement(470, 60, 250.6)), { up: false, maxHeight: 250 });
    // never shorter than a usable list, even with almost no room anywhere
    assert.deepEqual(plain(page.menuPlacement(470, 40, 30)), { up: true, maxHeight: 120 });
    // the exact fit still opens below
    assert.deepEqual(plain(page.menuPlacement(470, 600, 470)), { up: false, maxHeight: null });
});

test("versions are made from the original: the menu greys out the original's form and says so on another tab", () => {
    const TABLE = "Kurz [1].\n\n| A | Quelle |\n| --- | --- |\n| x | [1] |";
    const MAP = "Überblick [1].\n\n```mindmap\n# A\n## B\n- C [1]\n```";
    const turn = { id: "a", state: "done", citations: [{ n: 1 }], shape: "table", active: "mindmap", answer: MAP,
        versions: [{ name: "original", answer: TABLE }, { name: "mindmap", answer: MAP }] };
    const form = plain(page.menuEntries(turn, true))[0].items.map(i => [i.name, i.enabled]);
    assert.deepEqual(form, [["table", false], ["checklist", true], ["steps", true], ["mindmap", false]]);
    assert.equal(page.menuNote(turn, true), "Neue Fassung aus der Original-Antwort");
    assert.equal(page.menuNote(Object.assign({}, turn, { active: "original", answer: TABLE }), true), "");
    assert.equal(page.menuNote(Object.assign({}, turn, { active: undefined, versions: undefined }), true), "");
    assert.equal(page.menuNote(turn, false), "");   // an older answer offers export only: no note
});

test("the page detects every shape of the shared fixture like the backend and the evals checks", () => {
    const fixture = JSON.parse(readFileSync(new URL("./fixtures/answer_shapes.json", import.meta.url), "utf8"));
    assert.ok(fixture.samples.length >= 16);
    for (const sample of fixture.samples) {
        assert.equal(page.detectShape(sample.text), sample.shape, JSON.stringify(sample.text));
    }
});

function fakeWindow(hash, stored, blocked) {
    const kept = new Map(stored ? [["assistant-token-v1", stored]] : []);
    const replaced = [];
    return {
        kept, replaced,
        location: { hash, pathname: "/", search: "?m=nh" },
        history: { replaceState: (...args) => replaced.push(args) },
        sessionStorage: {
            getItem: key => { if (blocked) throw new Error("blocked"); return kept.has(key) ? kept.get(key) : null; },
            setItem: (key, value) => { if (blocked) throw new Error("blocked"); kept.set(key, value); },
        },
    };
}

test("the #token link: kept in sessionStorage, removed from the address bar, sent as a Bearer header", () => {
    const win = fakeWindow("#s3cret", "");
    assert.equal(page.takeToken(win), "s3cret");
    assert.equal(win.kept.get("assistant-token-v1"), "s3cret");
    assert.deepEqual(win.replaced, [[null, "", "/?m=nh"]]);   // the path and query stay, the fragment goes
    assert.deepEqual(plain(page.withToken({ cache: "no-store" })), { cache: "no-store", headers: { Authorization: "Bearer s3cret" } });
    // a reload of the tab has no fragment: the stored token stays; a new link replaces it
    assert.equal(page.takeToken(fakeWindow("", "s3cret")), "s3cret");
    assert.equal(page.takeToken(fakeWindow("#newer", "s3cret")), "newer");
});

test("without a token or with blocked storage the page still works", () => {
    const win = fakeWindow("", "");
    assert.equal(page.takeToken(win), "");
    assert.deepEqual(plain(page.withToken({ cache: "no-store" })), { cache: "no-store" });
    assert.deepEqual(win.replaced, []);
    assert.equal(page.takeToken(fakeWindow("#kept-in-memory", "", true)), "kept-in-memory");
    assert.equal(page.withToken({}).headers.Authorization, "Bearer kept-in-memory");
});

test("the error reference: meta's turn id, else the page's own", () => {
    // the first 8 characters of the turn id, as the backend logs them (ref=)
    assert.equal(page.refOf("A1B2C3D4-0000-4000-8000-000000000001"), "a1b2c3d4");
    assert.equal(page.refOf(""), "");
    assert.equal(page.refOf("p-xyz"), "");
    assert.equal(page.refOf(undefined), "");
    const own = "00000000-0000-4000-8000-000000000001";
    const turn = { id: own, state: "streaming" };
    page.handleFrame(turn, {}, { type: "meta", content: { format: 1, turn_id: own, trace_id: null } });
    assert.equal(turn.ref, undefined);   // the question's meta id is the page's own: the stored turn stays as it was
    page.handleFrame(turn, {}, { type: "meta", content: { format: 1, turn_id: "ABCDEF01-0000-4000-8000-000000000001", trace_id: null } });
    assert.equal(turn.ref, "abcdef01");
    assert.equal(page.referenceLine(Object.assign({}, turn, { state: "error" })).endsWith("Referenz abcdef01."), true);
});

test("the reference line shows under failed, interrupted and error-frame answers only", () => {
    const id = "00000000-0000-4000-8000-000000000001";
    const line = "Bitte nennen Sie bei Rückfragen die Referenz 00000000.";
    assert.equal(page.referenceLine({ id, state: "error", sawError: false }), line);
    assert.equal(page.referenceLine({ id, state: "interrupted", sawError: false }), line);
    assert.equal(page.referenceLine({ id, state: "done", sawError: true }), line);
    assert.equal(page.referenceLine({ id, state: "done", sawError: false }), "");
    assert.equal(page.referenceLine({ id, state: "stopped", sawError: false }), "");
    assert.equal(page.referenceLine({ id, state: "streaming", sawError: true }), "");
    assert.equal(page.referenceLine({ state: "error", sawError: false }), "");   // restored from before the ids: no reference
    assert.equal(page.referenceLine({ id: "p-xyz", state: "error" }), "");
});

test("angle-bracket placeholders survive the markdown converter; tags and code stay", async () => {
    // this repository is "type": "module", so require() would load the UMD file as ESM: run it as the browser does, a script
    const umd = { exports: {} };
    new Function("module", "exports", readFileSync(new URL("../vendor/showdown-2.1.0.min.js", import.meta.url), "utf8"))(umd, umd.exports);
    const showdown = umd.exports;
    const html = markdown => new showdown.Converter({ tables: true, simplifiedAutoLink: true })
        .makeHtml(page.keepPlaceholders(markdown));
    const out = html("Nennen Sie <Rechnungsnummer> und <Anschrift des Schuldners> sowie <name des kindes> und <datum> an <b>Amt</b> <https://a.de/X>.");
    assert.ok(out.includes("&lt;Rechnungsnummer&gt;") && out.includes("&lt;Anschrift des Schuldners&gt;"));
    assert.ok(out.includes("&lt;name des kindes&gt;") && out.includes("&lt;datum&gt;"));
    assert.ok(out.includes("<b>Amt</b>") && out.includes('href="https://a.de/X"'));
    const table = html("| A | B |\n|---|---|\n| eins<br>zwei | <Name> |");
    assert.ok(table.includes("eins<br />zwei") || table.includes("eins<br>zwei"));
    assert.ok(table.includes("&lt;Name&gt;"));
    assert.ok(html("`<Name>`").includes("<code>&lt;Name&gt;</code>"));
    assert.ok(html("```\n<Name>\n```").includes("&lt;Name&gt;"));
    assert.equal(page.keepPlaceholders("<br> `<Ä>` <Ä> <max@x.de>"), "<br> `<Ä>` &lt;Ä&gt; <max@x.de>");
});

// The sanitiser's behaviour needs a DOM (the browser suite renders the same inputs in Chromium); here its settings and its
// attribute hook are checked, with the markup showdown hands to it.
function showdownHtml(markdown) {
    const umd = { exports: {} };
    new Function("module", "exports", readFileSync(new URL("../vendor/showdown-2.1.0.min.js", import.meta.url), "utf8"))(umd, umd.exports);
    return new umd.exports.Converter({ tables: true, simplifiedAutoLink: true }).makeHtml(page.keepPlaceholders(markdown));
}

test("sanitiser: a tag split over two lines reaches it, and media tags are not among the tags it allows", () => {
    for (const probe of ["<video\nsrc=https://evil.example/v>", "<audio\nsrc=https://evil.example/a autoplay>"]) {
        assert.ok(/<(video|audio)\b/.test(showdownHtml(probe)), probe);
    }
    const allowed = page.SANITIZE_CONFIG.ALLOWED_TAGS;
    for (const tag of ["video", "audio", "source", "track", "img", "iframe", "svg", "form", "style", "script"]) {
        assert.ok(!allowed.includes(tag), tag);
    }
    assert.deepEqual(allowed, page.ANSWER_TAGS);   // what keepPlaceholders lets stand is what the sanitiser allows
});

test("sanitiser: data-, aria-, id, role and tabindex attributes are dropped (a forged citation marker has no tooltip data)", () => {
    const forged = '<a class="cite" data-n="1" data-tip-title="Prüfungsordnung 2023" data-tip-quote="x" href="https://evil.example/">1</a>';
    assert.ok(showdownHtml(forged).includes("data-tip-title"));   // the converter keeps it: the sanitiser must not
    assert.equal(page.SANITIZE_CONFIG.ALLOW_DATA_ATTR, false);
    assert.equal(page.SANITIZE_CONFIG.ALLOW_ARIA_ATTR, false);
    for (const name of ["style", "id", "role", "tabindex"]) assert.ok(page.SANITIZE_CONFIG.FORBID_ATTR.includes(name), name);
    assert.ok(page.SANITIZE_CONFIG.ADD_ATTR.includes("target"));
});

test("sanitiser: class survives only on code (a forged mind-map box or citation marker loses it, code.language-mindmap stays)", () => {
    const kept = (nodeName, attrName) => {
        const data = { attrName, keepAttr: true };
        page.dropForeignClass({ nodeName }, data);
        return data.keepAttr;
    };
    assert.equal(kept("DIV", "class"), false);   // <div class="mindmap" data-src="# A">
    assert.equal(kept("A", "class"), false);     // <a class="cite" ...>
    assert.equal(kept("SPAN", "class"), false);
    assert.equal(kept("CODE", "class"), true);   // ```mindmap
    assert.equal(kept("A", "href"), true);
    // the hook is registered once per sanitiser, and every call carries the settings
    const calls = [];
    const purify = { addHook: (...args) => calls.push(["hook", args[0]]), sanitize: (html, config) => { calls.push(["sanitize", config]); return html; } };
    page.sanitize(purify, "<p>a</p>");
    page.sanitize(purify, "<p>b</p>");
    assert.deepEqual(calls.map(c => c[0]), ["hook", "sanitize", "sanitize"]);
    assert.equal(calls[0][1], "uponSanitizeAttribute");
    assert.equal(calls[1][1], page.SANITIZE_CONFIG);
});

// A window that fires hashchange when the test sets a new fragment; a reload is counted.
function linkWindow(stored, blocked) {
    const win = fakeWindow("", stored, blocked);
    const listeners = [];
    win.reloads = 0;
    win.location.reload = () => { win.reloads += 1; };
    win.addEventListener = (type, listener) => { if (type === "hashchange") listeners.push(listener); };
    win.open = hash => {
        win.location.hash = hash;
        listeners.forEach(listener => listener());
    };
    return win;
}

test("a #token link opened again in the same tab is taken and the page reloads", () => {
    const win = linkWindow("");
    page.takeToken(win);
    let again = 0;
    page.followNewToken(win, () => { again += 1; });
    win.open("#fresh");
    assert.equal(win.kept.get("assistant-token-v1"), "fresh");
    assert.deepEqual(win.replaced, [[null, "", "/?m=nh"]]);   // the token leaves the address bar
    assert.equal(page.withToken({}).headers.Authorization, "Bearer fresh");
    assert.deepEqual([win.reloads, again], [1, 0]);
    win.open("");   // a change without a token is none of its business
    win.open("#");
    assert.deepEqual([win.reloads, again], [1, 0]);
});

test("with blocked storage a new #token link asks again instead of reloading", () => {
    const win = linkWindow("", true);
    let again = 0;
    page.followNewToken(win, () => { again += 1; });
    win.open("#in-memory");
    assert.equal(page.withToken({}).headers.Authorization, "Bearer in-memory");
    assert.deepEqual([win.reloads, again], [0, 1]);
});

test("chart numbers are read per column: an English column in English, an ambiguous one draws no chart", () => {
    const rows = cells => cells.map((cell, i) => [String(2000 + i), cell]);
    const values = cells => { const spec = page.chartSpec(["Jahr", "Wert"], rows(cells)); return spec && plain(spec.series[0].values); };
    assert.deepEqual(values(["1,234.5", "2,000", "3.5"]), [1234.5, 2000, 3.5]);     // the English form decides the column
    assert.deepEqual(values(["66,500", "70,250", "1,234,567"]), [66500, 70250, 1234567]);
    assert.deepEqual(values(["1,5", "2,000", "3,25"]), [1.5, 2, 3.25]);              // the German form decides it
    assert.deepEqual(values(["81.352", "82.019", "82.746"]), [81352, 82019, 82746]);   // thousands with points: German, as before
    assert.equal(values(["1,500", "2,500", "66,500"]), null);                         // only "d,ddd" cells: German 1.5 or English 1500
    assert.equal(values(["1,234.5", "1,5", "3"]), null);                              // both forms in one column
    assert.equal(values(["1,234.5", "–", "3"]), null);
    assert.equal(page.parseEnglishNumber("1,500"), 1500);
    assert.equal(page.parseEnglishNumber("12.5 %"), 12.5);
    assert.equal(page.parseEnglishNumber("0,5"), null);
    assert.equal(page.parseEnglishNumber("1,2,3"), null);
});

test("the counter and the limit text use the page language's number format; German stays 2.000", () => {
    page.setLang("en");
    const over = page.lengthState("x".repeat(2001));
    assert.equal(over.counter, "2,001 / 2,000");
    assert.ok(over.hint.includes("2,000") && !over.hint.includes("2.000"));
    assert.equal(page.lengthState("x".repeat(1700)).counter, "1,700 / 2,000");
    page.setLang("de");
    assert.equal(page.lengthState("x".repeat(1700)).counter, "1.700 / 2.000");
    assert.ok(page.lengthState("x".repeat(2001)).hint.includes("2.000"));
});

test("a grouped citation is split into one reference per number", () => {
    assert.deepEqual(plain(page.citationParts("Ja [1, 2] und [3,4][5].")), [
        { text: "Ja " }, { refs: [1, 2] }, { text: " und " }, { refs: [3, 4] }, { refs: [5] }, { text: "." },
    ]);
});
