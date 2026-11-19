// Look and language tests (DE/EN strings, footer, source cards, contact links, the watchdog; native ES modules).
// From web/page: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { questionUrl } from "../api/urls.js";
import { DEGREE_NAMES, LANGS, STRINGS, TEXT, VARIANT_LABELS, applyStatic, detectLang, getLang, providerLine, setLang }
    from "../i18n.js";
import { menuEntries } from "../model/formats.js";
import { applyStatus, filterText, formatSeconds, newThinking, summaryText } from "../model/thinking.js";
import { chipChoices } from "../ui/chips.js";
import { readFileSync } from "node:fs";
import { SILENCE_LIMIT_MS, TOTAL_LIMIT_MS } from "../config.js";
import { watchdogExpired, watchdogLimits } from "../model/streaming.js";
import { recordStatus } from "../model/turns.js";
import { sourceCards } from "../model/sources.js";
import { sourceElement } from "../render/sources.js";

const plain = value => JSON.parse(JSON.stringify(value));

// every test that switches the language puts German back
function inLang(t, lang) {
    setLang(lang);
    t.after(() => setLang("de"));
}

// ---- the watchdog

test("the watchdog waits as long as the backend's meta frame says (test_watchdog_uses_meta_deadline)", () => {
    // before a meta frame: the page's own limits
    assert.deepEqual(plain(watchdogLimits(undefined)), { silenceMs: SILENCE_LIMIT_MS, totalMs: TOTAL_LIMIT_MS });
    assert.deepEqual(plain(watchdogLimits({ deadline_s: "x", idle_s: -1 })), { silenceMs: SILENCE_LIMIT_MS, totalMs: TOTAL_LIMIT_MS });
    // a cold model: the card allows 240 s in all and 90 s of silence; the page waits a little longer than the backend
    const meta = watchdogLimits({ deadline_s: 240, idle_s: 90 });
    assert.ok(meta.totalMs > 240000 && meta.totalMs <= 250000);
    assert.ok(meta.silenceMs > 90000 && meta.silenceMs <= 100000);
    const session = Object.assign({ startedAt: 0, lastData: 190000 }, meta);
    // 200 s into the request, the backend spoke 10 s ago: still waiting (the fixed 150 s would have given up)
    assert.equal(watchdogExpired(session, 200000), false);
    assert.equal(watchdogExpired(Object.assign({ startedAt: 0, lastData: 190000 }, watchdogLimits(undefined)), 200000), true);
    // beyond the deadline, or silent longer than idle_s, the connection is treated as gone
    assert.equal(watchdogExpired(Object.assign({}, session, { lastData: 240000 }), 250001), true);
    assert.equal(watchdogExpired(Object.assign({}, session, { lastData: 100000 }), 200000), true);
    assert.equal(watchdogExpired(Object.assign({}, session, { lastData: 100000 }), 190000), false);
});

test("a retry after a network failure forgets the earlier refusal (test_retry_clears_stale_failure_flags)", () => {
    const turn = {};
    recordStatus(turn, 422);
    assert.deepEqual([turn.unauthorized, turn.refused], [false, true]);
    recordStatus(turn, 401);
    assert.deepEqual([turn.unauthorized, turn.refused], [true, false]);
    recordStatus(turn, 0);   // the start of the next attempt, before any answer of the backend
    assert.deepEqual([turn.unauthorized, turn.refused], [false, false]);
});

// ---- languages

test("the page language: stored choice, else browser, else German; the switch changes every text (test_i18n_switch)", t => {
    assert.deepEqual(LANGS, ["de", "en"]);
    assert.equal(detectLang("en", "de-DE"), "en");
    assert.equal(detectLang(null, "en-GB"), "en");
    assert.equal(detectLang(null, "de-AT"), "de");
    assert.equal(detectLang(null, "fr-FR"), "de");
    assert.equal(detectLang("xx", undefined), "de");
    assert.equal(getLang(), "de");
    assert.equal(TEXT.failed, "Die Antwort konnte nicht geladen werden. Bitte versuchen Sie es noch einmal.");
    assert.equal(DEGREE_NAMES.promotion, "Promotion");
    assert.equal(VARIANT_LABELS.table, "Tabelle");
    setLang("en");
    t.after(() => setLang("de"));
    assert.equal(getLang(), "en");
    assert.equal(TEXT.failed, "The answer could not be loaded. Please try again.");
    assert.equal(TEXT.unauthorized, "Access was not accepted. Please open the link from “rag up” again.");
    assert.equal(DEGREE_NAMES.promotion, "Doctorate");
    assert.equal(VARIANT_LABELS.table, "Table");
    assert.equal(VARIANT_LABELS.original, "Original");
    assert.deepEqual(menuEntries({ id: "a", answer: "Text [1]", state: "done", citations: [{ n: 1 }], shape: "text" }, true)
        .map(group => group.label), ["Form", "Length", "Language", "Export"]);
    // anything but de/en is German
    setLang("fr");
    assert.equal(getLang(), "de");
});

test("both languages have the same strings, with the same {marks} (test_i18n_switch)", () => {
    const marks = text => (String(text).match(/\{\w+\}/g) || []).sort().join();
    assert.deepEqual(Object.keys(STRINGS.en).sort(), Object.keys(STRINGS.de).sort());
    for (const key of Object.keys(STRINGS.de)) assert.equal(marks(STRINGS.en[key]), marks(STRINGS.de[key]), key);
});

test("the page's markup takes its text from the strings in use (test_i18n_switch)", t => {
    const make = attrs => {
        const el = { attrs: Object.assign({}, attrs), textContent: "",
            getAttribute(name) { return this.attrs[name]; }, setAttribute(name, value) { this.attrs[name] = value; } };
        return el;
    };
    const greeting = make({ "data-i18n": "greeting" });
    const button = make({ "data-i18n-attrs": "title=newChatTitle, aria-label=newChat" });
    const root = { querySelectorAll: selector => (selector === "[data-i18n]" ? [greeting] : [button]) };
    applyStatic(root, { name: "rag-tuda" });
    assert.match(greeting.textContent, /^Hallo, ich bin rag-tuda,/);
    assert.equal(button.attrs.title, "Neues Gespräch beginnen");
    inLang(t, "en");
    applyStatic(root, { name: "rag-tuda" });
    assert.match(greeting.textContent, /^Hello, I am rag-tuda,/);
    assert.equal(button.attrs["aria-label"], "New conversation");
});

test("thinking steps, summaries and numbers are worded in the page language (test_i18n_switch)", t => {
    const run = () => {
        const thinking = newThinking(0);
        applyStatus(thinking, { step: "analysing" }, "Q");
        applyStatus(thinking, { step: "understood", question: "Anmeldung zur Prüfung" }, "anmeldung");
        applyStatus(thinking, { step: "searching", queries: ["Prüfung"] }, "Q");
        applyStatus(thinking, { step: "searching_more" }, "Q");
        applyStatus(thinking, { step: "found", documents: 2 }, "Q");
        return thinking;
    };
    const german = run();
    assert.deepEqual(german.steps.map(step => step.text), [
        "Frage analysiert", "Verstanden als: „Anmeldung zur Prüfung“", "Gesucht nach: „Prüfung“",
        "Weitere Dokumente gesucht", "2 Dokumente gefunden"]);
    assert.equal(summaryText(Object.assign(german, { endedAt: 12500 })), "1 Suchbegriff · 2 Dokumente · 13 s");
    assert.equal(formatSeconds(2500), "2,5 s");
    inLang(t, "en");
    const english = run();
    assert.deepEqual(english.steps.map(step => step.text), [
        "Question analysed", "Understood as: “Anmeldung zur Prüfung”", "Searched for: “Prüfung”",
        "Searched for further documents", "2 documents found"]);
    assert.equal(summaryText(Object.assign(english, { endedAt: 12500 })), "1 search term · 2 documents · 13 s");
    assert.equal(formatSeconds(2500), "2.5 s");
});

test("the iterative round has its own status step, active while it runs (test_searching_more_step)", t => {
    const step = lang => {
        setLang(lang);
        const thinking = newThinking(0);
        applyStatus(thinking, { step: "searching", queries: ["a"] }, "Q");
        applyStatus(thinking, { step: "searching_more" }, "Q");
        return thinking.steps[thinking.steps.length - 1];
    };
    t.after(() => setLang("de"));
    assert.deepEqual(plain(step("de")), { key: "search_more", text: "Suche nach weiteren Dokumenten …", state: "active" });
    assert.deepEqual(plain(step("en")), { key: "search_more", text: "Searching for further documents …", state: "active" });
});

test("the chips show the English programme labels in English (test_chips_english)", t => {
    const prof = {
        degrees: ["master"], po_versions: [2023],
        programmes: [{ id: "msc-informatik", label_de: "M.Sc. Informatik", label_en: "M.Sc. Computer Science (Informatik)",
            degree: "master", po_versions: [2023] }],
    };
    assert.equal(chipChoices(prof, {}).programmes[0].label, "M.Sc. Informatik");
    inLang(t, "en");
    assert.equal(chipChoices(prof, {}).programmes[0].label, "M.Sc. Computer Science (Informatik)");
    // without an English label the German one stays
    const bare = { degrees: [], po_versions: [], programmes: [{ id: "x", label_de: "Nur deutsch", degree: "master", po_versions: [] }] };
    assert.equal(chipChoices(bare, {}).programmes[0].label, "Nur deutsch");
    assert.equal(filterText({ degree: ["promotion"], programme: ["x"] }, { x: "Name" }), "Filter: Doctorate · Name");
});

test("a question names the page language (test_question_url_carries_lang)", () => {
    assert.ok(questionUrl("Q", "c-1", "", "", undefined, undefined, "en").endsWith("&lang=en"));
    assert.ok(questionUrl("Q", "c-1", "tid", "rid", ["a"], { degree: "master", programme: "", po: "" }, "de")
        .endsWith("&turn_id=tid&replace_from=rid&kept=a&degree=master&lang=de"));
    // nothing is sent for a language the page does not know
    assert.ok(!questionUrl("Q", "c-1", "", "", undefined, undefined, "EN").includes("lang="));
    assert.ok(!questionUrl("Q", "c-1").includes("lang="));
});

// ---- the footer and the header

test("the footer says it is no official service and no legal advice, and names a remote answer provider (test_footer_and_provider_line)", t => {
    assert.equal(providerLine({ model: "llama-3.3", provider: "groq", remote: true }), "Antworten von llama-3.3 über groq.");
    assert.equal(providerLine({ model: "qwen3", provider: "ollama", remote: false }), "");
    assert.equal(providerLine(undefined), "");
    assert.equal(providerLine({ model: "m", provider: "p" }), "");
    assert.match(TEXT.footer, /kein offizieller Dienst der TU Darmstadt/);
    assert.match(TEXT.footer, /Keine Rechtsberatung/);
    assert.equal(TEXT.headerSub, "für den Fachbereich Informatik der TU Darmstadt");
    inLang(t, "en");
    assert.equal(providerLine({ model: "llama-3.3", provider: "groq", remote: true }), "Answers by llama-3.3 via groq.");
    assert.match(TEXT.footer, /not an official service of TU Darmstadt/);
    assert.match(TEXT.footer, /Not legal advice/);
    assert.equal(TEXT.headerSub, "for the Department of Computer Science at TU Darmstadt");
    // the markup carries both lines and the wordmark
    const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
    assert.match(html, /data-i18n="footer"/);
    assert.match(html, /data-i18n="headerSub"/);
    assert.match(html, /id="provider-line"/);
    assert.match(html, /rag-<span class="brand-hl">tuda<\/span>/);
});

// ---- source cards and contact links

const PO = "https://www.informatik.example.org/studium/po-2023.pdf";
const ISSUE = "https://www.example.org/satzungsbeilage/2023-14.pdf";
const poDoc = extra => ({ page_content: "x", metadata: Object.assign({ download_url: PO, page_number: 3, title: "PO B.Sc.",
    doc_type: "Prüfungsordnung", regulation: "po", po_version: [2023], official_url: ISSUE }, extra || {}) });

// the few things of a DOM that the element builder and the source card use
function fakeDocument() {
    const element = tag => ({ tag, className: "", textContent: "", attrs: {}, children: [],
        setAttribute(name, value) { this.attrs[name] = value; }, addEventListener() {},
        append(...items) { this.children.push(...items); }, set innerHTML(value) { this.html = value; }, nodeType: 1 });
    globalThis.document = { createElement: element };
}
const texts = el => [el.textContent, ...el.children.map(child => (typeof child === "string" ? child : texts(child)))].flat().filter(Boolean);
const links = el => [el.tag === "a" ? el : null, ...el.children.filter(child => typeof child !== "string").map(links)].flat().filter(Boolean);

test("a PO source card shows the PO version and links the official publication (test_source_card_po_and_official_link)", t => {
    fakeDocument();
    t.after(() => { delete globalThis.document; setLang("de"); });
    const [card] = sourceCards([poDoc(), poDoc({ page_number: 5 })], 2026);
    assert.deepEqual([card.poVersion, card.officialUrl], [[2023], ISSUE]);
    const li = sourceElement(Object.assign({ ns: [1], pages: [3], linkPage: 3 }, card));
    assert.ok(texts(li).includes("PO 2023"));
    const official = links(li).find(a => a.attrs.href === ISSUE);
    assert.equal(official.textContent, "Amtlich veröffentlicht");
    assert.equal(official.attrs.rel, "noopener noreferrer");
    assert.ok(texts(li).some(text => text.endsWith("S. 3")));
    setLang("en");
    const english = sourceElement(Object.assign({ ns: [1], pages: [3], linkPage: 3 }, card));
    assert.equal(links(english).find(a => a.attrs.href === ISSUE).textContent, "Officially published");
    assert.ok(texts(english).some(text => text.endsWith("p. 3")));
    // a source without a PO version or a publication, or with an unsafe link, shows neither
    const [other] = sourceCards([poDoc({ po_version: null, official_url: "javascript:alert(1)", regulation: "info" })], 2026);
    assert.deepEqual([other.poVersion, other.officialUrl], [[], ""]);
    const quiet = sourceElement(Object.assign({ ns: [], pages: [], linkPage: null }, other));
    assert.ok(!texts(quiet).includes("PO 2023"));
    assert.ok(!links(quiet).some(a => a.attrs.href === ISSUE || /^javascript/.test(a.attrs.href || "")));
});
