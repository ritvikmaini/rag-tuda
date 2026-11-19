import { withToken } from "../api/auth.js";
import { CHIPS_KEY, DEGREE_NAMES, PROFILE_PATH, TEXT } from "../config.js";
import { h } from "../render/dom.js";
import { store, ui } from "./state.js";

// ---- chips: degree, programme and PO version, offered from GET /api/profile (roles.yaml)
// A chip is text; "" is "not chosen" and is not sent. The selection belongs to the conversation (sessionStorage) and goes
// with every question.

const NONE = { degree: "", programme: "", po: "" };
const FIELDS = ["degree", "programme", "po"];

let profile = null;   // the cleaned /api/profile answer; null until it has loaded (or when it cannot)
let chips = Object.assign({}, NONE);
const selects = {};

// ---- choices

// What each selector offers for the current selection: the profile's degrees, the programmes of the chosen degree and the PO
// years roles.yaml lists for the chosen programme (all years without one).
export function chipChoices(prof, selection) {
    const programmes = prof.programmes.filter(p => !selection.degree || p.degree === selection.degree);
    const chosen = prof.programmes.find(p => p.id === selection.programme);
    return {
        degrees: prof.degrees,
        programmes: programmes.map(p => ({ id: p.id, label: p.label_de })),
        pos: chosen ? chosen.po_versions : prof.po_versions,
    };
}

// The selection with every value the selectors would not offer removed, degree first: a value the profile does not know, a
// programme of another degree, a PO year the programme lacks.
export function cleanChips(prof, selection) {
    if (!prof) return Object.assign({}, NONE);
    const degree = prof.degrees.includes(selection.degree) ? selection.degree : "";
    const programme = chipChoices(prof, { degree }).programmes.some(p => p.id === selection.programme) ? selection.programme : "";
    const po = chipChoices(prof, { degree, programme }).pos.map(String).includes(selection.po) ? selection.po : "";
    return { degree, programme, po };
}

// ---- sticky per conversation

// The chips stored for conversation `id`; none for another conversation, for blocked storage or for anything unreadable.
export function readChips(storage, id) {
    try {
        const data = JSON.parse(storage.getItem(CHIPS_KEY) || "null");
        if (!data || data.id !== id) return Object.assign({}, NONE);
        return Object.fromEntries(FIELDS.map(name => [name, typeof data[name] === "string" ? data[name] : ""]));
    } catch (error) {
        return Object.assign({}, NONE);
    }
}

export function writeChips(storage, id, selection) {
    try {
        storage.setItem(CHIPS_KEY, JSON.stringify(Object.assign({ id }, selection)));
    } catch (error) {
        // blocked or full: the selection lasts until the page reloads
    }
}

function session() {
    try {
        return window.sessionStorage;
    } catch (error) {
        return null;
    }
}

// ---- what the rest of the page reads

// The selection to send with a question.
export function currentChips() {
    return Object.assign({}, chips);
}

// Programme id -> label, for the filter line of the "understood" step.
export function programmeNames() {
    return profile ? Object.fromEntries(profile.programmes.map(p => [p.id, p.label_de])) : {};
}

// Takes the selection stored for the current conversation (call after the conversation id changes).
export function syncChips() {
    chips = cleanChips(profile, readChips(session(), store.id));
    renderChips();
}

// ---- profile

function cleanProfile(data) {
    const list = value => (Array.isArray(value) ? value : []);
    const years = value => list(value).filter(Number.isInteger);
    return {
        degrees: list(data && data.degrees).filter(id => typeof id === "string"),
        programmes: list(data && data.programmes)
            .filter(p => p && typeof p.id === "string" && typeof p.label_de === "string" && typeof p.degree === "string")
            .map(p => ({ id: p.id, label_de: p.label_de, degree: p.degree, po_versions: years(p.po_versions) })),
        po_versions: years(data && data.po_versions),
    };
}

async function loadProfile() {
    try {
        const response = await fetch(PROFILE_PATH, withToken({ cache: "no-store" }));
        if (!response.ok) return;
        profile = cleanProfile(await response.json());
    } catch (error) {
        return;  // no chips: questions are asked without a selection
    }
    syncChips();
}

// ---- selectors

function onPick() {
    chips = cleanChips(profile, Object.fromEntries(FIELDS.map(name => [name, selects[name].value])));
    writeChips(session(), store.id, chips);
    renderChips();
}

function fill(select, label, options, value) {
    select.replaceChildren(h("option", { value: "", text: label }),
        ...options.map(([id, text]) => h("option", { value: id, text })));
    select.value = value;
}

function renderChips() {
    if (!ui.chips) return;
    ui.chips.hidden = !profile || (!profile.degrees.length && !profile.programmes.length);
    if (!profile) return;
    const choices = chipChoices(profile, chips);
    fill(selects.degree, TEXT.chipDegree, choices.degrees.map(id => [id, DEGREE_NAMES[id] || id]), chips.degree);
    fill(selects.programme, TEXT.chipProgramme, choices.programmes.map(p => [p.id, p.label]), chips.programme);
    fill(selects.po, TEXT.chipPo, choices.pos.map(year => [String(year), String(year)]), chips.po);
}

// Puts the (still hidden) selectors above the composer and loads the profile that fills them.
export function startChips() {
    const make = (name, label) => {
        selects[name] = h("select", { class: "chip", "aria-label": label, onchange: onPick });
        return selects[name];
    };
    ui.chips = h("div", { class: "chips", role: "group", "aria-label": TEXT.chipsLabel, hidden: true },
        make("degree", TEXT.chipDegree), make("programme", TEXT.chipProgramme), make("po", TEXT.chipPo));
    ui.composerWrap.insertBefore(ui.chips, ui.composerWrap.firstChild);
    loadProfile();
}
