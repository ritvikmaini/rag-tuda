import { API_PATH, FORMATS_VERSION } from "../config.js";

// ---- request, copy, contacts, ids

// chips: { degree, programme, po } as text; an empty one is left out (the backend then applies no chip value for it).
// lang: the page language, "de" or "en" (lower case); it is left out for anything else (the backend then reads the question).
export function questionUrl(question, id, turnId, replaceFrom, kept, chips, lang) {
    let url = API_PATH + "?question=" + encodeURIComponent(question) +
        "&conversation_id=" + encodeURIComponent(id) + "&formats=" + FORMATS_VERSION;
    if (turnId) url += "&turn_id=" + encodeURIComponent(turnId);
    if (replaceFrom) {
        url += "&replace_from=" + encodeURIComponent(replaceFrom);
        if (Array.isArray(kept)) url += "&kept=" + encodeURIComponent(kept.join(","));
    }
    for (const name of ["degree", "programme", "po"]) {
        if (chips && chips[name]) url += "&" + name + "=" + encodeURIComponent(chips[name]);
    }
    if (lang === "de" || lang === "en") url += "&lang=" + lang;
    return url;
}

// A variant of the last answer (select: false) or a switch to one of its stored versions (select: true).
// lang: as for questionUrl (the backend words its busy and error texts in it).
export function variantUrl(question, id, variantOf, name, select, lang) {
    return API_PATH + "?question=" + encodeURIComponent(question) + "&conversation_id=" + encodeURIComponent(id) +
        "&formats=" + FORMATS_VERSION + "&variant_of=" + encodeURIComponent(variantOf) +
        (select ? "&variant_select=" : "&variant=") + encodeURIComponent(name) + (lang === "de" || lang === "en" ? "&lang=" + lang : "");
}
