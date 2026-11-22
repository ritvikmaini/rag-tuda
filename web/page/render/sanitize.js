import { ANSWER_TAGS } from "../model/text.js";

// Answers are text from a model and from crawled pages: they hold the tags of ANSWER_TAGS and links, and no attribute the page
// itself reads as a mark of its own (class, id, data-*, aria-*, role, tabindex): such text could pose as a citation marker,
// a mind map or a checkbox, or point a tooltip at a forged quote. The page sets its own marks after sanitising.
// target stays for the links; style never.
export const SANITIZE_CONFIG = {
    ALLOWED_TAGS: ANSWER_TAGS,
    ADD_ATTR: ["target"],
    ALLOW_DATA_ATTR: false,
    ALLOW_ARIA_ATTR: false,
    FORBID_ATTR: ["style", "id", "role", "tabindex"],
};

// The one class an answer keeps is the language class of a code block (code.language-mindmap becomes a mind map).
export function dropForeignClass(node, data) {
    if (data.attrName === "class" && node.nodeName !== "CODE") data.keepAttr = false;
}

const hooked = new WeakSet();

export function sanitize(purify, html) {
    if (!hooked.has(purify)) {
        purify.addHook("uponSanitizeAttribute", dropForeignClass);
        hooked.add(purify);
    }
    return purify.sanitize(html, SANITIZE_CONFIG);
}
