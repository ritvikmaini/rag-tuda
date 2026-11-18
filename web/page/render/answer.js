import { drawMindmaps } from "../features/mindmap.js";
import { closeOpenFence, shapeName } from "../model/formats.js";
import { cutAt, healMarkdown, revealCount } from "../model/streaming.js";
import { h, prefersReducedMotion } from "./dom.js";
import { decorateAnswer, renderMarkdown } from "./markdown.js";
import { keepFollowing } from "../ui/scroll.js";

export function lastMeaningful(el) {
    let node = el.lastChild;
    while (node && node.nodeType === 3 && !node.nodeValue.trim()) node = node.previousSibling;
    return node;
}

export function appendCaret(body) {
    let el = body;
    // the caret follows the last text, never inside a block it cannot sit in or a citation marker
    for (let node = lastMeaningful(el); node && node.nodeType === 1 && !/^(TABLE|PRE|DIV|HR|BR|BUTTON|IMG)$/.test(node.tagName) &&
        !node.classList.contains("cite"); node = lastMeaningful(el)) {
        el = node;
    }
    el.appendChild(h("span", { class: "caret", "aria-hidden": "true" }));
}

// The answer is rendered into a detached element and then merged into the page: finished paragraphs keep their
// nodes and are never repainted; only the growing part changes.
export function renderAnswer(view, turn, final) {
    const streaming = !final && turn.state === "streaming";
    const text = streaming ? healMarkdown(turn.answer.slice(0, view.shown)) : closeOpenFence(turn.answer);
    const next = document.createElement("div");
    next.innerHTML = renderMarkdown(text);
    decorateAnswer(next, turn, final);
    if (streaming && text) appendCaret(next);
    morphChildren(view.body, next);
    view.body.classList.toggle("shape-steps", shapeName(turn.shape) === "steps");
    view.card.setAttribute("aria-busy", streaming ? "true" : "false");
    if (final) drawMindmaps(view);
}

export function syncAttributes(current, next) {
    Array.prototype.slice.call(current.attributes).forEach(attr => {
        if (!next.hasAttribute(attr.name)) current.removeAttribute(attr.name);
    });
    Array.prototype.slice.call(next.attributes).forEach(attr => {
        if (current.getAttribute(attr.name) !== attr.value) current.setAttribute(attr.name, attr.value);
    });
}

// Makes target's children equal to source's children, reusing every node whose place and tag are unchanged.
export function morphChildren(target, source) {
    const current = Array.prototype.slice.call(target.childNodes);
    const next = Array.prototype.slice.call(source.childNodes);
    next.forEach((node, i) => {
        const old = current[i];
        if (!old) {
            target.appendChild(node);
        } else if (old.nodeType !== node.nodeType || old.nodeName !== node.nodeName) {
            target.replaceChild(node, old);
        } else if (node.nodeType === 1 && node.hasAttribute("data-keep") && old.getAttribute("data-keep") === node.getAttribute("data-keep")) {
            // a drawn mind map or chart stays as drawn
        } else if (node.nodeType !== 1) {
            if (old.nodeValue !== node.nodeValue) old.nodeValue = node.nodeValue;
        } else {
            syncAttributes(old, node);
            morphChildren(old, node);
        }
    });
    for (let i = next.length; i < current.length; i++) current[i].remove();
}

// Reveals the received text at a steady pace, one animation frame at a time, while the answer streams.
export function pump(turn, view) {
    if (view.pumping) return;
    view.pumping = true;
    const step = () => {
        const backlog = turn.answer.length - view.shown;
        if (turn.state !== "streaming" || backlog <= 0) {
            view.pumping = false;
            return;
        }
        view.shown = cutAt(turn.answer, view.shown + (prefersReducedMotion() ? backlog : revealCount(backlog)));
        renderAnswer(view, turn, false);
        keepFollowing();
        window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
}
