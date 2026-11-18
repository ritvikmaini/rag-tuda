import { TEXT } from "../config.js";
import { loadLibrary } from "./libraries.js";
import { citationHref, citedNumbers, entriesForTurn } from "../model/citations.js";
import { freeplaneXml, mindmapTree } from "../model/formats.js";
import { readableSlug } from "../model/sources.js";
import { morphChildren } from "../render/answer.js";
import { h, svg } from "../render/dom.js";
import { renderMarkdown } from "../render/markdown.js";
import { store } from "../ui/state.js";

// A ```mindmap block: a placeholder while it streams; once complete, a box drawn after the answer is in the page.
export function markMindmaps(body, turn, final) {
    body.querySelectorAll("pre > code.language-mindmap").forEach(code => {
        const pre = code.parentElement;
        if (!final) {
            pre.replaceWith(h("div", { class: "mindmap is-waiting" }, h("p", { class: "mindmap-wait", text: TEXT.mindmapWait })));
            return;
        }
        pre.replaceWith(mindmapBox(code.textContent, turn));
    });
}

// The tool buttons find their box when clicked: the first final render merges this box into the streaming placeholder
// (morphChildren keeps the placeholder element), so a box captured here would be a detached copy.
export function mindmapBox(src, turn) {
    const box = h("div", { class: "mindmap", "data-src": src, "data-keep": "mm:" + src });
    const own = event => event.currentTarget.closest(".mindmap");
    const tools = h("div", { class: "mindmap-tools" },
        h("button", { type: "button", class: "act mm-full", text: TEXT.mindmapFull, hidden: !document.fullscreenEnabled,
            onclick: event => fullMindmap(own(event)) }),
        h("button", { type: "button", class: "act", text: TEXT.mindmapSvg, onclick: event => saveMindmapSvg(own(event)) }),
        h("button", { type: "button", class: "act", text: TEXT.mindmapMm, onclick: event => saveMindmapMm(own(event)) }));
    const canvas = h("div", { class: "mindmap-canvas" }, h("p", { class: "mindmap-wait", text: TEXT.mindmapWait }));
    const list = h("div", { class: "mindmap-list", hidden: true });
    list.innerHTML = renderMarkdown(src);
    [tools, canvas, list, mindmapSources(src, turn)].filter(Boolean).forEach(el => box.appendChild(el));
    return box;
}

// The numbered sources of the map's [n] under it: the nodes show the numbers as text.
export function mindmapSources(src, turn) {
    const wanted = new Set(citedNumbers(src));
    const items = entriesForTurn(store.turns, store.turns.indexOf(turn)).filter(e => wanted.has(e.n) && e.url).map(e =>
        h("li", null, h("a", { href: citationHref(e.url, e.page), target: "_blank", rel: "noopener noreferrer",
            text: "[" + e.n + "] " + (e.title || readableSlug(e.url)) + (e.page ? ", S. " + e.page : "") })));
    return items.length ? h("ol", { class: "mindmap-sources" }, ...items) : null;
}

export const MINDMAP_NODE_WIDTH_PX = 240;

export async function drawMindmaps(view) {
    const boxes = Array.prototype.slice.call(view.body.querySelectorAll(".mindmap[data-src]:not([data-drawn])"));
    for (const box of boxes) {
        box.setAttribute("data-drawn", "pending");
        try {
            const tree = mindmapTree(box.getAttribute("data-src"));
            if (!tree) throw new Error("no map in the block");
            await loadLibrary("d3");
            await loadLibrary("markmap");
            const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svgEl.setAttribute("class", "mindmap-svg");
            box.querySelector(".mindmap-canvas").replaceChildren(svgEl);
            // long points wrap at MINDMAP_NODE_WIDTH_PX: unwrapped, a wide map is shrunk to the canvas width until its text is
            // too small to read
            box.markmap = window.markmap.Markmap.create(svgEl,
                { autoFit: true, initialExpandLevel: 3, duration: 200, maxWidth: MINDMAP_NODE_WIDTH_PX }, tree);
            box.setAttribute("data-drawn", "map");
        } catch (error) {
            console.warn("[Mindmap] shown as a list:", error);
            box.querySelector(".mindmap-canvas").hidden = true;
            box.querySelector(".mindmap-tools").hidden = true;
            box.querySelector(".mindmap-list").hidden = false;
            box.setAttribute("data-drawn", "list");
        }
    }
}

// The button toggles: in full screen it reads "Vollbild beenden" (Esc works too). Entering and leaving change the canvas
// size, so the map is fitted again both times (else it keeps the full-screen zoom in the small canvas). A frame without
// allowfullscreen refuses: the button is hidden there (mindmapBox).
export function fullMindmap(box) {
    if (document.fullscreenElement === box) {
        document.exitFullscreen().catch(() => {});
        return;
    }
    if (!box.requestFullscreen) return;
    box.onfullscreenchange = () => {
        const button = box.querySelector(".mm-full");
        if (button) button.textContent = document.fullscreenElement === box ? TEXT.mindmapFullExit : TEXT.mindmapFull;
        window.requestAnimationFrame(() => { if (box.markmap) box.markmap.fit(); });
    };
    box.requestFullscreen().catch(() => {});
}

export function download(name, type, text) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = h("a", { href: url, download: name });
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function saveMindmapSvg(box) {
    const svgEl = box.querySelector("svg.mindmap-svg");
    if (svgEl) download("mindmap.svg", "image/svg+xml", new XMLSerializer().serializeToString(svgEl));
}

export function saveMindmapMm(box) {
    const tree = mindmapTree(box.getAttribute("data-src"));
    if (tree) download("mindmap.mm", "application/x-freemind", freeplaneXml(tree));
}
