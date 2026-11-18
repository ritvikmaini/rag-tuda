import { h } from "./dom.js";
import { ui } from "../ui/state.js";

// ---- citation tooltip: one shared element in the page body (never clipped by a table's scroll box), placed above the
// marker under the pointer or keyboard focus, below it near the top edge, always inside the window

export function showTip(marker) {
    const tip = ui.tip;
    tip.textContent = "";
    tip.append(h("strong", { text: marker.dataset.tipTitle || "" }));
    if (marker.dataset.tipFacts) tip.append(h("span", { class: "tip-facts", text: marker.dataset.tipFacts }));
    if (marker.dataset.tipQuote) tip.append(h("q", { text: marker.dataset.tipQuote }));
    tip.hidden = false;
    placeTip(marker);
    if (ui.tipFor && ui.tipFor !== marker) ui.tipFor.removeAttribute("aria-describedby");
    marker.setAttribute("aria-describedby", "cite-tip");
    ui.tipFor = marker;
}

// Above the marker, below it near the top edge, always inside the window. The tip is measured at the left edge: at its
// previous place the window edge may squeeze it narrower (and taller).
export function placeTip(marker) {
    const tip = ui.tip;
    tip.style.left = "0px";
    tip.style.top = "0px";
    const r = marker.getBoundingClientRect();
    const left = Math.max(16, Math.min(r.left, window.innerWidth - tip.offsetWidth - 16));
    const above = r.top - tip.offsetHeight - 8;
    tip.style.left = left + "px";
    tip.style.top = (above >= 8 ? above : r.bottom + 8) + "px";
}

// A scroll (the page or a table's scroll box) moves the marker: the tip follows it while the marker keeps the keyboard focus
// or the pointer and is still in the window (Tab to a marker below the fold scrolls the page); otherwise it hides.
export function onTipScroll() {
    const marker = ui.tipFor;
    if (!marker || ui.tip.hidden) return;
    const r = marker.getBoundingClientRect();
    const inWindow = marker.isConnected && r.bottom > 0 && r.top < window.innerHeight;
    if (inWindow && (document.activeElement === marker || marker.matches(":hover"))) placeTip(marker);
    else hideTip();
}

export function hideTip() {
    if (!ui.tip || ui.tip.hidden) return;
    ui.tip.hidden = true;
    if (ui.tipFor) ui.tipFor.removeAttribute("aria-describedby");
    ui.tipFor = null;
}
