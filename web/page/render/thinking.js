import { summaryText } from "../model/thinking.js";
import { DOTS_HTML, h, icon, svg } from "./dom.js";

// ---- thinking

export function renderThinking(view, t) {
    const liveNow = t.endedAt === null;
    view.thinking.hidden = !liveNow && !t.retrieval;
    view.thinking.classList.toggle("is-live", liveNow);
    view.steps.setAttribute("aria-live", liveNow ? "polite" : "off");
    t.steps.forEach((step, i) => {
        let el = view.stepEls[i];
        if (!el) {
            el = h("li", { class: "step" }, h("span", { class: "step-icon" }), h("span", { class: "step-text" }), h("span", { class: "step-time", "aria-hidden": "true" }));
            view.stepEls.push(el);
            view.steps.appendChild(el);
        }
        if (el.dataset.state !== step.state) {
            el.dataset.state = step.state;
            el.className = "step is-" + step.state;
            el.children[0].innerHTML = step.state === "active" ? DOTS_HTML : svg(step.state === "done" ? "check" : "dash", 14);
        }
        if (el.children[1].textContent !== step.text) el.children[1].textContent = step.text;
        if (step.state !== "active") el.children[2].textContent = "";
    });
    while (view.stepEls.length > t.steps.length) view.stepEls.pop().remove();
    view.toggle.hidden = liveNow || !t.retrieval;
    view.toggleLabel.textContent = liveNow ? "" : summaryText(t);
    if (liveNow) {
        view.collapse.classList.add("open");
    } else if (!view.thinkingToggled) {
        view.collapse.classList.remove("open");
        view.toggle.setAttribute("aria-expanded", "false");
    }
}

export function tickTimer(view, t) {
    const i = t.steps.findIndex(step => step.state === "active");
    const el = i >= 0 ? view.stepEls[i] : null;
    if (!el) return;
    const seconds = Math.floor((Date.now() - t.startedAt) / 1000);
    el.children[2].textContent = seconds >= 1 ? seconds + " s" : "";
}

export function toggleThinking(view) {
    const open = !view.collapse.classList.contains("open");
    view.collapse.classList.toggle("open", open);
    view.toggle.setAttribute("aria-expanded", String(open));
    view.thinkingToggled = true;
}
