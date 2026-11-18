import { TEXT } from "../config.js";
import { parseTask, tickKey } from "../model/formats.js";
import { h } from "../render/dom.js";
import { store, views } from "../ui/state.js";
import { save } from "../ui/storage.js";

// Checklist items ("[ ] …", also in a loose list where the text sits in a <p>) get a tick box; ticks are stored with the
// turn per version and item.
export function markTasks(body, turn, final) {
    body.querySelectorAll("li").forEach((li, i) => {
        const holder = li.firstElementChild && li.firstElementChild === li.firstChild && li.firstElementChild.tagName === "P"
            ? li.firstElementChild : li;
        const first = holder.firstChild;
        if (!first || first.nodeType !== 3) return;
        const task = parseTask(first.nodeValue);
        if (!task) return;
        first.nodeValue = task.rest;
        const key = tickKey(turn.active, i);
        const ticks = turn.ticks || {};
        const checked = Object.prototype.hasOwnProperty.call(ticks, key) ? !!ticks[key] : task.checked;
        li.classList.add("task");
        holder.insertBefore(h("span", { class: "tick", role: "checkbox", tabindex: final ? "0" : "-1",
            "aria-checked": checked ? "true" : "false", "aria-label": TEXT.tick, "data-tick": key }), first);
    });
}

export function toggleTick(el) {
    const index = views.findIndex(view => view.body.contains(el));
    const turn = store.turns[index];
    if (!turn || turn.state === "streaming") return;
    const on = el.getAttribute("aria-checked") !== "true";
    turn.ticks = Object.assign({}, turn.ticks, { [el.getAttribute("data-tick")]: on });
    el.setAttribute("aria-checked", on ? "true" : "false");
    save();
}
