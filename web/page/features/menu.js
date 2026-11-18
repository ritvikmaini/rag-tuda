import { printTurn, sendMail } from "./export.js";
import { selectVersion, startVariant } from "./variants.js";
import { MENU_EDGE_PX, menuEntries, menuNote, menuPlacement } from "../model/formats.js";
import { h } from "../render/dom.js";
import { state, store, ui, views } from "../ui/state.js";

// ---- response formats v1: the "⋯" menu, variants of the last answer, version tabs, e-mail and print

export function toggleMenu(view, turn) {
    if (!view.menu.hidden) {
        closeMenu(view, true);
        return;
    }
    const isLast = store.turns.indexOf(turn) === store.turns.length - 1;
    const groups = menuEntries(turn, isLast);
    if (!groups.length) return;
    const note = menuNote(turn, isLast);
    // the note is announced with the menu (an element inside role="menu" that is no menu item is skipped by screen readers)
    const noteId = "menu-note-" + views.indexOf(view);
    if (note) view.menu.setAttribute("aria-describedby", noteId);
    else view.menu.removeAttribute("aria-describedby");
    view.menu.replaceChildren(...(note ? [h("p", { class: "menu-note", id: noteId, "aria-hidden": "true", text: note })] : []),
        ...groups.map(group => h("div", { class: "menu-group", role: "group", "aria-label": group.label },
        h("div", { class: "menu-label", "aria-hidden": "true", text: group.label }),
        ...group.items.map(item => h("button", { type: "button", class: "menu-item", role: "menuitem", tabindex: "-1",
            disabled: !item.enabled || (state.live !== null && group.kind === "model"), text: item.label, onclick: () => chooseMenu(view, turn, item.name) })))));
    view.menu.hidden = false;
    placeMenu(view);
    view.menuBtn.setAttribute("aria-expanded", "true");
    const first = view.menu.querySelector(".menu-item:not([disabled])");
    if (first) first.focus();
    view.closeOutside = event => {
        if (!view.menu.contains(event.target) && !view.menuBtn.contains(event.target)) closeMenu(view, false);
    };
    // a new window width (or the phone turned) closes the menu; a height change only (a phone's keyboard closing as the
    // focus moves into the menu) measures its place again
    const width = window.innerWidth;
    view.closeOnResize = () => (window.innerWidth !== width ? closeMenu(view, false) : placeMenu(view));
    window.addEventListener("resize", view.closeOnResize);
    window.setTimeout(() => { if (view.closeOutside) document.addEventListener("pointerdown", view.closeOutside); }, 0);
}

export function closeMenu(view, focusButton) {
    if (view.menu.hidden) return;
    view.menu.hidden = true;
    view.menuBtn.setAttribute("aria-expanded", "false");
    if (view.closeOutside) document.removeEventListener("pointerdown", view.closeOutside);
    view.closeOutside = null;
    if (view.closeOnResize) window.removeEventListener("resize", view.closeOnResize);
    view.closeOnResize = null;
    if (focusButton) view.menuBtn.focus();
}

export function placeMenu(view) {
    const menu = view.menu;
    menu.classList.remove("up");
    menu.style.maxHeight = "";
    const row = view.actions.getBoundingClientRect();
    const header = document.querySelector(".header");
    const ceiling = header && header.offsetParent !== null ? Math.max(0, header.getBoundingClientRect().bottom) : 0;
    const floor = Math.min(window.innerHeight, ui.composerWrap.getBoundingClientRect().top);
    const where = menuPlacement(menu.offsetHeight, row.top - ceiling - MENU_EDGE_PX, floor - row.bottom - MENU_EDGE_PX);
    menu.classList.toggle("up", where.up);
    if (where.maxHeight !== null) menu.style.maxHeight = where.maxHeight + "px";
}

// The menu-button keyboard pattern: arrows move, Home/End jump, Esc closes and returns focus, Tab leaves.
export function menuKey(view, event) {
    const items = Array.prototype.slice.call(view.menu.querySelectorAll(".menu-item:not([disabled])"));
    const at = items.indexOf(document.activeElement);
    let next = null;
    if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        closeMenu(view, true);
    } else if (event.key === "ArrowDown") {
        next = items[at < 0 ? 0 : (at + 1) % items.length];
    } else if (event.key === "ArrowUp") {
        next = items[at < 0 ? items.length - 1 : (at + items.length - 1) % items.length];
    } else if (event.key === "Home") {
        next = items[0];
    } else if (event.key === "End") {
        next = items[items.length - 1];
    } else if (event.key === "Tab") {
        closeMenu(view, false);
    }
    if (next) {
        event.preventDefault();
        event.stopPropagation();
        next.focus();
    }
}

export function chooseMenu(view, turn, name) {
    closeMenu(view, true);
    if (name === "mail") sendMail(turn);
    else if (name === "print") printTurn(view);
    else if ((turn.versions || []).some(v => v.name === name)) selectVersion(turn, view, name);
    else startVariant(turn, view, name);
}
