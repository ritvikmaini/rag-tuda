// ------------------------------------------------------------------ page

export const ICON_PATHS = {
    check: '<path d="M5 12.5l4.2 4.2L19 7"/>',
    dash: '<path d="M7 12h10"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M15 9V6.5A2.5 2.5 0 0 0 12.5 4h-6A2.5 2.5 0 0 0 4 6.5v6A2.5 2.5 0 0 0 6.5 15H9"/>',
    retry: '<path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v5h-5"/>',
    edit: '<path d="M4 20h4.5L19 9.5 14.5 5 4 15.5V20z"/><path d="M13 6.5l4.5 4.5"/>',
    chevron: '<path d="M9 6l6 6-6 6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    external: '<path d="M14 5h5v5M19 5l-8 8M18 14v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h4"/>',
    mark: '<path d="M5 5.5h14a1.5 1.5 0 0 1 1.5 1.5v8.5a1.5 1.5 0 0 1-1.5 1.5h-7l-4.5 3.5V17H5a1.5 1.5 0 0 1-1.5-1.5V7A1.5 1.5 0 0 1 5 5.5z"/><path d="M8 10h8M8 13h5"/>',
    more: '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
};
export const DOTS_HTML = '<span class="dots"><i></i><i></i><i></i></span>';

export function svg(name, size) {
    const s = size || 16;
    return '<svg viewBox="0 0 24 24" width="' + s + '" height="' + s + '" fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' + ICON_PATHS[name] + "</svg>";
}

// Shows a notice under a status role: the notice appears first and its text follows, so a screen reader announces it.
export function showNotice(box, textEl, message) {
    box.hidden = !message;
    textEl.textContent = message;
}

// Small element builder; "html" is only ever used with the constant icons above.
export function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    Object.keys(attrs || {}).forEach(key => {
        const value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === "class") el.className = value;
        else if (key === "text") el.textContent = value;
        else if (key === "html") el.innerHTML = value;
        else if (key.slice(0, 2) === "on") el.addEventListener(key.slice(2), value);
        else el.setAttribute(key, value === true ? "" : String(value));
    });
    children.forEach(child => {
        if (child === null || child === undefined || child === false) return;
        el.append(child.nodeType ? child : String(child));
    });
    return el;
}

export function icon(name, extraClass) {
    return h("span", { class: "icon" + (extraClass ? " " + extraClass : ""), "aria-hidden": "true", html: svg(name) });
}

export function prefersReducedMotion() {
    return !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
