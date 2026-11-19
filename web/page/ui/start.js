import { takeToken } from "../api/auth.js";
import { ASSISTANT_NAME, TEXT } from "../config.js";
import { toggleTick } from "../features/tasks.js";
import { LANG_KEY, applyStatic, detectLang, getLang, setLang } from "../i18n.js";
import { h } from "../render/dom.js";
import { hideTip, onTipScroll, showTip } from "../render/tooltip.js";
import { stopLive } from "./ask.js";
import { startChips } from "./chips.js";
import { autosize, cancelEdit, onKeyDown, onSubmit, updateSend } from "./composer.js";
import { onScroll, readerNearEnd, scrollToEnd, updateJump, userScrolled } from "./scroll.js";
import { state, ui } from "./state.js";
import { newConversation, restore } from "./storage.js";

// ---- language: the stored choice, else the browser's; the switch stores the other one and reloads the page (the
// conversation is kept in sessionStorage and comes back in the new language's page)

function storedLang() {
    try {
        return window.localStorage.getItem(LANG_KEY);
    } catch (error) {
        return null;
    }
}

function switchLang() {
    try {
        window.localStorage.setItem(LANG_KEY, getLang() === "de" ? "en" : "de");
    } catch (error) {
        return;  // blocked storage: the page stays in the language it has
    }
    window.location.reload();
}

// ---- start

export function init() {
    takeToken(window);
    setLang(detectLang(storedLang(), window.navigator.language));
    document.documentElement.lang = getLang();
    applyStatic(document, { name: ASSISTANT_NAME });
    ui.provider = document.getElementById("provider-line");
    const switcher = document.getElementById("lang-switch");
    switcher.lang = getLang() === "de" ? "en" : "de";  // its label is in the other language
    switcher.addEventListener("click", switchLang);
    ui.conversation = document.getElementById("conversation");
    ui.form = document.getElementById("composer");
    ui.question = document.getElementById("question");
    ui.send = document.getElementById("send");
    ui.newChat = document.getElementById("new-chat");
    ui.jump = document.getElementById("jump");
    ui.composerWrap = document.getElementById("composer-wrap");
    ui.editText = h("span", { class: "edit-text" });
    ui.editCancel = h("button", { type: "button", class: "edit-cancel", text: TEXT.cancelEdit, onclick: () => cancelEdit(true) });
    ui.editBar = h("div", { class: "edit-bar", hidden: true, role: "status" }, ui.editText, ui.editCancel);
    ui.composerWrap.insertBefore(ui.editBar, ui.form);
    ui.limitText = h("span", { class: "limit-text" });
    ui.limitCount = h("span", { class: "limit-count" });
    ui.limitBar = h("div", { class: "limit-bar", hidden: true, "aria-hidden": "true" }, ui.limitText, ui.limitCount);
    ui.composerWrap.insertBefore(ui.limitBar, ui.form);
    // screen readers hear the hint once per change of state (normal, long, too long), never per keystroke
    ui.limitLive = h("span", { class: "sr-only", id: "limit-live", role: "status" });
    ui.composerWrap.appendChild(ui.limitLive);
    ui.question.setAttribute("aria-describedby", "limit-live");

    startChips();

    document.title = ASSISTANT_NAME;
    if (new URLSearchParams(window.location.search).get("m") === "nh") document.body.classList.add("no-header");
    if (window.showdown) {
        state.converter = new window.showdown.Converter({
            tables: true,
            strikethrough: true,
            simplifiedAutoLink: true,
            literalMidWordUnderscores: true,
            openLinksInNewWindow: true,
            noHeaderId: true,
            ghCodeBlocks: true,
            requireSpaceBeforeHeadingText: true,
        });
    }

    ui.form.addEventListener("submit", onSubmit);
    ui.question.addEventListener("input", () => {
        autosize();
        updateSend();
    });
    ui.question.addEventListener("keydown", onKeyDown);
    ui.tip = h("div", { class: "cite-tip", id: "cite-tip", role: "tooltip", hidden: true });
    document.body.appendChild(ui.tip);
    ui.conversation.addEventListener("mouseover", event => {
        const marker = event.target.closest ? event.target.closest("a.cite") : null;
        if (marker && marker !== ui.tipFor) showTip(marker);
    });
    ui.conversation.addEventListener("mouseout", event => {
        const marker = event.target.closest ? event.target.closest("a.cite") : null;
        if (marker && !marker.contains(event.relatedTarget)) hideTip();
    });
    ui.conversation.addEventListener("focusin", event => {
        if (event.target.matches && event.target.matches("a.cite")) showTip(event.target);
    });
    ui.conversation.addEventListener("focusout", hideTip);
    ui.conversation.addEventListener("click", event => {
        const tick = event.target.closest ? event.target.closest(".tick") : null;
        if (tick) toggleTick(tick);
    });
    ui.conversation.addEventListener("keydown", event => {
        if ((event.key === " " || event.key === "Enter") && event.target.classList && event.target.classList.contains("tick")) {
            event.preventDefault();
            toggleTick(event.target);
        }
    });
    document.addEventListener("scroll", onTipScroll, { capture: true, passive: true });
    document.addEventListener("keydown", event => {
        if (event.key === "Escape") hideTip();
        if (event.key === "Escape" && state.editing && !state.live) cancelEdit(true);
        if (event.key === "Escape" && state.live) {
            event.preventDefault();
            stopLive();
        }
    });
    ui.newChat.addEventListener("click", newConversation);
    ui.jump.addEventListener("click", () => {
        state.follow = true;
        state.ignoreScrollUntil = Date.now() + 700;
        scrollToEnd(true);
        updateJump();
    });
    window.addEventListener("scroll", onScroll, { passive: true });
    // Scrolling up by the reader stops following at once and is never overridden (only the jump button shows);
    // scrolling down to the end resumes it.
    window.addEventListener("wheel", event => {
        if (event.ctrlKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;  // zoom, sideways in a wide table
        if (event.deltaY < 0) userScrolled();
        else if (event.deltaY > 0) readerNearEnd();
    }, { passive: true });
    window.addEventListener("touchstart", event => { state.touchY = event.touches[0].clientY; }, { passive: true });
    window.addEventListener("touchmove", event => {
        if (state.touchY === null) return;
        const dy = event.touches[0].clientY - state.touchY;
        if (dy > 4) userScrolled();
        else if (dy < -4) readerNearEnd();
    }, { passive: true });
    document.addEventListener("keydown", event => {
        if (event.target === ui.question) return;
        const up = ["PageUp", "ArrowUp", "Home"].indexOf(event.key) >= 0 || (event.key === " " && event.shiftKey);
        const down = ["PageDown", "ArrowDown", "End"].indexOf(event.key) >= 0 || (event.key === " " && !event.shiftKey);
        if (up) userScrolled();
        else if (down) readerNearEnd();
    });
    document.addEventListener("pointerdown", event => {
        if (event.target === document.documentElement) userScrolled();
    });
    window.addEventListener("resize", autosize);
    restore();
    autosize();
    updateSend();
    if (window.matchMedia && window.matchMedia("(pointer: fine)").matches) ui.question.focus({ preventScroll: true });
}
