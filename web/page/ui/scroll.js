import { NEAR_END_PX, RESUME_THRESHOLD_PX } from "../config.js";
import { prefersReducedMotion } from "../render/dom.js";
import { state, ui } from "./state.js";

// ---- scrolling

export function atBottom(threshold) {
    return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - threshold;
}

export function scrollToEnd(smooth) {
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: smooth && !prefersReducedMotion() ? "smooth" : "auto" });
}

export function keepFollowing() {
    // while a smooth scroll to the end runs (new question, jump button) the frames do not cut it short
    if (state.follow && Date.now() >= state.ignoreScrollUntil) scrollToEnd(false);
    updateJump();
}

export function userScrolled() {
    if (state.live === null) return;
    state.follow = false;
    state.ignoreScrollUntil = 0;
    updateJump();
}

// A downward scroll by the reader near the end resumes following.
export function readerNearEnd() {
    if (state.live === null || state.follow || !atBottom(NEAR_END_PX)) return;
    state.follow = true;
    updateJump();
}

// Following only resumes when the reader scrolls down to the very end (or uses the jump button); an upward scroll
// never re-enables it, so a smooth wheel scroll cannot be pulled back.
export function onScroll() {
    const movedDown = window.scrollY > state.lastScrollY;
    state.lastScrollY = window.scrollY;
    if (Date.now() < state.ignoreScrollUntil) return;
    if (state.follow && state.live !== null && !movedDown && !atBottom(2)) {
        // the page itself only ever scrolls down to the end while an answer streams, so an upward move away from the
        // end comes from the reader (find in page, a focused link, autoscroll, a screen reader)
        state.follow = false;
    } else if (!state.follow && movedDown && atBottom(RESUME_THRESHOLD_PX)) {
        state.follow = true;
    }
    updateJump();
}

export function updateJump() {
    const show = state.live !== null && !state.follow;
    ui.jump.classList.toggle("show", show);
    ui.jump.setAttribute("aria-hidden", show ? "false" : "true");
    ui.jump.tabIndex = show ? 0 : -1;
    if (show) ui.jump.style.bottom = ui.composerWrap.offsetHeight + 12 + "px";
}
