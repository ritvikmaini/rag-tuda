import { TOKEN_KEY } from "../config.js";

// ---- the Bearer token
// `rag up` prints a link ending in #<token>. A fragment never reaches a server: the page keeps it in sessionStorage (for this
// tab and its reloads), removes it from the address bar and sends it as `Authorization: Bearer` on every /api/ request.

let token = "";

// win: the window (a test passes a stand-in). Returns the token the tab holds, "" when it has none.
export function takeToken(win) {
    try {
        token = win.sessionStorage.getItem(TOKEN_KEY) || "";
    } catch (error) {
        token = "";  // storage blocked: the token lasts until the page reloads
    }
    const fromLink = win.location.hash.replace(/^#/, "");
    if (fromLink) {
        token = fromLink;
        try {
            win.sessionStorage.setItem(TOKEN_KEY, token);
        } catch (error) {
            // blocked: kept in memory only
        }
        win.history.replaceState(null, "", win.location.pathname + win.location.search);
    }
    return token;
}

// A link opened again in the tab that already holds the page changes only the fragment: the browser does not reload, so the
// page takes the new token here. It reloads (the profile it asked for before had no token), unless the token could not be stored
// (blocked storage: a reload would lose it), then `again` asks for what failed once more.
export function followNewToken(win, again) {
    win.addEventListener("hashchange", () => {
        if (win.location.hash.length <= 1) return;
        const taken = takeToken(win);
        let stored = "";
        try {
            stored = win.sessionStorage.getItem(TOKEN_KEY) || "";
        } catch (error) {
            // blocked: kept in memory only
        }
        if (stored === taken) win.location.reload();
        else again();
    });
}

// The init object of a fetch to /api/: the token when the tab has one, plus the caller's options.
export function withToken(options) {
    return token ? Object.assign({}, options, { headers: { Authorization: "Bearer " + token } }) : options;
}
