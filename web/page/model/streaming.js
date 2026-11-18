// ---- calm streaming: what part of the received answer is shown, and how

// Closes formatting that is still open in a streaming prefix, so raw markdown never flashes: an open code fence is
// closed, a table row that is still arriving is held back until its line is complete, a table until its separator row
// is complete, a half-arrived "*" or "`" and a line that is only a block marker are held back, an odd "**" or "`" is closed.
export function healMarkdown(text) {
    let out = String(text || "");
    if ((out.match(/```/g) || []).length % 2 === 1) return out + (out.endsWith("\n") ? "" : "\n") + "```";
    const lastBreak = out.lastIndexOf("\n");
    if (out.slice(lastBreak + 1).trim().charAt(0) === "|") out = out.slice(0, lastBreak + 1);
    out = holdBackTable(out);
    out = out.replace(/[*`]+$/, "").replace(/[ \t]+$/, "");  // a marker whose second half is still on its way
    // a last line that is only a block marker ("-", "1.", "##", "---") would briefly turn the line above into a heading
    out = out.replace(/(^|\n)[ \t]*([-*+]|\d+[.)]|#{1,6}|=+|-+)[ \t]*$/, "$1");
    const outside = out.replace(/```[\s\S]*?```/g, "");
    if ((outside.match(/\*\*/g) || []).length % 2 === 1) out = out.replace(/[ \t]+$/, "") + "**";
    if ((outside.replace(/\*\*/g, "").match(/`/g) || []).length % 2 === 1) out += "`";
    return out;
}

// A table at the end is shown only once its separator row is complete (before that, its rows would flash as text).
export function holdBackTable(text) {
    const lines = text.split("\n");
    let last = lines.length - 1;
    if (lines[last] === "") last--;
    let first = last;
    while (first >= 0 && lines[first].trim().charAt(0) === "|") first--;
    first++;
    if (first > last) return text;
    if (last > first && /^\s*\|?\s*:?-{3,}/.test(lines[first + 1])) return text;
    return lines.slice(0, first).join("\n") + (first > 0 ? "\n" : "");
}

// Characters to reveal in one animation frame: the received text is shown at a steady pace that adapts to the
// backlog (a burst drains within about ten frames), instead of jumping in network-sized chunks.
export function revealCount(backlog) {
    if (backlog <= 1) return Math.max(0, backlog);
    return Math.max(2, Math.ceil(backlog / 10));
}

// A cut position that never splits an emoji or other surrogate pair.
export function cutAt(text, n) {
    const end = Math.min(Math.max(0, n), text.length);
    const code = text.charCodeAt(end - 1);
    return end > 0 && end < text.length && code >= 0xD800 && code <= 0xDBFF ? end + 1 : end;
}
