import { FRAME_MARK } from "../config.js";

// ------------------------------------------------------ pure helpers (unit-tested)

// The backend writes frames as 'data: {json}' without terminators; the Node proxy's failure frame is
// 'event: error\ndata: {json}\n\n'. A frame ends where its JSON parses; the marker text inside an
// answer is kept by extending the piece to the next marker until it parses.
export function createFrameParser() {
    let buffer = "";
    function drain(final) {
        const frames = [];
        let pos = 0;
        for (;;) {
            const start = buffer.indexOf(FRAME_MARK, pos);
            if (start < 0) {
                // keep a marker that may be cut off at the end of this read
                pos = final ? buffer.length : Math.max(pos, buffer.length - FRAME_MARK.length + 1);
                break;
            }
            const bodyStart = start + FRAME_MARK.length;
            let end = buffer.indexOf(FRAME_MARK, bodyStart);
            let frame = null;
            let next = -1;
            for (;;) {
                const stop = end < 0 ? buffer.length : end;
                const piece = buffer.slice(bodyStart, stop).trim();
                if (piece.charAt(0) !== "{") {
                    // empty or not JSON (e.g. "[DONE]"): skip it once the next marker is there
                    next = end >= 0 ? end : (final ? buffer.length : -1);
                    break;
                }
                try {
                    frame = JSON.parse(piece);
                    next = stop;
                    break;
                } catch (error) {
                    // incomplete, or the marker is part of the content: try up to the next marker
                }
                if (end < 0) break;
                end = buffer.indexOf(FRAME_MARK, end + FRAME_MARK.length);
            }
            if (frame !== null) {
                frames.push(frame);
                pos = next;
            } else if (next >= 0) {
                pos = next;
            } else {
                pos = final ? buffer.length : start;
                break;
            }
        }
        buffer = buffer.slice(pos);
        return frames;
    }
    return {
        push(text) {
            buffer += text;
            return drain(false);
        },
        end() {
            return drain(true);
        },
    };
}
