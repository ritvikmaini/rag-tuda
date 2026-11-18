// Feeds recorded streams to the v2 page's own frame parser (the stream contract test):
//   node tests/frames_cli.mjs <folder> <seed>
// prints {"<name>.sse": [frames]} for every transcript, each read in pieces cut at seeded random byte positions.
import { readFileSync, readdirSync } from "node:fs";
import { createFrameParser } from "../api/frames.js";

function random(seed) { // mulberry32: the same cuts for the same seed on every machine
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const [folder, seed] = process.argv.slice(2);
const next = random(Number(seed));
const out = {};
for (const name of readdirSync(folder).filter(n => n.endsWith(".sse")).sort()) {
    const bytes = readFileSync(`${folder}/${name}`);
    const parser = createFrameParser();
    const decoder = new TextDecoder("utf-8");
    let frames = [];
    for (let at = 0; at < bytes.length;) {
        const cut = Math.min(bytes.length, at + 1 + Math.floor(next() * 64));
        frames = frames.concat(parser.push(decoder.decode(bytes.subarray(at, cut), { stream: true })));
        at = cut;
    }
    frames = frames.concat(parser.push(decoder.decode())).concat(parser.end());
    out[name] = JSON.parse(JSON.stringify(frames));
}
process.stdout.write(JSON.stringify(out));
