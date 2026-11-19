// The stream contract with the backend: the page's frame parser reads every recorded stream exactly as the evals runner's reader
// does (tests/contracts/test_transcripts.py makes the expected frames), wherever the network cuts it, and the page's question
// limit is the backend's.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { QUESTION_LIMIT } from "../config.js";

const path = rel => fileURLToPath(new URL(rel, import.meta.url));
const fixture = name => JSON.parse(readFileSync(path("./fixtures/" + name), "utf8"));
const expected = fixture("expected_frames.json");

for (const seed of [1, 2, 3]) {
    test(`the page parses every recorded stream like the evals runner (cuts of seed ${seed})`, () => {
        const out = execFileSync(process.execPath, [path("./frames_cli.mjs"), path("../../../tests/contracts/fixtures/transcripts_v1"), String(seed)],
            { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
        const frames = JSON.parse(out);
        assert.ok(Object.keys(frames).length >= 30);
        assert.deepEqual(frames, expected);
    });
}

test("the page's question limit is the backend's", () => {
    assert.equal(QUESTION_LIMIT, fixture("contract_constants.json").QUESTION_LIMIT);
});
