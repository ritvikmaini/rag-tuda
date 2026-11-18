// The browser runs a vendored library only when its bytes match the sha384 the page pins (SRI): an edit or a line-ending
// conversion would silently break markdown, charts and mind maps.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { LIBRARIES } from "../config.js";

const V2 = new URL("../", import.meta.url);
const sri = file => "sha384-" + createHash("sha384").update(readFileSync(new URL(file, V2))).digest("base64");
const html = readFileSync(new URL("index.html", V2), "utf8");
const pinned = [...html.matchAll(/src="(vendor\/[^"]+)" integrity="([^"]+)"/g)].map(m => [m[1], m[2]])
    .concat(Object.values(LIBRARIES).map(lib => [lib.src, lib.integrity]));

test("every vendored library matches the sha384 the page pins", () => {
    assert.equal(pinned.length, 5);
    for (const [file, integrity] of pinned) assert.equal(sri(file), integrity, file);
});

test("vendor.json records every vendored library with the sha384 the page pins", () => {
    const recorded = JSON.parse(readFileSync(new URL("../vendor.json", import.meta.url), "utf8")).libraries
        .map(lib => ["vendor/" + lib.file, lib.integrity]).sort();
    assert.deepEqual(recorded, [...pinned].sort());
});
