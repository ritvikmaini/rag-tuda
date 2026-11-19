// Look and language tests (DE/EN strings, footer, source cards, contact links, the watchdog; native ES modules).
// From web/page: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { SILENCE_LIMIT_MS, TOTAL_LIMIT_MS } from "../config.js";
import { watchdogExpired, watchdogLimits } from "../model/streaming.js";
import { recordStatus } from "../model/turns.js";

const plain = value => JSON.parse(JSON.stringify(value));

// ---- the watchdog

test("the watchdog waits as long as the backend's meta frame says (test_watchdog_uses_meta_deadline)", () => {
    // before a meta frame: the page's own limits
    assert.deepEqual(plain(watchdogLimits(undefined)), { silenceMs: SILENCE_LIMIT_MS, totalMs: TOTAL_LIMIT_MS });
    assert.deepEqual(plain(watchdogLimits({ deadline_s: "x", idle_s: -1 })), { silenceMs: SILENCE_LIMIT_MS, totalMs: TOTAL_LIMIT_MS });
    // a cold model: the card allows 240 s in all and 90 s of silence; the page waits a little longer than the backend
    const meta = watchdogLimits({ deadline_s: 240, idle_s: 90 });
    assert.ok(meta.totalMs > 240000 && meta.totalMs <= 250000);
    assert.ok(meta.silenceMs > 90000 && meta.silenceMs <= 100000);
    const session = Object.assign({ startedAt: 0, lastData: 190000 }, meta);
    // 200 s into the request, the backend spoke 10 s ago: still waiting (the fixed 150 s would have given up)
    assert.equal(watchdogExpired(session, 200000), false);
    assert.equal(watchdogExpired(Object.assign({ startedAt: 0, lastData: 190000 }, watchdogLimits(undefined)), 200000), true);
    // beyond the deadline, or silent longer than idle_s, the connection is treated as gone
    assert.equal(watchdogExpired(Object.assign({}, session, { lastData: 240000 }), 250001), true);
    assert.equal(watchdogExpired(Object.assign({}, session, { lastData: 100000 }), 200000), true);
    assert.equal(watchdogExpired(Object.assign({}, session, { lastData: 100000 }), 190000), false);
});

test("a retry after a network failure forgets the earlier refusal (test_retry_clears_stale_failure_flags)", () => {
    const turn = {};
    recordStatus(turn, 422);
    assert.deepEqual([turn.unauthorized, turn.refused], [false, true]);
    recordStatus(turn, 401);
    assert.deepEqual([turn.unauthorized, turn.refused], [true, false]);
    recordStatus(turn, 0);   // the start of the next attempt, before any answer of the backend
    assert.deepEqual([turn.unauthorized, turn.refused], [false, false]);
});
