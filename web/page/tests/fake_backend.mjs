// A fake backend for checking the page in a browser (response formats v1). Run from the
// web/page:   node tests/fake_backend.mjs   then open http://localhost:8765/index.html
// Question words pick the shape: "tabelle", "zahlen", "ablauf"/"wie beantrage", "mindmap"; "ich …" gets a checklist; anything
// else plain text. The "⋯" menu's variants and version switches answer like the backend (shape, chunk, citations, variant).
// --v1: stream format 1: "\n\n" after each frame, meta first (with the request's limits and the model), done last, code on a
// refused switch. GET /api/profile answers the recorded profile (fixtures/profile.json) for the chips and the footer.
// FAKE_PORT picks the port (default 8765).
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";

const V1 = process.argv.includes("--v1");
const PORT = Number(process.env.FAKE_PORT || 8765);

const DIR = new URL("../", import.meta.url);
const PROFILE = JSON.parse(readFileSync(new URL("fixtures/profile.json", import.meta.url), "utf8"));
const DOC = "https://docs.example.org/handbook/agreements/working-hours/?layout=file";
const ANSWERS = {
    text: "Die Kernarbeitszeit endet um 15:00 Uhr [1].",
    table: "Die Regelungen unterscheiden sich [1].\n\n| Gruppe | Urlaub | Übertrag bis |\n| --- | --- | --- |\n" +
        "| Beamte | 30 Tage [1] | 30. April [2] |\n| Tarifbeschäftigte | 30 Tage [2] | 31. Mai [2] |",
    numbers: "Die Studierendenzahl ist gestiegen [1].\n\n| Jahr | Studierende |\n| --- | --- |\n| 2019 | 81.352 [1] |\n" +
        "| 2020 | 82.019 [1] |\n| 2021 | 82.746 [2] |\n| 2022 | 83.590 [2] |",
    // a loose list (blank lines between items): showdown wraps each item in <p>
    checklist: "Vor der Elternzeit ist einiges zu erledigen [1].\n\n- [ ] Elternzeit beantragen – 7 Wochen vorher – Personalabteilung [1]\n\n" +
        "- [ ] Elterngeld beantragen – Elterngeldstelle [2]\n\n- [ ] Arbeitgeberbescheinigung anfordern – Personalabteilung [2]",
    steps: "So beantragen Sie eine Dienstreise [1].\n\n1. Antrag im Formular ausfüllen [1]\n2. Genehmigung der Abteilungsleitung " +
        "einholen [2]\n3. Reise nach der Rückkehr abrechnen – Kasse [2]",
    mindmap: "Ein Überblick über die Arbeitszeit [1].\n\n```mindmap\n# Arbeitszeit\n## Gleitzeit\n- Kernzeit 9 bis 15 Uhr [1]\n" +
        "- Rahmenzeit 6 bis 20 Uhr [1]\n## Mobile Arbeit\n- Dienstvereinbarung 2024 [2]\n- Antrag bei der Personalabteilung [2]\n" +
        "## Pausen\n- 30 Minuten ab 6 Stunden [1]\n```",
    shorter: "Kernzeit bis 15 Uhr [1].",
    longer: "Die Kernarbeitszeit endet um 15:00 Uhr [1]. Davor und danach gilt die Rahmenzeit von 6 bis 20 Uhr [1]. " +
        "Mobile Arbeit regelt die Dienstvereinbarung von 2024 [2].",
    simple: "Sie arbeiten in der Kernzeit bis 15 Uhr [1]. Danach können Sie gehen [1].",
    english: "Core working time ends at 3 p.m. (Kernarbeitszeit) [1].",
};
const CITATIONS = [
    { n: 1, url: DOC, page: 2, title: "Dienstvereinbarung Arbeitszeit", type: "Dienstvereinbarung", origin: "Handbook", date: "2024", quote: "Kernzeit 9 bis 15 Uhr" },
    { n: 2, url: DOC, page: 5, title: "Dienstvereinbarung Arbeitszeit", type: "Dienstvereinbarung", origin: "Handbook", date: "2024", quote: "Übertrag" },
];
const DOCS = [{ page_content: "x", metadata: { download_url: DOC, page_number: 2, title: "Dienstvereinbarung Arbeitszeit" } }];
const made = new Map();  // conversation id -> version names of its last answer
const originalShape = new Map();  // conversation id -> the shape of its last question's answer

function shapeFor(question) {
    const q = question.toLowerCase();
    if (q.includes("tabelle")) return "table";
    if (q.includes("zahlen")) return "numbers";
    if (q.includes("ablauf") || q.startsWith("wie beantrage")) return "steps";
    if (q.includes("mindmap")) return "mindmap";
    if (q.startsWith("ich ")) return "checklist";
    return "text";
}

const frame = obj => "data: " + JSON.stringify(obj) + (V1 ? "\n\n" : "");
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function open(res) {
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
    if (V1) {
        const { model, provider } = PROFILE.answer;
        res.write(frame({ type: "meta", content: { format: 1, turn_id: randomUUID(), trace_id: null, deadline_s: 240, idle_s: 90, model, provider } }));
    }
}

function close(res, outcome) {
    if (V1) res.write(frame({ type: "done", content: { outcome } }));
    res.end();
}

async function stream(res, before, text, shape, after) {
    open(res);
    before.forEach(f => res.write(frame(f)));
    res.write(frame({ type: "shape", content: shape === "numbers" ? "table" : shape }));
    for (const piece of text.match(/[\s\S]{1,12}/g)) {
        res.write(frame({ type: "chunk", content: piece }));
        await sleep(20);
    }
    res.write(frame({ type: "citations", content: CITATIONS }));
    after.forEach(f => res.write(frame(f)));
    close(res, "completed");
}

createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/api/chat") {
        const p = url.searchParams;
        const user = p.get("conversation_id");
        const select = p.get("variant_select");
        if (select) {
            const ok = (made.get(user) || []).includes(select);
            open(res);
            res.write(frame(Object.assign({ type: "variant", content: { name: select, ok } }, ok || !V1 ? {} : { code: "no_version" })));
            close(res, ok ? "completed" : "refused");
            return;
        }
        const variant = p.get("variant");
        if (variant) {
            made.set(user, (made.get(user) || ["original"]).concat(variant));
            // like the backend: a length or language variant keeps the original's form
            const shape = ["table", "checklist", "steps", "mindmap"].includes(variant) ? variant
                : (originalShape.get(user) || "text");
            await stream(res, [], ANSWERS[variant] || ANSWERS.text, shape, [{ type: "variant", content: { name: variant, ok: true } }]);
            return;
        }
        made.set(user, ["original"]);
        originalShape.set(user, shapeFor(p.get("question") || "") === "numbers" ? "table" : shapeFor(p.get("question") || ""));
        const shape = shapeFor(p.get("question") || "");
        await stream(res, [{ type: "docs", content: DOCS }], ANSWERS[shape], shape, []);
        return;
    }
    if (url.pathname === "/api/profile") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(PROFILE));
        return;
    }
    const name = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    const types = { html: "text/html; charset=utf-8", css: "text/css", js: "text/javascript" };
    try {
        const body = readFileSync(new URL(name, DIR));
        res.writeHead(200, { "Content-Type": types[name.split(".").pop()] || "application/octet-stream" });
        res.end(body);
    } catch (error) {
        res.writeHead(404);
        res.end("not found");
    }
}).listen(PORT, "127.0.0.1", () => console.log(`fake backend: http://localhost:${PORT}/index.html`));
