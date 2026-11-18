import { TEXT } from "../config.js";
import { loadLibrary } from "./libraries.js";
import { chartConfig, chartSpec } from "../model/formats.js";
import { morphChildren } from "../render/answer.js";
import { h } from "../render/dom.js";

// A table with a fully numeric column gets a switch between the table and a chart of exactly those numbers. Runs
// before the [n] markers are made, so the cells are still plain text.
export function markCharts(body) {
    body.querySelectorAll(".table-wrap > table").forEach((table, i) => {
        const header = Array.prototype.map.call(table.querySelectorAll("thead th"), th => th.textContent);
        const rows = Array.prototype.map.call(table.querySelectorAll("tbody tr"),
            tr => Array.prototype.map.call(tr.children, cell => cell.textContent));
        const spec = chartSpec(header, rows);
        if (!spec) return;
        const wrap = table.parentElement;
        const block = h("div", { class: "table-block" });
        const box = h("div", { class: "chart-box", "data-keep": "chart:" + i + ":" + JSON.stringify(spec),
            "data-spec": JSON.stringify(spec) },
            h("canvas", { role: "img", "aria-label": TEXT.chartLabel }));
        const button = h("button", { type: "button", class: "act chart-toggle", "aria-pressed": "false", text: TEXT.showChart });
        // found when clicked: the final render merges the block into the streaming table-wrap element (morphChildren)
        button.addEventListener("click", event => {
            const live = event.currentTarget.closest(".table-block");
            if (!live) return;
            const liveBox = live.querySelector(".chart-box");
            toggleChart(event.currentTarget, live, liveBox, JSON.parse(liveBox.getAttribute("data-spec")));
        });
        wrap.replaceWith(block);
        block.append(h("div", { class: "table-tools" }, button), wrap, box);
    });
}

export async function toggleChart(button, block, box, spec) {
    const show = !block.classList.contains("show-chart");
    if (show) {
        try {
            await loadLibrary("chart");
        } catch (error) {
            console.warn("[Chart] not available:", error);
            button.hidden = true;
            return;
        }
        if (box.chart && box.chartKey !== box.getAttribute("data-keep")) {
            box.chart.destroy();
            box.chart = null;
        }
        if (!box.chart) box.chart = new window.Chart(box.querySelector("canvas"), chartConfig(spec));
        box.chartKey = box.getAttribute("data-keep");
    }
    block.classList.toggle("show-chart", show);
    button.setAttribute("aria-pressed", show ? "true" : "false");
    button.textContent = show ? TEXT.showTable : TEXT.showChart;
}
