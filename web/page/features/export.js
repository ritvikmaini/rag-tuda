import { TEXT } from "../config.js";
import { mailPlan } from "../model/formats.js";
import { copyText } from "../model/text.js";
import { h, svg } from "../render/dom.js";
import { displayCards } from "../render/sources.js";

export async function sendMail(turn) {
    const plan = mailPlan(turn.question, turn.answer, displayCards(turn));
    if (plan.clipboard) {
        try {
            await navigator.clipboard.writeText(plan.clipboard);
        } catch (error) {
            legacyCopy(plan.clipboard);
        }
    }
    const link = h("a", { href: plan.href });
    document.body.appendChild(link);
    link.click();
    link.remove();
}

// Prints only this turn: question, date, answer as shown, sources (print stylesheet); the browser offers "Als PDF".
export function printTurn(view) {
    view.root.setAttribute("data-print-date", new Date().toLocaleDateString(TEXT.printLocale));
    view.root.classList.add("print-target");
    document.body.classList.add("print-one");
    const done = () => {
        document.body.classList.remove("print-one");
        view.root.classList.remove("print-target");
        window.removeEventListener("afterprint", done);
    };
    window.addEventListener("afterprint", done);
    window.print();
}

// ---- copy, new conversation, storage

export async function copyTurn(view, turn) {
    const cites = Array.prototype.slice.call(view.body.querySelectorAll(".cite"));
    const labels = cites.map(el => el.textContent);
    const tools = Array.prototype.slice.call(view.body.querySelectorAll(".mindmap-tools, .table-tools"));
    const toolsHidden = tools.map(el => el.hidden);
    cites.forEach(el => { el.textContent = "[" + el.dataset.n + "]"; });
    tools.forEach(el => { el.hidden = true; });
    const text = copyText(view.body.innerText, displayCards(turn));
    tools.forEach((el, i) => { el.hidden = toolsHidden[i]; });
    cites.forEach((el, i) => { el.textContent = labels[i]; });
    let ok = false;
    try {
        await navigator.clipboard.writeText(text);
        ok = true;
    } catch (error) {
        ok = legacyCopy(text);
    }
    if (!ok) return;
    view.copyLabel.textContent = TEXT.copied;
    view.copyIcon.innerHTML = svg("check");
    window.clearTimeout(view.copyTimer);
    view.copyTimer = window.setTimeout(() => {
        view.copyLabel.textContent = TEXT.copy;
        view.copyIcon.innerHTML = svg("copy");
    }, 2000);
}

export function legacyCopy(text) {
    const area = h("textarea", { readonly: true, "aria-hidden": "true" });
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
        ok = document.execCommand("copy");
    } catch (error) {
        ok = false;
    }
    area.remove();
    return ok;
}
