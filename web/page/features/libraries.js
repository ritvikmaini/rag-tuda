import { LIBRARIES } from "../config.js";

// ---- response formats v1: rendering

export const libraryLoads = {};

// One of LIBRARIES, loaded once (pinned and integrity-checked); a failed load may be tried again later.
export function loadLibrary(name) {
    if (!libraryLoads[name]) {
        libraryLoads[name] = new Promise((resolve, reject) => {
            const lib = LIBRARIES[name];
            const script = document.createElement("script");
            script.src = lib.src;
            script.integrity = lib.integrity;
            script.crossOrigin = "anonymous";
            script.onload = () => resolve();
            script.onerror = () => {
                delete libraryLoads[name];
                reject(new Error("could not load " + name));
            };
            document.head.appendChild(script);
        });
    }
    return libraryLoads[name];
}
