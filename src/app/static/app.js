// UI for the base vs. fine-tuned streaming comparison. All data comes from a
// "source" with health(), clips(), start(options, onEvent) and stop(); in
// phase 1 that is the mock in mock-source.js.
(function () {
  "use strict";

  const PLACEHOLDER = "Press live mic or play a random Greek clip.";
  const STAR_SVG =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5l2.94 6.07 6.56.92-4.78 4.6 1.17 6.53L12 17.5l-5.89 3.12 1.17-6.53-4.78-4.6 6.56-.92z"/></svg>';

  const source = window.createMockSource();

  const els = {
    status: document.getElementById("status"),
    statusText: document.getElementById("status-text"),
    language: document.getElementById("language"),
    micBtn: document.getElementById("mic-btn"),
    micLabel: document.getElementById("mic-label"),
    clipBtn: document.getElementById("clip-btn"),
    clipCount: document.getElementById("clip-count"),
    reference: document.getElementById("reference"),
    referenceText: document.getElementById("reference-text"),
  };

  const panels = {};
  for (const model of ["base", "finetuned"]) {
    const root = document.getElementById(`panel-${model}`);
    panels[model] = {
      root,
      transcript: root.querySelector(".transcript"),
      badge: root.querySelector(".wer-badge"),
      step: root.querySelector(".step"),
      banner: root.querySelector(".best-banner"),
    };
  }

  let mode = null; // "mic" | "clip" | null
  let finals = {};
  let micStream = null;

  function releaseMic() {
    if (micStream) {
      for (const track of micStream.getTracks()) track.stop();
      micStream = null;
    }
  }

  function showPanelMessage(message) {
    for (const panel of Object.values(panels)) {
      panel.transcript.textContent = message;
      panel.transcript.classList.add("placeholder");
    }
  }

  function werLevel(wer) {
    if (wer < 0.15) return "good";
    if (wer < 0.6) return "mid";
    return "bad";
  }

  function formatWer(wer) {
    return `WER ${Math.round(wer * 100)}%`;
  }

  function renderStatus(health) {
    if (!health || health.status !== "ok") {
      els.status.dataset.state = "down";
      els.statusText.textContent = "server unavailable";
      return;
    }
    const hasFinetuned = health.models.includes("finetuned");
    els.status.dataset.state = hasFinetuned ? "ok" : "warn";
    let text = hasFinetuned ? "server ok · base + finetuned" : "server ok · base only";
    if (health.mock) text += " · mock";
    els.statusText.textContent = text;
  }

  function renderTranscript(panel, text, streaming) {
    panel.transcript.classList.remove("placeholder");
    panel.transcript.textContent = text;
    if (streaming) {
      if (text) panel.transcript.append(" ");
      const cursor = document.createElement("span");
      cursor.className = "cursor";
      cursor.setAttribute("aria-hidden", "true");
      panel.transcript.append(cursor);
    }
  }

  function resetPanels(placeholder) {
    for (const panel of Object.values(panels)) {
      panel.root.classList.remove("best");
      panel.badge.hidden = true;
      panel.banner.hidden = true;
      panel.step.textContent = "";
      if (placeholder) {
        panel.transcript.textContent = PLACEHOLDER;
        panel.transcript.classList.add("placeholder");
      } else {
        renderTranscript(panel, "", true);
      }
    }
  }

  function setMicActive(active) {
    els.micBtn.setAttribute("aria-pressed", String(active));
    els.micLabel.textContent = active ? "stop mic" : "live mic";
  }

  function markBest() {
    const base = finals.base;
    const tuned = finals.finetuned;
    if (typeof base !== "number" || typeof tuned !== "number" || base === tuned) return;
    const winner = base < tuned ? "base" : "finetuned";
    const panel = panels[winner];
    panel.root.classList.add("best");
    panel.banner.innerHTML =
      `${STAR_SVG}<span class="best-wer">${formatWer(finals[winner])}</span><span class="best-tag">BEST</span>`;
    panel.banner.hidden = false;
  }

  function handleEvent(event) {
    switch (event.type) {
      case "reference":
        els.referenceText.textContent = event.text;
        els.reference.hidden = false;
        break;
      case "partial": {
        const panel = panels[event.model];
        renderTranscript(panel, event.text, true);
        panel.step.textContent = `step ${event.step}`;
        break;
      }
      case "final": {
        const panel = panels[event.model];
        renderTranscript(panel, event.text, false);
        finals[event.model] = event.wer;
        if (typeof event.wer === "number") {
          panel.badge.textContent = formatWer(event.wer);
          panel.badge.dataset.level = werLevel(event.wer);
          panel.badge.hidden = false;
        }
        break;
      }
      case "end":
        mode = null;
        releaseMic();
        setMicActive(false);
        markBest();
        break;
    }
  }

  function startSession(nextMode) {
    if (nextMode !== "mic") releaseMic();
    mode = nextMode;
    finals = {};
    setMicActive(nextMode === "mic");
    els.reference.hidden = true;
    resetPanels(false);
    source.start({ mode: nextMode, targetLang: els.language.value }, handleEvent);
  }

  // The stream is captured now so the browser asks for permission; sending
  // its audio to the server comes with the real source.
  async function startMic() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      resetPanels(false);
      showPanelMessage("Microphone access was blocked or no microphone was found.");
      return;
    }
    startSession("mic");
  }

  els.micBtn.addEventListener("click", () => {
    if (mode === "mic") {
      source.stop();
    } else {
      startMic();
    }
  });

  els.clipBtn.addEventListener("click", () => startSession("clip"));

  resetPanels(true);
  source.health().then(renderStatus, () => renderStatus(null));
  // Mock clips are scripted sentences with no audio, so they are not counted.
  source.clips().then((list) => {
    const real = list.filter((clip) => !clip.mock);
    els.clipCount.textContent = real.length ? `(${real.length} clips)` : "";
  });
})();
