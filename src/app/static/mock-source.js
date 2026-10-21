// Phase 1 stand-in for the ASR server. It emits the same events the
// /ws/stream WebSocket will send, so app.js does not change when the real
// server replaces it:
//   {type: "reference", text}
//   {type: "partial", model, text, step}
//   {type: "final", model, text, wer}   (wer is null without a reference)
//   {type: "end"}
(function () {
  "use strict";

  const FRAME_MS = 80;
  const WORD_S = 0.42;
  const MODELS = ["base", "finetuned"];
  // Seconds of extra delay before each model emits a word.
  const MODEL_LAG_S = { base: 0.12, finetuned: 0.06 };

  // Invented Greek sentences, not FLEURS data. The base hypotheses carry the
  // kind of spelling and word-boundary errors an un-adapted model makes.
  const CLIPS = [
    {
      id: "el-mock-001",
      reference: "Η Αθήνα είναι μία από τις αρχαιότερες πόλεις της Ευρώπης.",
      hypotheses: {
        base: "Η Αθήνα είναι μια από τις αρχαιότερες πόλης της Ευρώπη.",
        finetuned: "Η Αθήνα είναι μία από τις αρχαιότερες πόλεις της Ευρώπης.",
      },
    },
    {
      id: "el-mock-002",
      reference: "Το πλοίο αναχωρεί από τον Πειραιά κάθε πρωί στις επτά.",
      hypotheses: {
        base: "Το πλίο ανα χωρεί από το Πειραιά κάθε πρωί στης επτά.",
        finetuned: "Το πλοίο αναχωρεί από τον Πειραιά κάθε πρωί στις επτά.",
      },
    },
    {
      id: "el-mock-003",
      reference: "Οι επιστήμονες μελετούν τις αλλαγές του κλίματος στη Μεσόγειο.",
      hypotheses: {
        base: "Η επιστήμονες μελετούν τις αλλαγές του κλήματος στημεσόγειο.",
        finetuned: "Οι επιστήμονες μελετούν τις αλλαγές του κλίματος στη Μεσόγειο.",
      },
    },
    {
      id: "el-mock-004",
      reference: "Το μουσείο θα παραμείνει κλειστό για ανακαίνιση μέχρι τον Μάρτιο.",
      hypotheses: {
        base: "Το μουσίο θα παραμείνει κλειστό για ανακαίνηση μέχρι το Μάρτιο.",
        finetuned: "Το μουσείο θα παραμείνει κλειστό για ανακαίνιση μέχρι το Μάρτιο.",
      },
    },
    {
      id: "el-mock-005",
      reference: "Η βροχή σταμάτησε και ο ουρανός καθάρισε πάνω από τα βουνά.",
      hypotheses: {
        base: "Η βροχή σταμάτησε κι ο ουρανός καθάρισε πάνω απ τα βουνά.",
        finetuned: "Η βροχή σταμάτησε και ο ουρανός καθάρισε πάνω από τα βουνά.",
      },
    },
    {
      id: "el-mock-006",
      reference: "Οι μαθητές επισκέφθηκαν το εργαστήριο φυσικής του πανεπιστημίου.",
      hypotheses: {
        base: "Οι μαθητές επισκέφτηκαν τον εργαστήριο φισικής του πανεπιστήμιου.",
        finetuned: "Οι μαθητές επισκέφτηκαν το εργαστήριο φυσικής του πανεπιστημίου.",
      },
    },
  ];

  function normalizeWords(text) {
    return text
      .toLowerCase()
      .replace(/[\p{P}\p{S}]/gu, " ")
      .split(/\s+/)
      .filter(Boolean);
  }

  // Word-level Levenshtein distance divided by reference length.
  function wordErrorRate(reference, hypothesis) {
    const ref = normalizeWords(reference);
    const hyp = normalizeWords(hypothesis);
    if (ref.length === 0) {
      return hyp.length === 0 ? 0 : 1;
    }
    let previous = Array.from({ length: hyp.length + 1 }, (_, j) => j);
    for (let i = 1; i <= ref.length; i++) {
      const current = [i];
      for (let j = 1; j <= hyp.length; j++) {
        const cost = ref[i - 1] === hyp[j - 1] ? 0 : 1;
        current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost);
      }
      previous = current;
    }
    return previous[hyp.length] / ref.length;
  }

  // Time (seconds) at which each word of a hypothesis becomes visible.
  function revealTimes(text, lagS) {
    return text.split(/\s+/).map((word, i) => ({
      word,
      at: 0.35 + lagS + i * WORD_S + ((i * 37) % 5) * 0.03,
    }));
  }

  function textAt(schedule, seconds) {
    return schedule
      .filter((entry) => entry.at <= seconds)
      .map((entry) => entry.word)
      .join(" ");
  }

  function createMockSource() {
    let timer = null;
    let session = null;
    let lastClipId = null;

    function health() {
      return Promise.resolve({ status: "ok", models: MODELS.slice(), mock: true });
    }

    function clips() {
      return Promise.resolve(CLIPS.map((clip) => ({ id: clip.id, language: "el-GR", mock: true })));
    }

    function pickClip() {
      const choices = CLIPS.filter((clip) => clip.id !== lastClipId);
      const clip = choices[Math.floor(Math.random() * choices.length)];
      lastClipId = clip.id;
      return clip;
    }

    function clearTimer() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    }

    function start(options, onEvent) {
      clearTimer();
      const mode = options.mode;
      const clip = mode === "clip" ? pickClip() : null;
      // The mock cannot recognize speech, so live-mic sessions get empty
      // transcripts; only clips have scripted hypotheses.
      const schedules = {};
      let lastWordAt = 0;
      for (const model of MODELS) {
        schedules[model] = clip ? revealTimes(clip.hypotheses[model], MODEL_LAG_S[model]) : [];
        for (const entry of schedules[model]) {
          lastWordAt = Math.max(lastWordAt, entry.at);
        }
      }
      const totalSteps = Math.ceil(((lastWordAt + 0.5) * 1000) / FRAME_MS);

      session = { clip, schedules, onEvent, step: 0 };
      if (clip) {
        onEvent({ type: "reference", text: clip.reference });
      }

      timer = setInterval(() => {
        session.step += 1;
        const seconds = (session.step * FRAME_MS) / 1000;
        for (const model of MODELS) {
          onEvent({ type: "partial", model, text: textAt(schedules[model], seconds), step: session.step });
        }
        // Clips end on their own; the mic keeps listening until stop().
        if (clip && session.step >= totalSteps) {
          finish();
        }
      }, FRAME_MS);
    }

    function finish() {
      clearTimer();
      if (!session) {
        return;
      }
      const { clip, schedules, onEvent } = session;
      const seconds = (session.step * FRAME_MS) / 1000;
      for (const model of MODELS) {
        const text = textAt(schedules[model], seconds);
        const wer = clip ? wordErrorRate(clip.reference, text) : null;
        onEvent({ type: "final", model, text, wer });
      }
      onEvent({ type: "end" });
      session = null;
    }

    // Ends the current session the way the server will: finals, then "end".
    function stop() {
      finish();
    }

    return { health, clips, start, stop };
  }

  window.createMockSource = createMockSource;
})();
