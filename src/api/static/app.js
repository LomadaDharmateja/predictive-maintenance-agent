/* The demo page. One file, no framework, no build step, no CDN.
 *
 * Everything is created with document.createElement and textContent rather
 * than innerHTML. Tool results are data from the database and answers are
 * model output -- neither is trusted markup, and building the DOM node by node
 * means a stray angle bracket in a maintenance record renders as text instead
 * of becoming an element.
 */
'use strict';

var PRESETS = [];
var DEMO_MODE = true;

/* ---------------------------------------------------------------- utils */

function el(tag, cls, text) {
  var node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function fmtMs(ms) {
  if (ms === null || ms === undefined) return '—';
  if (ms < 1) return ms.toFixed(2) + ' ms';
  if (ms < 1000) return Math.round(ms) + ' ms';
  return (ms / 1000).toFixed(2) + ' s';
}

function fmtCost(usd) {
  if (!usd) return '$0.00';
  if (usd < 0.01) return '$' + usd.toFixed(4);
  return '$' + usd.toFixed(2);
}

function fmtProb(p) {
  if (p === null || p === undefined) return '—';
  return (p * 100).toFixed(1) + '%';
}

/* ------------------------------------------------------------- presets */

function loadPresets() {
  return fetch('/v1/demo/presets')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      PRESETS = data.presets || [];
      DEMO_MODE = !!data.demo_mode;

      var text = DEMO_MODE
        ? 'Demo mode — replaying ' + PRESETS.length +
          ' recorded runs. No model is called, nothing costs anything, works offline.'
        : 'Live mode — ' + data.provider + ' / ' + data.model + '. Each question calls a model.';
      document.getElementById('modetext').textContent = text;

      document.getElementById('hint').textContent = DEMO_MODE
        ? 'Demo mode answers the preset questions only — they are the runs that were recorded. ' +
          'Free text needs a configured provider (PDM_DEMO_MODE=0).'
        : 'Answered live against the configured provider.';

      var box = document.getElementById('presets');
      box.textContent = '';
      if (!PRESETS.length) {
        box.appendChild(el('p', 'empty', 'No recorded transcripts found.'));
        return;
      }
      PRESETS.forEach(function (p) {
        var b = el('button', 'preset');
        b.type = 'button';
        b.setAttribute('data-kind', p.kind);
        b.appendChild(el('span', 'kind', p.kind));
        b.appendChild(document.createTextNode(p.label));
        b.addEventListener('click', function () { runPreset(p, b); });
        box.appendChild(b);
      });
    })
    .catch(function () {
      document.getElementById('modetext').textContent = 'Could not reach the service.';
    });
}

/* ----------------------------------------------------------------- ask */

function setBusy(busy) {
  var buttons = document.querySelectorAll('.preset');
  for (var i = 0; i < buttons.length; i++) buttons[i].disabled = busy;
}

function runPreset(preset, button) {
  setBusy(true);
  post({ question: preset.question, scenario_id: preset.scenario_id }, preset)
    .then(function () { setBusy(false); });
}

function runFreeText(question) {
  return post({ question: question }, null);
}

function post(body, preset) {
  var out = document.getElementById('output');
  out.textContent = '';
  out.appendChild(el('p', 'empty', 'Running…'));

  return fetch('/v1/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
    .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
    .then(function (res) {
      out.textContent = '';
      if (!res.ok) {
        out.appendChild(renderError(res.body));
        return;
      }
      out.appendChild(renderRun(res.body, body.question, preset));
    })
    .catch(function (e) {
      out.textContent = '';
      var card = el('div', 'card');
      card.appendChild(el('p', 'err', 'Request failed: ' + e.message));
      out.appendChild(card);
    });
}

/* -------------------------------------------------------------- render */

function renderError(body) {
  var card = el('div', 'card');
  card.appendChild(el('h3', null, 'Not answered'));
  card.appendChild(el('p', 'err', (body && body.code ? body.code + ' — ' : '') +
    (body && body.message ? body.message : 'unknown error')));
  return card;
}

function renderRun(run, question, preset) {
  var frag = document.createDocumentFragment();

  /* answer */
  var card = el('div', 'card');
  if (question) card.appendChild(el('p', 'q', question));
  card.appendChild(el('p', 'answer', run.answer || '(no answer)'));
  if (preset && preset.takeaway) {
    card.appendChild(el('div', 'takeaway', 'What to notice: ' + preset.takeaway));
  }
  frag.appendChild(card);

  /* the two fields that matter */
  if (run.highlights && run.highlights.length) {
    frag.appendChild(el('h2', null, 'Risk, warning adequacy and calibration'));
    run.highlights.forEach(function (h) { frag.appendChild(renderHighlight(h)); });
  }

  /* accounting */
  frag.appendChild(el('h2', null, 'Run accounting' + (run.replayed ? ' (replayed)' : '')));
  frag.appendChild(renderAccounting(run.accounting, run.replayed));

  /* trace */
  var n = (run.tool_calls || []).length;
  frag.appendChild(el('h2', null, 'Tool calls (' + n + ')'));
  if (!n) {
    frag.appendChild(el('p', 'empty', 'The agent answered without calling a tool.'));
  } else {
    run.tool_calls.forEach(function (c, i) { frag.appendChild(renderCall(c, i)); });
  }

  return frag;
}

function badge(cls, label, detail) {
  var b = el('span', 'badge ' + cls, label);
  if (detail) {
    b.appendChild(document.createTextNode(' '));
    b.appendChild(el('small', null, detail));
  }
  return b;
}

var ADEQUACY_CLASS = {
  sufficient: 'b-ok',
  marginal: 'b-warn',
  insufficient: 'b-bad'
};

function renderHighlight(h) {
  var box = el('div', 'risk');

  var head = el('div', 'risk-head');
  head.appendChild(el('span', 'risk-name',
    'machine ' + h.machine_id + ' · ' + h.component));
  head.appendChild(el('span', 'prob', fmtProb(h.probability)));
  box.appendChild(head);

  /* The interval on the contract is the *model's* PR-AUC interval, not a
   * predictive interval for this machine. Printing it beside the probability
   * as "95% CI" invites exactly the misreading src/agent/risk.py warns about
   * -- on three of four components it does not even bracket the number it sat
   * next to. It is labelled for what it is, on its own line. */
  if (h.model_prauc_ci_low !== undefined && h.model_prauc_ci_low !== null) {
    box.appendChild(el('div', 'ci',
      'model PR-AUC 95% CI [' + h.model_prauc_ci_low.toFixed(3) + ', ' +
      h.model_prauc_ci_high.toFixed(3) +
      '] — how well the model separates this component overall, not uncertainty about this machine'));
  }

  var badges = el('div', 'badges');

  /* calibrated: the flag that says whether the number may be believed */
  if (h.calibrated === true) {
    badges.appendChild(badge('b-ok', 'calibrated',
      '— held-out Brier skill excludes zero'));
  } else if (h.calibrated === false) {
    badges.appendChild(badge('b-bad', 'not calibrated',
      '— not established as better than the base rate'));
  }

  /* warning_adequacy: whether the warning is long enough to act on */
  if (h.warning_adequacy) {
    var cls = ADEQUACY_CLASS[h.warning_adequacy] || 'b-warn';
    badges.appendChild(badge(cls, 'warning ' + h.warning_adequacy,
      h.warning_adequacy === 'insufficient' ? '— too short to order a part' : ''));
  }

  if (h.exceeds_threshold === true) {
    badges.appendChild(badge('b-warn', 'above threshold', ''));
  }
  box.appendChild(badges);

  if (h.parts && h.parts.length) {
    var list = el('ul', 'parts');
    h.parts.forEach(function (p) {
      var li = el('li');
      li.appendChild(el('span', null, p.part_id));
      li.appendChild(el('span', null, 'lead ' + p.lead_time_days + ' d'));
      li.appendChild(el('span', null, 'warning ' +
        (p.detection_lead_hours === null || p.detection_lead_hours === undefined
          ? 'not detected'
          : p.detection_lead_hours + ' h')));
      var v = el('span', 'verdict ' + (ADEQUACY_CLASS[p.adequacy] || ''), p.adequacy);
      li.appendChild(v);
      list.appendChild(li);
    });
    box.appendChild(list);
  }
  return box;
}

function renderAccounting(a, replayed) {
  var grid = el('div', 'meters');
  if (!a) { grid.appendChild(el('div', 'meter', 'no accounting')); return grid; }

  function meter(key, value) {
    var m = el('div', 'meter');
    m.appendChild(el('div', 'v', value));
    m.appendChild(el('div', 'k', key));
    grid.appendChild(m);
  }

  meter('tokens in', a.tokens_in);
  meter('tokens out', a.tokens_out);
  meter(replayed ? 'cost (recorded)' : 'cost', fmtCost(a.estimated_cost_usd));
  meter('wall clock', fmtMs(a.wall_clock_ms));
  meter('tool time', fmtMs(a.tool_ms));
  meter('iterations', a.iterations + '/' + a.max_iterations);
  meter('tool calls', a.tool_calls + (a.tool_errors ? ' (' + a.tool_errors + ' err)' : ''));
  if (a.cache_read) meter('cache read', a.cache_read);
  return grid;
}

function renderCall(call, index) {
  var d = el('details', 'call');
  /* The trace is the point of the page, so the first call is open on arrival
   * -- a visitor should not have to guess that these expand. Errors are always
   * open; the rest fold so a four-tool run still fits on a screen. */
  if (index === 0 || call.status !== 'ok') d.open = true;

  var s = el('summary');
  s.appendChild(el('span', null, (index + 1) + '. ' + call.tool));
  s.appendChild(el('span', 'status ' + (call.status === 'ok' ? 's-ok' : 's-err'),
    call.status === 'ok' ? 'Success' : 'ToolError' +
      (call.error_code ? ' · ' + call.error_code : '')));
  if (call.truncated) s.appendChild(el('span', 'status s-err', 'truncated'));
  s.appendChild(el('span', 'ms', fmtMs(call.duration_ms)));
  d.appendChild(s);

  var body = el('div', 'body');
  body.appendChild(el('h4', null, 'Arguments'));
  body.appendChild(el('pre', null, JSON.stringify(call.arguments || {}, null, 2)));
  body.appendChild(el('h4', null, 'Result'));
  body.appendChild(el('pre', null, JSON.stringify(call.result, null, 2)));
  d.appendChild(body);
  return d;
}

/* ----------------------------------------------------------------- init */

document.getElementById('askform').addEventListener('submit', function (e) {
  e.preventDefault();
  var q = document.getElementById('question').value.trim();
  if (q.length < 3) return;
  runFreeText(q);
});

loadPresets();
