async function refreshStatus() {
  const el = document.getElementById('status-bar');
  if (!el) return;
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const evidence = data.evidence || {};
    const chat = data.chat || {};
    let ev;
    const label = evidence.label || 'Evidence';
    if (!evidence.connected) {
      ev = `Evidence: ${label} \u2014 Not available`;
    } else if (!evidence.evidence_access) {
      ev = `Evidence: ${label} \u2014 Permission required`;
    } else {
      ev = `Evidence: ${label} \u2014 Ready`;
    }
    const chatLabel = chat.label || 'Chat';
    const ol = chat.connected ? `${chatLabel}: Connected` : `${chatLabel}: Not connected`;
    el.textContent = `${ev}  |  ${ol} (${chat.model})`;
    el.title = [evidence.hint, chat.hint].filter(Boolean).join(' | ');
    const allOk = evidence.connected && evidence.evidence_access && chat.connected;
    el.classList.toggle('warn', !allOk);
  } catch (err) {
    el.textContent = 'Unable to check service status.';
    el.classList.add('warn');
  }
}

async function startSession(exerciseId, studentDifficulty) {
  const body = { exercise_id: exerciseId };
  if (studentDifficulty) body.student_difficulty = studentDifficulty;
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    alert('Could not start session.');
    return;
  }
  const session = await res.json();
  window.location.href = `/sessions/${session.id}`;
}

function initSessionPage() {
  const body = document.body;
  const sessionId = body.dataset.sessionId;
  const startedAt = new Date(body.dataset.startedAt);

  setInterval(() => {
    const el = document.getElementById('duration');
    if (!el) return;
    const seconds = Math.max(0, Math.floor((Date.now() - startedAt.getTime()) / 1000));
    const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
    const ss = String(seconds % 60).padStart(2, '0');
    el.textContent = `${mm}:${ss}`;
  }, 1000);

  setInterval(refreshStatus, 15000);

  const chatForm = document.getElementById('chat-form');
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    if (!question) return;
    appendChatMessage('student', question);
    input.value = '';
    input.disabled = true;
    try {
      const res = await fetch(`/api/sessions/${sessionId}/mentor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      appendChatMessage('mentor', data.answer || data.detail || 'No response.');
    } catch (err) {
      appendChatMessage('mentor', 'Could not reach the mentor service.');
    } finally {
      input.disabled = false;
      input.focus();
    }
  });

  document.getElementById('finish-btn').addEventListener('click', async () => {
    if (!confirm('Finish the exercise and evaluate your work?')) return;
    const res = await fetch(`/api/sessions/${sessionId}/finish`, { method: 'POST' });
    if (!res.ok) {
      alert('Could not finish the session.');
      return;
    }
    window.location.href = `/sessions/${sessionId}/results`;
  });

  document.getElementById('load-evidence-btn').addEventListener('click', async () => {
    const out = document.getElementById('debug-output');
    out.textContent = 'Loading...';
    const res = await fetch(`/api/sessions/${sessionId}/evidence`);
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  });
}

function appendChatMessage(role, text) {
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = `chat-message ${role}`;
  const label = role === 'student' ? 'You' : 'Mentor';
  div.innerHTML = `<strong>${label}:</strong> ${escapeHtml(text)}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ---- Ask About This Session (completed-session Q&A) ----

function appendSessionQA(role, text, renderMarkdown = false) {
  const log = document.getElementById('qa-log');
  if (!log) return null;
  const div = document.createElement('div');
  div.className = `chat-message ${role}`;
  const label = role === 'student' ? 'You' : 'Answer';
  // Student input and transient placeholders stay escaped plain text; only
  // mentor answers are rendered as Markdown (renderMentorReviewText escapes
  // internally, so this can never inject raw HTML from the model).
  const body = renderMarkdown ? renderMentorReviewText(text) : escapeHtml(text);
  div.innerHTML = `<strong>${label}:</strong> ${body}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function initSessionQueryPage() {
  const form = document.getElementById('session-query-form');
  if (!form) return;
  const sessionId = document.body.dataset.sessionId;
  const input = document.getElementById('session-query-input');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    appendSessionQA('student', question);
    input.value = '';
    input.disabled = true;
    const answerEl = appendSessionQA('mentor', 'Thinking...');
    try {
      const res = await fetch(`/api/sessions/${sessionId}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      answerEl.innerHTML = `<strong>Answer:</strong> ${renderMentorReviewText(data.answer || data.detail || 'No response.')}`;
    } catch (err) {
      answerEl.innerHTML = `<strong>Answer:</strong> Could not reach the session query service.`;
    } finally {
      input.disabled = false;
      input.focus();
    }
  });
}

async function loadResults() {
  const sessionId = document.body.dataset.sessionId;
  const container = document.getElementById('results-content');
  try {
    const res = await fetch(`/api/sessions/${sessionId}/evaluation`);
    if (!res.ok) {
      container.textContent = 'No evaluation found for this session yet.';
      return;
    }
    const data = await res.json();
    container.innerHTML = renderResults(data);
  } catch (err) {
    container.textContent = 'Could not load evaluation.';
  }
}

// Re-evaluates the ALREADY-CAPTURED evidence for this session (no new
// capture session is started) -- offered on the results page when
// evaluation was unavailable (e.g. a model timeout), so a student is never
// stuck with a stale "unavailable" result without redoing the exercise.
async function retryEvaluation() {
  const sessionId = document.body.dataset.sessionId;
  const btn = document.getElementById('retry-evaluation-btn');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Retrying...';
  }
  try {
    const res = await fetch(`/api/sessions/${sessionId}/retry-evaluation`, { method: 'POST' });
    if (!res.ok) {
      alert('Retry failed -- please try again in a moment.');
      if (btn) { btn.disabled = false; btn.textContent = 'Retry Evaluation'; }
      return;
    }
    await loadResults();
  } catch (err) {
    alert('Retry failed -- please try again in a moment.');
    if (btn) { btn.disabled = false; btn.textContent = 'Retry Evaluation'; }
  }
}

function renderOutcomeItem(o) {
  // verification_state "unverifiable" means -- by its own definition (see
  // app.services.prompts.EVALUATION_SYSTEM_PROMPT) -- "the capture/evaluation
  // mechanism could not reliably determine completion", i.e. an evaluator or
  // capture-infrastructure limitation, NOT evidence the student failed. This
  // covers both an evidence_error and an ai_unavailable (e.g. model timeout)
  // outcome uniformly: render it as unresolved, never as a failure.
  const unresolved = !o.passed && o.verification_state === 'unverifiable';
  const icon = unresolved ? '\u2022' : (o.passed ? '\u2713' : (o.confidence === 'inferred' ? '\u25B3' : '\u2717'));
  const cssClass = unresolved ? 'unresolved' : (o.passed ? 'pass' : 'fail');
  const scoreText = unresolved ? 'not yet scored' : `${o.score}/${o.max_score} pts`;
  return `<li class="outcome ${cssClass}">
    <span class="confidence">${o.confidence}</span>
    <span class="icon">${icon}</span>
    <strong>${o.id}</strong> - ${scoreText}
    <p class="evidence">${escapeHtml(o.evidence)}</p>
  </li>`;
}

function renderOutcomes(outcomes) {
  const hasSteps = outcomes.some((o) => o.step_id);
  if (!hasSteps) {
    return `<ul class="outcome-list">${outcomes.map(renderOutcomeItem).join('')}</ul>`;
  }
  const groups = [];
  const seen = new Set();
  for (const o of outcomes) {
    const key = o.step_id || '';
    if (!seen.has(key)) {
      seen.add(key);
      groups.push({ stepId: key, stepTitle: o.step_title || 'Step', items: [] });
    }
    groups.find((g) => g.stepId === key).items.push(o);
  }
  return groups
    .map(
      (g) => `<h4>${escapeHtml(g.stepTitle)}</h4><ul class="outcome-list">${g.items.map(renderOutcomeItem).join('')}</ul>`
    )
    .join('');
}

function renderResults(data) {
  const details = data.details || {};
  const outcomes = details.outcomes || [];
  const outcomeRows = renderOutcomes(outcomes);

  const list = (items) =>
    items && items.length
      ? `<ul>${items.map((i) => `<li>${escapeHtml(i)}</li>`).join('')}</ul>`
      : '<p>None noted.</p>';

  const evidenceErrorBanner = details.evidence_error
    ? `<div class="evidence-error-banner">
        <strong>Evidence retrieval failed</strong> -- this does NOT mean the learner did nothing;
        the capture system could not be reached during this session.<br>${escapeHtml(details.evidence_error)}
      </div>`
    : '';

  // ai_unavailable means the AI evaluator itself could not complete scoring
  // (e.g. a model timeout, even after the built-in retry with a smaller
  // payload) -- distinct from evidence_error (evidence RETRIEVAL failing).
  // Both are evaluator/infrastructure problems, not evidence of student
  // failure, so neither may present a final "/100" total or a failure mark
  // on the outcomes that could not be resolved (see renderOutcomeItem).
  const evaluationUnavailable = Boolean(details.ai_unavailable);
  const aiUnavailableBanner = evaluationUnavailable
    ? `<div class="evidence-error-banner">
        <strong>Evaluation unavailable</strong> -- the AI evaluator could not complete scoring for this
        attempt, even after automatically retrying with a smaller evidence packet. This is NOT a
        reflection of your work: the outcomes below marked "not yet scored" simply could not be
        resolved yet.<br>${escapeHtml(details.ai_unavailable)}
        <div style="margin-top:0.5em">
          <button type="button" id="retry-evaluation-btn" class="button" onclick="retryEvaluation()">Retry Evaluation</button>
        </div>
      </div>`
    : '';

  const scoreBanner = (evaluationUnavailable || details.evidence_error)
    ? `<div class="score-banner">Evaluation incomplete</div>`
    : `<div class="score-banner">Score: ${data.score} / 100</div>`;

  return `
    ${scoreBanner}
    ${evidenceErrorBanner}
    ${aiUnavailableBanner}
    <h3>Outcome Results</h3>
    ${outcomeRows}
    <h3>Summary</h3>
    <p>${escapeHtml(data.summary || '')}</p>
    <h3>What You Did</h3>
    ${list(details.observed_approach)}
    <h3>What You Did Well</h3>
    ${list(details.strengths)}
    <h3>What Could Be Improved</h3>
    ${list(details.improvements)}
    <h3>Risky or Unnecessary Steps</h3>
    ${list(details.risky_or_unnecessary_steps)}
    <h3>Alternative Valid Approaches</h3>
    ${list(details.alternative_approaches)}
  `;
}

// ---- Settings page ----

async function initSettingsPage() {
  const input = document.getElementById('student-name');
  const courseInput = document.getElementById('course-id');
  const status = document.getElementById('settings-status');
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();
    input.value = data.student_name || '';
    courseInput.value = data.course_id || '';
  } catch (err) {
    status.textContent = 'Could not load current settings.';
  }

  document.getElementById('settings-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    status.textContent = 'Saving...';
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_name: input.value.trim() || null,
          course_id: courseInput.value.trim() || null,
        }),
      });
      if (!res.ok) throw new Error('save failed');
      status.textContent = 'Saved.';
    } catch (err) {
      status.textContent = 'Could not save settings.';
    }
  });
}

// ---- Exercise authoring (chat) page ----

let authoringSessionId = null;

async function initAuthoringPage() {
  const res = await fetch('/api/authoring', { method: 'POST' });
  const session = await res.json();
  authoringSessionId = session.id;

  initReferenceAttachPanel();
  initSeedDocumentPanel();

  const chatForm = document.getElementById('authoring-chat-form');
  const chatInput = document.getElementById('authoring-chat-input');

  // Enter sends the message; Shift+Enter inserts a newline (textarea default).
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('authoring-chat-input');
    const message = input.value.trim();
    if (!message) return;
    hideSeedDocumentPanel();
    appendChatMessage('student', message);
    input.value = '';
    input.disabled = true;
    try {
      const res = await fetch(`/api/authoring/${authoringSessionId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      appendChatMessage('mentor', data.reply || data.detail || 'No response.');
    } catch (err) {
      appendChatMessage('mentor', 'Could not reach the AI model.');
    } finally {
      input.disabled = false;
      input.focus();
    }
  });

  const researchForm = document.getElementById('research-form');
  const researchInput = document.getElementById('research-input');
  researchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      researchForm.requestSubmit();
    }
  });

  researchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = researchInput.value.trim();
    if (!question) return;
    const status = document.getElementById('research-status');
    status.textContent = 'Researching (this calls an external AI service)...';
    researchInput.disabled = true;
    try {
      const res = await fetch(`/api/authoring/${authoringSessionId}/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = data.detail || 'Research failed.';
        return;
      }
      appendChatMessage('student', `[Research request] ${question}`);
      appendChatMessage('mentor', data.answer);
      researchInput.value = '';
      status.textContent = 'Added to the conversation below -- review it, then keep chatting or finalize.';
    } catch (err) {
      status.textContent = 'Could not reach the research service.';
    } finally {
      researchInput.disabled = false;
    }
  });

  document.getElementById('finalize-btn').addEventListener('click', async () => {
    const finalizeBtn = document.getElementById('finalize-btn');
    finalizeBtn.disabled = true;
    finalizeBtn.textContent = 'Finalizing...';
    try {
      const res = await fetch(`/api/authoring/${authoringSessionId}/finalize`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail || 'Could not finalize the exercise.');
        return;
      }
      document.getElementById('draft-section').style.display = 'block';
      document.getElementById('draft-yaml').value = data.draft_yaml;
      const difficultyMatch = data.draft_yaml.match(/^difficulty:\s*(\S+)/m);
      document.getElementById('difficulty-select').value = difficultyMatch ? difficultyMatch[1] : 'intermediate';
    } finally {
      finalizeBtn.disabled = false;
      finalizeBtn.textContent = 'Finalize Exercise';
    }
  });

  document.getElementById('save-btn').addEventListener('click', async () => {
    const saveStatus = document.getElementById('save-status');
    saveStatus.textContent = 'Saving...';
    const difficulty = document.getElementById('difficulty-select').value;
    const draftYamlEl = document.getElementById('draft-yaml');
    // The instructor's manually selected difficulty always wins over whatever
    // the draft YAML happens to contain -- overwrite (or add) the field.
    let yamlText = /^difficulty:\s*\S+/m.test(draftYamlEl.value)
      ? draftYamlEl.value.replace(/^difficulty:\s*\S+/m, `difficulty: ${difficulty}`)
      : `${draftYamlEl.value}\ndifficulty: ${difficulty}\n`;
    draftYamlEl.value = yamlText;
    try {
      const res = await fetch(`/api/authoring/${authoringSessionId}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml: yamlText }),
      });
      const data = await res.json();
      if (!res.ok) {
        saveStatus.textContent = data.detail || 'Could not save the exercise.';
        return;
      }
      saveStatus.textContent = `Saved as "${data.title}" (${data.exercise_id}). Export it from Manage Exercises to share it.`;
    } catch (err) {
      saveStatus.textContent = 'Could not save the exercise.';
    }
  });
}

// ---- Exercise authoring: start-from-document seed panel ----

function hideSeedDocumentPanel() {
  const panel = document.getElementById('seed-document-panel');
  if (panel) panel.style.display = 'none';
}

function initSeedDocumentPanel() {
  const form = document.getElementById('seed-document-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('seed-document-file');
    const status = document.getElementById('seed-document-status');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    status.textContent = 'Reading and distilling the document...';
    fileInput.disabled = true;
    try {
      const res = await fetch(`/api/authoring/${authoringSessionId}/seed-document`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = data.detail || 'Could not use that document.';
        return;
      }
      appendChatMessage('student', data.seeded_message);
      appendChatMessage('mentor', data.reply);
      hideSeedDocumentPanel();
    } catch (err) {
      status.textContent = 'Could not reach the AI model.';
    } finally {
      fileInput.disabled = false;
    }
  });
}

// ---- Exercise authoring: reference attach panel ----

async function initReferenceAttachPanel() {
  const list = document.getElementById('reference-attach-list');
  if (!list) return;
  try {
    const [allRes, attachedRes] = await Promise.all([
      fetch('/api/references'),
      fetch(`/api/authoring/${authoringSessionId}/references`),
    ]);
    const all = await allRes.json();
    const attached = await attachedRes.json();
    const attachedIds = new Set(attached.map((r) => r.id));

    if (!all.length) {
      list.innerHTML = '<li class="muted">No references in the library yet.</li>';
      return;
    }

    list.innerHTML = all
      .map(
        (r) => `<li>
          <label>
            <input type="checkbox" data-reference-id="${r.id}" ${attachedIds.has(r.id) ? 'checked' : ''}>
            ${escapeHtml(r.title)} <span class="muted">(${escapeHtml(r.category.replace('_', ' '))})</span>
          </label>
        </li>`
      )
      .join('');

    list.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
      checkbox.addEventListener('change', async () => {
        const referenceId = checkbox.dataset.referenceId;
        const method = checkbox.checked ? 'POST' : 'DELETE';
        checkbox.disabled = true;
        try {
          const res = await fetch(
            `/api/authoring/${authoringSessionId}/references/${referenceId}`,
            { method }
          );
          if (!res.ok) throw new Error('attachment update failed');
        } catch (err) {
          checkbox.checked = !checkbox.checked;
          alert('Could not update the attached reference.');
        } finally {
          checkbox.disabled = false;
        }
      });
    });
  } catch (err) {
    list.innerHTML = '<li class="muted">Could not load the reference library.</li>';
  }
}

// ---- Reference library page ----

async function initReferencesPage() {
  document.getElementById('add-reference-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const title = document.getElementById('reference-title').value.trim();
    const category = document.getElementById('reference-category').value;
    const fileInput = document.getElementById('reference-file');
    const url = document.getElementById('reference-url').value.trim();
    const status = document.getElementById('add-reference-status');

    if (!fileInput.files.length && !url) {
      status.textContent = 'Provide either a file upload or a URL.';
      return;
    }

    const formData = new FormData();
    formData.append('title', title);
    formData.append('category', category);
    if (fileInput.files.length) formData.append('file', fileInput.files[0]);
    if (url) formData.append('url', url);

    status.textContent = 'Adding reference (this may take a moment for large documents or URLs)...';
    try {
      const res = await fetch('/api/references', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = data.detail || 'Could not add reference.';
        return;
      }
      status.textContent = `Added "${data.title}".`;
      setTimeout(() => window.location.reload(), 600);
    } catch (err) {
      status.textContent = 'Could not add reference.';
    }
  });

  document.querySelectorAll('.delete-reference-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      if (!confirm('Delete this reference? This cannot be undone.')) return;
      const referenceId = btn.dataset.referenceId;
      try {
        const res = await fetch(`/api/references/${referenceId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('delete failed');
        document.querySelector(`tr[data-reference-id="${referenceId}"]`).remove();
      } catch (err) {
        alert('Could not delete reference.');
      }
    });
  });
}

// ---- Manage exercises: import ----

function initImportExercisePage() {
  document.getElementById('import-exercise-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('import-exercise-file');
    const status = document.getElementById('import-exercise-status');
    if (!fileInput.files.length) return;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    status.textContent = 'Importing...';
    try {
      const res = await fetch('/api/exercises/import', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = data.detail || 'Import failed.';
        return;
      }
      status.textContent = `Imported "${data.title}". Reloading...`;
      setTimeout(() => window.location.reload(), 800);
    } catch (err) {
      status.textContent = 'Import failed.';
    }
  });
}

// ---- Instructor submissions ----

function initSubmissionsPage() {
  loadSubmissionsTable();

  document.getElementById('import-submissions-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const filesInput = document.getElementById('import-submissions-files');
    const status = document.getElementById('import-submissions-status');
    if (!filesInput.files.length) return;
    const formData = new FormData();
    for (const file of filesInput.files) formData.append('files', file);
    status.textContent = 'Importing...';
    try {
      const res = await fetch('/api/submissions/import', { method: 'POST', body: formData });
      const data = await res.json();
      status.innerHTML = (data.results || [])
        .map((r) => {
          if (r.error) return `<p>${escapeHtml(r.filename)}: ${escapeHtml(r.error)}</p>`;
          const sig = r.signature_valid === true ? 'valid' : r.signature_valid === false ? 'INVALID' : 'unverifiable';
          return `<p>${escapeHtml(r.filename)}: imported (signature: ${sig})</p>`;
        })
        .join('');
      loadSubmissionsTable();
    } catch (err) {
      status.textContent = 'Import failed.';
    }
  });
}

async function loadSubmissionsTable() {
  const container = document.getElementById('submissions-table');
  try {
    const res = await fetch('/api/submissions');
    const submissions = await res.json();
    if (!submissions.length) {
      container.innerHTML = '<p>No submissions imported yet.</p>';
      return;
    }
    const rows = submissions
      .map((s) => {
        const sig =
          s.signature_valid === true
            ? '<span class="sig-ok">valid</span>'
            : s.signature_valid === false
            ? '<span class="sig-bad">INVALID</span>'
            : '<span class="sig-unknown">unverifiable</span>';
        return `<tr>
          <td>${escapeHtml(s.student_name || 'Unknown')}</td>
          <td>${escapeHtml(s.exercise_title)}</td>
          <td>${s.score}</td>
          <td>${sig}</td>
          <td><a href="/instructor/submissions/${s.id}">View</a></td>
        </tr>`;
      })
      .join('');
    container.innerHTML = `<table class="data-table">
      <thead><tr><th>Student</th><th>Exercise</th><th>Score</th><th>Signature</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  } catch (err) {
    container.textContent = 'Could not load submissions.';
  }
}

async function loadSubmissionDetail() {
  const submissionId = document.body.dataset.submissionId;
  const metaEl = document.getElementById('submission-meta');
  const container = document.getElementById('results-content');
  try {
    const res = await fetch(`/api/submissions/${submissionId}`);
    if (!res.ok) {
      container.textContent = 'Submission not found.';
      return;
    }
    const data = await res.json();
    const s = data.submission;
    const sig =
      s.signature_valid === true
        ? '<span class="sig-ok">Signature valid</span>'
        : s.signature_valid === false
        ? '<span class="sig-bad">Signature INVALID -- this file may have been edited</span>'
        : `<span class="sig-unknown">Signature unverifiable -- ${escapeHtml(s.signature_note || '')}</span>`;
    metaEl.innerHTML = `<p><strong>${escapeHtml(s.student_name || 'Unknown')}</strong> -- ${escapeHtml(s.exercise_title)}</p><p>${sig}</p>`;
    container.innerHTML = renderResults({ score: s.score, summary: s.summary, details: data.details });
  } catch (err) {
    container.textContent = 'Could not load submission.';
  }
}

// ---- Mentor Review page ----

function renderMentorReviewText(text) {
  // Renders the Markdown subset the mentor-review and session-query prompts
  // produce: `##`/`###` section headings, `-` bullet lists, `1.` numbered
  // lists, GFM-style pipe tables, and inline **bold**, *italic*, and `code`.
  // Everything is escaped first, then the markup tokens are reintroduced, so
  // model output can never inject raw HTML. Plain prose (no Markdown) still
  // renders as <p> paragraphs, so older/non-markdown text degrades gracefully.
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let para = [];
  let listType = null;   // 'ul' | 'ol' | null
  let listItems = [];

  const inline = (s) => {
    let v = escapeHtml(s);
    v = v.replace(/`([^`]+)`/g, '<code>$1</code>');
    v = v.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    v = v.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    return v;
  };
  // A GFM table row starts and ends with `|`; returns the trimmed cells, or
  // null if the line isn't a row. A separator row's cells are only dashes
  // with optional leading/trailing colons (e.g. `---`, `:---`, `---:`).
  const tableRowCells = (line) => {
    const t = (line || '').trim();
    if (!t.startsWith('|') || !t.endsWith('|')) return null;
    return t.slice(1, -1).split('|').map((c) => c.trim());
  };
  const isTableSeparator = (line) => {
    const cells = tableRowCells(line);
    if (!cells) return false;
    return cells.every((c) => /^:?-{1,}:?$/.test(c));
  };

  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${inline(para.join(' '))}</p>`);
      para = [];
    }
  };
  const flushList = () => {
    if (listType) {
      const items = listItems.map((it) => `<li>${inline(it)}</li>`).join('');
      out.push(`<${listType}>${items}</${listType}>`);
      listType = null;
      listItems = [];
    }
  };
  const flushAll = () => { flushPara(); flushList(); };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i].replace(/\s+$/, '');

    // Table: a header row, a separator row, then zero or more body rows.
    const headerCells = tableRowCells(line);
    if (headerCells && headerCells.length >= 2 && isTableSeparator(lines[i + 1])) {
      flushAll();
      const bodyRows = [];
      let j = i + 2;
      while (j < lines.length) {
        const cells = tableRowCells(lines[j]);
        if (!cells) break;
        bodyRows.push(cells);
        j++;
      }
      const headHtml = headerCells.map((c) => `<th>${inline(c)}</th>`).join('');
      const bodyHtml = bodyRows
        .map((row) => `<tr>${row.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`)
        .join('');
      out.push(
        `<table class="data-table"><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`
      );
      i = j;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^-\s+(.*)$/);
    const ordered = line.match(/^\d+\.\s+(.*)$/);

    if (heading) {
      flushAll();
      const depth = heading[1].length;
      const level = depth >= 4 ? 5 : Math.max(3, depth + 1);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      i++;
      continue;
    }
    if (bullet) {
      flushPara();
      if (listType !== 'ul') { flushList(); listType = 'ul'; listItems = []; }
      listItems.push(bullet[1]);
      i++;
      continue;
    }
    if (ordered) {
      flushPara();
      if (listType !== 'ol') { flushList(); listType = 'ol'; listItems = []; }
      listItems.push(ordered[1]);
      i++;
      continue;
    }
    if (!line.trim()) {
      flushAll();
      i++;
      continue;
    }
    flushList();
    para.push(line.trim());
    i++;
  }
  flushAll();
  return `<div class="review-body">${out.join('\n')}</div>`;
}

function initMentorReviewPage() {
  document.getElementById('generate-review-btn').addEventListener('click', async () => {
    const btn = document.getElementById('generate-review-btn');
    const content = document.getElementById('mentor-review-content');
    btn.disabled = true;
    btn.textContent = 'Generating...';
    content.innerHTML = '<p class="muted">Looking back across your completed labs...</p>';
    try {
      const res = await fetch('/api/mentor-review/generate', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        content.innerHTML = `<p class="muted">${escapeHtml(data.detail || 'Could not generate a review.')}</p>`;
        return;
      }
      content.innerHTML = renderMentorReviewText(data.review || '');
    } catch (err) {
      content.innerHTML = '<p class="muted">Could not reach the mentor service.</p>';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate My Mentor Review';
    }
  });
}

// ---- Instructor Mentor Insights page ----

async function loadMentorLearnersTable() {
  const container = document.getElementById('mentor-learners-table');
  try {
    const res = await fetch('/api/instructor/mentor-insights/learners');
    const learners = await res.json();
    if (!learners.length) {
      container.innerHTML = '<p>No anonymous mentor data imported yet.</p>';
      return;
    }
    const rows = learners
      .map(
        (l) => `<tr>
          <td>Learner ${escapeHtml(l.anonymous_student_id)}</td>
          <td>${escapeHtml(l.course_id || 'default')}</td>
          <td>${l.import_count}</td>
          <td><button class="button learner-review-btn" data-learner-id="${escapeHtml(l.anonymous_student_id)}">Generate Review</button></td>
        </tr>`
      )
      .join('');
    container.innerHTML = `<table class="data-table">
      <thead><tr><th>Learner</th><th>Course</th><th>Imports</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

    container.querySelectorAll('.learner-review-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const learnerId = btn.dataset.learnerId;
        const content = document.getElementById('learner-review-content');
        btn.disabled = true;
        content.innerHTML = `<p class="muted">Generating a review for Learner ${escapeHtml(learnerId)}...</p>`;
        try {
          const res = await fetch(`/api/instructor/mentor-insights/learners/${learnerId}/review`, { method: 'POST' });
          const data = await res.json();
          if (!res.ok) {
            content.innerHTML = `<p class="muted">${escapeHtml(data.detail || 'Could not generate a review.')}</p>`;
            return;
          }
          content.innerHTML = `<h4>Learner ${escapeHtml(learnerId)}</h4>${renderMentorReviewText(data.review || '')}`;
        } catch (err) {
          content.innerHTML = '<p class="muted">Could not reach the mentor service.</p>';
        } finally {
          btn.disabled = false;
        }
      });
    });
  } catch (err) {
    container.textContent = 'Could not load imported learners.';
  }
}

function initMentorInsightsPage() {
  loadMentorLearnersTable();

  document.getElementById('import-mentor-data-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const filesInput = document.getElementById('import-mentor-data-files');
    const status = document.getElementById('import-mentor-data-status');
    if (!filesInput.files.length) return;
    const formData = new FormData();
    for (const file of filesInput.files) formData.append('files', file);
    status.textContent = 'Importing...';
    try {
      const res = await fetch('/api/instructor/mentor-insights/import', { method: 'POST', body: formData });
      const data = await res.json();
      status.innerHTML = (data.results || [])
        .map((r) => {
          if (r.error) return `<p>${escapeHtml(r.filename)}: ${escapeHtml(r.error)}</p>`;
          return `<p>${escapeHtml(r.filename)}: imported as Learner ${escapeHtml(r.anonymous_student_id)}</p>`;
        })
        .join('');
      loadMentorLearnersTable();
    } catch (err) {
      status.textContent = 'Import failed.';
    }
  });

  document.getElementById('class-review-btn').addEventListener('click', async () => {
    const btn = document.getElementById('class-review-btn');
    const content = document.getElementById('class-review-content');
    btn.disabled = true;
    btn.textContent = 'Generating...';
    content.innerHTML = '<p class="muted">Looking for patterns across imported learners...</p>';
    try {
      const res = await fetch('/api/instructor/mentor-insights/class-review', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        content.innerHTML = `<p class="muted">${escapeHtml(data.detail || 'Could not generate a class-wide review.')}</p>`;
        return;
      }
      content.innerHTML = renderMentorReviewText(data.review || '');
    } catch (err) {
      content.innerHTML = '<p class="muted">Could not reach the mentor service.</p>';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate Class-Wide Analysis';
    }
  });
}

// ---- Common errors dashboard ----

async function loadCommonErrors() {
  const exerciseId = document.body.dataset.exerciseId;
  const container = document.getElementById('common-errors-content');
  try {
    const res = await fetch(`/api/exercises/${exerciseId}/common-errors`);
    const data = await res.json();
    if (!data.submission_count) {
      container.innerHTML = '<p>No submissions imported for this exercise yet.</p>';
      return;
    }

    const outcomeRows = (data.outcome_pass_rates || [])
      .map(
        (o) => `<tr>
          <td>${escapeHtml(o.id)}</td>
          <td>${o.pass_rate === null ? 'n/a' : o.pass_rate + '%'}</td>
          <td>${o.passed}/${o.total}</td>
        </tr>`
      )
      .join('');

    const grouped = (items) =>
      items && items.length
        ? `<ul>${items.map((i) => `<li>${escapeHtml(i.text)} <span class="muted">(${i.count}x)</span></li>`).join('')}</ul>`
        : '<p>None noted.</p>';

    container.innerHTML = `
      <p class="muted">${data.submission_count} submission(s) analyzed.</p>
      <h3>Outcome Pass Rates (worst first)</h3>
      <table class="data-table">
        <thead><tr><th>Outcome</th><th>Pass Rate</th><th>Passed / Total</th></tr></thead>
        <tbody>${outcomeRows}</tbody>
      </table>
      <h3>Common Risky/Unnecessary Steps</h3>
      ${grouped(data.common_risky_steps)}
      <h3>Common Improvement Suggestions</h3>
      ${grouped(data.common_improvements)}
    `;
  } catch (err) {
    container.textContent = 'Could not load common-errors data.';
  }
}
