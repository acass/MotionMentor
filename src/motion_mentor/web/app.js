/**
 * MotionMentor - Interactive Web Dashboard & Playback Engine
 */

// MediaPipe Hand Connection Graph (21 landmarks)
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm transverse
  [5, 9], [9, 13], [13, 17]
];

// App State
const state = {
  activities: [],
  sessions: [],
  references: [],
  currentActivity: null,
  currentAttempt: null,
  currentReference: null,
  currentAssessment: null,

  // Landmarking & Telemetry Data
  expertLandmarks: [],
  traineeLandmarks: [],
  traineeFeatures: null,
  referenceEnvelope: null,

  // Playback Control
  isPlaying: false,
  currentTime: 0.0,
  duration: 5.0,
  playbackRate: 1.0,
  dtwWarpSync: true,
  warpingPath: [], // list of [i_ref, j_trainee]
  deviatingJointIndices: [5, 8], // default highlight joints
};

// DOM Elements
const dom = {
  activitySelect: document.getElementById('activitySelect'),
  attemptSelect: document.getElementById('attemptSelect'),
  referenceSelect: document.getElementById('referenceSelect'),
  btnRecompare: document.getElementById('btnRecompare'),
  btnToggleHistory: document.getElementById('btnToggleHistory'),
  btnPurgeAttempts: document.getElementById('btnPurgeAttempts'),
  btnPurgeExperts: document.getElementById('btnPurgeExperts'),
  historyModal: document.getElementById('historyModal'),
  btnCloseHistory: document.getElementById('btnCloseHistory'),
  historyTableBody: document.getElementById('historyTableBody'),

  // Videos & Canvas
  expertVideo: document.getElementById('expertVideo'),
  traineeVideo: document.getElementById('traineeVideo'),
  expertCanvas: document.getElementById('expertCanvas'),
  traineeCanvas: document.getElementById('traineeCanvas'),
  superimposedCanvas: document.getElementById('superimposedCanvas'),
  expertPlaceholder: document.getElementById('expertPlaceholder'),
  traineePlaceholder: document.getElementById('traineePlaceholder'),
  expertTelemetry: document.getElementById('expertTelemetry'),
  traineeTelemetry: document.getElementById('traineeTelemetry'),

  // Master Playback
  btnPlayPause: document.getElementById('btnPlayPause'),
  playIcon: document.getElementById('playIcon'),
  pauseIcon: document.getElementById('pauseIcon'),
  btnStepBack: document.getElementById('btnStepBack'),
  btnStepForward: document.getElementById('btnStepForward'),
  masterScrubber: document.getElementById('masterScrubber'),
  currentTimeDisplay: document.getElementById('currentTimeDisplay'),
  totalTimeDisplay: document.getElementById('totalTimeDisplay'),
  playbackSpeed: document.getElementById('playbackSpeed'),
  chkDtwWarp: document.getElementById('chkDtwWarp'),
  chkSuperimposed: document.getElementById('chkSuperimposed'),
  superimposedPanel: document.getElementById('superimposedPanel'),

  // Scorecard
  gaugeFill: document.getElementById('gaugeFill'),
  overallScoreNumber: document.getElementById('overallScoreNumber'),
  interpretationBandPill: document.getElementById('interpretationBandPill'),
  reliabilityPill: document.getElementById('reliabilityPill'),
  coverageVal: document.getElementById('coverageVal'),
  criticalAlert: document.getElementById('criticalAlert'),
  criticalText: document.getElementById('criticalText'),
  componentsList: document.getElementById('componentsList'),
  feedbackList: document.getElementById('feedbackList'),
  feedbackCountBadge: document.getElementById('feedbackCountBadge'),
  featureChartSelect: document.getElementById('featureChartSelect'),
  timelineChartCanvas: document.getElementById('timelineChartCanvas'),

  // Live Camera & Record Modal
  btnOpenRecordModal: document.getElementById('btnOpenRecordModal'),
  recordModal: document.getElementById('recordModal'),
  btnCloseRecordModal: document.getElementById('btnCloseRecordModal'),
  liveCameraVideo: document.getElementById('liveCameraVideo'),
  liveCamStatus: document.getElementById('liveCamStatus'),
  countdownOverlay: document.getElementById('countdownOverlay'),
  countdownNumber: document.getElementById('countdownNumber'),
  countdownSub: document.getElementById('countdownSub'),
  recBanner: document.getElementById('recBanner'),
  recCountdownTimer: document.getElementById('recCountdownTimer'),
  recordProgressBar: document.getElementById('recordProgressBar'),
  recordMsgArea: document.getElementById('recordMsgArea'),
  recordRole: document.getElementById('recordRole'),
  btnStartLiveRecord: document.getElementById('btnStartLiveRecord'),
  btnSyntheticRecord: document.getElementById('btnSyntheticRecord'),
};

// Recording State
const recordState = {
  liveStream: null,
  mediaRecorder: null,
  recordedChunks: [],
  isRecording: false,
};

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
  initEventListeners();
  await loadInitialData();
  requestAnimationFrame(renderLoop);
});

// Event Listeners
function initEventListeners() {
  dom.activitySelect.addEventListener('change', onActivityChange);
  dom.attemptSelect.addEventListener('change', onAttemptChange);
  dom.referenceSelect.addEventListener('change', onReferenceChange);
  dom.btnRecompare.addEventListener('click', runRecomparison);

  // Playback Controls
  dom.btnPlayPause.addEventListener('click', togglePlayPause);
  dom.btnStepBack.addEventListener('click', () => stepFrame(-1));
  dom.btnStepForward.addEventListener('click', () => stepFrame(1));
  dom.masterScrubber.addEventListener('input', onScrubberInput);
  dom.playbackSpeed.addEventListener('change', (e) => {
    state.playbackRate = parseFloat(e.target.value);
    dom.expertVideo.playbackRate = state.playbackRate;
    dom.traineeVideo.playbackRate = state.playbackRate;
  });

  dom.chkDtwWarp.addEventListener('change', (e) => {
    state.dtwWarpSync = e.target.checked;
  });

  dom.chkSuperimposed.addEventListener('change', (e) => {
    dom.superimposedPanel.style.display = e.target.checked ? 'block' : 'none';
  });

  // History Modal
  dom.btnToggleHistory.addEventListener('click', () => dom.historyModal.classList.remove('hidden'));
  dom.btnCloseHistory.addEventListener('click', () => dom.historyModal.classList.add('hidden'));

  if (dom.btnPurgeAttempts) {
    dom.btnPurgeAttempts.addEventListener('click', () => purgeSessions('trainee'));
  }
  if (dom.btnPurgeExperts) {
    dom.btnPurgeExperts.addEventListener('click', () => purgeSessions('expert'));
  }

  // Live Camera & Record Modal
  if (dom.btnOpenRecordModal) {
    dom.btnOpenRecordModal.addEventListener('click', openRecordModal);
  }
  if (dom.btnCloseRecordModal) {
    dom.btnCloseRecordModal.addEventListener('click', closeRecordModal);
  }
  if (dom.btnStartLiveRecord) {
    dom.btnStartLiveRecord.addEventListener('click', startLiveRecording);
  }
  if (dom.btnSyntheticRecord) {
    dom.btnSyntheticRecord.addEventListener('click', startSyntheticRecording);
  }

  // Feature Timeline Select
  dom.featureChartSelect.addEventListener('change', drawFeatureChart);

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') {
      e.preventDefault();
      togglePlayPause();
    } else if (e.key === '[') {
      stepFrame(-1);
    } else if (e.key === ']') {
      stepFrame(1);
    }
  });

  // Resize listener for canvas resolution matching
  window.addEventListener('resize', syncCanvasSizes);
}

function syncCanvasSizes() {
  [dom.expertCanvas, dom.traineeCanvas].forEach(canvas => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
  });
}

// API Data Fetching
async function loadInitialData() {
  try {
    const [acts, sess, refs] = await Promise.all([
      fetch('/api/activities').then(r => r.json()),
      fetch('/api/sessions').then(r => r.json()),
      fetch('/api/references').then(r => r.json()),
    ]);

    state.activities = acts;
    state.sessions = sess;
    state.references = refs;

    populateActivityDropdown();
    populateReferenceDropdown();
    populateAttemptDropdown();
    await onAttemptChange();
  } catch (err) {
    console.error('Failed to load initial data:', err);
  }
}

// Preview a purge, confirm the counts with the user, then delete.
async function purgeSessions(role) {
  const isExpert = role === 'expert';
  const btn = isExpert ? dom.btnPurgeExperts : dom.btnPurgeAttempts;
  const noun = isExpert ? 'expert capture' : 'trainee attempt';
  const original = btn.cloneNode(true); // icon + label, restored when we are done
  btn.disabled = true;
  try {
    btn.textContent = 'Checking...';
    const preview = await purgeRequest(role, false);

    if (preview.sessions.length === 0) {
      alert(`No ${noun}s to delete.`);
      return;
    }

    const lines = [
      `Permanently delete ${preview.sessions.length} ${noun}(s), ` +
      `${preview.references.length} reference profile(s), ` +
      `${preview.assessments.length} assessment(s) and ${preview.files.length} media file(s)?`,
      '',
      isExpert
        ? 'Every reference profile built from these captures goes too, so comparison will not work until a new expert capture is recorded.'
        : 'Expert captures and references are not touched.',
      'The database is backed up first.',
    ];
    if (!confirm(lines.join('\n'))) return;

    btn.textContent = 'Deleting...';
    const result = await purgeRequest(role, true);

    state.currentAttempt = null;
    if (isExpert) state.currentReference = null;
    await loadInitialData();
    alert(`Deleted ${result.deleted.sessions} ${noun}(s). Backup: ${result.backup}`);
  } catch (err) {
    console.error('Purge failed:', err);
    alert(`Purge failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.replaceChildren(...original.childNodes);
  }
}

async function purgeRequest(role, apply) {
  const res = await fetch(`/api/sessions/purge?role=${role}&apply=${apply}`, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

function populateActivityDropdown() {
  dom.activitySelect.innerHTML = '';
  state.activities.forEach(a => {
    const opt = document.createElement('option');
    opt.value = a.activity_id;
    opt.textContent = a.name.replace(/_/g, ' ').toUpperCase();
    dom.activitySelect.appendChild(opt);
  });
  if (state.activities.length > 0) {
    state.currentActivity = state.activities[0];
  }
}

function populateReferenceDropdown() {
  dom.referenceSelect.innerHTML = '';
  const activityReferences = state.currentActivity
    ? state.references.filter(r => r.activity_id === state.currentActivity.activity_id)
    : state.references;

  activityReferences.forEach((r) => {
    const opt = document.createElement('option');
    opt.value = r.reference_id;
    const representative = r.representative_session;
    const sourceLabel = {
      recorded: 'RECORDED',
      generated: 'GENERATED',
    }[representative?.capture_kind] || 'UNAVAILABLE';
    const representativeId = representative?.session_id
      ? representative.session_id.substring(0, 8)
      : 'unknown';
    opt.textContent = `${sourceLabel} · ${representativeId} · ${r.total_demonstrations} take${r.total_demonstrations > 1 ? 's' : ''} · ${r.duration_mean_sec}s`;
    dom.referenceSelect.appendChild(opt);
  });
  state.currentReference = activityReferences[0] || null;
}

function populateAttemptDropdown() {
  dom.attemptSelect.innerHTML = '';
  const activitySessions = state.currentActivity
    ? state.sessions.filter(s => s.activity_id === state.currentActivity.activity_id)
    : state.sessions;
  const trainees = activitySessions.filter(s => s.role === 'trainee');
  const items = trainees.length > 0 ? trainees : activitySessions;

  items.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s.session_id;
    const dateStr = s.started_at ? new Date(s.started_at).toLocaleTimeString() : '';
    const cov = s.quality_summary ? `${s.quality_summary.detection_coverage_pct.toFixed(0)}% cov` : '';
    opt.textContent = `[${s.role.toUpperCase()}] ${s.session_id.substring(0, 8)} (${s.duration_seconds}s, ${cov}) ${dateStr}`;
    dom.attemptSelect.appendChild(opt);
  });
}

// Selection Handlers
async function onActivityChange() {
  const actId = dom.activitySelect.value;
  state.currentActivity = state.activities.find(a => a.activity_id === actId);
  populateReferenceDropdown();
  populateAttemptDropdown();
  await onAttemptChange();
}

async function onReferenceChange() {
  const refId = dom.referenceSelect.value;
  state.currentReference = state.references.find(r => r.reference_id === refId);
  await loadExpertSessionData();
  if (state.currentAttempt) {
    await loadAssessment(state.currentAttempt.session_id, state.currentReference?.reference_id);
  }
}

async function onAttemptChange() {
  const sessId = dom.attemptSelect.value;
  const attempt = state.sessions.find(s => s.session_id === sessId);
  if (!attempt) {
    clearAttemptData();
    await loadExpertSessionData();
    return;
  }
  state.currentAttempt = attempt;

  // Set duration
  state.duration = Math.max(state.currentAttempt.duration_seconds || 5.0, 1.0);
  dom.totalTimeDisplay.textContent = formatTime(state.duration);
  dom.masterScrubber.max = Math.floor(state.duration * 30);

  // Setup Videos
  setupVideoSources();

  // Load Landmarks & Features
  await Promise.all([
    loadLandmarks(state.currentAttempt.session_id, 'trainee'),
    loadExpertSessionData(),
    loadAssessment(state.currentAttempt.session_id, state.currentReference?.reference_id),
    loadFeatures(state.currentAttempt.session_id),
  ]);

  updateQualityAndMetaDisplay();
  drawFeatureChart();
  populateHistoryTable();
  syncCanvasSizes();
}

function clearAttemptData() {
  state.currentAttempt = null;
  state.currentAssessment = null;
  state.traineeLandmarks = [];
  state.traineeFeatures = null;
  state.warpingPath = [];
  state.currentTime = 0;
  state.duration = 5.0;

  if (state.isPlaying) togglePlayPause();
  clearVideo(dom.traineeVideo, dom.traineePlaceholder, 'No Trainee Attempt Available');
  dom.coverageVal.textContent = '--';
  dom.currentTimeDisplay.textContent = formatTime(0);
  dom.totalTimeDisplay.textContent = formatTime(state.duration);
  dom.masterScrubber.value = 0;
  dom.masterScrubber.max = Math.floor(state.duration * 30);

  dom.overallScoreNumber.textContent = '--';
  dom.gaugeFill.style.strokeDashoffset = 314.16;
  dom.interpretationBandPill.textContent = 'No Attempt';
  dom.interpretationBandPill.className = 'band-pill';
  dom.reliabilityPill.textContent = '--';
  dom.reliabilityPill.className = 'reliability-pill';
  dom.criticalAlert.classList.add('hidden');
  dom.componentsList.innerHTML = '';
  dom.feedbackCountBadge.textContent = '0 tips';
  dom.feedbackList.innerHTML = '<div style="font-size: 0.75rem; color: var(--text-muted); padding: 8px;">Select a trainee attempt to view coaching feedback.</div>';
}

function clearVideo(video, placeholder, message) {
  video.pause();
  video.removeAttribute('src');
  video.load();
  placeholder.classList.remove('hidden');
  placeholder.textContent = message;
}

function setupVideoSources() {
  if (state.currentAttempt && state.currentAttempt.video_path) {
    const filename = state.currentAttempt.video_path.split('/').pop();
    dom.traineeVideo.src = `/api/videos/${filename}`;
    dom.traineePlaceholder.classList.add('hidden');
    dom.traineeVideo.load();
  } else {
    dom.traineePlaceholder.classList.remove('hidden');
    dom.traineePlaceholder.textContent = 'No Video Available (Synthetic / Landmarks Only)';
  }
}

async function loadExpertSessionData() {
  state.referenceEnvelope = null;
  state.expertLandmarks = [];
  clearVideo(
    dom.expertVideo,
    dom.expertPlaceholder,
    'No Recorded Expert Reference Available',
  );

  if (!state.currentReference) return;
  try {
    const refData = await fetch(`/api/references/${state.currentReference.reference_id}`).then(r => r.json());
    state.referenceEnvelope = refData.envelope;

    const medoidId = state.currentReference.medoid_session_id;
    if (medoidId) {
      const expertSession = await fetch(`/api/sessions/${medoidId}`).then(r => r.json());
      if (expertSession && expertSession.video_path) {
        const filename = expertSession.video_path.split('/').pop();
        dom.expertVideo.src = `/api/videos/${filename}`;
        dom.expertPlaceholder.classList.add('hidden');
        dom.expertVideo.load();
      }
      await loadLandmarks(medoidId, 'expert');
    }
  } catch (err) {
    console.error('Failed loading expert session:', err);
  }
}

async function loadLandmarks(sessionId, role) {
  try {
    const landmarks = await fetch(`/api/sessions/${sessionId}/landmarks`).then(r => r.json());
    if (role === 'expert') {
      state.expertLandmarks = landmarks;
    } else {
      state.traineeLandmarks = landmarks;
    }
  } catch (err) {
    console.warn(`Landmarks not found for ${sessionId}:`, err);
  }
}

async function loadFeatures(sessionId) {
  try {
    const feat = await fetch(`/api/sessions/${sessionId}/features`).then(r => r.json());
    state.traineeFeatures = feat.records;
  } catch (err) {
    console.warn('Features not available:', err);
  }
}

async function loadAssessment(attemptSessionId, referenceId) {
  try {
    const query = new URLSearchParams({ attempt_session_id: attemptSessionId });
    if (referenceId) query.set('reference_id', referenceId);
    const assessments = await fetch(`/api/assessments?${query}`).then(r => r.json());
    if (assessments && assessments.length > 0) {
      state.currentAssessment = assessments[0];
      renderScorecard(state.currentAssessment);
    } else {
      // Auto-run comparison
      await runRecomparison();
    }
  } catch (err) {
    console.error('Error fetching assessment:', err);
  }
}

async function runRecomparison() {
  if (!state.currentAttempt) return;
  dom.btnRecompare.disabled = true;
  dom.btnRecompare.textContent = 'Analyzing...';

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        attempt_session_id: state.currentAttempt.session_id,
        reference_id: state.currentReference?.reference_id,
      })
    });
    if (res.ok) {
      state.currentAssessment = await res.json();
      renderScorecard(state.currentAssessment);
    }
  } catch (err) {
    console.error('Re-comparison failed:', err);
  } finally {
    dom.btnRecompare.disabled = false;
    dom.btnRecompare.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
      </svg>
      Re-Evaluate
    `;
  }
}

// Render Scorecard UI
function renderScorecard(assessment) {
  if (!assessment) return;

  const score = assessment.overall_score || 0.0;
  dom.overallScoreNumber.textContent = score.toFixed(1);

  // Radial Gauge Stroke (circumference = 2 * PI * 50 ~= 314.16)
  const circumference = 314.16;
  const offset = circumference - (score / 100) * circumference;
  dom.gaugeFill.style.strokeDashoffset = offset;

  // Band styling
  const band = assessment.interpretation_band || 'Evaluating';
  dom.interpretationBandPill.textContent = band;
  dom.interpretationBandPill.className = 'band-pill';

  if (band.includes('Excellent')) {
    dom.interpretationBandPill.classList.add('band-excellent');
    dom.gaugeFill.style.stroke = 'var(--color-success)';
  } else if (band.includes('Good')) {
    dom.interpretationBandPill.classList.add('band-good');
    dom.gaugeFill.style.stroke = 'var(--accent-cyan)';
  } else if (band.includes('Developing')) {
    dom.interpretationBandPill.classList.add('band-developing');
    dom.gaugeFill.style.stroke = 'var(--color-warning)';
  } else {
    dom.interpretationBandPill.classList.add('band-review');
    dom.gaugeFill.style.stroke = 'var(--color-danger)';
  }

  // Reliability
  const rel = (assessment.reliability || 'HIGH').toUpperCase();
  dom.reliabilityPill.textContent = rel;
  dom.reliabilityPill.className = `reliability-pill ${rel === 'HIGH' ? 'rel-high' : 'rel-low'}`;

  // Critical alert
  if (assessment.critical_failures && assessment.critical_failures.length > 0) {
    dom.criticalAlert.classList.remove('hidden');
    dom.criticalText.innerHTML = assessment.critical_failures.join('<br>');
  } else {
    dom.criticalAlert.classList.add('hidden');
  }

  // Component Dimensions
  renderComponentScores(assessment.component_scores);

  // Coaching Feedback
  renderFeedbackItems(assessment.feedback);
}

function renderComponentScores(comps) {
  if (!comps) return;
  dom.componentsList.innerHTML = '';

  const definitions = [
    { key: 'pose', label: 'Local Hand Pose', weight: '25%' },
    { key: 'trajectory', label: 'Global Trajectory', weight: '25%' },
    { key: 'orientation', label: 'Hand Orientation', weight: '15%' },
    { key: 'timing', label: 'Timing & Duration', weight: '15%' },
    { key: 'smoothness', label: 'Speed & Smoothness', weight: '10%' },
    { key: 'sequence', label: 'Sequence & Checkpoints', weight: '10%' },
  ];

  definitions.forEach(d => {
    const val = comps[d.key] || 0.0;
    const color = val >= 90 ? 'var(--color-success)' : val >= 75 ? 'var(--accent-cyan)' : val >= 60 ? 'var(--color-warning)' : 'var(--color-danger)';

    const item = document.createElement('div');
    item.className = 'component-item';
    item.innerHTML = `
      <div class="comp-header">
        <span class="comp-name">${d.label} <small style="color: var(--text-muted)">(${d.weight})</small></span>
        <span class="comp-score" style="color: ${color}">${val.toFixed(1)}</span>
      </div>
      <div class="comp-bar-bg">
        <div class="comp-bar-fill" style="width: ${val}%; background: ${color}"></div>
      </div>
    `;
    dom.componentsList.appendChild(item);
  });
}

function renderFeedbackItems(feedbackList) {
  dom.feedbackList.innerHTML = '';
  if (!feedbackList || feedbackList.length === 0) {
    dom.feedbackCountBadge.textContent = '0 tips';
    dom.feedbackList.innerHTML = `<div style="font-size: 0.75rem; color: var(--text-muted); padding: 8px;">No critical deviations detected.</div>`;
    return;
  }

  dom.feedbackCountBadge.textContent = `${feedbackList.length} tips`;

  feedbackList.forEach(item => {
    const card = document.createElement('div');
    const sevClass = item.severity === 'high' ? 'high-sev' : item.severity === 'medium' ? 'med-sev' : 'low-sev';
    card.className = `feedback-card ${sevClass}`;

    card.innerHTML = `
      <div class="feedback-top">
        <span class="feedback-tag">[${item.component}]</span>
        <button class="btn-jump" data-time="${item.time_sec || 0.0}">
          Jump to ${item.time_sec ? item.time_sec.toFixed(1) + 's' : '0.0s'}
        </button>
      </div>
      <div class="feedback-msg">${item.message}</div>
      ${item.recommendation ? `<div class="feedback-tip">Tip: ${item.recommendation}</div>` : ''}
    `;

    card.querySelector('.btn-jump').addEventListener('click', () => {
      seekToTime(item.time_sec || 0.0);
    });

    dom.feedbackList.appendChild(card);
  });
}

function updateQualityAndMetaDisplay() {
  if (state.currentAttempt && state.currentAttempt.quality_summary) {
    const q = state.currentAttempt.quality_summary;
    dom.coverageVal.textContent = `${q.detection_coverage_pct.toFixed(0)}% (${q.effective_fps.toFixed(1)} FPS)`;
  } else {
    dom.coverageVal.textContent = '--';
  }
}

// Master Transport & Sync Playback
function togglePlayPause() {
  state.isPlaying = !state.isPlaying;
  if (state.isPlaying) {
    dom.playIcon.classList.add('hidden');
    dom.pauseIcon.classList.remove('hidden');
    if (state.currentTime >= state.duration) {
      seekToTime(0);
    }
    dom.expertVideo.play().catch(() => {});
    dom.traineeVideo.play().catch(() => {});
  } else {
    dom.playIcon.classList.remove('hidden');
    dom.pauseIcon.classList.add('hidden');
    dom.expertVideo.pause();
    dom.traineeVideo.pause();
  }
}

function stepFrame(frames) {
  if (state.isPlaying) togglePlayPause();
  const dt = frames * (1.0 / 30.0);
  seekToTime(Math.max(0, Math.min(state.duration, state.currentTime + dt)));
}

function onScrubberInput(e) {
  const frame = parseInt(e.target.value);
  const t = frame / 30.0;
  seekToTime(t);
}

function seekToTime(targetTime) {
  state.currentTime = Math.max(0, Math.min(state.duration, targetTime));
  dom.masterScrubber.value = Math.floor(state.currentTime * 30);
  dom.currentTimeDisplay.textContent = formatTime(state.currentTime);

  // Sync Videos
  dom.traineeVideo.currentTime = state.currentTime;

  if (state.dtwWarpSync && dom.expertVideo.duration) {
    // Map time through warping path if duration varies
    const expertDur = dom.expertVideo.duration || state.duration;
    const progress = state.currentTime / Math.max(0.1, state.duration);
    dom.expertVideo.currentTime = progress * expertDur;
  } else {
    dom.expertVideo.currentTime = state.currentTime;
  }

  dom.traineeTelemetry.textContent = `${state.currentTime.toFixed(1)}s / Frame ${Math.floor(state.currentTime * 30)}`;
  dom.expertTelemetry.textContent = `${dom.expertVideo.currentTime.toFixed(1)}s / Frame ${Math.floor(dom.expertVideo.currentTime * 30)}`;
}

// Main Render Loop (60 FPS)
let lastTimestamp = performance.now();

function renderLoop(now) {
  const dt = (now - lastTimestamp) / 1000.0;
  lastTimestamp = now;

  if (state.isPlaying) {
    state.currentTime += dt * state.playbackRate;
    if (state.currentTime >= state.duration) {
      state.currentTime = state.duration;
      togglePlayPause();
    }
    dom.masterScrubber.value = Math.floor(state.currentTime * 30);
    dom.currentTimeDisplay.textContent = formatTime(state.currentTime);
    dom.traineeTelemetry.textContent = `${state.currentTime.toFixed(1)}s / Frame ${Math.floor(state.currentTime * 30)}`;
    dom.expertTelemetry.textContent = `${dom.expertVideo.currentTime.toFixed(1)}s / Frame ${Math.floor(dom.expertVideo.currentTime * 30)}`;
  }

  // Draw overlay skeletons on videos
  const expertDur = dom.expertVideo.duration || state.currentReference?.duration_mean_sec || state.duration;
  let expertTime = (dom.expertVideo.duration && !dom.expertVideo.paused && !state.dtwWarpSync)
    ? dom.expertVideo.currentTime
    : (state.currentTime / Math.max(0.1, state.duration)) * expertDur;

  drawOverlaySkeleton(dom.expertCanvas, state.expertLandmarks, expertTime, 'cyan');
  drawOverlaySkeleton(dom.traineeCanvas, state.traineeLandmarks, state.currentTime, 'coral');

  // Draw superimposed 3D ghost canvas
  drawSuperimposedSkeleton(expertTime);

  requestAnimationFrame(renderLoop);
}

// Landmark Interpolation & Drawing
function getLandmarkFrame(landmarkRecords, targetTimeSec) {
  if (!landmarkRecords || landmarkRecords.length === 0) return null;

  const t0 = landmarkRecords[0].timestamp_ms;
  const tn = landmarkRecords[landmarkRecords.length - 1].timestamp_ms;
  const totalDurationMs = tn - t0;
  const targetElapsedMs = Math.max(0, targetTimeSec * 1000.0);

  let closest = landmarkRecords[0];
  let minDiff = Infinity;

  for (let i = 0; i < landmarkRecords.length; i++) {
    const rec = landmarkRecords[i];
    // Calculate elapsed time from the start of the recording
    const elapsedMs = (totalDurationMs > 0) ? (rec.timestamp_ms - t0) : (i * (1000.0 / 30.0));
    const diff = Math.abs(elapsedMs - targetElapsedMs);
    if (diff < minDiff) {
      minDiff = diff;
      closest = rec;
    } else if (diff > minDiff) {
      // Past the closest frame
      break;
    }
  }

  return closest;
}

function drawOverlaySkeleton(canvas, landmarkRecords, timeSec, style) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const frame = getLandmarkFrame(landmarkRecords, timeSec);
  if (!frame || !frame.hands || frame.hands.length === 0) return;

  const hand = frame.hands[0];
  const pts = hand.landmarks;
  if (!pts || pts.length < 21) return;

  const w = canvas.width;
  const h = canvas.height;
  const mainColor = style === 'cyan' ? '#00f0ff' : '#ff5376';

  // Draw bones
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = mainColor;
  ctx.lineCap = 'round';

  HAND_CONNECTIONS.forEach(([i, j]) => {
    ctx.beginPath();
    ctx.moveTo(pts[i][0] * w, pts[i][1] * h);
    ctx.lineTo(pts[j][0] * w, pts[j][1] * h);
    ctx.stroke();
  });

  // Draw landmark points
  pts.forEach((pt, idx) => {
    ctx.beginPath();
    ctx.arc(pt[0] * w, pt[1] * h, idx === 4 || idx === 8 ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = idx === 4 || idx === 8 ? '#ffffff' : mainColor;
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function drawSuperimposedSkeleton(expertTime) {
  const canvas = dom.superimposedCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const w = canvas.width;
  const h = canvas.height;

  // Grid background
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const expTime = (expertTime !== undefined) ? expertTime : dom.expertVideo.currentTime;
  const expFrame = getLandmarkFrame(state.expertLandmarks, expTime);
  const traineeFrame = getLandmarkFrame(state.traineeLandmarks, state.currentTime);

  // 1. Draw Expert Ghost Skeleton (Cyan glow)
  if (expFrame && expFrame.hands && expFrame.hands.length > 0) {
    const pts = expFrame.hands[0].landmarks;
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
    ctx.lineWidth = 3;
    ctx.shadowColor = 'rgba(0, 240, 255, 0.8)';
    ctx.shadowBlur = 8;

    HAND_CONNECTIONS.forEach(([i, j]) => {
      ctx.beginPath();
      ctx.moveTo(pts[i][0] * w, pts[i][1] * h);
      ctx.lineTo(pts[j][0] * w, pts[j][1] * h);
      ctx.stroke();
    });

    pts.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#00f0ff';
      ctx.fill();
    });
    ctx.restore();
  }

  // 2. Draw Trainee Skeleton (Coral with Deviating Rings)
  if (traineeFrame && traineeFrame.hands && traineeFrame.hands.length > 0) {
    const pts = traineeFrame.hands[0].landmarks;
    ctx.save();
    ctx.strokeStyle = '#ff5376';
    ctx.lineWidth = 2.5;

    HAND_CONNECTIONS.forEach(([i, j]) => {
      ctx.beginPath();
      ctx.moveTo(pts[i][0] * w, pts[i][1] * h);
      ctx.lineTo(pts[j][0] * w, pts[j][1] * h);
      ctx.stroke();
    });

    pts.forEach((pt, idx) => {
      const isDeviating = state.deviatingJointIndices.includes(idx);
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, isDeviating ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = isDeviating ? '#ef4444' : '#ffffff';
      ctx.fill();

      // Pulsing error ring
      if (isDeviating) {
        ctx.beginPath();
        const pulse = 8 + Math.sin(performance.now() / 150) * 3;
        ctx.arc(pt[0] * w, pt[1] * h, pulse, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
    ctx.restore();
  }
}

// Feature Mini Timeline Chart
function drawFeatureChart() {
  const canvas = dom.timelineChartCanvas;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const featKey = dom.featureChartSelect.value;
  if (!state.traineeFeatures || state.traineeFeatures.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;
  const records = state.traineeFeatures;
  const n = records.length;

  // Find min and max
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let r of records) {
    const v = r[featKey] || 0.0;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  const range = Math.max(0.001, maxVal - minVal);

  // Background gradient
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(0, 240, 255, 0.25)');
  grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

  // Plot curve
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w;
    const val = records[i][featKey] || 0.0;
    const y = h - ((val - minVal) / range) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Playhead line
  const progress = state.currentTime / Math.max(0.1, state.duration);
  const playheadX = progress * w;
  ctx.beginPath();
  ctx.moveTo(playheadX, 0);
  ctx.lineTo(playheadX, h);
  ctx.strokeStyle = '#ff5376';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// Performance History Table
function populateHistoryTable() {
  dom.historyTableBody.innerHTML = '';
  state.sessions.forEach(s => {
    const tr = document.createElement('tr');
    const dateStr = s.started_at ? new Date(s.started_at).toLocaleString() : 'N/A';
    const cov = s.quality_summary ? `${s.quality_summary.detection_coverage_pct.toFixed(0)}%` : '--';
    const dur = `${s.duration_seconds.toFixed(1)}s`;

    tr.innerHTML = `
      <td>${dateStr}</td>
      <td><span class="header-badge ${s.role === 'expert' ? 'expert-badge' : 'trainee-badge'}">${s.role.toUpperCase()}</span></td>
      <td>${dur}</td>
      <td>${cov}</td>
      <td><strong style="color: var(--accent-cyan);">--</strong></td>
      <td><span class="sub-badge">Recorded</span></td>
      <td><button class="btn btn-outline" style="padding: 3px 8px; font-size: 0.7rem;" data-id="${s.session_id}">Load</button></td>
    `;

    tr.querySelector('button').addEventListener('click', () => {
      dom.attemptSelect.value = s.session_id;
      dom.historyModal.classList.add('hidden');
      onAttemptChange();
    });

    dom.historyTableBody.appendChild(tr);
  });
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
}

// ==========================================================================
// Live Camera & Recording Functions
// ==========================================================================

async function openRecordModal() {
  if (!dom.recordModal) return;
  dom.recordModal.classList.remove('hidden');
  dom.recordProgressBar.style.width = '0%';
  dom.countdownOverlay.classList.add('hidden');
  dom.recBanner.classList.add('hidden');
  dom.btnStartLiveRecord.disabled = true;
  dom.btnSyntheticRecord.disabled = false;
  dom.recordMsgArea.textContent = 'Requesting camera access from browser...';

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    });
    recordState.liveStream = stream;
    dom.liveCameraVideo.srcObject = stream;
    dom.liveCamStatus.innerHTML = '<span class="status-dot green-dot"></span> Camera Active (30 FPS)';
    dom.recordMsgArea.textContent = 'Position your hand inside the guide, then click "Start 5s Recording".';
    dom.btnStartLiveRecord.disabled = false;
  } catch (err) {
    console.warn('Camera access denied or unavailable:', err);
    dom.liveCamStatus.innerHTML = '<span class="status-dot coral-dot"></span> Camera Inactive';
    dom.recordMsgArea.textContent = 'Camera permission was not granted or no webcam was found. You can allow permissions in your browser or click "Quick Synthetic Take".';
    dom.btnStartLiveRecord.disabled = true;
  }
}

function closeRecordModal() {
  if (!dom.recordModal) return;
  dom.recordModal.classList.add('hidden');
  if (recordState.liveStream) {
    recordState.liveStream.getTracks().forEach(t => t.stop());
    recordState.liveStream = null;
  }
  recordState.isRecording = false;
  dom.countdownOverlay.classList.add('hidden');
  dom.recBanner.classList.add('hidden');
}

async function startLiveRecording() {
  if (!recordState.liveStream || recordState.isRecording) return;

  const captureRole = dom.recordRole.value;

  dom.btnStartLiveRecord.disabled = true;
  dom.btnSyntheticRecord.disabled = true;
  recordState.isRecording = true;

  // 3-Second Countdown
  dom.countdownOverlay.classList.remove('hidden');
  for (let count = 3; count >= 1; count--) {
    dom.countdownNumber.textContent = count;
    dom.countdownSub.textContent = count === 1 ? 'Ready...' : 'Get Ready!';
    await new Promise(r => setTimeout(r, 1000));
  }
  dom.countdownOverlay.classList.add('hidden');

  // Start MediaRecorder
  recordState.recordedChunks = [];
  let mimeType = 'video/webm';
  if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
    mimeType = 'video/webm;codecs=vp9';
  } else if (MediaRecorder.isTypeSupported('video/mp4')) {
    mimeType = 'video/mp4';
  }

  try {
    recordState.mediaRecorder = new MediaRecorder(recordState.liveStream, { mimeType });
  } catch (e) {
    recordState.mediaRecorder = new MediaRecorder(recordState.liveStream);
    mimeType = recordState.mediaRecorder.mimeType || 'video/webm';
  }

  recordState.mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      recordState.recordedChunks.push(e.data);
    }
  };

  recordState.mediaRecorder.start(100);

  // Active Recording Timer & Progress Bar (5.0s)
  dom.recBanner.classList.remove('hidden');
  const recordDuration = 5.0;
  const startTime = performance.now();

  const timerInterval = setInterval(() => {
    const elapsed = (performance.now() - startTime) / 1000.0;
    const remaining = Math.max(0.0, recordDuration - elapsed);
    dom.recCountdownTimer.textContent = remaining.toFixed(1);
    const pct = Math.min(100, (elapsed / recordDuration) * 100);
    dom.recordProgressBar.style.width = `${pct}%`;

    if (elapsed >= recordDuration) {
      clearInterval(timerInterval);
    }
  }, 50);

  await new Promise(r => setTimeout(r, recordDuration * 1000));
  clearInterval(timerInterval);

  dom.recBanner.classList.add('hidden');
  dom.recordMsgArea.innerHTML = captureRole === 'expert'
    ? '<span class="status-dot cyan-dot"></span> Processing expert take and rebuilding the reference profile...'
    : '<span class="status-dot cyan-dot"></span> Processing video with MediaPipe 3D Hand Landmarker & DTW...';

  recordState.mediaRecorder.onstop = async () => {
    const blob = new Blob(recordState.recordedChunks, { type: mimeType });
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Data = reader.result;
      try {
        const actId = state.currentActivity ? state.currentActivity.activity_id : 'reach-and-pinch-001';
        const res = await fetch('/api/record_upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            video_base64: base64Data,
            mime_type: mimeType,
            activity_id: actId,
            role: captureRole,
            participant_id: 'browser-user',
          }),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Capture processing failed');
        }

        closeRecordModal();

        await loadInitialData();

        // A capture that produced no reference or no assessment is not a success,
        // so say what blocked it instead of showing a checkmark over an empty
        // evaluation panel.
        const reasons = data.session?.quality_summary?.status_reasons || [];
        const reasonText = reasons.length ? ` Reasons: ${reasons.join(' ')}` : '';
        const ok = captureRole === 'expert' ? Boolean(data.reference) : Boolean(data.assessment);

        let message;
        if (captureRole === 'expert') {
          message = ok
            ? `✅ New Expert Take Captured! Reference rebuilt from ${data.reference.total_demonstrations} expert take${data.reference.total_demonstrations === 1 ? '' : 's'}.`
            : `⚠️ Expert take saved, but no reference was built, so trainee attempts cannot be scored yet.${reasonText} Re-record with your hand fully visible for the whole take.`;
        } else if (ok) {
          message = `✅ New Attempt Captured! Score: ${data.assessment.overall_score.toFixed(1)} / 100 (${data.assessment.interpretation_band}). Review synchronized ghost overlay below.`;
        } else if (state.references.length === 0) {
          message = `⚠️ Attempt saved, but there is no expert reference to score against, so the evaluation area stays empty. Record a quality-passing expert take first.${reasonText}`;
        } else {
          message = `⚠️ Attempt saved, but it could not be scored.${reasonText}`;
        }

        dom.criticalAlert.classList.remove('hidden');
        dom.criticalText.textContent = message;
      } catch (err) {
        console.error('Record upload error:', err);
        dom.recordMsgArea.textContent = `Upload failed: ${err.message}`;
        dom.btnStartLiveRecord.disabled = false;
        dom.btnSyntheticRecord.disabled = false;
        recordState.isRecording = false;
      }
    };
    reader.readAsDataURL(blob);
  };

  recordState.mediaRecorder.stop();
}

async function startSyntheticRecording() {
  // A synthetic take is drawn, not captured, so it can only ever be a trainee
  // attempt. The role select is shared with live recording, where expert is valid.
  const requestedRole = dom.recordRole.value;
  const captureRole = 'trainee';
  dom.btnStartLiveRecord.disabled = true;
  dom.btnSyntheticRecord.disabled = true;
  dom.recordMsgArea.innerHTML = `<span class="status-dot cyan-dot"></span> Generating synthetic ${captureRole} take...`;

  try {
    const actId = state.currentActivity ? state.currentActivity.activity_id : 'reach-and-pinch-001';
    const res = await fetch('/api/record_synthetic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        activity_id: actId,
        duration: 5.0,
        role: captureRole,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Synthetic take failed');

    closeRecordModal();

    await loadInitialData();

    dom.criticalAlert.classList.remove('hidden');
    dom.criticalText.textContent = requestedRole === 'expert'
      ? 'Synthetic Attempt Loaded. Note: synthetic takes are generated, not recorded, so they cannot serve as an expert reference — record a real capture for that.'
      : 'Synthetic Attempt Loaded! Ready for playback review.';
  } catch (err) {
    console.error('Synthetic take failed:', err);
    dom.recordMsgArea.textContent = `Synthetic generation failed: ${err.message}`;
    dom.btnStartLiveRecord.disabled = false;
    dom.btnSyntheticRecord.disabled = false;
  }
}
