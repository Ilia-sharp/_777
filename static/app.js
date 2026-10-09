(() => {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const defaultState = [0, 0, 0, 0];
  const simulation = {
    sessionId: makeSessionId(), seed: 116, state: [...defaultState], targetState: [...defaultState],
    hasReset: false, running: false, busy: false, resetPending: false, online: false, trainEnabled: false,
    operationRunning: false, mode: 'agent', manualAction: 0, step: 0, totalReward: 0,
    action: null, engine: 'CartPole-v1', generation: 0, loopToken: 0,
    animation: null, lastFrameAt: 0, runStartedAt: 0, history: [], activity: [],
    activityHidden: false, healthFailures: 0, evaluationStartedAt: 0, lastResponse: null,
  };
  let canvasContext = null;
  let canvasWidth = 0;
  let canvasHeight = 0;
  let lastHudUpdate = 0;

  function makeSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  async function api(url, options = {}, timeout = 15000) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      });
      let body = {};
      try { body = await response.json(); } catch (_error) { body = {}; }
      if (!response.ok) {
        const error = new Error(typeof body.detail === 'string' ? body.detail : 'Не удалось выполнить запрос. Попробуйте ещё раз.');
        error.status = response.status;
        throw error;
      }
      return body;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Сервис отвечает слишком долго. Проверьте соединение и повторите попытку.');
      if (error instanceof TypeError) throw new Error('Не удалось связаться с сервисом. Проверьте соединение.');
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function toast(message, kind = 'error') {
    const item = document.createElement('div');
    item.className = 'toast';
    item.dataset.kind = kind;
    item.textContent = message;
    byId('toast-region').appendChild(item);
    window.setTimeout(() => item.remove(), 4300);
  }

  function setServiceStatus(status, text) {
    byId('service-status').dataset.state = status;
    byId('status-text').textContent = text;
  }

  async function checkHealth() {
    try {
      const health = await api('/health', {}, 85000);
      simulation.healthFailures = 0;
      simulation.online = Boolean(health.model_loaded);
      if (health.model_version) byId('model-version').textContent = health.model_version;
      if (health.model_loaded) setServiceStatus('online', 'Онлайн');
      else setServiceStatus('offline', 'Модель не загружена');
    } catch (_error) {
      simulation.healthFailures += 1;
      if (simulation.healthFailures === 1) {
        setServiceStatus('sleeping', 'Просыпается…');
        byId('footer-message').textContent = 'Сервер просыпается. Это может занять до минуты.';
        toast('Сервер просыпается. Через несколько секунд попробуем подключиться снова.', 'info');
      } else if (simulation.healthFailures >= 4) setServiceStatus('offline', 'Недоступен');
      else setServiceStatus('sleeping', 'Просыпается…');
      simulation.online = false;
    }
    updateButtons();
  }

  function updateButtons() {
    byId('run-button').disabled = !simulation.online || !simulation.hasReset || !byId('terminal-state').hidden;
    byId('reset-button').disabled = !simulation.online || simulation.resetPending;
    byId('reset-small').disabled = !simulation.online || simulation.resetPending;
    byId('evaluate-button').disabled = !simulation.online || simulation.operationRunning;
    byId('train-button').disabled = !simulation.online || !simulation.trainEnabled || simulation.operationRunning;
    byId('run-icon').textContent = simulation.running ? 'Ⅱ' : '▶';
    byId('run-label').textContent = simulation.running ? 'Приостановить симуляцию' : 'Запустить симуляцию';
    byId('simulation-dot').classList.toggle('is-running', simulation.running);
    byId('simulation-state-label').textContent = simulation.running ? 'СИМУЛЯЦИЯ ИДЁТ' : simulation.hasReset ? 'СРЕДА НА ПАУЗЕ' : 'ГОТОВО К ЗАПУСКУ';
    byId('physics-status').textContent = simulation.engine.includes('offline') ? 'СОВМЕСТИМАЯ ФИЗИКА' : 'ФИЗИКА В РЕАЛЬНОМ ВРЕМЕНИ';
    byId('train-explanation').textContent = simulation.trainEnabled
      ? 'Кандидат обучается отдельно и не заменяет рабочую модель до проверки.'
      : 'На бесплатном сервере обучение отключено. Модель обучена заранее в симуляторе.';
    byId('train-status-title').textContent = simulation.trainEnabled ? 'Обучение доступно' : 'Модель готова';
  }

  function formatDuration(seconds) {
    const safe = Math.max(0, Math.floor(Number(seconds) || 0));
    if (safe < 60) return `${safe} с`;
    return `${Math.floor(safe / 60)} мин ${safe % 60} с`;
  }

  function safeNumber(value, digits = 3) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '0'.repeat(digits + 1);
  }

  function displayState(values, animate = true) {
    const state = Array.isArray(values) ? values : defaultState;
    const thetaDegrees = (Number(state[2]) || 0) * 180 / Math.PI;
    const items = [
      { id: 'state-x', bar: 'bar-x', value: Number(state[0]) || 0, text: `${safeNumber(state[0])} <small>м</small>`, bound: 2.4 },
      { id: 'state-xdot', bar: 'bar-xdot', value: Number(state[1]) || 0, text: `${safeNumber(state[1])} <small>м/с</small>`, bound: 3.0 },
      { id: 'state-theta', bar: 'bar-theta', value: thetaDegrees, text: `${safeNumber(thetaDegrees, 2)} <small>°</small>`, bound: 12 },
      { id: 'state-thetadot', bar: 'bar-thetadot', value: Number(state[3]) || 0, text: `${safeNumber(state[3])} <small>рад/с</small>`, bound: 4.0 },
    ];
    items.forEach((item) => {
      byId(item.id).innerHTML = item.text;
      const magnitude = Math.min(1, Math.abs(item.value) / item.bound) * 50;
      const bar = byId(item.bar);
      bar.style.left = item.value < 0 ? `${50 - magnitude}%` : '50%';
      bar.style.width = `${magnitude}%`;
      const parent = byId(item.id).closest('.state-metric');
      if (parent) {
        parent.classList.toggle('is-changing', animate && Math.abs(item.value) > item.bound * .65);
      }
    });
  }

  function setAction(action, step, mode = simulation.mode) {
    simulation.action = action;
    const hasAction = action === 0 || action === 1;
    const direction = action === 1 ? 'ВПРАВО' : 'ВЛЕВО';
    byId('decision-display').dataset.action = hasAction ? String(action) : 'none';
    byId('decision-arrow').textContent = hasAction ? (action === 1 ? '→' : '←') : '↔';
    byId('decision-title').textContent = hasAction ? `Толчок ${direction.toLowerCase()}` : 'Готов к запуску';
    byId('decision-subtitle').textContent = hasAction
      ? `${mode === 'agent' ? 'Политика PPO' : 'Ручное управление'} · шаг ${String(step).padStart(3, '0')}`
      : 'Агент выберет направление силы';
    byId('decision-caption').textContent = hasAction ? `Действие ${action} · ${direction.toLowerCase()}` : 'Ждём первый шаг';
    byId('decision-chip').textContent = mode === 'agent' ? 'PPO' : 'MANUAL';
    byId('decision-state-indicator').lastChild.textContent = hasAction ? ' ACTION' : ' READY';
    byId('decision-state-indicator').querySelector('i').style.background = hasAction ? '#25b899' : '#b9c2d0';
  }

  function updateFrameHud(frame) {
    simulation.step = Number(frame.steps) || simulation.step;
    simulation.totalReward = Number(frame.total_reward) || 0;
    byId('episode-reward').textContent = String(Math.round(simulation.totalReward));
    byId('step-counter').textContent = String(simulation.step).padStart(3, '0');
    byId('episode-progress-label').textContent = `${Math.min(100, simulation.step / 5).toFixed(0)}%`;
    byId('episode-progress-fill').style.width = `${Math.min(100, simulation.step / 5)}%`;
    byId('reward-delta').textContent = `+${safeNumber(frame.reward, 0)} за шаг`;
    byId('elapsed-label').textContent = simulation.running ? `В работе · ${formatDuration((Date.now() - simulation.runStartedAt) / 1000)}` : 'Пауза';
    setAction(frame.action, frame.steps);
    byId('footer-message').textContent = frame.done
      ? `Эпизод завершён на шаге ${frame.steps}. Награда: ${Math.round(frame.total_reward)}.`
      : `Действие ${frame.action} применено в среде. Награда эпизода: ${Math.round(frame.total_reward)}.`;
    simulation.history.push({ state: [...frame.state], action: frame.action, step: frame.steps });
    if (simulation.history.length > 60) simulation.history.shift();
    simulation.activity.unshift({ action: frame.action, step: frame.steps, reward: frame.reward, mode: simulation.mode, time: new Date() });
    simulation.activity = simulation.activity.slice(0, 3);
    simulation.activityHidden = false;
    renderActivity();
    if (frame.done) {
      simulation.running = false;
      byId('terminal-state').hidden = false;
      byId('terminal-message').textContent = frame.terminated
        ? `Шест потерял равновесие на шаге ${frame.steps}. Награда — ${Math.round(frame.total_reward)}.`
        : `Достигнут лимит в 500 шагов. Награда — ${Math.round(frame.total_reward)}.`;
      byId('simulation-dot').classList.remove('is-running');
      byId('simulation-dot').classList.add('is-error');
      byId('simulation-state-label').textContent = 'ЭПИЗОД ЗАВЕРШЁН';
      byId('run-label').textContent = 'Эпизод завершён';
      byId('run-button').disabled = true;
    }
  }

  function animateFrame(frame, generation) {
    return new Promise((resolve) => {
      if (generation !== simulation.generation) { resolve(false); return; }
      const from = [...simulation.state];
      const to = frame.state.map(Number);
      simulation.targetState = to;
      simulation.animation = {
        from, to, frame, start: performance.now(), duration: reduceMotion ? 1 : 68,
        resolve: (valid) => resolve(valid), generation,
      };
    });
  }

  async function pumpSimulation(token, generation) {
    if (!simulation.running || token !== simulation.loopToken || generation !== simulation.generation || simulation.busy) return;
    simulation.busy = true;
    try {
      const payload = {
        session_id: simulation.sessionId,
        steps: 4,
        mode: simulation.mode,
        ...(simulation.mode === 'manual' ? { action: simulation.manualAction } : {}),
      };
      const result = await api('/api/sim/step', { method: 'POST', body: JSON.stringify(payload) }, 12000);
      if (generation !== simulation.generation) return;
      if (result.engine) simulation.engine = result.engine;
      for (const frame of (result.frames || [])) {
        if (generation !== simulation.generation) return;
        const completed = await animateFrame(frame, generation);
        if (!completed || generation !== simulation.generation) return;
      }
      if (result.done) {
        simulation.running = false;
        updateButtons();
        return;
      }
    } catch (error) {
      simulation.running = false;
      updateButtons();
      byId('footer-message').textContent = error.message || 'Не удалось продолжить симуляцию.';
      toast(error.message || 'Не удалось продолжить симуляцию. Попробуйте сбросить эпизод.');
      if (error.status === 404) simulation.hasReset = false;
    } finally {
      simulation.busy = false;
      updateButtons();
    }
    if (simulation.running && generation === simulation.generation && byId('terminal-state').hidden) {
      // Если пользователь поставил паузу и сразу продолжил, используем актуальный токен цикла.
      pumpSimulation(simulation.loopToken, generation);
    }
  }

  async function resetSimulation() {
    if (!simulation.online || simulation.resetPending) return;
    const wasInitialized = simulation.hasReset;
    const previousSessionId = simulation.sessionId;
    simulation.sessionId = makeSessionId();
    simulation.resetPending = true;
    simulation.running = false;
    simulation.loopToken += 1;
    simulation.generation += 1;
    const generation = simulation.generation;
    simulation.busy = true;
    simulation.hasReset = false;
    simulation.animation = null;
    simulation.state = [...defaultState];
    simulation.targetState = [...defaultState];
    simulation.step = 0;
    simulation.runStartedAt = 0;
    simulation.totalReward = 0;
    simulation.action = null;
    simulation.history = [];
    simulation.activity = [];
    simulation.activityHidden = false;
    if (wasInitialized) simulation.seed += 1;
    else if (simulation.seed < 116) simulation.seed = 116;
    byId('terminal-state').hidden = true;
    byId('simulation-dot').classList.remove('is-error');
    byId('simulation-state-label').textContent = 'СБРОС СРЕДЫ…';
    byId('episode-reward').textContent = '0';
    byId('step-counter').textContent = '000';
    byId('episode-progress-label').textContent = '0%';
    byId('episode-progress-fill').style.width = '0%';
    byId('elapsed-label').textContent = 'Ожидание запуска';
    byId('reward-delta').textContent = '+0 за шаг';
    byId('seed-label').textContent = `SEED ${simulation.seed}`;
    displayState(defaultState, false);
    setAction(null, 0);
    renderActivity();
    updateButtons();
    try {
      const result = await api('/api/sim/reset', {
        method: 'POST', body: JSON.stringify({ session_id: simulation.sessionId, seed: simulation.seed }),
      }, 12000);
      if (generation !== simulation.generation) return;
      simulation.state = result.state.map(Number);
      simulation.targetState = [...simulation.state];
      simulation.engine = result.engine || 'Gymnasium CartPole-v1';
      simulation.hasReset = true;
      if (previousSessionId && previousSessionId !== simulation.sessionId) {
        api('/api/sim/close', { method: 'POST', body: JSON.stringify({ session_id: previousSessionId }) }, 5000).catch(() => {});
      }
      byId('engine-label').textContent = simulation.engine;
      byId('seed-label').textContent = `SEED ${result.seed}`;
      byId('footer-message').textContent = 'Среда сброшена. Запусти агента — каждое действие будет применено к физической модели.';
    } catch (error) {
      byId('engine-label').textContent = 'Среда недоступна';
      toast(error.message || 'Не удалось создать эпизод CartPole.');
    } finally {
      if (generation === simulation.generation) {
        simulation.busy = false;
        simulation.resetPending = false;
        updateButtons();
      }
    }
  }

  function toggleSimulation() {
    if (!simulation.online || !simulation.hasReset) return;
    if (simulation.running) {
      simulation.running = false;
      simulation.loopToken += 1;
      byId('elapsed-label').textContent = 'Пауза';
      byId('footer-message').textContent = `Пауза на шаге ${simulation.step}. Нажми запуск, чтобы продолжить с текущего состояния.`;
      updateButtons();
      return;
    }
    if (byId('terminal-state').hidden === false) return;
    simulation.running = true;
    simulation.runStartedAt = simulation.runStartedAt || Date.now();
    simulation.loopToken += 1;
    updateButtons();
    byId('canvas-tip').style.opacity = '0';
    byId('footer-message').textContent = simulation.mode === 'agent'
      ? 'Политика PPO выбирает действие на каждом шаге среды.'
      : `Ручной режим: постоянный толчок ${simulation.manualAction === 0 ? 'влево' : 'вправо'}.`;
    pumpSimulation(simulation.loopToken, simulation.generation);
  }

  function renderActivity() {
    const host = byId('activity-list');
    host.replaceChildren();
    if (simulation.activityHidden) {
      host.innerHTML = '<div class="activity-empty"><span class="empty-pulse"></span> Отображение журнала очищено. Следующее действие появится здесь.</div>';
      return;
    }
    if (!simulation.activity.length) {
      host.innerHTML = '<div class="activity-empty"><span class="empty-pulse"></span> Запусти эпизод — здесь появится история решений.</div>';
      return;
    }
    simulation.activity.forEach((record) => {
      const item = document.createElement('div');
      item.className = 'activity-item';
      item.dataset.action = String(record.action);
      const icon = document.createElement('span');
      icon.className = 'activity-action';
      icon.textContent = record.action === 1 ? '→' : '←';
      const details = document.createElement('span');
      const title = document.createElement('strong');
      title.textContent = `${record.action === 1 ? 'Вправо' : 'Влево'} · шаг ${record.step}`;
      const subtitle = document.createElement('small');
      subtitle.textContent = `${record.mode === 'agent' ? 'Агент PPO' : 'Ручной режим'} · ${record.time.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
      details.append(title, subtitle);
      const reward = document.createElement('span');
      reward.className = 'activity-reward';
      reward.textContent = `+${safeNumber(record.reward, 0)}`;
      item.append(icon, details, reward);
      host.appendChild(item);
    });
  }

  function renderLogs(items) {
    if (simulation.activityHidden || simulation.running || simulation.step > 0) return;
    simulation.activity = (Array.isArray(items) ? items.slice(-4).reverse() : []).map((item) => ({
      action: Number(item.action) === 1 ? 1 : 0,
      step: null,
      reward: item.reward === null || item.reward === undefined ? 0 : Number(item.reward),
      mode: 'agent',
      time: new Date(item.time),
    }));
    renderActivity();
    [...byId('activity-list').querySelectorAll('.activity-item')].forEach((node, index) => {
      const record = simulation.activity[index];
      const title = node.querySelector('strong');
      if (record && record.step === null) title.textContent = `${record.action === 1 ? 'Вправо' : 'Влево'} · журнал API`;
    });
  }

  async function loadLogs() {
    try {
      const result = await api('/api/logs?limit=8', {}, 8000);
      renderLogs(result.items || []);
    } catch (_error) {
      // Текущая симуляция остаётся доступной даже при временной ошибке журнала.
    }
  }

  function setTab(name) {
    document.querySelectorAll('.sidebar-tab').forEach((tab) => {
      const active = tab.dataset.view === name;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('.side-view').forEach((panel) => {
      const active = panel.dataset.panel === name;
      panel.classList.toggle('is-active', active);
      panel.hidden = !active;
    });
  }

  function makeSegments() {
    const host = byId('episode-segments');
    host.replaceChildren();
    for (let index = 0; index < 20; index += 1) {
      const segment = document.createElement('span');
      segment.className = 'episode-segment';
      segment.setAttribute('aria-hidden', 'true');
      host.appendChild(segment);
    }
  }

  function setEvaluationProgress(done, total = 20) {
    const current = Math.max(0, Math.min(20, Math.round((Number(done) || 0) / Math.max(1, Number(total) || 20) * 20)));
    [...byId('episode-segments').children].forEach((segment, index) => {
      segment.classList.toggle('is-done', index < current);
      segment.classList.toggle('is-last', index === current - 1 && current > 0);
    });
    byId('eval-progress-text').textContent = `${current} / 20`;
  }

  function updateStepper(stage, finished, failed) {
    const stages = { idle: 0, initializing: 0, training: 1, evaluation: 2, done: 3, error: 3 };
    const currentIndex = stages[stage] ?? 0;
    byId('train-stepper').querySelectorAll('span').forEach((item, index) => {
      item.classList.remove('is-current', 'is-done', 'is-error');
      const marker = item.querySelector('i');
      if (failed && index === currentIndex) { item.classList.add('is-error'); marker.textContent = '!'; }
      else if (finished || index < currentIndex) { item.classList.add('is-done'); marker.textContent = '✓'; }
      else if (index === currentIndex) { item.classList.add('is-current'); marker.textContent = String(index + 1); }
      else marker.textContent = String(index + 1);
    });
  }

  function fillMetric(prefix, values) {
    byId(`${prefix}-mean`).textContent = Number(values.mean).toFixed(1);
    byId(`${prefix}-std`).textContent = Number(values.std).toFixed(1);
    byId(`${prefix}-min`).textContent = String(values.min);
    byId(`${prefix}-max`).textContent = String(values.max);
    byId(`${prefix}-fill`).style.width = `${Math.max(0, Math.min(100, Number(values.mean) / 5))}%`;
  }

  async function loadEvaluationResult() {
    try {
      const result = await api('/api/evaluate/result', {}, 9000);
      if (!result.ppo || !result.random) return;
      fillMetric('ppo', result.ppo);
      fillMetric('random', result.random);
      byId('improvement-text').textContent = `PPO лучше случайного агента в ${Number(result.improvement_ratio).toFixed(2)} раза.`;
      byId('evaluation-message').textContent = 'Последняя оценка загружена';
      setEvaluationProgress(20, 20);
    } catch (error) {
      if (error.status !== 404) console.warn('Не удалось загрузить результаты оценки.');
    }
  }

  async function loadProgress() {
    try {
      const progress = await api('/api/progress', {}, 8000);
      simulation.trainEnabled = Boolean(progress.train_enabled);
      simulation.operationRunning = Boolean(progress.job_type && !progress.finished_at);
      updateButtons();
      if (progress.job_type === 'evaluate') {
        byId('evaluation-message').textContent = progress.message || 'Выполняется оценка…';
        setEvaluationProgress(progress.done, progress.total);
        if (progress.started_at) byId('eval-duration').textContent = `Прошло ${formatDuration((Date.now() - new Date(progress.started_at).getTime()) / 1000)}`;
      }
      if (progress.job_type === 'train') {
        const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
        byId('train-progress-fill').style.width = `${percent}%`;
        byId('train-percent').textContent = `${Math.round(percent)}%`;
        byId('train-progress-message').textContent = progress.message || 'Выполняется обучение';
        byId('train-step-count').textContent = `${progress.done || 0} / ${progress.total || 0} шагов`;
        byId('train-eta').textContent = progress.eta_seconds === null || progress.eta_seconds === undefined ? 'Осталось —' : `Осталось ${formatDuration(progress.eta_seconds)}`;
        updateStepper(progress.stage, Boolean(progress.finished_at && !progress.error), Boolean(progress.error));
        if (progress.error) {
          byId('train-result').hidden = false;
          byId('train-result').textContent = progress.message || 'Обучение не завершилось.';
        } else if (progress.finished_at && progress.result) {
          byId('train-result').hidden = false;
          const result = progress.result;
          byId('train-result').textContent = `Текущая модель: ${Number(result.current_model_mean_reward).toFixed(1)}; кандидат: ${Number(result.candidate_mean_reward).toFixed(1)}. Основная модель не заменена.`;
        }
      }
      if (progress.finished_at && progress.job_type === 'evaluate') {
        byId('evaluation-message').textContent = progress.error ? progress.message : 'Оценка завершена';
        if (!progress.error) {
          setEvaluationProgress(20, 20);
          await loadEvaluationResult();
        }
      }
      if (progress.finished_at && progress.job_type === 'train') byId('train-status-title').textContent = progress.error ? 'Обучение завершилось с ошибкой' : 'Кандидат обучен';
    } catch (_error) {
      // Фоновая проверка повторится автоматически.
    }
  }

  async function startEvaluation() {
    if (simulation.operationRunning) return;
    simulation.operationRunning = true;
    updateButtons();
    byId('evaluation-message').textContent = 'Подготовка оценки…';
    byId('eval-duration').textContent = 'Оценка началась';
    setEvaluationProgress(0, 20);
    try {
      await api('/api/evaluate', { method: 'POST', body: '{}' }, 15000);
      simulation.evaluationStartedAt = Date.now();
      toast('Оценка запущена. Результат появится здесь.', 'success');
      await loadProgress();
    } catch (error) {
      simulation.operationRunning = false;
      updateButtons();
      toast(error.message || 'Не удалось запустить оценку.');
    }
  }

  async function startTraining() {
    if (!simulation.trainEnabled || simulation.operationRunning) return;
    simulation.operationRunning = true;
    updateButtons();
    byId('train-result').hidden = true;
    try {
      await api('/api/train', { method: 'POST', body: JSON.stringify({ timesteps: 30000 }) }, 15000);
      toast('Обучение кандидатной модели запущено.', 'success');
      await loadProgress();
    } catch (error) {
      simulation.operationRunning = false;
      updateButtons();
      toast(error.message || 'Не удалось запустить обучение.');
    }
  }

  function resizeCanvas() {
    const canvas = byId('simulation-canvas');
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    canvasWidth = width;
    canvasHeight = height;
    canvasContext = canvas.getContext('2d');
    if (canvasContext) canvasContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function drawCartPole(state, timestamp) {
    if (!canvasContext) return;
    const ctx = canvasContext;
    const width = canvasWidth / Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const height = canvasHeight / Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    if (!width || !height) return;
    ctx.clearRect(0, 0, width, height);
    const background = ctx.createLinearGradient(0, 0, 0, height);
    background.addColorStop(0, '#fbfcff');
    background.addColorStop(1, '#f3f7fc');
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    const compact = width < 460;
    const trackLeft = width * (compact ? .12 : .11);
    const trackRight = width * (compact ? .88 : .89);
    const trackWidth = trackRight - trackLeft;
    const floorY = height * .735;
    const cartWidth = Math.max(52, Math.min(74, width * .112));
    const cartHeight = compact ? 25 : 29;
    const poleLength = Math.max(76, Math.min(height * .40, width * .19));
    const x = Number(state[0]) || 0;
    const theta = Number(state[2]) || 0;
    const maxShift = trackWidth * .43;
    const cartX = width / 2 + Math.max(-2.4, Math.min(2.4, x)) / 2.4 * maxShift;
    const cartY = floorY - 15;
    const jointX = cartX;
    const jointY = cartY - cartHeight / 2 + 2;
    const poleX = jointX + Math.sin(theta) * poleLength;
    const poleY = jointY - Math.cos(theta) * poleLength;

    // Ненавязчивая координатная сетка помогает видеть направление движения.
    ctx.save();
    ctx.strokeStyle = 'rgba(151, 168, 194, .12)';
    ctx.lineWidth = 1;
    for (let gx = trackLeft; gx <= trackRight; gx += 34) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, floorY + 19); ctx.stroke();
    }
    for (let gy = 34; gy < floorY; gy += 34) {
      ctx.beginPath(); ctx.moveTo(trackLeft - 12, gy); ctx.lineTo(trackRight + 12, gy); ctx.stroke();
    }
    ctx.restore();

    // Мягкие концентрические дуги обозначают область управления, а не декоративный макет.
    ctx.save();
    ctx.translate(width / 2, floorY - 14);
    ctx.strokeStyle = 'rgba(117, 151, 222, .13)';
    ctx.lineWidth = 1;
    for (const radius of [Math.min(width, height) * .16, Math.min(width, height) * .25, Math.min(width, height) * .34]) {
      ctx.beginPath(); ctx.arc(0, 0, radius, Math.PI, 2 * Math.PI); ctx.stroke();
    }
    ctx.restore();

    // Рельс и метки шкалы CartPole: от -2.4 м до +2.4 м.
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#dce4f0';
    ctx.lineWidth = 7;
    ctx.beginPath(); ctx.moveTo(trackLeft, floorY); ctx.lineTo(trackRight, floorY); ctx.stroke();
    ctx.strokeStyle = '#aebdd2';
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(trackLeft, floorY + 7); ctx.lineTo(trackRight, floorY + 7); ctx.stroke();
    [-2.4, -1.2, 0, 1.2, 2.4].forEach((value) => {
      const px = width / 2 + value / 2.4 * maxShift;
      ctx.strokeStyle = value === 0 ? '#98acd0' : '#c4cfde';
      ctx.lineWidth = value === 0 ? 1.6 : 1;
      ctx.beginPath(); ctx.moveTo(px, floorY - 5); ctx.lineTo(px, floorY + 9); ctx.stroke();
      ctx.fillStyle = '#a0adbf'; ctx.font = '9px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
      ctx.fillText(value === 0 ? '0' : value.toFixed(1), px, floorY + 23);
    });

    // История последних положений тележки оставляет след её реального движения.
    if (simulation.history.length > 1) {
      ctx.save();
      ctx.beginPath();
      const recent = simulation.history.slice(-28);
      recent.forEach((entry, index) => {
        const hx = width / 2 + Math.max(-2.4, Math.min(2.4, entry.state[0])) / 2.4 * maxShift;
        const hy = floorY - 21;
        if (index === 0) ctx.moveTo(hx, hy); else ctx.lineTo(hx, hy);
      });
      ctx.strokeStyle = 'rgba(91, 133, 239, .27)';
      ctx.lineWidth = 2;
      ctx.setLineDash([2, 5]);
      ctx.stroke();
      ctx.restore();
    }

    // Тень под тележкой.
    ctx.save();
    ctx.fillStyle = 'rgba(54, 78, 122, .10)';
    ctx.beginPath(); ctx.ellipse(cartX, floorY + 13, cartWidth * .55, 5, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    // Направление силы показывается на мгновение после решения агента.
    if (simulation.action === 0 || simulation.action === 1) {
      const direction = simulation.action === 1 ? 1 : -1;
      const pulse = .55 + Math.sin(timestamp / 180) * .10;
      const startX = cartX + direction * cartWidth * .48;
      const endX = startX + direction * (compact ? 25 : 34);
      const arrowY = cartY + 1;
      ctx.save();
      ctx.globalAlpha = simulation.running ? pulse : .55;
      ctx.strokeStyle = simulation.action === 1 ? '#21aa91' : '#5c82ec';
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2.3;
      ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(startX, arrowY); ctx.lineTo(endX, arrowY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(endX - direction * 6, arrowY - 5); ctx.lineTo(endX, arrowY); ctx.lineTo(endX - direction * 6, arrowY + 5); ctx.stroke();
      ctx.restore();
    }

    // Полюс строится из текущего состояния среды, а не из заранее заданного угла.
    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(55, 81, 128, .10)';
    ctx.lineWidth = compact ? 13 : 15;
    ctx.beginPath(); ctx.moveTo(jointX + 2, jointY + 3); ctx.lineTo(poleX + 2, poleY + 3); ctx.stroke();
    const poleGradient = ctx.createLinearGradient(jointX, jointY, poleX, poleY);
    poleGradient.addColorStop(0, '#5178d9'); poleGradient.addColorStop(.5, '#7194ee'); poleGradient.addColorStop(1, '#a6bcfa');
    ctx.strokeStyle = poleGradient;
    ctx.lineWidth = compact ? 8 : 10;
    ctx.beginPath(); ctx.moveTo(jointX, jointY); ctx.lineTo(poleX, poleY); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,.75)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(jointX - 1, jointY - 3); ctx.lineTo(poleX - 1, poleY + 6); ctx.stroke();
    ctx.fillStyle = '#7294ed';
    ctx.beginPath(); ctx.arc(poleX, poleY, compact ? 5 : 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e5edff';
    ctx.beginPath(); ctx.arc(poleX, poleY, compact ? 2 : 2.3, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    // Корпус тележки и колёса.
    const cartXLeft = cartX - cartWidth / 2;
    const cartYTop = cartY - cartHeight / 2;
    ctx.save();
    const cartGradient = ctx.createLinearGradient(cartXLeft, cartYTop, cartXLeft, cartYTop + cartHeight);
    cartGradient.addColorStop(0, '#34496f'); cartGradient.addColorStop(1, '#233550');
    roundedRect(ctx, cartXLeft, cartYTop, cartWidth, cartHeight, 7);
    ctx.fillStyle = cartGradient; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,.26)'; ctx.lineWidth = 1; ctx.stroke();
    roundedRect(ctx, cartX - cartWidth * .25, cartYTop + 5, cartWidth * .5, 3, 2);
    ctx.fillStyle = 'rgba(176, 200, 255, .55)'; ctx.fill();
    [-cartWidth * .27, cartWidth * .27].forEach((offset) => {
      ctx.fillStyle = '#d2ddec'; ctx.beginPath(); ctx.arc(cartX + offset, cartYTop + cartHeight + 3, 5.6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#778aa8'; ctx.beginPath(); ctx.arc(cartX + offset, cartYTop + cartHeight + 3, 2.5, 0, Math.PI * 2); ctx.fill();
    });
    // Шарнир соединяет шест с тележкой и реагирует на угол.
    ctx.fillStyle = '#f7faff'; ctx.beginPath(); ctx.arc(jointX, jointY, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#537ce4'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#537ce4'; ctx.beginPath(); ctx.arc(jointX, jointY, 2, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    // Короткая измерительная линия показывает текущий угол относительно вертикали.
    ctx.save();
    ctx.strokeStyle = 'rgba(105, 131, 181, .33)'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(jointX, jointY); ctx.lineTo(jointX, jointY - poleLength * .73); ctx.stroke();
    ctx.restore();

    // Текущие координаты выводятся непосредственно поверх сцены.
    ctx.save();
    const angleDeg = theta * 180 / Math.PI;
    const labelX = Math.max(64, Math.min(width - 64, poleX + (poleX > width * .66 ? -44 : 44)));
    const labelY = Math.max(78, poleY - 13);
    ctx.fillStyle = 'rgba(255,255,255,.92)';
    roundedRect(ctx, labelX - 31, labelY - 13, 62, 24, 7); ctx.fill();
    ctx.strokeStyle = 'rgba(225,233,245,.95)'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = Math.abs(angleDeg) > 9 ? '#d58a4b' : '#667a9d';
    ctx.font = '10px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
    ctx.fillText(`${angleDeg >= 0 ? '+' : ''}${angleDeg.toFixed(1)}°`, labelX, labelY + 3);
    ctx.restore();

    // Едва заметный центр сцены служит ориентиром при движении по рельсу.
    ctx.save();
    ctx.strokeStyle = 'rgba(100, 132, 195, .20)'; ctx.setLineDash([3, 5]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(width / 2, height * .18); ctx.lineTo(width / 2, floorY - 11); ctx.stroke();
    ctx.restore();
  }

  function renderLoop(timestamp) {
    if (simulation.animation) {
      const animation = simulation.animation;
      if (animation.generation !== simulation.generation) {
        simulation.animation = null;
        animation.resolve(false);
      } else {
        const progress = Math.min(1, (timestamp - animation.start) / animation.duration);
        const eased = progress * progress * (3 - 2 * progress);
        simulation.state = animation.from.map((value, index) => value + (animation.to[index] - value) * eased);
        if (timestamp - lastHudUpdate > 35 || progress >= 1) {
          displayState(simulation.state, true);
          lastHudUpdate = timestamp;
        }
        if (progress >= 1) {
          simulation.state = [...animation.to];
          simulation.animation = null;
          updateFrameHud(animation.frame);
          animation.resolve(true);
        }
      }
    }
    drawCartPole(simulation.state, timestamp);
    window.requestAnimationFrame(renderLoop);
  }

  function setMode(mode) {
    simulation.mode = mode;
    document.querySelectorAll('.mode-button').forEach((button) => button.classList.toggle('is-active', button.dataset.mode === mode));
    byId('manual-controls').hidden = mode !== 'manual';
    byId('decision-chip').textContent = mode === 'agent' ? 'PPO' : 'MANUAL';
    byId('footer-message').textContent = mode === 'agent'
      ? 'Агент PPO сам выбирает направление силы на каждом шаге.'
      : `Ручное управление включено. Выбрано направление: ${simulation.manualAction === 0 ? 'влево' : 'вправо'}.`;
    updateButtons();
  }

  function bindEvents() {
    byId('run-button').addEventListener('click', toggleSimulation);
    byId('reset-button').addEventListener('click', resetSimulation);
    byId('reset-small').addEventListener('click', resetSimulation);
    byId('terminal-reset').addEventListener('click', resetSimulation);
    byId('evaluate-button').addEventListener('click', startEvaluation);
    byId('train-button').addEventListener('click', startTraining);
    byId('clear-activity').addEventListener('click', () => { simulation.activityHidden = true; renderActivity(); });
    document.querySelectorAll('.sidebar-tab').forEach((tab) => tab.addEventListener('click', () => setTab(tab.dataset.view)));
    document.querySelectorAll('.mode-button').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
    document.querySelectorAll('.manual-key').forEach((button) => button.addEventListener('click', () => {
      simulation.manualAction = Number(button.dataset.action);
      document.querySelectorAll('.manual-key').forEach((key) => key.classList.toggle('is-selected', key === button));
      if (!simulation.running) setAction(simulation.manualAction, simulation.step, 'manual');
      if (simulation.mode === 'manual') byId('footer-message').textContent = `Выбран толчок ${simulation.manualAction === 0 ? 'влево' : 'вправо'}. Запусти или продолжи эпизод.`;
    }));
    window.addEventListener('resize', resizeCanvas, { passive: true });
    window.addEventListener('keydown', (event) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (event.code === 'Space') { event.preventDefault(); toggleSimulation(); }
      if (event.key.toLowerCase() === 'r') resetSimulation();
      if (event.key === 'ArrowLeft' && simulation.mode === 'manual') byId('manual-left').click();
      if (event.key === 'ArrowRight' && simulation.mode === 'manual') byId('manual-right').click();
    });
  }

  async function boot() {
    bindEvents();
    makeSegments();
    resizeCanvas();
    window.requestAnimationFrame(renderLoop);
    await checkHealth();
    if (simulation.online) await resetSimulation();
    await Promise.all([loadProgress(), loadLogs(), loadEvaluationResult()]);
    window.setInterval(checkHealth, 5000);
    window.setInterval(loadProgress, 1000);
    window.setInterval(loadLogs, 3000);
    window.setInterval(() => {
      if (simulation.running) byId('elapsed-label').textContent = `В работе · ${formatDuration((Date.now() - simulation.runStartedAt) / 1000)}`;
    }, 500);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
