/**
 * AeroPulse Intelligent Operations Assistant
 * Multi-model fallback client with token budgeting and rate limiting.
 */
(function () {
    var p = window.parent;
    if (!p || !p.document) return;

    var SERVER_KEY = window.AEROPULSE_SERVER_KEY || '';
    var hasServerKey = !!(SERVER_KEY && SERVER_KEY.trim());

    // ── Persistent State on window.parent ─────────────────────────────────────
    p.__ap_chat = p.__ap_chat || {
        history: [],
        messages: [],
        apiKey: '',
        isOpen: false,
        msgCount: 0,
        lastSend: 0
    };

    var state = p.__ap_chat;
    if (hasServerKey) {
        state.apiKey = SERVER_KEY;
    }

    // ── Fallback Model Chain & Budget Protections ─────────────────────────────
    var MODELS = [
        { id: 'google/gemini-2.0-flash-001', name: 'Gemini 2.0 Flash' },
        { id: 'google/gemini-flash-1.5', name: 'Gemini 1.5 Flash' },
        { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B' },
        { id: 'mistralai/mistral-small-3.2-24b-instruct', name: 'Mistral Small 24B' }
    ];
    var MAX_MSGS   = 25;    // Session limit to prevent budget exhaustion
    var MAX_CHARS  = 400;   // Character ceiling per prompt
    var COOLDOWN   = 3500;  // 3.5s cooldown between queries
    var MAX_TOKENS = 450;   // Max output tokens per reply

    var SYS_PROMPT = 'You are the AeroPulse Flight Operations AI Assistant. '
        + 'Provide concise, professional insights on aviation operations, delay analytics, and predictive models. '
        + 'Key facts: 469,968 real U.S. DOT flight records (Jan 2024); HistGBM regression + classification; '
        + 'chronological split Days 1-23 train (345,440 flights) / Days 24-31 test (111,573 flights); '
        + 'FAA OTP-15 threshold; $101.90/min FAA delay cost benchmark; ROC-AUC 0.6096; MAE 16.56 min; '
        + 'top predictor route_avg_delay_minutes (68.6% importance); hub buffering saves $1.8M/mo; '
        + 'corridor padding +4.2% OTP; crew staging avoids $3.4M/quarter; star schema warehouse; '
        + 'stack: Python, PostgreSQL, Pandas, Scikit-Learn, Streamlit, Plotly, PyArrow, JavaScript, Docker. '
        + 'Keep responses under 3 paragraphs. If asked about non-aviation/data topics, politely decline in 1 sentence to conserve tokens.';

    if (state.history.length === 0) {
        state.history.push({ role: 'system', content: SYS_PROMPT });
    }

    // ── Clean Up Stale DOM from Previous Remounts ─────────────────────────────
    try {
        var oldFab = p.document.getElementById('ap-fab');
        if (oldFab) oldFab.remove();
        var oldWin = p.document.getElementById('ap-win');
        if (oldWin) oldWin.remove();
        var oldStyle = p.document.getElementById('ap-style');
        if (oldStyle) oldStyle.remove();
    } catch (e) {
        console.warn('[AeroPulse] Error cleaning old widget DOM:', e);
    }

    // ── Build DOM Elements in Parent ─────────────────────────────────────────
    var pd = p.document;

    var fab = pd.createElement('button');
    fab.id = 'ap-fab';
    fab.title = 'AeroPulse Operations Assistant';
    fab.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="currentColor" viewBox="0 0 16 16"><path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4.414a1 1 0 0 0-.707.293L.854 15.146A.5.5 0 0 1 0 14.793V2z"/></svg>';
    pd.body.appendChild(fab);

    var win = pd.createElement('div');
    win.id = 'ap-win';
    win.innerHTML =
        '<div id="ap-hdr">' +
            '<div class="ap-ttl"><span class="ap-dot"></span> AeroPulse Assistant</div>' +
            '<button id="ap-cls" title="Close">&#x2715;</button>' +
        '</div>' +
        (hasServerKey ? '' : (
            '<div id="ap-key-row">' +
                '<input id="ap-key" type="password" placeholder="Paste OpenRouter API key (sk-or-v1-...)..." autocomplete="off">' +
            '</div>'
        )) +
        '<div id="ap-msgs"></div>' +
        '<div id="ap-footer-info">' +
            '<span id="ap-status-hint">Gemini 2.0 Flash + 3 fallbacks</span>' +
            '<span id="ap-quota">0 / ' + MAX_MSGS + ' msgs</span>' +
        '</div>' +
        '<div id="ap-inp-row">' +
            '<textarea id="ap-inp" placeholder="Ask about delay metrics, routes, models..." rows="1" maxlength="' + MAX_CHARS + '"></textarea>' +
            '<button id="ap-snd">Send</button>' +
        '</div>';
    pd.body.appendChild(win);

    // ── Restore State ────────────────────────────────────────────────────────
    var keyInput = pd.getElementById('ap-key');
    if (keyInput) {
        if (state.apiKey && !hasServerKey) {
            keyInput.value = state.apiKey;
        }
        keyInput.addEventListener('input', function () {
            state.apiKey = this.value.trim();
        });
    }

    if (state.isOpen) {
        win.style.display = 'flex';
    }

    var msgsContainer = pd.getElementById('ap-msgs');
    if (state.messages.length === 0) {
        var welcomeMsg = hasServerKey
            ? 'Hello. I am the AeroPulse Operations Assistant. Ask me anything about flight delay metrics, HistGBM model predictions, route bottlenecks, or operational cost savings.'
            : 'Hello. I am the AeroPulse Operations Assistant. Ask me about OTP-15 delay benchmarks, HistGBM model performance, route congestion, or financial savings. Enter your OpenRouter key above to start.';
        appendMsgUI(welcomeMsg, 'b');
    } else {
        for (var i = 0; i < state.messages.length; i++) {
            var item = state.messages[i];
            appendMsgUI(item.text, item.cls, item.via);
        }
    }
    updateQuotaUI();

    // ── Bind Listeners ───────────────────────────────────────────────────────
    fab.addEventListener('click', toggleChat);
    pd.getElementById('ap-cls').addEventListener('click', toggleChat);
    pd.getElementById('ap-snd').addEventListener('click', handleSend);

    var inp = pd.getElementById('ap-inp');
    inp.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    inp.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 84) + 'px';
    });

    function toggleChat() {
        state.isOpen = !state.isOpen;
        win.style.display = state.isOpen ? 'flex' : 'none';
        if (state.isOpen) {
            pd.getElementById('ap-inp').focus();
        }
    }

    // ── Robust Model Fallback Chain ──────────────────────────────────────────
    async function callWithFallback(messages, apiKey) {
        var lastError = 'All models unavailable';
        for (var idx = 0; idx < MODELS.length; idx++) {
            var modelObj = MODELS[idx];
            try {
                var response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Authorization': 'Bearer ' + apiKey,
                        'Content-Type': 'application/json',
                        'HTTP-Referer': p.location.href,
                        'X-Title': 'AeroPulse Flight Intelligence'
                    },
                    body: JSON.stringify({
                        model: modelObj.id,
                        messages: messages,
                        max_tokens: MAX_TOKENS,
                        temperature: 0.35
                    })
                });

                if (response.status === 429 || response.status === 503) {
                    lastError = modelObj.name + ' rate-limited (' + response.status + '). Trying fallback...';
                    console.warn('[AeroPulse Fallback]', lastError);
                    continue;
                }

                var data = await response.json();
                if (data.error) {
                    lastError = data.error.message || 'API error';
                    console.warn('[AeroPulse API Error]', modelObj.name, lastError);
                    continue;
                }

                if (data.choices && data.choices[0] && data.choices[0].message) {
                    return {
                        content: data.choices[0].message.content,
                        modelName: modelObj.name
                    };
                }
            } catch (err) {
                lastError = 'Network error contacting ' + modelObj.name;
                console.warn('[AeroPulse Network Error]', err);
            }
        }
        throw new Error(lastError);
    }

    // ── Send Handler with Budget Restrictions ────────────────────────────────
    var isSending = false;

    async function handleSend() {
        if (isSending) return;
        var keyInputEl = pd.getElementById('ap-key');
        var key = (SERVER_KEY || (keyInputEl ? keyInputEl.value.trim() : '') || state.apiKey || '').trim();
        var inpEl = pd.getElementById('ap-inp');
        var text = inpEl.value.trim();
        if (!text) return;

        if (!key) {
            appendMsgUI('OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in .env or secrets.toml.', 's');
            return;
        }

        if (state.msgCount >= MAX_MSGS) {
            appendMsgUI('Session budget limit of ' + MAX_MSGS + ' messages reached. Refresh page to reset.', 's');
            return;
        }

        if (text.length > MAX_CHARS) {
            appendMsgUI('Query exceeds ' + MAX_CHARS + ' characters. Please keep questions concise.', 's');
            return;
        }

        var now = Date.now();
        if (now - state.lastSend < COOLDOWN) {
            var waitSec = Math.ceil((COOLDOWN - (now - state.lastSend)) / 1000);
            appendMsgUI('Rate limit: please wait ' + waitSec + 's before sending another question.', 's');
            return;
        }

        appendMsgUI(text, 'u');
        state.messages.push({ text: text, cls: 'u' });
        state.history.push({ role: 'user', content: text });
        inpEl.value = '';
        inpEl.style.height = 'auto';
        state.msgCount++;
        state.lastSend = Date.now();
        updateQuotaUI();

        isSending = true;
        var sendBtn = pd.getElementById('ap-snd');
        sendBtn.disabled = true;
        var thinkingDiv = appendMsgUI('Consulting operations models...', 's', null, true);

        try {
            var result = await callWithFallback(state.history, key);
            thinkingDiv.remove();
            appendMsgUI(result.content, 'b', result.modelName);
            state.messages.push({ text: result.content, cls: 'b', via: result.modelName });
            state.history.push({ role: 'assistant', content: result.content });
        } catch (err) {
            thinkingDiv.remove();
            appendMsgUI('Service notice: ' + err.message, 's');
        }

        isSending = false;
        sendBtn.disabled = false;
        inpEl.focus();
    }

    function appendMsgUI(text, cls, via, returnElement) {
        var el = pd.createElement('div');
        el.className = 'ap-m ' + cls;
        el.textContent = text;
        if (via) {
            var viaSpan = pd.createElement('div');
            viaSpan.className = 'ap-via';
            viaSpan.textContent = '— via ' + via;
            el.appendChild(viaSpan);
        }
        msgsContainer.appendChild(el);
        msgsContainer.scrollTop = msgsContainer.scrollHeight;
        if (returnElement) return el;
    }

    function updateQuotaUI() {
        var q = pd.getElementById('ap-quota');
        if (!q) return;
        q.textContent = state.msgCount + ' / ' + MAX_MSGS + ' msgs';
        if (state.msgCount >= MAX_MSGS - 5) {
            q.style.color = '#C0392B';
        } else if (state.msgCount >= MAX_MSGS - 10) {
            q.style.color = '#D4A017';
        }
    }
})();
