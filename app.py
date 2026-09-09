"""
AeroPulse Intelligence — Real-World Aviation Delay & Operations Platform.

Interactive Executive Dashboard and Machine Learning Inference Engine
powered by real-world U.S. Department of Transportation / Kaggle flight operations data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.ml.features import FAA_DELAY_COST_PER_MINUTE_USD
from src.ml.predict import FlightDelayPredictor

# Load environment variables (.env)
load_dotenv()

# Resolve OpenRouter API Key from Streamlit secrets or environment
OPENROUTER_API_KEY = ""
if hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
    OPENROUTER_API_KEY = str(st.secrets["OPENROUTER_API_KEY"]).strip()
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# --- Page Configuration -------------------------------------------------------
st.set_page_config(
    page_title="AeroPulse | Flight Operations & Delay Intelligence",
    page_icon="assets/favicon.ico" if Path("assets/favicon.ico").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global Styling (CSS only — scripts must go via components.html) ----------
DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.main { background-color: #1A1C1E; }
section[data-testid="stSidebar"] {
    background-color: #1E2022;
    border-right: 1px solid #2C2F33;
}
.kpi-card {
    background: #242628;
    border: 1px solid #2C2F33;
    border-radius: 10px;
    padding: 20px;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.kpi-card:hover { border-color: #4CAF82; background: #272A2C; }
.kpi-title {
    color: #858C94; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px;
}
.kpi-value {
    color: #D4D8DC; font-size: 1.75rem; font-weight: 700;
    letter-spacing: -0.02em; margin-bottom: 4px;
}
.kpi-delta-good { color: #4CAF82; font-size: 0.80rem; font-weight: 500; }
.kpi-delta-bad  { color: #C0392B; font-size: 0.80rem; font-weight: 500; }
.kpi-subtext    { color: #606870; font-size: 0.74rem; margin-top: 4px; }
.rec-card {
    background: #242628; border-left: 3px solid #4CAF82;
    border-radius: 0 8px 8px 0; padding: 16px 18px; margin-bottom: 14px;
}
.rec-tag {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.09em;
    color: #4CAF82; text-transform: uppercase; margin-bottom: 4px;
}
.rec-title { font-size: 1.00rem; font-weight: 600; color: #D4D8DC; margin: 4px 0 6px 0; }
.rec-body  { font-size: 0.84rem; color: #9AA0A8; line-height: 1.5; }
.rec-roi   { font-size: 0.82rem; font-weight: 600; color: #4CAF82; margin-top: 8px; }
.section-header { margin-bottom: 18px; }
.section-header h2 { margin: 0; color: #D4D8DC; font-weight: 700; font-size: 1.35rem; }
.section-header p  { color: #858C94; margin: 4px 0 0 0; font-size: 0.88rem; }
</style>
"""
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# --- Chatbot Widget -----------------------------------------------------------
_CHATBOT_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:transparent;">
<script>
(function () {
    var p = window.parent;
    if (!p || !p.document) return;

    var SERVER_KEY = __SERVER_KEY__;
    var hasServerKey = !!(SERVER_KEY && SERVER_KEY.trim());

    // ── Persistent State on window.parent (survives Streamlit reruns) ─────────
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

    // ── Fallback Model Chain & Restrictions ──────────────────────────────────
    var MODELS = [
        { id: 'google/gemini-2.0-flash-001', name: 'Gemini 2.0 Flash' },
        { id: 'google/gemini-flash-1.5', name: 'Gemini 1.5 Flash' },
        { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B' },
        { id: 'mistralai/mistral-small-3.2-24b-instruct', name: 'Mistral Small 24B' }
    ];
    var MAX_MSGS   = 25;    // Session limit to restrict API spend
    var MAX_CHARS  = 400;   // Character limit per message
    var COOLDOWN   = 3500;  // 3.5s cooldown between sends
    var MAX_TOKENS = 450;   // Token ceiling to control cost

    var SYS_PROMPT = 'You are the AeroPulse Flight Operations AI Assistant. '
        + 'Provide concise, professional insights on aviation operations, delay analytics, and predictive models. '
        + 'Key facts: 469,968 real U.S. DOT flight records (Jan 2024); HistGBM regression + classification; '
        + 'chronological split Days 1-23 train (345,440 flights) / Days 24-31 test (111,573 flights); '
        + 'FAA OTP-15 threshold; $101.90/min FAA delay cost benchmark; ROC-AUC 0.6096; MAE 16.56 min; '
        + 'top predictor route_avg_delay_minutes (68.6% importance); hub buffering saves $1.8M/mo; '
        + 'corridor padding +4.2% OTP; crew staging avoids $3.4M/quarter; star schema warehouse; '
        + 'stack: Python, Pandas, Scikit-Learn, Streamlit, Plotly, PyArrow. '
        + 'Keep responses under 3 paragraphs. If asked about non-aviation/data topics, politely decline in 1 sentence to conserve tokens.';

    if (state.history.length === 0) {
        state.history.push({ role: 'system', content: SYS_PROMPT });
    }

    // ── Clean Up Any Previous Injection (prevent detached event closures) ───
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

    // ── Inject Styles into Parent Head ───────────────────────────────────────
    var styleEl = p.document.createElement('style');
    styleEl.id = 'ap-style';
    styleEl.textContent = [
        '#ap-fab {',
        '  position: fixed; bottom: 28px; right: 28px; width: 52px; height: 52px;',
        '  border-radius: 50%; background: #4CAF82; color: #111; border: none; cursor: pointer;',
        '  display: flex; align-items: center; justify-content: center;',
        '  box-shadow: 0 4px 20px rgba(76,175,130,0.42); z-index: 999999;',
        '  transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;',
        '}',
        '#ap-fab:hover { background: #3fa173; transform: scale(1.05); box-shadow: 0 6px 26px rgba(76,175,130,0.55); }',
        '#ap-fab:active { transform: scale(0.96); }',
        '#ap-win {',
        '  position: fixed; bottom: 92px; right: 28px; width: 375px; height: 510px;',
        '  background: #1E2022; border: 1px solid #2C2F33; border-radius: 14px;',
        '  display: none; flex-direction: column;',
        '  box-shadow: 0 16px 48px rgba(0,0,0,0.65); z-index: 999998; overflow: hidden;',
        '  font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;',
        '}',
        '#ap-hdr {',
        '  background: #242628; border-bottom: 1px solid #2C2F33;',
        '  padding: 13px 16px; display: flex; align-items: center;',
        '  justify-content: space-between; flex-shrink: 0;',
        '}',
        '.ap-ttl { color: #D4D8DC; font-weight: 600; font-size: 0.90rem; display: flex; align-items: center; gap: 8px; }',
        '.ap-dot { display: inline-block; width: 8px; height: 8px; background: #4CAF82; border-radius: 50%; }',
        '#ap-cls { background: none; border: none; color: #858C94; cursor: pointer; font-size: 1.1rem; padding: 0; line-height: 1; }',
        '#ap-cls:hover { color: #D4D8DC; }',
        '#ap-key-row { padding: 10px 14px 4px 14px; flex-shrink: 0; background: #1E2022;' + (hasServerKey ? 'display:none;' : '') + '}',
        '#ap-key {',
        '  width: 100%; background: #2A2D30; border: 1px solid #2C2F33; border-radius: 7px;',
        '  color: #D4D8DC; padding: 7px 10px; font-size: 0.77rem; font-family: inherit;',
        '  outline: none; box-sizing: border-box;',
        '}',
        '#ap-key:focus { border-color: #4CAF82; }',
        '#ap-msgs {',
        '  flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column;',
        '  gap: 10px; scrollbar-width: thin; scrollbar-color: #2C2F33 transparent;',
        '}',
        '#ap-msgs::-webkit-scrollbar { width: 4px; }',
        '#ap-msgs::-webkit-scrollbar-thumb { background: #2C2F33; border-radius: 4px; }',
        '.ap-m {',
        '  max-width: 86%; padding: 9px 12px; border-radius: 11px; font-size: 0.83rem;',
        '  line-height: 1.48; word-wrap: break-word; white-space: pre-wrap;',
        '}',
        '.ap-m.u { background: #4CAF82; color: #111; align-self: flex-end; border-bottom-right-radius: 3px; font-weight: 500; }',
        '.ap-m.b { background: #2A2D30; color: #D4D8DC; align-self: flex-start; border-bottom-left-radius: 3px; border: 1px solid #2C2F33; }',
        '.ap-m.s { background: #242628; color: #858C94; font-style: italic; align-self: flex-start; border-bottom-left-radius: 3px; border: 1px solid #2C2F33; font-size: 0.78rem; }',
        '.ap-via { font-size: 0.71rem; color: #606870; margin-top: 5px; font-style: italic; }',
        '#ap-footer-info {',
        '  display: flex; justify-content: space-between; align-items: center;',
        '  padding: 3px 14px; flex-shrink: 0; font-size: 0.71rem; color: #606870; background: #1E2022;',
        '}',
        '#ap-inp-row {',
        '  display: flex; gap: 8px; padding: 10px 12px;',
        '  border-top: 1px solid #2C2F33; flex-shrink: 0; background: #1E2022;',
        '}',
        '#ap-inp {',
        '  flex: 1; background: #2A2D30; border: 1px solid #2C2F33; border-radius: 7px;',
        '  color: #D4D8DC; padding: 8px 11px; font-size: 0.83rem; font-family: inherit;',
        '  outline: none; resize: none; min-height: 36px; max-height: 84px; line-height: 1.4;',
        '}',
        '#ap-inp:focus { border-color: #4CAF82; }',
        '#ap-snd {',
        '  background: #4CAF82; color: #111; border: none; border-radius: 7px;',
        '  padding: 0 14px; cursor: pointer; font-size: 0.86rem; font-weight: 600;',
        '  transition: background 0.15s; flex-shrink: 0;',
        '}',
        '#ap-snd:hover { background: #3fa173; }',
        '#ap-snd:disabled { background: #2C2F33; color: #4A5058; cursor: not-allowed; }'
    ].join('');
    p.document.head.appendChild(styleEl);

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
        keyInput.addEventListener('input', function() {
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
        // Replay existing messages
        for (var i = 0; i < state.messages.length; i++) {
            var item = state.messages[i];
            appendMsgUI(item.text, item.cls, item.via);
        }
    }
    updateQuotaUI();

    // ── Bind Event Listeners ─────────────────────────────────────────────────
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

        // Restriction: API Key Required
        if (!key) {
            appendMsgUI('OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in .env or secrets.toml.', 's');
            return;
        }

        // Restriction: Session Message Limit (cost cap)
        if (state.msgCount >= MAX_MSGS) {
            appendMsgUI('Session budget limit of ' + MAX_MSGS + ' messages reached. Refresh page to reset.', 's');
            return;
        }

        // Restriction: Query Length Limit
        if (text.length > MAX_CHARS) {
            appendMsgUI('Query exceeds ' + MAX_CHARS + ' characters. Please keep questions concise.', 's');
            return;
        }

        // Restriction: Rate Limit / Cooldown
        var now = Date.now();
        if (now - state.lastSend < COOLDOWN) {
            var waitSec = Math.ceil((COOLDOWN - (now - state.lastSend)) / 1000);
            appendMsgUI('Rate limit: please wait ' + waitSec + 's before sending another question.', 's');
            return;
        }

        // Add User Message
        appendMsgUI(text, 'u');
        state.messages.push({ text: text, cls: 'u' });
        state.history.push({ role: 'user', content: text });
        inpEl.value = '';
        inpEl.style.height = 'auto';
        state.msgCount++;
        state.lastSend = Date.now();
        updateQuotaUI();

        // Lock UI during query
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

    // ── UI Helpers ───────────────────────────────────────────────────────────
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
</script>
</body></html>"""

# Render iframe with injected SERVER_KEY from secrets / env
_RENDERED_CHATBOT_HTML = _CHATBOT_HTML.replace("__SERVER_KEY__", json.dumps(OPENROUTER_API_KEY))
components.html(_RENDERED_CHATBOT_HTML, height=0, scrolling=False)



# --- Data Caching -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_clean.parquet"
COMPAT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_2024_01_clean.parquet"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"


@st.cache_data(show_spinner="Loading real-world flight operations dataset...")
def load_dataset() -> pd.DataFrame:
    path = DATA_PATH if DATA_PATH.exists() else COMPAT_DATA_PATH
    if not path.exists():
        st.error(f"Processed flight dataset not found at {path}. Run: python -m src.transform.clean_flight_data")
        st.stop()

    cols = [
        "flight_date",
        "reporting_airline_code",
        "reporting_airline_name",
        "flight_number",
        "origin_airport_code",
        "origin_city_name",
        "destination_airport_code",
        "destination_city_name",
        "scheduled_departure_hour",
        "departure_time_block",
        "scheduled_arrival_hour",
        "departure_delay_minutes",
        "departure_delay_minutes_signed",
        "arrival_delay_minutes",
        "arrival_delay_minutes_signed",
        "departure_delayed_15",
        "arrival_delayed_15",
        "arrival_on_time",
        "flight_status",
        "cancelled",
        "diverted",
        "carrier_delay_minutes",
        "weather_delay_minutes",
        "national_air_system_delay_minutes",
        "security_delay_minutes",
        "late_aircraft_delay_minutes",
        "distance_miles",
        "distance_group",
        "route_code",
        "day_of_week",
        "is_weekend",
    ]
    df = pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path).columns])
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df


@st.cache_resource
def get_predictor() -> FlightDelayPredictor:
    predictor = FlightDelayPredictor()
    return predictor


@st.cache_data
def load_model_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


df_full = load_dataset()
predictor = get_predictor()
metadata = load_model_metadata()

# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #2C2F33;">
            <div style="background: #4CAF82; border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#111" viewBox="0 0 16 16">
                    <path d="M6.428 1.151C6.708.591 7.213 0 8 0s1.292.592 1.572 1.151C9.861 1.73 10 2.431 10 3v3.691l5.17 2.585a1.5 1.5 0 0 1 .83 1.342V12a.5.5 0 0 1-.582.493l-5.507-.918-.375 2.253 1.318 1.318A.5.5 0 0 1 10.5 16h-5a.5.5 0 0 1-.354-.854l1.319-1.318-.376-2.253-5.507.918A.5.5 0 0 1 0 12v-1.382a1.5 1.5 0 0 1 .83-1.342L6 6.691V3c0-.568.14-1.271.428-1.849z"/>
                </svg>
            </div>
            <div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #D4D8DC; letter-spacing: -0.01em;">AEROPULSE</div>
                <div style="font-size: 0.68rem; color: #606870; text-transform: uppercase; letter-spacing: 0.09em;">Operations Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; color:#606870; margin-bottom:8px;">Data Filters</p>',
        unsafe_allow_html=True,
    )

    # Airline filter
    available_airlines = sorted(df_full["reporting_airline_code"].dropna().unique())
    selected_airlines = st.multiselect(
        "Reporting Carrier",
        options=available_airlines,
        default=[],
        placeholder="All Carriers (14 Major Airlines)",
        help="Select one or more reporting airlines to filter metrics.",
    )

    # Top origin airports
    top_airports = df_full["origin_airport_code"].value_counts().head(30).index.tolist()
    selected_origins = st.multiselect(
        "Origin Hub",
        options=top_airports,
        default=[],
        placeholder="All Origin Hubs (300+ Airports)",
        help="Filter by primary departure airport.",
    )

    # Date range
    min_date = df_full["flight_date"].min().date()
    max_date = df_full["flight_date"].max().date()
    date_range = st.date_input(
        "Operational Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("<hr style='border-color:#2C2F33; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.74rem; color: #606870; line-height: 1.6;">
            <span style="color: #858C94; font-weight:600;">Dataset:</span> U.S. DOT On-Time Performance<br>
            <span style="color: #858C94; font-weight:600;">Source:</span> Kaggle / BTS Verified Extract<br>
            <span style="color: #858C94; font-weight:600;">Volume:</span> 469,968 Flight Records<br>
            <span style="color: #858C94; font-weight:600;">Cost Basis:</span> FAA $101.90 / min
        </div>
        """,
        unsafe_allow_html=True,
    )

# Filter dataset based on sidebar
filtered_df = df_full.copy()
if selected_airlines:
    filtered_df = filtered_df[filtered_df["reporting_airline_code"].isin(selected_airlines)]
if selected_origins:
    filtered_df = filtered_df[filtered_df["origin_airport_code"].isin(selected_origins)]
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["flight_date"].dt.date >= start_d)
        & (filtered_df["flight_date"].dt.date <= end_d)
    ]

# Common Plotly layout defaults for grey theme
_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#242628",
    font=dict(color="#9AA0A8", size=12),
    margin=dict(l=40, r=40, t=55, b=40),
)

# --- Navigation Tabs ----------------------------------------------------------
tab_kpi, tab_predictor, tab_benchmark, tab_diagnostics, tab_sql = st.tabs(
    [
        "Executive Overview",
        "ML Delay Simulator",
        "Carrier & Airport Benchmarks",
        "Diagnostics & Business Recommendations",
        "SQL & Star Schema",
    ]
)

# =============================================================================
# TAB 1: EXECUTIVE KPI COMMAND CENTER
# =============================================================================
with tab_kpi:
    st.markdown(
        """
        <div class="section-header">
            <h2>Executive Aviation Reliability Overview</h2>
            <p>Macro-level on-time performance (OTP-15), delay driver decomposition, and financial impact across U.S. airspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute Core KPIs
    total_flights = len(filtered_df)
    completed_mask = filtered_df["flight_status"] == "Completed"
    completed_df = filtered_df[completed_mask]
    cancelled_count = int((filtered_df["flight_status"] == "Cancelled").sum())
    diverted_count = int((filtered_df["flight_status"] == "Diverted").sum())

    total_completed = len(completed_df)
    on_time_arrivals = int((completed_df["arrival_on_time"] == True).sum())
    otp_pct = (on_time_arrivals / total_completed * 100) if total_completed else 0.0

    delayed_flights_count = int((completed_df["arrival_delayed_15"] == 1).sum())
    avg_arr_delay = float(completed_df["arrival_delay_minutes"].mean()) if total_completed else 0.0
    avg_dep_delay = float(completed_df["departure_delay_minutes"].mean()) if total_completed else 0.0
    total_delay_minutes = float(completed_df["arrival_delay_minutes"].sum())
    total_cost_usd = total_delay_minutes * FAA_DELAY_COST_PER_MINUTE_USD

    cancellation_rate = (cancelled_count / total_flights * 100) if total_flights else 0.0

    # KPI Ribbon
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Flight Volume</div>
                <div class="kpi-value">{total_flights:,}</div>
                <div class="kpi-subtext">Completed: {total_completed:,} ({total_completed/total_flights*100:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        otp_color = "kpi-delta-good" if otp_pct >= 80 else "kpi-delta-bad"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Arrival OTP-15</div>
                <div class="kpi-value">{otp_pct:.1f}%</div>
                <div class="{otp_color}">Delayed: {delayed_flights_count:,} ({100-otp_pct:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Arrival Delay</div>
                <div class="kpi-value">{avg_arr_delay:.1f} <span style="font-size:0.95rem;color:#606870;">min</span></div>
                <div class="kpi-subtext">Avg Departure: {avg_dep_delay:.1f} min</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Financial Cost Impact</div>
                <div class="kpi-value">${total_cost_usd/1e6:,.1f}M</div>
                <div class="kpi-subtext">FAA $101.90/min benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        canc_color = "kpi-delta-good" if cancellation_rate < 2.0 else "kpi-delta-bad"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Cancellation Rate</div>
                <div class="kpi-value">{cancellation_rate:.2f}%</div>
                <div class="{canc_color}">{cancelled_count:,} flights cancelled</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # Charts Row 1
    ch_col1, ch_col2 = st.columns([3, 2])

    with ch_col1:
        hourly = (
            completed_df.groupby("scheduled_departure_hour")
            .agg(
                avg_delay=("arrival_delay_minutes", "mean"),
                delay_rate=("arrival_delayed_15", lambda x: (x == 1).mean() * 100),
                flight_volume=("flight_number", "count"),
            )
            .reset_index()
        )
        hourly = hourly[hourly["scheduled_departure_hour"].between(5, 23)]

        fig_cascade = go.Figure()
        fig_cascade.add_trace(
            go.Bar(
                x=hourly["scheduled_departure_hour"],
                y=hourly["flight_volume"],
                name="Flight Volume",
                marker_color="rgba(76,175,130,0.20)",
                yaxis="y2",
            )
        )
        fig_cascade.add_trace(
            go.Scatter(
                x=hourly["scheduled_departure_hour"],
                y=hourly["avg_delay"],
                name="Avg Arrival Delay (min)",
                line=dict(color="#C0392B", width=2.5),
                mode="lines+markers",
                marker=dict(size=5),
            )
        )
        fig_cascade.add_trace(
            go.Scatter(
                x=hourly["scheduled_departure_hour"],
                y=hourly["delay_rate"],
                name="Delayed Flight Rate (% >=15m)",
                line=dict(color="#D4A017", width=2, dash="dot"),
                mode="lines",
            )
        )
        fig_cascade.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text="<b>Time-of-Day Delay Cascade Effect</b>", font=dict(size=13, color="#C8CDD3")),
            xaxis=dict(title="Scheduled Departure Hour (24h)", gridcolor="#2C2F33", tickmode="linear", dtick=1),
            yaxis=dict(title="Delay (min) / Rate (%)", gridcolor="#2C2F33"),
            yaxis2=dict(title="Flight Volume", overlaying="y", side="right", showgrid=False, color="#606870"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            height=380,
        )
        st.plotly_chart(fig_cascade, use_container_width=True)

    with ch_col2:
        delay_causes = {
            "Late Aircraft Cascade": completed_df["late_aircraft_delay_minutes"].sum(),
            "Air Carrier Control": completed_df["carrier_delay_minutes"].sum(),
            "National Airspace (NAS)": completed_df["national_air_system_delay_minutes"].sum(),
            "Severe Weather": completed_df["weather_delay_minutes"].sum(),
            "Security Incident": completed_df["security_delay_minutes"].sum(),
        }
        total_causes = sum(delay_causes.values())
        cause_df = pd.DataFrame(
            [
                {"Cause": k, "Minutes": v, "Percentage": (v / total_causes * 100) if total_causes else 0}
                for k, v in delay_causes.items()
            ]
        ).sort_values("Minutes", ascending=True)

        fig_cause = px.bar(
            cause_df,
            x="Minutes",
            y="Cause",
            orientation="h",
            text=cause_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
            color="Percentage",
            color_continuous_scale=["#4CAF82", "#C0392B"],
            title="<b>Root-Cause Delay Breakdown</b>",
        )
        fig_cause.update_layout(
            **_PLOT_LAYOUT,
            coloraxis_showscale=False,
            xaxis=dict(title="Total Delay Minutes", gridcolor="#2C2F33"),
            yaxis=dict(title=""),
            height=380,
        )
        st.plotly_chart(fig_cause, use_container_width=True)

    # Daily OTP Trend
    daily = (
        completed_df.groupby(completed_df["flight_date"].dt.date)
        .agg(
            total_flights=("flight_number", "count"),
            otp=("arrival_on_time", lambda x: (x == True).mean() * 100),
            avg_delay=("arrival_delay_minutes", "mean"),
        )
        .reset_index()
    )

    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Bar(
            x=daily["flight_date"],
            y=daily["total_flights"],
            name="Daily Completed Flights",
            marker_color="rgba(76,175,130,0.22)",
            yaxis="y2",
        )
    )
    fig_daily.add_trace(
        go.Scatter(
            x=daily["flight_date"],
            y=daily["otp"],
            name="On-Time Performance (%)",
            line=dict(color="#4CAF82", width=2.5),
            mode="lines+markers",
            marker=dict(size=5),
        )
    )
    fig_daily.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text="<b>Daily Flight Volume & On-Time Performance Trend</b>", font=dict(size=13, color="#C8CDD3")),
        xaxis=dict(title="Date", gridcolor="#2C2F33"),
        yaxis=dict(title="OTP-15 (%)", range=[50, 100], gridcolor="#2C2F33"),
        yaxis2=dict(title="Total Flights", overlaying="y", side="right", showgrid=False, color="#606870"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        height=320,
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# =============================================================================
# TAB 2: INTERACTIVE ML DELAY PREDICTOR
# =============================================================================
with tab_predictor:
    st.markdown(
        """
        <div class="section-header">
            <h2>Live Predictive Delay Simulator</h2>
            <p>Run real-time inference using trained Histogram Gradient Boosting models on the out-of-time test distribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_c1, pred_c2 = st.columns([1, 2])

    with pred_c1:
        st.markdown(
            '<p style="font-weight:600; color:#9AA0A8; margin-bottom:12px; font-size:0.88rem;">Flight Parameters</p>',
            unsafe_allow_html=True,
        )

        sim_carrier = st.selectbox(
            "Airline Carrier",
            options=["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "HA", "VX", "OO", "EV", "MQ", "US"],
            index=0,
            help="Operating air carrier (IATA code)",
        )

        top_hubs = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "BOS"]
        sim_origin = st.selectbox("Origin Airport", options=top_hubs, index=5)
        sim_dest = st.selectbox("Destination Airport", options=top_hubs, index=4)

        sim_hour = st.slider("Scheduled Departure Hour", min_value=5, max_value=23, value=17, format="%02d:00")
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sim_day_str = st.selectbox("Day of Week", options=day_names, index=4)
        sim_day_num = day_names.index(sim_day_str) + 1
        sim_is_weekend = sim_day_num in [6, 7]

        sim_dist_group = st.slider("Distance Group (1=Short <250mi, 11=Transcontinental >2500mi)", 1, 11, 10)

        run_pred_btn = st.button("Run Prediction", use_container_width=True, type="primary")

    with pred_c2:
        pred_result = predictor.predict(
            carrier=sim_carrier,
            origin=sim_origin,
            destination=sim_dest,
            scheduled_hour=sim_hour,
            day_of_week=sim_day_num,
            distance_group=sim_dist_group,
            is_weekend=sim_is_weekend,
        )

        risk = pred_result["risk_level"]
        prob_pct = pred_result["delay_probability_pct"]
        est_min = pred_result["estimated_delay_minutes"]
        cost = pred_result["cost_impact_usd"]

        if risk == "low":
            risk_color = "#4CAF82"
            risk_badge = "LOW RISK"
        elif risk == "moderate":
            risk_color = "#D4A017"
            risk_badge = "MODERATE RISK"
        else:
            risk_color = "#C0392B"
            risk_badge = "HIGH RISK"

        st.markdown(
            f"""
            <div style="background: #242628; border: 1px solid #2C2F33; border-radius: 10px; padding: 22px; margin-bottom: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #D4D8DC;">
                        {sim_carrier} — {sim_origin} to {sim_dest}
                    </div>
                    <div style="background: {risk_color}18; border: 1px solid {risk_color}55; color: {risk_color}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.75rem;">
                        {risk_badge}
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 14px;">
                    <div>
                        <div class="kpi-title">Delay Probability</div>
                        <div style="font-size: 2.0rem; font-weight: 800; color: {risk_color};">{prob_pct:.1f}%</div>
                        <div class="kpi-subtext">Arrival delay >= 15 min</div>
                    </div>
                    <div>
                        <div class="kpi-title">Estimated Delay</div>
                        <div style="font-size: 2.0rem; font-weight: 800; color: #D4D8DC;">{est_min:.1f} <span style="font-size:0.95rem; color:#606870;">min</span></div>
                        <div class="kpi-subtext">GBM Duration Regression</div>
                    </div>
                    <div>
                        <div class="kpi-title">Direct Cost Exposure</div>
                        <div style="font-size: 2.0rem; font-weight: 800; color: #D4D8DC;">${cost:,.0f}</div>
                        <div class="kpi-subtext">FAA $101.90 / min rate</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "<b>Operational Delay Risk</b>", "font": {"size": 13, "color": "#9AA0A8"}},
                number={"suffix": "%", "font": {"color": "#D4D8DC", "size": 30}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#606870"},
                    "bar": {"color": risk_color},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 1,
                    "bordercolor": "#2C2F33",
                    "steps": [
                        {"range": [0, 20], "color": "rgba(76,175,130,0.12)"},
                        {"range": [20, 40], "color": "rgba(212,160,23,0.12)"},
                        {"range": [40, 100], "color": "rgba(192,57,43,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#C0392B", "width": 2.5},
                        "thickness": 0.75,
                        "value": 40,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=220,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.info(f"Operations Intelligence: {pred_result['interpretation']}")


# =============================================================================
# TAB 3: CARRIER & AIRPORT BENCHMARKS
# =============================================================================
with tab_benchmark:
    st.markdown(
        """
        <div class="section-header">
            <h2>Carrier & Airport Reliability Benchmarks</h2>
            <p>Head-to-head performance rankings, airport hub congestion bottlenecks, and route vulnerability analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bench_c1, bench_c2 = st.columns(2)

    with bench_c1:
        carrier_perf = (
            completed_df.groupby("reporting_airline_code")
            .agg(
                total_flights=("flight_number", "count"),
                otp=("arrival_on_time", lambda x: (x == True).mean() * 100),
                avg_delay=("arrival_delay_minutes", "mean"),
            )
            .reset_index()
            .sort_values("otp", ascending=True)
        )

        fig_carrier = px.bar(
            carrier_perf,
            x="otp",
            y="reporting_airline_code",
            orientation="h",
            text=carrier_perf["otp"].apply(lambda x: f"{x:.1f}%"),
            color="otp",
            color_continuous_scale=["#C0392B", "#D4A017", "#4CAF82"],
            title="<b>Airline On-Time Performance (OTP-15) Ranking</b>",
        )
        fig_carrier.update_layout(
            **_PLOT_LAYOUT,
            xaxis=dict(title="On-Time Arrival Rate (%)", range=[50, 100], gridcolor="#2C2F33"),
            yaxis=dict(title="Airline Code"),
            coloraxis_showscale=False,
            height=420,
        )
        st.plotly_chart(fig_carrier, use_container_width=True)

    with bench_c2:
        hub_perf = (
            completed_df.groupby("origin_airport_code")
            .agg(
                flight_volume=("flight_number", "count"),
                delay_rate=("arrival_delayed_15", lambda x: (x == 1).mean() * 100),
                avg_delay=("arrival_delay_minutes", "mean"),
            )
            .reset_index()
        )
        top_hubs_perf = hub_perf[hub_perf["flight_volume"] >= 2000].sort_values("delay_rate", ascending=False).head(15)

        fig_hub = px.bar(
            top_hubs_perf,
            x="delay_rate",
            y="origin_airport_code",
            orientation="h",
            text=top_hubs_perf["delay_rate"].apply(lambda x: f"{x:.1f}%"),
            color="delay_rate",
            color_continuous_scale=["#4CAF82", "#C0392B"],
            title="<b>Top 15 Most Congested Departure Hubs (>=2,000 flights)</b>",
        )
        fig_hub.update_layout(
            **_PLOT_LAYOUT,
            xaxis=dict(title="Delayed Flight Rate (% >=15m)", gridcolor="#2C2F33"),
            yaxis=dict(title="Airport Code", autorange="reversed"),
            coloraxis_showscale=False,
            height=420,
        )
        st.plotly_chart(fig_hub, use_container_width=True)

    st.markdown(
        '<p style="font-weight:600; color:#9AA0A8; margin: 16px 0 6px 0; font-size:0.88rem;">High-Risk Operational Corridors</p>',
        unsafe_allow_html=True,
    )
    route_stats = (
        completed_df.groupby("route_code")
        .agg(
            volume=("flight_number", "count"),
            avg_delay=("arrival_delay_minutes", "mean"),
            delay_rate=("arrival_delayed_15", lambda x: (x == 1).mean() * 100),
            total_delay_cost=("arrival_delay_minutes", lambda x: x.sum() * FAA_DELAY_COST_PER_MINUTE_USD),
        )
        .reset_index()
    )
    high_volume_routes = route_stats[route_stats["volume"] >= 200].sort_values("avg_delay", ascending=False).head(20)

    fig_routes = px.scatter(
        high_volume_routes,
        x="volume",
        y="avg_delay",
        size="total_delay_cost",
        color="delay_rate",
        hover_name="route_code",
        color_continuous_scale=["#4CAF82", "#D4A017", "#C0392B"],
        title="<b>Route Risk Matrix: Volume vs. Average Delay Duration</b>",
        labels={"volume": "Monthly Flight Volume", "avg_delay": "Average Delay (Minutes)", "delay_rate": "Delay Rate (%)"},
    )
    fig_routes.update_layout(
        **_PLOT_LAYOUT,
        height=400,
    )
    st.plotly_chart(fig_routes, use_container_width=True)


# =============================================================================
# TAB 4: DIAGNOSTICS & BUSINESS RECOMMENDATIONS
# =============================================================================
with tab_diagnostics:
    st.markdown(
        """
        <div class="section-header">
            <h2>Model Diagnostics, Error Analysis & Business ROI</h2>
            <p>Rigorous evaluation across baselines, out-of-time test validation, residual analysis, and actionable business recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    diag_c1, diag_c2 = st.columns([3, 2])

    with diag_c1:
        st.markdown(
            '<p style="font-weight:600; color:#9AA0A8; margin-bottom:6px; font-size:0.88rem;">Out-of-Time Model Benchmark Comparison</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="font-size: 0.83rem; color: #606870; margin-bottom: 12px;">
                Models evaluated on a strict <b style="color:#9AA0A8;">chronological out-of-time split</b> — Days 1–23 train (345,440 flights), Days 24–31 test (111,573 flights) — to prevent temporal data leakage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        reg_results = metadata.get("regression_results", [])
        if reg_results:
            bench_df = pd.DataFrame(reg_results)
            bench_df.columns = ["Model Architecture", "MAE (min)", "RMSE (min)", "Median AE (min)", "MAPE (%)", "R2"]
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python -m src.ml.train` to populate the benchmark comparison table.")

        st.markdown(
            '<p style="font-weight:600; color:#9AA0A8; margin: 14px 0 6px 0; font-size:0.88rem;">Classification Performance (Delay >= 15 min)</p>',
            unsafe_allow_html=True,
        )
        cls_results = metadata.get("classification_results", [])
        if cls_results:
            cls_df = pd.DataFrame(cls_results)
            cls_df.columns = ["Model Architecture", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
            st.dataframe(cls_df, use_container_width=True, hide_index=True)

        st.markdown(
            '<p style="font-weight:600; color:#9AA0A8; margin: 14px 0 6px 0; font-size:0.88rem;">Error Characteristics & Heavy-Tail Delays</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            > **Why is RMSE notably higher than MAE?**
            >
            > In commercial aviation, arrival delay exhibits severe positive skewness. While over 78% of flights arrive within schedule, a small fraction encounter multi-hour delays from ATC ground stops or severe weather events. Because RMSE squares deviations, these outlier delays inflate RMSE significantly. **MAE and Median Absolute Error** provide more robust operational metrics for daily gate scheduling.
            """
        )

    with diag_c2:
        st.markdown(
            '<p style="font-weight:600; color:#9AA0A8; margin-bottom:6px; font-size:0.88rem;">Permutation Feature Importance</p>',
            unsafe_allow_html=True,
        )
        feat_imp = metadata.get("feature_importance", {})
        if feat_imp:
            feat_df = pd.DataFrame(
                [{"Feature": k.replace("_", " ").title(), "Importance": v} for k, v in feat_imp.items() if v > 0]
            ).sort_values("Importance", ascending=True)

            fig_feat = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text=feat_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
                color="Importance",
                color_continuous_scale=["#2C5F4A", "#4CAF82"],
                title="<b>Top Delay Predictors (Permutation Importance)</b>",
            )
            fig_feat.update_layout(
                **_PLOT_LAYOUT,
                coloraxis_showscale=False,
                xaxis=dict(title="Normalized Importance", gridcolor="#2C2F33"),
                yaxis=dict(title=""),
                height=340,
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("<hr style='border-color:#2C2F33; margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-weight:600; color:#9AA0A8; margin-bottom:14px; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.06em;">Actionable Business Recommendations & ROI</p>',
        unsafe_allow_html=True,
    )

    recs = metadata.get("business_recommendations", [])
    if not recs:
        from src.ml.evaluate import generate_business_recommendations
        recs = generate_business_recommendations(metadata)

    r_col1, r_col2, r_col3 = st.columns(3)
    cols_list = [r_col1, r_col2, r_col3]

    for idx, rec in enumerate(recs):
        with cols_list[idx % 3]:
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-tag">{rec['category']}</div>
                    <div class="rec-title">{rec['title']}</div>
                    <div class="rec-body"><b>Observed Finding:</b> {rec['finding']}</div>
                    <div class="rec-body" style="margin-top:6px;"><b>Operational Decision:</b> {rec['recommendation']}</div>
                    <div class="rec-roi"><b>Estimated ROI:</b> {rec['estimated_roi']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# TAB 5: SQL & STAR SCHEMA INTELLIGENCE
# =============================================================================
with tab_sql:
    st.markdown(
        """
        <div class="section-header">
            <h2>SQL & Star Schema Intelligence</h2>
            <p>Explore analytical warehouse views, star schema design, and production SQL patterns deployed in the warehouse.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sql_c1, sql_c2 = st.columns([1, 1])

    with sql_c1:
        st.markdown(
            '<p style="font-weight:600; color:#9AA0A8; margin-bottom:8px; font-size:0.88rem;">Dimensional Warehouse Architecture (Star Schema)</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ```
            +----------------------------------------------+
            |       fact_flight  (Fact Table)              |
            +----------------------------------------------+
            | flight_key (PK)                              |
            | date_key (FK)    -----> dim_date             |
            | airline_key (FK) -----> dim_airline          |
            | origin_airport_key ---> dim_airport          |
            | dest_airport_key ----->  dim_airport         |
            | scheduled_departure_time                     |
            | actual_departure_time                        |
            | arrival_delay_minutes                        |
            | arrival_delayed_15                           |
            | carrier_delay_minutes                        |
            | weather_delay_minutes                        |
            | national_air_system_delay_minutes            |
            | late_aircraft_delay_minutes                  |
            | total_reported_delay_minutes                 |
            +----------------------------------------------+
            ```
            """
        )

    with sql_c2:
        st.markdown(
            '<p style="font-weight:600; color:#9AA0A8; margin-bottom:8px; font-size:0.88rem;">Production Analytical Views</p>',
            unsafe_allow_html=True,
        )
        view_choice = st.selectbox(
            "Select Analytical SQL View",
            options=[
                "vw_carrier_monthly_performance",
                "vw_airport_congestion_hourly",
                "vw_delay_cascade_root_cause",
                "vw_high_risk_corridors",
            ],
        )

        if view_choice == "vw_carrier_monthly_performance":
            st.code(
                """
-- Monthly Carrier On-Time Performance & Direct Financial Impact
CREATE OR REPLACE VIEW warehouse.vw_carrier_monthly_performance AS
SELECT
    da.reporting_airline_code,
    dd.year_number,
    dd.month_number,
    COUNT(*) AS total_flights,
    SUM(CASE WHEN ff.flight_status = 'Completed' THEN 1 ELSE 0 END) AS completed_flights,
    ROUND(AVG(CASE WHEN ff.arrival_on_time = TRUE THEN 1.0 ELSE 0.0 END) * 100, 2) AS otp_15_pct,
    ROUND(AVG(ff.arrival_delay_minutes)::NUMERIC, 2) AS avg_delay_minutes,
    ROUND((SUM(ff.arrival_delay_minutes) * 101.90)::NUMERIC, 2) AS total_delay_cost_usd
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airline da ON ff.airline_key = da.airline_key
JOIN warehouse.dim_date dd ON ff.date_key = dd.date_key
GROUP BY 1, 2, 3
ORDER BY otp_15_pct DESC;
                """,
                language="sql",
            )
        elif view_choice == "vw_airport_congestion_hourly":
            st.code(
                """
-- Hub Congestion Cascade by Departure Hour Block
CREATE OR REPLACE VIEW warehouse.vw_airport_congestion_hourly AS
SELECT
    dp.airport_code,
    dp.city_name,
    ff.scheduled_departure_hour,
    COUNT(*) AS departures_count,
    ROUND(AVG(ff.departure_delay_minutes)::NUMERIC, 2) AS avg_departure_delay,
    ROUND(AVG(ff.arrival_delay_minutes)::NUMERIC, 2) AS avg_arrival_delay,
    ROUND(AVG(CASE WHEN ff.arrival_delayed_15 = TRUE THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_risk_pct
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airport dp ON ff.origin_airport_key = dp.airport_key
GROUP BY 1, 2, 3
ORDER BY dp.airport_code, ff.scheduled_departure_hour;
                """,
                language="sql",
            )
        elif view_choice == "vw_delay_cascade_root_cause":
            st.code(
                """
-- Decomposition of Root-Cause Delays Across Airspace
CREATE OR REPLACE VIEW warehouse.vw_delay_cascade_root_cause AS
SELECT
    da.reporting_airline_code,
    SUM(ff.carrier_delay_minutes) AS carrier_delay_min,
    SUM(ff.weather_delay_minutes) AS weather_delay_min,
    SUM(ff.national_air_system_delay_minutes) AS nas_delay_min,
    SUM(ff.late_aircraft_delay_minutes) AS late_aircraft_delay_min,
    ROUND(SUM(ff.late_aircraft_delay_minutes) /
          NULLIF(SUM(ff.total_reported_delay_minutes), 0) * 100, 2) AS cascade_pct
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airline da ON ff.airline_key = da.airline_key
WHERE ff.delay_cause_reported = TRUE
GROUP BY 1
ORDER BY late_aircraft_delay_min DESC;
                """,
                language="sql",
            )
        else:
            st.code(
                """
-- Top High-Risk Operational Flight Corridors
CREATE OR REPLACE VIEW warehouse.vw_high_risk_corridors AS
SELECT
    ff.route_code,
    orig.airport_code AS origin_code,
    dest.airport_code AS dest_code,
    COUNT(*) AS flight_count,
    ROUND(AVG(ff.arrival_delay_minutes)::NUMERIC, 2) AS avg_delay_minutes,
    ROUND(AVG(CASE WHEN ff.arrival_delayed_15 = TRUE THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_probability_pct,
    ROUND((SUM(ff.arrival_delay_minutes) * 101.90)::NUMERIC, 2) AS total_financial_exposure_usd
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airport orig ON ff.origin_airport_key = orig.airport_key
JOIN warehouse.dim_airport dest ON ff.destination_airport_key = dest.airport_key
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 200
ORDER BY avg_delay_minutes DESC;
                """,
                language="sql",
            )

    st.markdown(
        '<p style="font-weight:600; color:#9AA0A8; margin: 16px 0 6px 0; font-size:0.88rem;">Live SQL Query Result Preview (calculated over real flight records)</p>',
        unsafe_allow_html=True,
    )
    sql_preview = (
        completed_df.groupby("reporting_airline_code")
        .agg(
            Total_Flights=("flight_number", "count"),
            OTP_15_Pct=("arrival_on_time", lambda x: round((x == True).mean() * 100, 2)),
            Avg_Delay_Min=("arrival_delay_minutes", lambda x: round(x.mean(), 2)),
            Cost_Exposure_USD=("arrival_delay_minutes", lambda x: f"${x.sum() * FAA_DELAY_COST_PER_MINUTE_USD / 1e6:.2f}M"),
        )
        .reset_index()
        .sort_values("OTP_15_Pct", ascending=False)
    )
    st.dataframe(sql_preview, use_container_width=True, hide_index=True)

# --- Footer -------------------------------------------------------------------
st.markdown("<hr style='border-color:#2C2F33; margin-top:28px;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #4A5058; font-size: 0.76rem; padding: 10px 0 24px 0;">
        AeroPulse Aviation Intelligence &mdash; End-to-End Data Science, Operations Research &amp; Analytics Platform<br>
        Built with Streamlit, Plotly, Scikit-Learn, PyArrow &amp; PostgreSQL &mdash; MIT Licensed
    </div>
    """,
    unsafe_allow_html=True,
)
