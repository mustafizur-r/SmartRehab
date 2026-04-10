"""
gradio_ui.py — SmartRehab Gait Chat Interface  v2
Run: python gradio_ui.py
Connects to app_server.py running on http://localhost:8000

CHANGES FROM v1:
  1. _format_refine_result() replaces _format_impairments() — shows BOTH
     gait impairments AND custom joint offsets in the assistant reply
  2. intent badge ("gait" or "offset") shown per refinement turn
  3. _call_refine() now reads the new fields: intent, custom_offsets, labels
  4. Session state panel shows both impairment count and offset count
"""
from __future__ import annotations

import gradio as gr
import requests
import os

API_BASE   = "http://localhost:8000"
VIDEO_PATH = "video_result/Final_Fbx_Mesh_Animation.mp4"


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_gen(prompt: str) -> dict:
    """Call /gen_text2motion/ with video_render=true."""
    try:
        r = requests.get(
            f"{API_BASE}/gen_text2motion/",
            params={"text_prompt": prompt, "video_render": "true"},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e), "session_id": "",
                "base_video_url": ""}


def _call_refine(session_id: str, prompt: str) -> dict:
    """Call /refine_motion/ with video_render=true.

    New response fields (v6 app_server):
      intent           : "gait" | "offset"
      impairment_state : dict   {key: severity}
      custom_offsets   : list   [{joint_key, delta, phase, label}, ...]
      labels           : list   of human-readable strings
      message          : str
    """
    try:
        r = requests.post(
            f"{API_BASE}/refine_motion/",
            params={"video_render": "true"},
            json={"session_id": session_id, "prompt": prompt},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {
            "status":           "error",
            "message":          str(e),
            "intent":           "",
            "impairment_state": {},
            "custom_offsets":   [],
            "labels":           [],
        }


def _call_session_state(session_id: str) -> dict:
    """Call /session_state/{session_id}."""
    try:
        r = requests.get(f"{API_BASE}/session_state/{session_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _call_cleanup(session_id: str) -> None:
    """Call DELETE /cleanup_session/ to free BVH + video files for a session."""
    if not session_id:
        return
    try:
        requests.delete(
            f"{API_BASE}/cleanup_session/",
            params={"session_id": session_id},
            timeout=15,
        )
    except Exception:
        pass


def _get_base_video(session_id: str) -> str | None:
    """Return local path for the base video of this session, or None."""
    if not session_id:
        return None
    path = f"video_result/base_{session_id}.mp4"
    return path if os.path.exists(path) else None


def _get_video() -> str | None:
    return VIDEO_PATH if os.path.exists(VIDEO_PATH) else None


# ─────────────────────────────────────────────────────────────────────────────
# Result formatters
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable labels for gait impairment keys
_IMP_LABELS = {
    "ankle_drop_right":              "Ankle drop (R)",
    "ankle_drop_left":               "Ankle drop (L)",
    "knee_stiffness_right":          "Stiff knee (R)",
    "knee_stiffness_left":           "Stiff knee (L)",
    "stride_asymmetry_right":        "Stride asymmetry (R)",
    "stride_asymmetry_left":         "Stride asymmetry (L)",
    "trunk_lean_right":              "Trunk lean (R)",
    "trunk_lean_left":               "Trunk lean (L)",
    "arm_swing_reduction_right":     "Arm swing ↓ (R)",
    "arm_swing_reduction_left":      "Arm swing ↓ (L)",
    "hip_hike_right":                "Hip hike (R)",
    "hip_hike_left":                 "Hip hike (L)",
    "cadence_reduction":             "Cadence ↓",
    "forward_lean":                  "Forward lean",
    "wide_base":                     "Wide base",
    "hemiplegic_right":              "Hemiplegic (R)",
    "hemiplegic_left":               "Hemiplegic (L)",
    "parkinsonian_shuffle":          "Parkinsonian shuffle",
    "festinating_gait":              "Festinating gait",
    "freezing_of_gait":              "Freezing of gait",
    "ataxic_gait":                   "Ataxic gait",
    "choreic_gait":                  "Choreic gait",
    "cerebellar_ataxia":             "Cerebellar ataxia",
    "crouch_gait":                   "Crouch gait",
    "scissor_gait":                  "Scissor gait",
    "diplegic":                      "Diplegic (bilateral CP)",
    "myopathic":                     "Myopathic",
    "sensory_ataxia":                "Sensory ataxia",
    "waddling_gait":                 "Waddling gait",
    "dystonic_right":                "Dystonic (R)",
    "dystonic_left":                 "Dystonic (L)",
    "equinus_right":                 "Equinus / toe-walk (R)",
    "equinus_left":                  "Equinus / toe-walk (L)",
    "antalgic_right":                "Antalgic (R)",
    "antalgic_left":                 "Antalgic (L)",
    "hip_extensor_weakness_right":   "Hip extensor weak (R)",
    "hip_extensor_weakness_left":    "Hip extensor weak (L)",
    "leg_length_short_right":        "Leg length discrepancy (R short)",
    "leg_length_short_left":         "Leg length discrepancy (L short)",
}


def _bar(v: float) -> str:
    """Render a tiny 10-cell progress bar for a 0-1 severity value."""
    filled = int(round(v * 10))
    return "█" * filled + "░" * (10 - filled)


def _format_refine_result(result: dict) -> str:
    """
    Build the assistant reply markdown for a /refine_motion/ response.
    Handles both gait impairments and custom joint offsets.
    """
    intent         = result.get("intent", "")
    imp_state      = result.get("impairment_state", {})
    custom_offsets = result.get("custom_offsets", [])
    labels         = result.get("labels", [])
    msg            = result.get("message", "")

    lines = []

    # ── Intent badge ──────────────────────────────────────────────────────────
    if intent == "gait":
        lines.append("**Motion type:** `clinical gait syndrome`")
    elif intent == "offset":
        lines.append("**Motion type:** `custom joint offset`")

    lines.append("")

    # ── Gait impairments section ──────────────────────────────────────────────
    if imp_state:
        lines.append("**Active gait impairments:**")
        for k, v in imp_state.items():
            label = _IMP_LABELS.get(k, k.replace("_", " ").title())
            lines.append(f"- {label}: `{_bar(v)}` {v:.2f}")
    else:
        if intent == "gait":
            lines.append("*No gait impairments active — motion reset to normal.*")

    # ── Custom joint offsets section ──────────────────────────────────────────
    if custom_offsets:
        lines.append("")
        lines.append("**Active joint offsets:**")
        for o in custom_offsets:
            joint = o.get("joint_key", "?").replace("_", " ")
            delta = float(o.get("delta", 0.0))
            phase = o.get("phase", "all")
            lbl   = o.get("label", joint)
            sign  = "+" if delta >= 0 else ""
            ph    = f" *(swing only)*" if "swing" in phase else \
                    f" *(stance only)*" if "stance" in phase else ""
            lines.append(f"- {lbl}: `{sign}{delta:.1f}°`{ph}")
    else:
        if intent == "offset":
            lines.append("")
            lines.append("*All joint offsets cleared.*")

    lines.append("")
    lines.append(f"*{msg}*")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Chat logic
# ─────────────────────────────────────────────────────────────────────────────

def send_message(user_msg, history, session_id, is_first_message):
    """
    Process one chat turn.
    Yields: (history, session_id, is_first, base_video, modified_video, status)
    """
    if not user_msg.strip():
        return history, session_id, is_first_message, _get_base_video(session_id), _get_video(), ""

    history = history + [{"role": "user", "content": user_msg}]

    yield (
        history + [{"role": "assistant", "content": "⏳ Generating motion..."}],
        session_id, is_first_message, None, None, "🔄 Processing...",
    )

    if is_first_message:
        result     = _call_gen(user_msg)
        new_sid    = result.get("session_id", "")
        status     = result.get("status", "error")
        exp_prompt = result.get("expressive_prompt", "")
        eng_prompt = result.get("english_prompt", user_msg)

        if status == "success":
            reply = (
                f"**Base motion generated.**\n\n"
                f"**English prompt:** {eng_prompt}\n\n"
                f"**Motion description:**\n> {exp_prompt}\n\n"
                "---\n"
                "Refine with clinical syndromes or body-part adjustments:\n\n"
                "**Gait syndromes:** 'right foot drop', 'Parkinson with freezing', 'moderate right hemiplegia'\n\n"
                "**Joint adjustments:** 'patient hands on chest', 'left knee 20 degrees more bent', 'head down'"
            )
            history    = history + [{"role": "assistant", "content": reply}]
            base_vid   = _get_base_video(new_sid)
            mod_vid    = _get_video()
            yield history, new_sid, False, base_vid, mod_vid, "✅ Done"
        else:
            msg = result.get("message", "Unknown error")
            history = history + [{"role": "assistant", "content": f"❌ Error: {msg}"}]
            yield history, session_id, True, None, None, f"❌ {msg}"

    else:
        if not session_id:
            history = history + [{"role": "assistant", "content":
                "❌ No active session. Please start a new chat first."}]
            yield history, session_id, is_first_message, None, None, "❌ No session"
            return

        result = _call_refine(session_id, user_msg)
        status = result.get("status", "error")

        if status == "success":
            reply    = _format_refine_result(result)
            history  = history + [{"role": "assistant", "content": reply}]
            base_vid = _get_base_video(session_id)
            mod_vid  = _get_video()
            yield history, session_id, False, base_vid, mod_vid, "✅ Done"
        else:
            err = result.get("message", "Unknown error")
            history = history + [{"role": "assistant", "content": f"❌ Error: {err}"}]
            yield history, session_id, is_first_message, _get_base_video(session_id), _get_video(), f"❌ {err}"


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --border:   #2a2d3a;
    --accent:   #4f8ef7;
    --accent2:  #7c5cbf;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --success:  #34d399;
    --error:    #f87171;
    --radius:   12px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

.main-row { gap: 0 !important; }

.left-panel {
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 0;
}

.panel-header {
    padding: 20px 16px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}

.panel-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

#new-chat-btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: calc(100% - 32px) !important;
    margin: 12px 16px !important;
}
#new-chat-btn:hover { opacity: 0.85 !important; }

.chat-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.chat-list-item {
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background 0.15s;
}
.chat-list-item:hover  { background: var(--border); color: var(--text); }
.chat-list-item.active { background: rgba(79,142,247,0.15); color: var(--accent); }

#chatbot {
    background: var(--bg) !important;
    border: none !important;
    flex: 1 !important;
}
#chatbot * { color: #e2e8f0 !important; }

#chatbot .user,
#chatbot [data-testid="user"] > div,
#chatbot [data-testid="user"] > div > div {
    background: rgba(79,142,247,0.18) !important;
    border: 1px solid rgba(79,142,247,0.3) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #e2e8f0 !important;
}

#chatbot .bot,
#chatbot [data-testid="bot"] > div,
#chatbot [data-testid="bot"] > div > div {
    background: #1e2235 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #e2e8f0 !important;
}

#chatbot p, #chatbot span, #chatbot li, #chatbot ul, #chatbot ol,
#chatbot strong, #chatbot em, #chatbot b,
#chatbot h1, #chatbot h2, #chatbot h3, #chatbot h4 {
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
}

#chatbot blockquote {
    border-left: 3px solid #4f8ef7 !important;
    margin: 8px 0 !important;
    padding: 4px 12px !important;
    background: rgba(79,142,247,0.08) !important;
    border-radius: 4px !important;
}
#chatbot blockquote * { color: #94a3b8 !important; }

#chatbot code {
    background: rgba(255,255,255,0.1) !important;
    padding: 2px 5px !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
    color: #7dd3fc !important;
}

#chatbot hr { border-color: var(--border) !important; margin: 8px 0 !important; }
#chatbot [class*="placeholder"] { color: #64748b !important; }
#chatbot [class*="placeholder"] * { color: #64748b !important; }
#chatbot [class*="avatar"] { background: transparent !important; }

#msg-input textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    padding: 12px !important;
    resize: none !important;
}
#msg-input textarea:focus { border-color: var(--accent) !important; outline: none !important; }
#msg-input textarea::placeholder { color: var(--muted) !important; }

#send-btn {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
#send-btn:hover { opacity: 0.85 !important; }
#send-btn:disabled,
#send-btn[disabled] {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
}

.right-panel {
    background: var(--bg);
    display: flex;
    flex-direction: column;
    height: 100vh;
}

.video-header {
    padding: 20px 20px 12px;
    border-bottom: 1px solid var(--border);
}

.video-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

#video-base, #video-modified {
    border-radius: var(--radius) !important;
    overflow: hidden;
    margin: 0 16px 0;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

/* Kill Gradio's default border/padding on the spinner wrapper */
#spinner-wrap {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
#spinner-wrap > div {
    border: none !important;
    padding: 0 !important;
}

/* Spinner between videos */
.vid-spinner-hidden { display: none !important; }
.vid-spinner-show {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px 0 8px;
    flex-direction: column;
    gap: 8px;
    background: transparent;
}
.vid-spinner-ring {
    width: 30px;
    height: 30px;
    border: 3px solid rgba(79,142,247,0.2);
    border-top-color: #4f8ef7;
    border-radius: 50%;
    animation: spin 0.85s linear infinite;
}
.vid-spinner-label {
    font-size: 11px;
    color: #4f8ef7;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}
@keyframes spin { to { transform: rotate(360deg); } }

#status-text {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    color: var(--muted) !important;
    padding: 8px 20px !important;
    border-top: 1px solid var(--border) !important;
    background: var(--surface) !important;
    min-height: 36px !important;
}

#session-badge {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    padding: 6px 20px !important;
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}

.logo-area {
    padding: 16px 20px 0;
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.logo-main { font-size: 20px; font-weight: 600; color: var(--text); letter-spacing: -0.02em; }
.logo-sub  { font-size: 12px; color: var(--muted); font-family: 'DM Mono', monospace; }

::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
"""

with gr.Blocks(css=CSS, title="SmartRehab — Gait Simulator", theme=gr.themes.Base()) as demo:

    session_id_state   = gr.State("")
    is_first_msg_state = gr.State(True)
    chat_titles_state  = gr.State([])

    with gr.Row(elem_classes="main-row"):

        # ── LEFT PANEL ────────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="left-panel", min_width=280):

            gr.HTML("""
                <div class="logo-area">
                    <span class="logo-main">SmartRehab</span>
                    <span class="logo-sub">gait simulator v2</span>
                </div>
            """)

            new_chat_btn = gr.Button("＋  New Chat", elem_id="new-chat-btn")

            gr.HTML('<div class="panel-header"><span class="panel-title">Conversations</span></div>')

            chat_list_html = gr.HTML(
                '<div class="chat-list">'
                '<div class="chat-list-item active">New conversation</div>'
                '</div>'
            )

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                height=460,
                show_label=False,
                placeholder=(
                    "**Start by describing a walking motion.**\n\n"
                    "Examples:\n"
                    "- *person walking normally*\n"
                    "- *elderly person walking slowly*\n"
                    "- *右足を引きずりながら歩く*\n\n"
                    "After generating, refine with:\n\n"
                    "**Gait syndromes:**\n"
                    "- *right foot drags on the floor*\n"
                    "- *Parkinson's with freezing*\n"
                    "- *moderate right hemiplegia*\n\n"
                    "**Body-part adjustments:**\n"
                    "- *patient's hands on their chest*\n"
                    "- *left knee 20 degrees more bent*\n"
                    "- *head looking slightly down*"
                ),
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Describe gait syndrome or body-part adjustment...",
                    show_label=False,
                    lines=2,
                    max_lines=4,
                    elem_id="msg-input",
                    scale=4,
                )
                send_btn = gr.Button("Send", elem_id="send-btn", scale=1)

            session_badge = gr.Textbox(
                value="No active session",
                show_label=False,
                interactive=False,
                elem_id="session-badge",
            )

        # ── RIGHT PANEL — dual video with spinner between ────────────────────
        with gr.Column(scale=1, elem_classes="right-panel"):

            gr.HTML("""
                <div class="video-header">
                    <span class="video-label">Motion Preview</span>
                </div>
            """)

            gr.HTML('<div style="padding:6px 16px 2px;font-size:11px;font-weight:600;'
                    'color:#64748b;text-transform:uppercase;letter-spacing:0.07em;">'
                    'Base motion</div>')

            video_base = gr.Video(
                label=None,
                show_label=False,
                elem_id="video-base",
                height=400,
                autoplay=True,
            )

            # Spinner shown between the two videos while processing
            spinner_html = gr.HTML(
                value='<div id="vid-spinner" class="vid-spinner-hidden"></div>',
                elem_id="spinner-wrap",
            )

            gr.HTML('<div style="padding:10px 16px 2px;font-size:11px;font-weight:600;'
                    'color:#64748b;text-transform:uppercase;letter-spacing:0.07em;">'
                    'Modified motion</div>')

            video_modified = gr.Video(
                label=None,
                show_label=False,
                elem_id="video-modified",
                height=400,
                autoplay=True,
            )

            status_text = gr.Textbox(
                value="",
                show_label=False,
                interactive=False,
                elem_id="status-text",
                placeholder="Status will appear here...",
            )

    # ── Event wiring ──────────────────────────────────────────────────────────

    def _update_sidebar(titles, new_title=None):
        if new_title:
            titles = [new_title] + [t for t in titles if t != new_title]
        items = ""
        for i, t in enumerate(titles[:20]):
            active = "active" if i == 0 else ""
            short  = t[:35] + "…" if len(t) > 35 else t
            items += f'<div class="chat-list-item {active}">💬 {short}</div>'
        return f'<div class="chat-list">{items}</div>', titles

    def _badge(sid):
        return f"Session: {sid[:8]}…" if sid else "No active session"

    # Shared helper: pack all outputs in the right order
    # outputs order: chatbot, session_id, is_first, video_base, video_modified,
    #                status_text, chat_list_html, chat_titles, session_badge,
    #                send_btn (interactive), msg_input (interactive)

    _SPINNER_ON  = ('<div id="vid-spinner" class="vid-spinner-show">'
                    '<div class="vid-spinner-ring"></div>'
                    '<div class="vid-spinner-label">Generating...</div>'
                    '</div>')
    _SPINNER_OFF = '<div id="vid-spinner" class="vid-spinner-hidden"></div>'

    def on_send(user_msg, history, session_id, is_first, titles):
        if not user_msg.strip():
            yield (history, session_id, is_first,
                   _get_base_video(session_id), _get_video(),
                   _SPINNER_OFF, "", gr.update(), titles, "",
                   gr.update(interactive=True))
            return

        sidebar_html, new_titles = _update_sidebar(titles, user_msg)

        # ── Disable send button, show spinner ─────────────────────────────────
        yield (history, session_id, is_first,
               _get_base_video(session_id), _get_video(),
               _SPINNER_ON, "🔄 Processing...", sidebar_html, new_titles,
               _badge(session_id), gr.update(interactive=False))

        last_out = None
        for history_out, sid_out, first_out, base_vid, mod_vid, status_out in \
                send_message(user_msg, history, session_id, is_first):
            last_out = (history_out, sid_out, first_out, base_vid, mod_vid, status_out)
            yield (history_out, sid_out, first_out, base_vid, mod_vid,
                   _SPINNER_ON, status_out, sidebar_html, new_titles,
                   _badge(sid_out), gr.update(interactive=False))

        # ── Hide spinner, re-enable button ─────────────────────────────────────
        if last_out:
            history_out, sid_out, first_out, base_vid, mod_vid, status_out = last_out
            yield (history_out, sid_out, first_out, base_vid, mod_vid,
                   _SPINNER_OFF, status_out, sidebar_html, new_titles,
                   _badge(sid_out), gr.update(interactive=True))

    send_btn.click(
        fn=on_send,
        inputs=[msg_input, chatbot, session_id_state, is_first_msg_state, chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state,
                 video_base, video_modified,
                 spinner_html, status_text,
                 chat_list_html, chat_titles_state, session_badge,
                 send_btn],
    ).then(fn=lambda: "", outputs=msg_input)

    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, chatbot, session_id_state, is_first_msg_state, chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state,
                 video_base, video_modified,
                 spinner_html, status_text,
                 chat_list_html, chat_titles_state, session_badge,
                 send_btn],
    ).then(fn=lambda: "", outputs=msg_input)

    def on_new_chat(session_id, titles):
        _call_cleanup(session_id)
        sidebar_html, new_titles = _update_sidebar(titles)
        return ([], "", True, None, None, "",
                sidebar_html, new_titles, "No active session")

    new_chat_btn.click(
        fn=on_new_chat,
        inputs=[session_id_state, chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state,
                 video_base, video_modified,
                 status_text, chat_list_html, chat_titles_state, session_badge],
    )

    # ── Video loop: seek back before 'ended' fires — bypasses Gradio's player ──
    gr.HTML("""
    <script>
    (function() {
      var watched = new Map(); // video el → { duration, active }

      function wire(v) {
        if (watched.has(v)) return;
        watched.set(v, { active: true });

        v.addEventListener('timeupdate', function() {
          if (!v.duration || v.duration === Infinity) return;
          // Seek back 0.15s before end — fires before Gradio's 'ended' handler
          if (v.currentTime >= v.duration - 0.15) {
            v.currentTime = 0;
            v.play().catch(function(){});
          }
        });

        // Also set loop attribute as belt-and-suspenders
        v.loop = true;

        // Re-apply loop if Gradio clears it after a src swap
        v.addEventListener('loadeddata', function() {
          v.loop = true;
          v.play().catch(function(){});
        });
      }

      function cleanup() {
        watched.forEach(function(_, v) {
          if (!document.body.contains(v)) watched.delete(v);
        });
      }

      function scan() {
        cleanup();
        document.querySelectorAll('video').forEach(function(v) {
          // Only target videos inside our two panels
          var id = v.closest('[id]') ? v.closest('[id]').id : '';
          if (id.indexOf('video-base') !== -1 || id.indexOf('video-modified') !== -1) {
            wire(v);
          }
        });
      }

      if (document.readyState !== 'loading') scan();
      else document.addEventListener('DOMContentLoaded', scan);

      setInterval(scan, 500);
    })();
    </script>
    """)


if __name__ == "__main__":
    print("[SmartRehab UI] Starting on http://localhost:7860")
    print("[SmartRehab UI] Make sure app_server.py is running on http://localhost:8000")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)