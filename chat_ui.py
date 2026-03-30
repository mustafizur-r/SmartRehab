"""
gradio_ui.py — SmartRehab Gait Chat Interface
Run: python gradio_ui.py
Connects to app_server.py running on http://localhost:8000
"""
from __future__ import annotations

import gradio as gr
import requests
import json
import time
import os

from sympy import true

API_BASE = "http://localhost:8000"
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
        return {"status": "error", "message": str(e), "session_id": ""}


def _call_refine(session_id: str, prompt: str) -> dict:
    """Call /refine_motion/ with video_render=true."""
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
        return {"status": "error", "message": str(e), "impairment_state": {}}


def _call_session_state(session_id: str) -> dict:
    """Call /session_state/{session_id}."""
    try:
        r = requests.get(f"{API_BASE}/session_state/{session_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _get_video() -> str | None:
    """Return video path if it exists."""
    if os.path.exists(VIDEO_PATH):
        return VIDEO_PATH
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Chat logic
# ─────────────────────────────────────────────────────────────────────────────

def _format_impairments(state: dict) -> str:
    if not state:
        return "Normal walking — no impairments active."
    lines = ["**Active impairments:**"]
    labels = {
        "ankle_drop_right":          "Ankle drop (right)",
        "ankle_drop_left":           "Ankle drop (left)",
        "knee_stiffness_right":      "Knee stiffness (right)",
        "knee_stiffness_left":       "Knee stiffness (left)",
        "stride_asymmetry_right":    "Stride asymmetry (right)",
        "stride_asymmetry_left":     "Stride asymmetry (left)",
        "trunk_lean_right":          "Trunk lean (right)",
        "trunk_lean_left":           "Trunk lean (left)",
        "arm_swing_reduction_right": "Arm swing reduction (right)",
        "arm_swing_reduction_left":  "Arm swing reduction (left)",
        "cadence_reduction":         "Cadence reduction",
    }
    for k, v in state.items():
        label = labels.get(k, k)
        bar   = "█" * int(v * 10) + "░" * (10 - int(v * 10))
        lines.append(f"- {label}: {bar} {v:.1f}")
    return "\n".join(lines)


def send_message(user_msg, history, session_id, is_first_message):
    """
    Process one chat turn.
    - First message → /gen_text2motion/
    - Subsequent messages → /refine_motion/
    Returns: updated history, session_id, is_first_message flag, video path, status text
    """
    if not user_msg.strip():
        return history, session_id, is_first_message, _get_video(), ""

    # Add user message to history
    history = history + [{"role": "user", "content": user_msg}]

    # Show thinking indicator
    yield history + [{"role": "assistant", "content": "⏳ Generating motion..."}], \
          session_id, is_first_message, None, "🔄 Processing..."

    if is_first_message:
        # ── First turn: generate base motion ─────────────────────────────────
        result      = _call_gen(user_msg)
        new_sid     = result.get("session_id", "")
        status      = result.get("status", "error")
        exp_prompt  = result.get("expressive_prompt", "")
        eng_prompt  = result.get("english_prompt", user_msg)

        if status == "success":
            reply = (
                f"**Base motion generated.**\n\n"
                f"**English prompt:** {eng_prompt}\n\n"
                f"**Rewrite description:**\n> {exp_prompt}\n\n"
                f"You can now refine the motion — try:\n"
                f'- *"Add a moderate right leg limp"*\n'
                f'- *"Make it more severe"*\n'
                f'- *"Add slow cadence"*\n'
                f'- *"Reset to normal"*'
            )
            history = history + [{"role": "assistant", "content": reply}]
            video   = _get_video()
            yield history, new_sid, False, video, "✅ Done"
        else:
            msg = result.get("message", "Unknown error")
            history = history + [{"role": "assistant", "content": f"❌ Error: {msg}"}]
            yield history, session_id, True, None, f"❌ {msg}"

    else:
        # ── Subsequent turns: iterative refinement ────────────────────────────
        if not session_id:
            history = history + [{"role": "assistant", "content":
                "❌ No active session. Please start a new chat first."}]
            yield history, session_id, is_first_message, None, "❌ No session"
            return

        result  = _call_refine(session_id, user_msg)
        status  = result.get("status", "error")
        imp_state = result.get("impairment_state", {})
        msg     = result.get("message", "")

        if status == "success":
            imp_summary = _format_impairments(imp_state)
            reply = (
                f"**Motion refined.**\n\n"
                f"{imp_summary}\n\n"
                f"*{msg}*"
            )
            history = history + [{"role": "assistant", "content": reply}]
            video   = _get_video()
            yield history, session_id, False, video, "✅ Done"
        else:
            err = result.get("message", "Unknown error")
            history = history + [{"role": "assistant", "content": f"❌ Error: {err}"}]
            yield history, session_id, is_first_message, _get_video(), f"❌ {err}"


def new_chat():
    """Reset everything for a new conversation."""
    return [], "", True, None, ""


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

/* ── Layout ── */
.main-row { gap: 0 !important; }

/* ── Left panel ── */
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

/* ── New chat button ── */
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
    width: 100% !important;
    margin: 12px 16px !important;
    width: calc(100% - 32px) !important;
}
#new-chat-btn:hover { opacity: 0.85 !important; }

/* ── Chat history list ── */
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
.chat-list-item:hover { background: var(--border); color: var(--text); }
.chat-list-item.active { background: rgba(79,142,247,0.15); color: var(--accent); }

/* ── Chatbox — aggressive white text override ── */
#chatbot {
    background: var(--bg) !important;
    border: none !important;
    flex: 1 !important;
}

/* Nuclear option: force white on every element inside chatbot */
#chatbot * { color: #e2e8f0 !important; }

/* User bubble */
#chatbot .user,
#chatbot [class*="user"] div,
#chatbot [data-testid="user"],
#chatbot [data-testid="user"] > div,
#chatbot [data-testid="user"] > div > div {
    background: rgba(79,142,247,0.18) !important;
    border: 1px solid rgba(79,142,247,0.3) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #e2e8f0 !important;
}

/* Bot bubble */
#chatbot .bot,
#chatbot [class*="bot"] div,
#chatbot [data-testid="bot"],
#chatbot [data-testid="bot"] > div,
#chatbot [data-testid="bot"] > div > div {
    background: #1e2235 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #e2e8f0 !important;
}

/* All paragraph/span/text inside messages */
#chatbot p,
#chatbot span,
#chatbot li,
#chatbot ul,
#chatbot ol,
#chatbot strong,
#chatbot em,
#chatbot b,
#chatbot h1, #chatbot h2, #chatbot h3, #chatbot h4 {
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
}

/* Blockquote inside messages */
#chatbot blockquote {
    border-left: 3px solid #4f8ef7 !important;
    margin: 8px 0 !important;
    padding: 4px 12px !important;
    background: rgba(79,142,247,0.08) !important;
    border-radius: 4px !important;
    color: #94a3b8 !important;
}
#chatbot blockquote * { color: #94a3b8 !important; }

/* Inline code */
#chatbot code {
    background: rgba(255,255,255,0.1) !important;
    padding: 2px 5px !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    color: #7dd3fc !important;
}

/* Placeholder text */
#chatbot [class*="placeholder"] { color: #64748b !important; }
#chatbot [class*="placeholder"] * { color: #64748b !important; }

/* Avatar icons */
#chatbot [class*="avatar"] { background: transparent !important; }

/* ── Input area ── */
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

/* ── Send button ── */
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

/* ── Right panel ── */
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

#video-out {
    flex: 1;
    border-radius: var(--radius) !important;
    overflow: hidden;
    margin: 16px;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

/* ── Status bar ── */
#status-text {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    color: var(--muted) !important;
    padding: 8px 20px !important;
    border-top: 1px solid var(--border) !important;
    background: var(--surface) !important;
    min-height: 36px !important;
}

/* ── Session badge ── */
#session-badge {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    padding: 6px 20px !important;
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Logo / title ── */
.logo-area {
    padding: 16px 20px 0;
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.logo-main {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}
.logo-sub {
    font-size: 12px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
}

/* ── Empty video placeholder ── */
.video-placeholder {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 13px;
    gap: 12px;
    margin: 16px;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
}
.video-placeholder-icon { font-size: 40px; opacity: 0.4; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
"""

with gr.Blocks(css=CSS, title="SmartRehab — Gait Simulator", theme=gr.themes.Base()) as demo:

    # ── State ──────────────────────────────────────────────────────────────────
    session_id_state    = gr.State("")
    is_first_msg_state  = gr.State(True)
    chat_titles_state   = gr.State([])   # list of first messages for sidebar

    with gr.Row(elem_classes="main-row"):

        # ── LEFT PANEL — Chat ─────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="left-panel", min_width=280):

            gr.HTML("""
                <div class="logo-area">
                    <span class="logo-main">SmartRehab</span>
                    <span class="logo-sub">gait simulator</span>
                </div>
            """)

            new_chat_btn = gr.Button("＋  New Chat", elem_id="new-chat-btn")

            gr.HTML('<div class="panel-header"><span class="panel-title">Conversations</span></div>')

            # Chat history sidebar (shows previous session titles)
            chat_list_html = gr.HTML('<div class="chat-list"><div class="chat-list-item active">New conversation</div></div>')

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                height=460,
                show_label=False,
                placeholder=(
                    "**Start by describing the gait motion you want to generate.**\n\n"
                    "Examples:\n"
                    "- *person walking normally*\n"
                    "- *elderly person walking slowly*\n"
                    "- *右足を引きずりながら歩く*\n\n"
                    "After the base motion is generated, you can refine it:\n"
                    "- *add a moderate right leg limp*\n"
                    "- *make it more severe*\n"
                    "- *reset to normal*"
                ),
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Describe gait or refinement...",
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

        # ── RIGHT PANEL — Video output ────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="right-panel"):

            gr.HTML("""
                <div class="video-header">
                    <span class="video-label">Motion Preview</span>
                </div>
            """)

            video_out = gr.Video(
                label=None,
                show_label=False,
                elem_id="video-out",
                height=480,
                autoplay=True,
            )

            status_text = gr.Textbox(
                value="",
                show_label=False,
                interactive=False,
                elem_id="status-text",
                placeholder="Status will appear here...",
            )

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _update_sidebar(titles, new_title=None):
        """Rebuild sidebar HTML from chat title list."""
        if new_title:
            titles = [new_title] + [t for t in titles if t != new_title]
        items = ""
        for i, t in enumerate(titles[:20]):
            active = "active" if i == 0 else ""
            short  = t[:35] + "…" if len(t) > 35 else t
            items += f'<div class="chat-list-item {active}">💬 {short}</div>'
        return f'<div class="chat-list">{items}</div>', titles

    def _update_badge(sid):
        if sid:
            return f"Session: {sid[:8]}…"
        return "No active session"

    # Send on button click
    def on_send(user_msg, history, session_id, is_first, titles):
        if not user_msg.strip():
            yield history, session_id, is_first, _get_video(), "", \
                  gr.update(), titles, ""
            return

        # Update sidebar with this conversation
        sidebar_html, new_titles = _update_sidebar(titles, user_msg)

        gen = send_message(user_msg, history, session_id, is_first)
        for history_out, sid_out, first_out, video_out, status_out in gen:
            badge = _update_badge(sid_out)
            yield (history_out, sid_out, first_out, video_out, status_out,
                   sidebar_html, new_titles, badge)

    send_btn.click(
        fn=on_send,
        inputs=[msg_input, chatbot, session_id_state, is_first_msg_state, chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state, video_out,
                 status_text, chat_list_html, chat_titles_state, session_badge],
    ).then(fn=lambda: "", outputs=msg_input)   # clear input after send

    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, chatbot, session_id_state, is_first_msg_state, chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state, video_out,
                 status_text, chat_list_html, chat_titles_state, session_badge],
    ).then(fn=lambda: "", outputs=msg_input)

    # New chat button
    def on_new_chat(titles):
        sidebar_html, new_titles = _update_sidebar(titles)
        return [], "", True, None, "", sidebar_html, new_titles, "No active session"

    new_chat_btn.click(
        fn=on_new_chat,
        inputs=[chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state, video_out,
                 status_text, chat_list_html, chat_titles_state, session_badge],
    )


if __name__ == "__main__":
    print("[SmartRehab UI] Starting on http://localhost:7860")
    print("[SmartRehab UI] Make sure app_server.py is running on http://localhost:8000")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)