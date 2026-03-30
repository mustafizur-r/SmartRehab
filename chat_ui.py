"""
gradio_ui.py — SmartRehab Gait Chat Interface
Run: python gradio_ui.py
Connects to app_server.py running on http://localhost:8000
"""

import gradio as gr
import requests
import os

API_BASE   = "http://localhost:8000"
VIDEO_PATH = "video_result/Final_Fbx_Mesh_Animation.mp4"


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_gen(prompt: str) -> dict:
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


def _get_video():
    return VIDEO_PATH if os.path.exists(VIDEO_PATH) else None


# ─────────────────────────────────────────────────────────────────────────────
# Chat logic
# ─────────────────────────────────────────────────────────────────────────────

def _format_impairments(state: dict) -> str:
    if not state:
        return "Normal walking — no impairments active."
    labels = {
        "ankle_drop_right":          "Ankle drop (R)",
        "ankle_drop_left":           "Ankle drop (L)",
        "knee_stiffness_right":      "Knee stiffness (R)",
        "knee_stiffness_left":       "Knee stiffness (L)",
        "stride_asymmetry_right":    "Stride asymmetry (R)",
        "stride_asymmetry_left":     "Stride asymmetry (L)",
        "trunk_lean_right":          "Trunk lean (R)",
        "trunk_lean_left":           "Trunk lean (L)",
        "arm_swing_reduction_right": "Arm swing↓ (R)",
        "arm_swing_reduction_left":  "Arm swing↓ (L)",
        "cadence_reduction":         "Cadence↓",
        "hemiplegic_right":          "Hemiplegic (R)",
        "hemiplegic_left":           "Hemiplegic (L)",
        "parkinsonian_shuffle":      "Parkinsonian shuffle",
        "crouch_gait":               "Crouch gait",
        "scissor_gait":              "Scissor gait",
        "antalgic_right":            "Antalgic (R)",
        "antalgic_left":             "Antalgic (L)",
        "hip_hike_right":            "Hip hike (R)",
        "hip_hike_left":             "Hip hike (L)",
    }
    lines = ["**Active impairments:**"]
    for k, v in state.items():
        label = labels.get(k, k)
        bar   = "█" * int(float(v) * 10) + "░" * (10 - int(float(v) * 10))
        lines.append(f"- {label}: {bar} {float(v):.1f}")
    return "\n".join(lines)


def send_message(user_msg, history, session_id, is_first):
    if not user_msg.strip():
        return history, session_id, is_first, _get_video(), ""

    history = history + [{"role": "user", "content": user_msg}]

    # Show thinking
    yield (history + [{"role": "assistant", "content": "⏳ Generating motion..."}],
           session_id, is_first, None, "🔄 Processing...")

    if is_first:
        result     = _call_gen(user_msg)
        new_sid    = result.get("session_id", "")
        status     = result.get("status", "error")
        exp_prompt = result.get("expressive_prompt", "")
        eng_prompt = result.get("english_prompt", user_msg)

        if status == "success":
            reply = (
                f"✅ **Base motion generated.**\n\n"
                f"**English:** {eng_prompt}\n\n"
                f"**SnapMoGen description:**\n> {exp_prompt}\n\n"
                f"Now refine it — try:\n"
                f'- *"add moderate right leg limp"*\n'
                f'- *"hemiplegic gait right side"*\n'
                f'- *"make it more severe"*\n'
                f'- *"reset to normal"*'
            )
            history = history + [{"role": "assistant", "content": reply}]
            yield history, new_sid, False, _get_video(), "✅ Done"
        else:
            msg = result.get("message", "Unknown error")
            history = history + [{"role": "assistant", "content": f"❌ Error: {msg}"}]
            yield history, session_id, True, None, f"❌ {msg}"

    else:
        if not session_id:
            history = history + [{"role": "assistant",
                                   "content": "❌ No active session. Start a new chat first."}]
            yield history, session_id, is_first, None, "❌ No session"
            return

        result    = _call_refine(session_id, user_msg)
        status    = result.get("status", "error")
        imp_state = result.get("impairment_state", {})

        if status == "success":
            reply = f"✅ **Motion refined.**\n\n{_format_impairments(imp_state)}"
            history = history + [{"role": "assistant", "content": reply}]
            yield history, session_id, False, _get_video(), "✅ Done"
        else:
            err = result.get("message", "Unknown error")
            history = history + [{"role": "assistant", "content": f"❌ Error: {err}"}]
            yield history, session_id, is_first, _get_video(), f"❌ {err}"


# ─────────────────────────────────────────────────────────────────────────────
# CSS — clean, simple, correct
# All text at readable sizes. Heights fixed so everything fits on screen.
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2a2d3a;
    --accent:  #4f8ef7;
    --accent2: #7c5cbf;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --radius:  10px;
}

* { box-sizing: border-box; }

html, body {
    height: 100%;
    overflow: hidden;
    background: var(--bg);
    margin: 0;
    padding: 0;
}

/* ── Gradio outer shell ── */
.gradio-container {
    background: var(--bg) !important;
    max-width: 1400px !important;
    width: 100% !important;
    height: 100vh !important;
    margin: 0 auto !important;
    padding: 0 !important;
    overflow: hidden !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Gradio footer completely */
footer, .footer, [class*="footer"],
.gradio-container ~ footer,
#footer, .svelte-footer,
.built-with { display: none !important; height: 0 !important; }

/* Kill gradio inner padding */
.gradio-container > .main,
.gradio-container .contain {
    padding: 0 !important;
    margin: 0 !important;
    height: 100% !important;
}

/* ── Two-column layout ── */
.main-row {
    display: flex !important;
    height: 100vh !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* ══════════════════════════
   LEFT PANEL
══════════════════════════ */
.left-panel {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    overflow: hidden !important;
}

.logo-area {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.logo-main { font-size: 18px; font-weight: 600; color: var(--text); }
.logo-sub  { font-size: 12px; color: var(--muted); font-family: 'DM Mono', monospace; }

#new-chat-btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: white !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    margin: 10px 12px 4px !important;
    width: calc(100% - 24px) !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
}
#new-chat-btn:hover { opacity: 0.85 !important; }

.panel-header {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}
.panel-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.chat-list {
    padding: 6px;
    overflow-y: auto;
    max-height: 100px;
    flex-shrink: 0;
}
.chat-list-item {
    padding: 7px 10px;
    border-radius: 7px;
    font-size: 13px;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background 0.12s;
}
.chat-list-item:hover  { background: var(--border); color: var(--text); }
.chat-list-item.active { background: rgba(79,142,247,0.14); color: var(--accent); }

/* Chatbot — takes all remaining height */
#chatbot {
    flex: 1 !important;
    min-height: 0 !important;
    background: var(--bg) !important;
    border: none !important;
    overflow: hidden !important;
}
#chatbot * { color: #e2e8f0 !important; }
#chatbot [data-testid="user"],
#chatbot [data-testid="user"] > div {
    background: rgba(79,142,247,0.16) !important;
    border: 1px solid rgba(79,142,247,0.28) !important;
    border-radius: 9px !important;
    padding: 10px 14px !important;
}
#chatbot [data-testid="bot"],
#chatbot [data-testid="bot"] > div {
    background: #1e2235 !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    padding: 10px 14px !important;
}
#chatbot p, #chatbot span, #chatbot li, #chatbot strong, #chatbot em {
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #e2e8f0 !important;
}
#chatbot blockquote {
    border-left: 3px solid var(--accent) !important;
    padding: 4px 10px !important;
    background: rgba(79,142,247,0.07) !important;
    border-radius: 3px !important;
    margin: 6px 0 !important;
}
#chatbot blockquote * { color: #94a3b8 !important; }
#chatbot code { color: #7dd3fc !important; font-size: 12px !important; }
#chatbot [class*="placeholder"] * { color: var(--muted) !important; }

/* Input row — fixed height at bottom */
.input-row {
    flex-shrink: 0;
    padding: 8px 10px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    gap: 8px;
    align-items: center;
}
#msg-input textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    padding: 9px 12px !important;
    resize: none !important;
    min-height: 40px !important;
    max-height: 80px !important;
}
#msg-input textarea:focus { border-color: var(--accent) !important; outline: none !important; }
#msg-input textarea::placeholder { color: var(--muted) !important; }

#send-btn {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: white !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 9px 20px !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
    height: 40px !important;
}
#send-btn:hover { opacity: 0.85 !important; }

/* Session badge */
#session-badge {
    flex-shrink: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    padding: 20px 20px 60px !important;
    background: var(--bg) !important;
    border-top: 1px solid var(--border) !important;
    height: 26px !important;
    overflow: hidden !important;
}

/* ══════════════════════════
   RIGHT PANEL
══════════════════════════ */
.right-panel {
    background: var(--bg) !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    overflow: hidden !important;
}

.video-header {
    flex-shrink: 0;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
}
.video-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

#video-out {
    flex: 1 !important;
    min-height: 0 !important;
    margin: 12px !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
}
#video-out video {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}

/* Status bar */
#status-text {
    flex-shrink: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    padding: 20px 20px 60px !important;
    border-top: 1px solid var(--border) !important;
    background: var(--surface) !important;
    height: 26px !important;
    overflow: hidden !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="SmartRehab — Gait Simulator", theme=gr.themes.Base()) as demo:

    session_id_state   = gr.State("")
    is_first_msg_state = gr.State(True)
    chat_titles_state  = gr.State([])

    with gr.Row(elem_classes="main-row"):

        # ── LEFT PANEL ──────────────────────────────────────────────────────
        with gr.Column(scale=5, elem_classes="left-panel", min_width=380):

            gr.HTML("""
                <div class="logo-area">
                    <span class="logo-main">SmartRehab</span>
                    <span class="logo-sub">gait simulator</span>
                </div>
            """)

            new_chat_btn = gr.Button("＋  New Chat", elem_id="new-chat-btn")

            gr.HTML('<div class="panel-header"><span class="panel-title">Conversations</span></div>')

            chat_list_html = gr.HTML(
                '<div class="chat-list"><div class="chat-list-item active">New conversation</div></div>'
            )

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                height=160,
                show_label=False,
                placeholder=(
                    "**Start by describing the gait motion you want to generate.**\n\n"
                    "Examples:\n"
                    "- *person walking normally*\n"
                    "- *elderly person walking slowly*\n"
                    "- *右足を引きずりながら歩く*\n\n"
                    "After the base motion is generated, refine it:\n"
                    "- *add a moderate right leg limp*\n"
                    "- *hemiplegic gait right side*\n"
                    "- *make it more severe*\n"
                    "- *reset to normal*"
                ),
            )

            with gr.Row(elem_classes="input-row"):
                msg_input = gr.Textbox(
                    placeholder="Describe gait or refinement...",
                    show_label=False,
                    lines=1,
                    max_lines=3,
                    elem_id="msg-input",
                    scale=5,
                )
                send_btn = gr.Button("Send", elem_id="send-btn", scale=1)

            session_badge = gr.Textbox(
                value="No active session",
                show_label=False,
                interactive=False,
                elem_id="session-badge",
            )

        # ── RIGHT PANEL ─────────────────────────────────────────────────────
        with gr.Column(scale=6, elem_classes="right-panel"):

            gr.HTML("""
                <div class="video-header">
                    <span class="video-label">Motion Preview</span>
                </div>
            """)

            video_out = gr.Video(
                label=None,
                show_label=False,
                elem_id="video-out",
                height=160,
                autoplay=True,
            )

            status_text = gr.Textbox(
                value="",
                show_label=False,
                interactive=False,
                elem_id="status-text",
                placeholder="Status will appear here...",
            )

    # ── Event handlers ───────────────────────────────────────────────────────

    def _sidebar(titles, new_title=None):
        if new_title:
            titles = [new_title] + [t for t in titles if t != new_title]
        items = "".join(
            f'<div class="chat-list-item {"active" if i==0 else ""}">💬 {t[:38]}{"…" if len(t)>38 else ""}</div>'
            for i, t in enumerate(titles[:20])
        )
        return f'<div class="chat-list">{items}</div>', titles

    def _badge(sid):
        return f"Session: {sid[:8]}…" if sid else "No active session"

    def on_send(user_msg, history, session_id, is_first, titles):
        if not user_msg.strip():
            yield history, session_id, is_first, _get_video(), "", gr.update(), titles, ""
            return

        sidebar_html, new_titles = _sidebar(titles, user_msg)

        for h_out, sid_out, first_out, vid_out, status_out in \
                send_message(user_msg, history, session_id, is_first):
            yield (h_out, sid_out, first_out, vid_out, status_out,
                   sidebar_html, new_titles, _badge(sid_out))

    send_btn.click(
        fn=on_send,
        inputs=[msg_input, chatbot, session_id_state, is_first_msg_state, chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state, video_out,
                 status_text, chat_list_html, chat_titles_state, session_badge],
    ).then(fn=lambda: "", outputs=msg_input)

    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, chatbot, session_id_state, is_first_msg_state, chat_titles_state],
        outputs=[chatbot, session_id_state, is_first_msg_state, video_out,
                 status_text, chat_list_html, chat_titles_state, session_badge],
    ).then(fn=lambda: "", outputs=msg_input)

    def on_new_chat(titles):
        sidebar_html, new_titles = _sidebar(titles)
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_api=False)