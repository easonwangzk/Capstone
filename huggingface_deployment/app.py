# streamlit_app.py
import os
from datetime import datetime

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================== Page Basic Settings (Must be first) ==================
st.set_page_config(
    page_title="Medical Assistant Demo",
    page_icon="",
    layout="wide",
)

# ================== Model Configuration ==================
BASE_MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER_REPO = "Easonwangzk/lora-llama31-med-adapter"
USE_LORA = True

# Get HuggingFace token from environment (for gated models)
HF_TOKEN = os.getenv("HF_TOKEN", None)


# ================== Load Model (Cached) ==================
@st.cache_resource
def load_model():
    """Load the model and tokenizer with LoRA adapter."""
    # Check if model is pre-loaded in Colab (from globals)
    if 'preloaded_model' in globals() and 'preloaded_tokenizer' in globals():
        return globals()['preloaded_model'], globals()['preloaded_tokenizer']

    # Otherwise, load model normally
    # Determine dtype based on GPU capability
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Load tokenizer with token for gated models
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_REPO,
        use_fast=True,
        token=HF_TOKEN
    )

    # Load base model with token for gated models
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_REPO,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=HF_TOKEN
    )
    base_model.eval()

    # Load LoRA adapter if enabled
    if USE_LORA:
        model = PeftModel.from_pretrained(
            base_model,
            LORA_ADAPTER_REPO,
            torch_dtype=torch_dtype,
            token=HF_TOKEN
        )
        model.eval()
    else:
        model = base_model

    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

# ---------- Session Initialization ----------
if "theme" not in st.session_state:
    st.session_state.theme = "light"  # "light" / "dark"

if "messages" not in st.session_state:
    # Initialize with a greeting message
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I am your medical assistant. How can I help you today? "
                "Please share your medical question with me."
            ),
            "timestamp": datetime.now().strftime("%I:%M %p").lstrip("0"),
            "reasoning": None,
        }
    ]

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

# ================== Theme & Global Styles ==================
is_dark = st.session_state.theme == "dark"

BACKGROUND = "#0F172A" if is_dark else "#E2E8F0"
CARD_BG = "#020617" if is_dark else "#FFFFFF"
CARD_BORDER = "#1E293B" if is_dark else "#CBD5E1"
TEXT_PRIMARY = "#E5E7EB" if is_dark else "#0F172A"
TEXT_SECONDARY = "#9CA3AF" if is_dark else "#64748B"
ACCENT = "#2563EB"
ACCENT_SOFT = "rgba(37, 99, 235, 0.10)"

CUSTOM_CSS = f"""
<style>
/* Page background */
.stApp {{
    background: {BACKGROUND};
}}

/* Top header styles */
.app-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px 4px 20px;
}}
.app-header-left {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.app-logo {{
    width: 32px;
    height: 32px;
    border-radius: 999px;
    background: #EFF6FF;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:18px;
    color:#1D4ED8;
    font-weight:600;
}}
.app-title-block {{
    display:flex;
    flex-direction:column;
}}
.app-title {{
    font-weight: 600;
    font-size: 18px;
    color: {TEXT_PRIMARY};
}}
.app-subtitle {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}
.app-header-right {{
    display:flex;
    align-items:center;
    gap:10px;
}}

/* Top icon buttons (theme toggle) */
.icon-btn {{
    width: 32px;
    height: 32px;
    border-radius: 999px;
    border: 1px solid {CARD_BORDER};
    background: {CARD_BG};
    display:flex;
    align-items:center;
    justify-content:center;
    cursor:pointer;
    font-size:14px;
    color:{TEXT_SECONDARY};
}}
.icon-btn:hover {{
    border-color: {ACCENT};
    color:{ACCENT};
}}

/* Disclaimer banner */
.notice-banner {{
    margin: 4px 20px 16px 20px;
    border-radius: 999px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    padding: 8px 16px;
    font-size: 13px;
    text-align:center;
    color:#1D4ED8;
}}

/* Chat area container (bottom padding for fixed bar) */
.chat-wrapper {{
    padding: 0 20px 180px 20px;
    max-width: 980px;
    margin: 0 auto;
}}

/* Chat bubble cards (unified blue theme) */
.chat-card {{
    background: {CARD_BG};
    border-radius: 18px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border: 1px solid {CARD_BORDER};
    box-shadow: 0 8px 16px rgba(15, 23, 42, 0.16);
}}
.chat-card.assistant {{
    border-left: 3px solid {ACCENT};
}}
.chat-card.user {{
    border-left: 3px solid {ACCENT};
    background: linear-gradient(135deg, {ACCENT_SOFT}, {CARD_BG});
}}
.chat-role {{
    font-size: 12px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    margin-bottom: 2px;
}}
.chat-content {{
    font-size: 14px;
    color: {TEXT_PRIMARY};
}}
.chat-meta {{
    font-size: 11px;
    color: {TEXT_SECONDARY};
    margin-top: 4px;
}}

/* Quick questions bar - fixed at bottom (same color scheme as input box) */
.quick-questions-bar {{
    position: fixed;
    left: 50%;
    transform: translateX(-50%);
    bottom: 88px;  /* Just above chat input */
    max-width: 980px;
    width: calc(100% - 40px);
    padding: 6px 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    z-index: 900;
    pointer-events: none;  /* Let internal buttons handle clicks */
}}
.quick-chip {{
    pointer-events: auto;
}}
.quick-chip button {{
    border-radius:999px;
    border: 1px solid {CARD_BORDER};
    padding: 6px 12px;
    font-size: 12px;
    background: {CARD_BG};
    color:{TEXT_PRIMARY};
}}
.quick-chip button:hover {{
    border-color:{ACCENT};
    background: {ACCENT_SOFT};
    color:{ACCENT};
}}

/* Bottom input area (gradient background, sticky) */
[data-testid="stChatInputRoot"] {{
    background: linear-gradient(to top, rgba(15, 23, 42, 0.90), rgba(15, 23, 42, 0.0));
    padding: 16px 20px 20px 20px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ================== Top Header ==================
header_col = st.container()
with header_col:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo">MA</div>
                <div class="app-title-block">
                    <div class="app-title">Medical Assistant Demo</div>
                    <div class="app-subtitle">Powered by Llama 3.1 with LoRA</div>
                </div>
            </div>
            <div class="app-header-right">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Theme toggle and clear chat buttons (placed on the right side below header)
theme_cols = st.columns([8, 1, 1])
with theme_cols[1]:
    if st.button("🗑️ Clear", key="clear_chat", help="Clear conversation history"):
        # Reset messages to initial greeting only
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I am your medical assistant. How can I help you today? "
                    "Please share your medical question with me."
                ),
                "timestamp": datetime.now().strftime("%I:%M %p").lstrip("0"),
                "reasoning": None,
            }
        ]
        # Clear all flags to prevent any pending responses
        st.session_state["processing"] = False
        st.session_state["last_processed"] = None
        st.rerun()
with theme_cols[2]:
    if st.button("🌙" if not is_dark else "☀️", key="toggle_theme", help="Toggle dark/light theme"):
        st.session_state.theme = "dark" if not is_dark else "light"
        st.rerun()

# Disclaimer banner
st.markdown(
    '<div class="notice-banner">'
    'This system is for demonstration purposes only and cannot replace professional medical advice.'
    '</div>',
    unsafe_allow_html=True,
)

# ================== Message Display Area ==================
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    for idx, msg in enumerate(st.session_state.messages):
        role_label = "Assistant" if msg["role"] == "assistant" else "You"
        role_class = "assistant" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        ts = msg.get("timestamp")

        with st.container():
            st.markdown(
                f"""
                <div class="chat-card {role_class}">
                    <div class="chat-role">{role_label}</div>
                    <div class="chat-content">{content}</div>
                    <div class="chat-meta">{ts}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ================== Common Medical Questions (fixed at bottom) ==================
QUICK_QUESTIONS = [
    "What is the diagnosis?",
    "What are the findings?",
    "What is the most likely cause?",
    "What is the next step?",
]

# Use columns + buttons, but wrap the entire bar with fixed positioning
bar_placeholder = st.empty()
with bar_placeholder.container():
    st.markdown('<div class="quick-questions-bar">', unsafe_allow_html=True)
    qq_cols = st.columns(len(QUICK_QUESTIONS))
    for i, q in enumerate(QUICK_QUESTIONS):
        with qq_cols[i]:
            with st.container():
                st.markdown('<div class="quick-chip">', unsafe_allow_html=True)
                if st.button(q, key=f"qq_{i}", use_container_width=True):
                    # Clear any existing processing flag and set pending text
                    st.session_state["processing"] = False
                    st.session_state["pending_user_text"] = q
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ================== Call Local Model (LoRA) ==================
@torch.no_grad()
def generate_answer(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate model output for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens (skip the input prompt)
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def generate_assistant_reply(user_text: str):
    """
    Generate a reply using the local LoRA model.
    """
    try:
        # Build conversation messages for chat template
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": (
                "You are a careful and knowledgeable medical assistant. "
                "Provide helpful, accurate information based on medical knowledge. "
                "Be concise and never give definitive treatment decisions. "
                "Always recommend consulting with healthcare professionals for medical advice."
            )
        })

        # Add conversation history - only completed exchanges (user-assistant pairs)
        # Exclude the greeting message and the current user message
        history_messages = []
        for msg in st.session_state.messages:
            # Skip the initial greeting
            if msg.get("content", "").startswith("Hello. I am your medical assistant"):
                continue
            # Only add messages that are not the current user question
            if msg["role"] in ("user", "assistant") and msg["content"] != user_text:
                history_messages.append(msg)

        # Take last 6 messages (3 exchanges) for context
        for msg in history_messages[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user question
        messages.append({
            "role": "user",
            "content": user_text
        })

        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate response
        reply = generate_answer(prompt, max_new_tokens=512)
        reasoning = None
        return reply, reasoning
    except Exception as e:
        # Fallback: print error for debugging
        fallback = (
            f"Error generating response: {str(e)}\n\n"
            f"Please try again or rephrase your question."
        )
        return fallback, None


# ================== Handle User Input (including quick questions) ==================
# Initialize processing flag and last processed message if not exists
if "processing" not in st.session_state:
    st.session_state["processing"] = False
if "last_processed" not in st.session_state:
    st.session_state["last_processed"] = None

# Get user text from quick questions or chat input
user_text = st.session_state.pop("pending_user_text", None)

if user_text is None:
    # If processing, disable input
    if st.session_state["processing"]:
        st.chat_input("Processing...", disabled=True, key=f"chat_input_disabled_{st.session_state.input_counter}")
        st.stop()
    else:
        user_text = st.chat_input("Type your message...", key=f"chat_input_{st.session_state.input_counter}")

if user_text and user_text.strip():
    # CRITICAL: Check if we're already processing - this should NEVER happen
    if st.session_state["processing"]:
        st.warning("Already processing a message. Please wait.")
        st.stop()

    # Check if the last message in history is already this user text
    # This prevents duplicate processing after rerun
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg["role"] == "user" and last_msg["content"] == user_text:
            # Already processed this message
            st.stop()

    # Check if last two messages form a complete Q&A pair with this question
    if len(st.session_state.messages) >= 2:
        if (st.session_state.messages[-2]["role"] == "user" and
            st.session_state.messages[-2]["content"] == user_text and
            st.session_state.messages[-1]["role"] == "assistant"):
            # This Q&A pair already exists
            st.stop()

    import hashlib
    message_id = hashlib.md5(f"{user_text}{datetime.now().timestamp()}".encode()).hexdigest()

    # Mark as processing FIRST
    st.session_state["processing"] = True
    st.session_state["last_processed"] = user_text

    try:
        # Add user message immediately
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
                "timestamp": datetime.now().strftime("%I:%M %p").lstrip("0"),
                "reasoning": None,
                "message_id": message_id,
            }
        )

        # Generate response immediately (no rerun in between)
        with st.spinner("Thinking..."):
            full_reply, reasoning = generate_assistant_reply(user_text)

        # Store complete assistant message in history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_reply,
                "timestamp": datetime.now().strftime("%I:%M %p").lstrip("0"),
                "reasoning": reasoning,
                "message_id": message_id,
            }
        )
    finally:
        # Always clear processing flag, even if error occurs
        st.session_state["processing"] = False

    # Increment input counter to reset the chat input widget
    st.session_state.input_counter += 1

    # Rerun once to display the complete conversation
    st.rerun()