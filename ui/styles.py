"""Shared Streamlit presentation styles."""

import streamlit as st

from ui.theme import (
    AMBER,
    BG,
    BUTTON_TEXT,
    CARD,
    GRID,
    MUTED,
    SAGE,
    SILVER,
    TEXT,
    WALNUT,
)

_STYLES_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --sage: __SAGE__;
  --walnut: __WALNUT__;
  --amber: __AMBER__;
  --silver: __SILVER__;
  --border: __SILVER__;
  --bg: __BG__;
  --card: __CARD__;
  --muted: __MUTED__;
  --grid: __GRID__;
  --text: __TEXT__;
  --button-text: __BUTTON_TEXT__;
}

body, p, h1, h2, h3, h4, h5, h6, label, input, textarea, button, .stMarkdown {
  font-family: 'Inter', sans-serif;
}
span[data-testid="stIconMaterial"],
[class*="material"],
[data-testid="stSidebarCollapseButton"] {
  font-family: "Material Symbols Rounded" !important;
}

.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
}
[data-testid="stMain"] {
  background: var(--bg);
}
[data-testid="stMainBlockContainer"] {
  padding: 1.5rem 1.5rem 2rem;
}
[data-testid="stHeader"], [data-testid="stDecoration"] {
  background: var(--bg);
}
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
  color: var(--text);
}
h1, h2, h3 { letter-spacing: -0.035em; }

[data-testid="stSidebar"] {
  background: var(--walnut);
  border-right: 1px solid var(--walnut);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  color: var(--bg);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--bg);
}
[data-testid="stSidebarNav"] a {
  border-radius: 0.7rem;
  color: var(--bg);
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: var(--amber);
  color: var(--text);
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--card);
  color: var(--text);
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMetric"],
[data-testid="stChatMessage"],
[data-testid="stExpander"],
[data-testid="stStatusWidget"],
[data-testid="stPopover"] > div {
  background: var(--card);
  border: 1px solid var(--silver);
  border-radius: 1rem;
  box-shadow: 0 0.35rem 1.1rem color-mix(in srgb, var(--walnut) 7%, transparent);
}
[data-testid="stMetric"] {
  padding: 1rem 1.1rem;
  border-bottom: 2px solid var(--amber);
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
  color: var(--text);
}
[data-testid="stChatMessage"] {
  width: 75%;
  max-width: 75%;
  padding: 1rem;
  margin-bottom: 1rem;
  border: 1px solid var(--silver);
  box-shadow: none;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: var(--sage);
  border-color: var(--sage);
  align-self: flex-end;
  margin-left: auto;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
  color: var(--text);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  background: var(--card);
  border: 1px solid var(--silver);
  align-self: flex-start;
  margin-right: auto;
}
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] {
  border-radius: 50%;
}

button[kind="primary"], [data-testid="stFormSubmitButton"] button {
  background: var(--sage);
  color: var(--walnut);
  border: 1px solid var(--sage);
  border-radius: 0.75rem;
  font-weight: 600;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  background: var(--sage);
  color: var(--walnut);
  border: 1px solid var(--sage);
  border-radius: 0.75rem;
}
button:hover, [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--amber);
  box-shadow: 0 0 0 0.15rem color-mix(in srgb, var(--amber) 30%, transparent);
  color: var(--walnut);
}
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background: var(--card) !important;
  background-color: var(--card) !important;
  color: var(--button-text) !important;
  border: 1px solid var(--sage) !important;
}
[data-testid="stSidebar"] .stButton button *,
[data-testid="stSidebar"] [data-testid="stButton"] button *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * {
  color: var(--button-text) !important;
}
[data-testid="stSidebar"] .stButton button:hover,
[data-testid="stSidebar"] .stButton button:active,
[data-testid="stSidebar"] .stButton button:focus,
[data-testid="stSidebar"] .stButton button:focus-visible,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:active,
[data-testid="stSidebar"] [data-testid="stButton"] button:focus,
[data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:active,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus-visible {
  background: var(--walnut) !important;
  background-color: var(--walnut) !important;
  border-color: var(--walnut) !important;
  box-shadow: none !important;
  color: var(--bg) !important;
}
[data-testid="stSidebar"] .stButton button:hover p,
[data-testid="stSidebar"] .stButton button:hover span,
[data-testid="stSidebar"] .stButton button:active p,
[data-testid="stSidebar"] .stButton button:active span,
[data-testid="stSidebar"] .stButton button:focus p,
[data-testid="stSidebar"] .stButton button:focus span,
[data-testid="stSidebar"] .stButton button:focus-visible p,
[data-testid="stSidebar"] .stButton button:focus-visible span {
  color: var(--bg) !important;
}
[data-testid="stSidebar"] .stButton button:hover *,
[data-testid="stSidebar"] .stButton button:active *,
[data-testid="stSidebar"] .stButton button:focus *,
[data-testid="stSidebar"] .stButton button:focus-visible *,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover *,
[data-testid="stSidebar"] [data-testid="stButton"] button:active *,
[data-testid="stSidebar"] [data-testid="stButton"] button:focus *,
[data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:active *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus-visible * {
  color: var(--bg) !important;
}
[data-testid="stSidebar"] .stButton button {
  min-height: 3rem;
  height: 3rem;
}
.resolveai-sidebar-label {
  margin: 1.5rem 0 0.75rem;
  color: var(--silver);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.1em;
}
.resolveai-status {
  display: inline-block;
  padding: 0.45rem 0.8rem;
  border: 1px solid var(--silver);
  border-radius: 999px;
  color: var(--text);
  font-size: 0.85rem;
  font-weight: 600;
  white-space: nowrap;
}
.resolveai-status.online {
  background: var(--sage);
  border-color: var(--sage);
}
.resolveai-status.offline {
  background: var(--card);
  border-color: var(--amber);
}
.resolveai-page-header {
  margin-bottom: 1.5rem;
}
.resolveai-page-title {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.resolveai-page-title h1 {
  margin: 0;
  color: var(--text);
}
.resolveai-page-header > p {
  margin: 0.25rem 0 0;
  color: var(--muted);
  font-size: 1rem;
}
.resolveai-welcome {
  margin: 1rem 0;
  padding: 1.5rem;
  background: var(--card);
  border: 1px solid var(--silver);
  border-radius: 1rem;
}
.resolveai-welcome h3 {
  margin: 0 0 0.5rem;
  color: var(--text);
}
.resolveai-welcome p {
  margin: 0;
  color: var(--muted);
}

input, textarea, [data-baseweb="input"], [data-baseweb="textarea"],
[data-testid="stChatInput"] {
  background: var(--card);
  color: var(--text);
  border-color: var(--border);
}
[data-testid="stChatInput"] {
  border: 1px solid var(--border);
  border-radius: 1rem;
  background: var(--card);
  box-shadow: none;
  padding: 0.5rem;
}
[data-testid="stChatInput"] textarea {
  color: var(--text);
  background: var(--card);
  font-size: 1rem;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--muted);
  opacity: 1;
}
[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within,
[data-baseweb="select"]:focus-within {
  border-color: var(--amber);
  box-shadow: 0 0 0 0.15rem color-mix(in srgb, var(--amber) 30%, transparent);
}
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
  border-color: var(--sage);
  box-shadow: none;
}
[data-testid="stFeedback"] button[aria-pressed="true"],
[data-testid="stFeedback"] button[aria-checked="true"] {
  color: var(--amber);
}
[data-testid="stFeedback"] button {
  min-width: 2.75rem;
  min-height: 2.75rem;
}
[data-testid="stFeedback"] button [data-testid="stIconMaterial"] {
  font-size: 1.75rem !important;
}
[data-testid="stFeedback"] button[aria-pressed="true"] [data-testid="stIconMaterial"],
[data-testid="stFeedback"] button[aria-checked="true"] [data-testid="stIconMaterial"] {
  color: var(--amber);
}
[data-testid="stForm"] {
  background: var(--card);
  border: 1px solid var(--silver);
  border-radius: 1rem;
  padding: 1.5rem;
}
[data-testid="stForm"] [data-baseweb="input"],
[data-testid="stForm"] input {
  background: var(--card);
  color: var(--text);
  border: 1px solid var(--silver);
  border-radius: 0.7rem;
}
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div,
[data-baseweb="select"] [data-baseweb="select"] {
  background: var(--card);
  color: var(--text);
  border-color: var(--silver);
  border-radius: 0.7rem;
}
[data-baseweb="select"] svg {
  color: var(--text);
  fill: var(--text);
}
[role="listbox"], [role="option"], [data-baseweb="popover"] > div {
  background: var(--card);
  color: var(--text);
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
  background: var(--amber);
  color: var(--text);
}
[data-testid="stPopover"] button,
[data-testid="stToolbar"] button {
  color: var(--walnut);
  background: var(--silver);
  border-radius: 0.65rem;
}
[data-testid="stToolbar"] {
  background: var(--bg);
}
[data-testid="stDeployButton"] button {
  background: var(--sage);
  color: var(--walnut);
  border-color: var(--sage);
}

[data-testid="stExpander"] {
  border-color: var(--silver);
  border-left: 2px solid var(--amber);
}
.resolveai-trace-duration {
  font-size: 0.95rem;
}
[data-testid="stExpander"] summary:hover {
  color: var(--text);
  border-bottom-color: var(--amber);
}
[data-testid="stTabs"] [role="tab"] {
  color: var(--walnut);
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--amber);
  border-bottom-color: var(--amber);
}
[data-testid="stAlert"] {
  background: var(--card);
  color: var(--walnut);
  border: 1px solid var(--silver);
  border-left: 0.25rem solid var(--amber);
  border-radius: 0.8rem;
}
[role="tooltip"] {
  background: var(--walnut);
  color: var(--card);
  border-radius: 0.5rem;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: var(--card);
  border: 1px solid var(--silver);
  border-radius: 0.8rem;
  overflow: hidden;
}
hr { border-color: var(--silver); }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb {
  background: var(--sage);
  border: 2px solid var(--bg);
  border-radius: 999px;
}

.resolveai-typing {
  display: inline-flex;
  align-items: center;
  gap: 0.2rem;
  color: var(--muted);
  font-size: 0.9rem;
}
.resolveai-typing .dot {
  width: 0.35rem;
  height: 0.35rem;
  border-radius: 50%;
  background: var(--amber);
  animation: resolveai-bounce 1.2s infinite ease-in-out;
}
.resolveai-typing .dot:nth-child(2) { animation-delay: 0.15s; }
.resolveai-typing .dot:nth-child(3) { animation-delay: 0.3s; }
.resolveai-trace-heading {
  border-left: 0.2rem solid var(--amber);
  padding-left: 0.7rem;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text);
}
.resolveai-trace-duration {
  color: var(--muted);
  font-size: 0.85rem;
  text-align: right;
}
.resolveai-trace-step {
  color: var(--text);
  font-weight: 600;
}
@keyframes resolveai-bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.55; }
  30% { transform: translateY(-0.3rem); opacity: 1; }
}
</style>
"""

STYLES = (
    _STYLES_TEMPLATE.replace("__SAGE__", SAGE)
    .replace("__WALNUT__", WALNUT)
    .replace("__AMBER__", AMBER)
    .replace("__SILVER__", SILVER)
    .replace("__BG__", BG)
    .replace("__CARD__", CARD)
    .replace("__MUTED__", MUTED)
    .replace("__GRID__", GRID)
    .replace("__TEXT__", TEXT)
    .replace("__BUTTON_TEXT__", BUTTON_TEXT)
)


def apply_styles() -> None:
    """Apply the shared theme once at the top of each Streamlit page."""
    st.markdown(STYLES, unsafe_allow_html=True)


def render_typing_indicator() -> None:
    """Render the animated assistant typing label."""
    st.markdown(
        '<div class="resolveai-typing">ResolveAI is typing...'
        '<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
        "</div>",
        unsafe_allow_html=True,
    )
