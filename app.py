from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from customer_support_agent import CustomerSupportAgent
from customer_support_agent.feedback import FeedbackStore

st.set_page_config(page_title="Resolve AI", page_icon="✨", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(145deg,#fff7ed,#f8fafc 50%,#eef2ff)}
.banner{background:linear-gradient(120deg,#4f46e5,#7c3aed,#db2777);color:white;padding:2.2rem;border-radius:28px;box-shadow:0 20px 50px #6366f133;animation:rise .55s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}html:focus-within{scroll-behavior:auto!important}}
[data-testid=stMetric]{background:white;border-radius:16px;padding:12px;border:1px solid #e2e8f0}
</style><div class=banner><h1>✨ Resolve AI</h1><p>Turn support messages into grounded, reviewable answers in seconds.</p></div>""", unsafe_allow_html=True)

store = FeedbackStore(ROOT / "feedback" / "ratings.jsonl")
stats = store.summary()
with st.sidebar:
    st.header("Quality loop")
    st.metric("Feedback", stats["responses"])
    st.metric("Satisfaction", f"{stats['satisfaction']:.0%}")
    count = st.slider("Response candidates", 1, 3, 3)
    st.caption("The default engine runs locally. No message leaves this machine.")

subject = st.text_input("Ticket subject", "Package has not moved")
message = st.text_area("Customer message", "I'm frustrated because tracking has not updated in four days. Please help.", height=160)
if st.button("Create resolution", type="primary", use_container_width=True):
    try:
        st.session_state.answer = CustomerSupportAgent().answer(message, subject, count)
    except Exception as exc:
        st.error(str(exc))

if answer := st.session_state.get("answer"):
    cols = st.columns(4)
    for column, label, value in zip(cols, ("Category", "Urgency", "Sentiment", "Confidence"), (answer.category, answer.urgency, answer.sentiment, f"{answer.confidence:.0%}")):
        column.metric(label, str(value).title())
    st.subheader("Recommended response")
    st.success(answer.answer)
    with st.expander("Compare candidates and evidence", expanded=False):
        for candidate in answer.candidates:
            st.markdown(f"**{candidate.strategy.title()} · {candidate.score:.0%}**\n\n{candidate.text}")
        st.caption("Sources: " + ", ".join(answer.citations))
    if answer.human_review:
        st.warning("Human review is required due to urgency, sensitive content, or low retrieval confidence.")
    left, right = st.columns(2)
    if left.button("👍 Helpful", use_container_width=True):
        store.add(answer.ticket_id, 1); st.toast("Feedback saved")
    if right.button("👎 Needs work", use_container_width=True):
        store.add(answer.ticket_id, -1); st.toast("Feedback saved")
