import streamlit as st
from distress_score import compute_dps, classify_risk

st.set_page_config(page_title="PIIBI Distress Score", layout="centered")
st.title("🏭 Pre-Insolvency Intelligence Bureau")
st.subheader("Free Distress Probability Score (India)")

st.markdown("Enter company indicators (all fields optional, but more data = better prediction)")

col1, col2 = st.columns(2)
with col1:
    gst_delay = st.number_input("GST returns delayed (months)", min_value=0, max_value=12, value=0)
    legal_cases = st.number_input("Active recovery suits (e-Courts)", min_value=0, max_value=50, value=0)
    auditor_resigned = st.selectbox("Auditor resigned in last 6 months?", [0,1], format_func=lambda x: "Yes" if x else "No")
    director_changes = st.number_input("Director changes (last 2 years)", min_value=0, max_value=10, value=0)

with col2:
    neg_networth = st.selectbox("Negative net worth (last balance sheet)?", [0,1], format_func=lambda x: "Yes" if x else "No")
    payment_days = st.number_input("Average payment days outstanding", min_value=0, max_value=360, value=30)
    nclt_mention = st.selectbox("Already mentioned in NCLT?", [0,1], format_func=lambda x: "Yes" if x else "No")

features = {
    'gst_delayed_months': gst_delay,
    'legal_cases_count': legal_cases,
    'auditor_resigned': auditor_resigned,
    'director_change_frequency': director_changes,
    'negative_net_worth': neg_networth,
    'payment_days_outstanding': payment_days,
    'nclt_mention': nclt_mention
}

if st.button("Compute Distress Score"):
    score = compute_dps(features)
    risk = classify_risk(score)
    st.metric("Distress Probability Score", f"{score} / 100")
    st.error(risk) if score >= 50 else st.info(risk)
    
    st.caption("Data sources: MCA, GST portal, e-Courts. This is an analytical opinion only.")
