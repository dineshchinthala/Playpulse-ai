import streamlit as st
import pandas as pd
import json
from typing import Literal
from pydantic import BaseModel
from google import genai
from google.genai import types

st.set_page_config(page_title="PlayPulse AI | Telemetry Radar", layout="wide")

st.title("PlayPulse AI: Autonomous App Review & Safety Radar")
st.caption("Transforming unstructured Play Store telemetry into prioritized product backlogs.")

# Structured output schema for deterministic classification
class ReviewAudit(BaseModel):
    category: Literal[
        "Critical Security/Auth",
        "Data Privacy",
        "Feature Request",
        "UI/Functional Bug",
    ]
    severity: Literal["P0", "P1", "P2"]
    root_cause: str

# Load API key from Streamlit Cloud Secrets or sidebar fallback
api_key = ""
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

model_choice = st.sidebar.selectbox(
    "Select Gemini Model",
    ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    index=0,
)

default_data = pd.DataFrame({
    "Review_ID": ["REV-101", "REV-102", "REV-103", "REV-104", "REV-105"],
    "User_Review": [
        "My account got locked after UPI payment failed and Rs 400 was debited! Fix this scam!",
        "Why does a calculator app access my contacts and microphone in the background? Suspicious.",
        "The dark mode looks clean, but please add a one-click PDF export button on the dashboard.",
        "App freezes on the splash screen every time I open it on Android 14.",
        "I entered my OTP and immediately lost access to my account. Unauthorized login from unknown device.",
    ],
})

uploaded_file = st.sidebar.file_uploader("Upload App Reviews (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "User_Review" not in df.columns:
        text_cols = df.select_dtypes(include="object").columns
        if len(text_cols) > 0:
            df = df.rename(columns={text_cols[-1]: "User_Review"})
        else:
            st.error("Uploaded CSV must contain at least one text review column.")
            st.stop()
    if "Review_ID" not in df.columns:
        df.insert(0, "Review_ID", [f"REV-{i+101}" for i in range(len(df))])
else:
    df = default_data

st.subheader("1. Raw User Telemetry")
st.dataframe(df, width="stretch")

if st.button("Run AI Triage & Safety Audit"):
    if not api_key:
        st.error("Please configure GEMINI_API_KEY in Secrets or enter it in the sidebar.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            results = []
            with st.spinner(f"Auditing telemetry streams with {model_choice}..."):
                for _, row in df.iterrows():
                    prompt = (
                        "Analyze this mobile app review and classify its category, "
                        "engineering severity (P0=critical security/payment/auth, "
                        "P1=major bug/privacy, P2=minor bug/feature), and 1-sentence root cause.\n\n"
                        f"Review: \"{row['User_Review']}\""
                    )
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ReviewAudit,
                            temperature=0.0,
                        ),
                    )
                    parsed = json.loads(response.text)
                    parsed["Review_ID"] = row["Review_ID"]
                    results.append(parsed)
            st.session_state["results"] = pd.DataFrame(results)[
                ["Review_ID", "category", "severity", "root_cause"]
            ]
        except Exception as e:
            st.error(f"Audit failed: {e}")

if "results" in st.session_state:
    res_df = st.session_state["results"]
    st.subheader("2. AI Telemetry & Triage Matrix")

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews Processed", len(res_df))
    c2.metric("P0 Critical Alerts", int((res_df["severity"] == "P0").sum()))
    c3.metric("Feature Requests", int((res_df["category"] == "Feature Request").sum()))

    st.dataframe(res_df, width="stretch")

    st.divider()
    st.subheader("3. Autonomous PRD Micro-Spec")
    if st.button("Generate Feature PRD from Top Backlog"):
        if not api_key:
            st.error("Please provide a Gemini API Key first.")
        else:
            try:
                client = genai.Client(api_key=api_key)
                prd_prompt = f"""
                Act as a Senior Product Manager. Based on these triaged app reviews:
                {res_df.to_dict(orient='records')}

                Write a concise 1-page Product Requirement Document (PRD) addressing the highest-priority user pain point.
                Include:
                1. Problem Statement
                2. Target User Story
                3. Functional Acceptance Criteria
                4. North Star & Guardrail Metrics
                """
                with st.spinner("Drafting PRD specification..."):
                    prd_response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prd_prompt,
                        config=types.GenerateContentConfig(temperature=0.2),
                    )
                st.markdown(prd_response.text)
            except Exception as e:
                st.error(f"PRD generation failed: {e}")
                
