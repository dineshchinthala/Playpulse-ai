import streamlit as st
import pandas as pd
import json
import time
from typing import Literal
from pydantic import BaseModel
from google import genai
from google.genai import types

st.set_page_config(page_title="PlayPulse AI | Telemetry Radar", layout="wide")

st.title("PlayPulse AI: Autonomous App Review & Safety Radar")
st.caption("Transforming unstructured Play Store telemetry into prioritized product backlogs.")

class ReviewAudit(BaseModel):
    Review_ID: str
    category: Literal[
        "Critical Security/Auth",
        "Data Privacy",
        "Feature Request",
        "UI/Functional Bug",
    ]
    severity: Literal["P0", "P1", "P2"]
    root_cause: str

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

max_reviews = st.sidebar.select_slider(
    "Max Reviews per Run (Latency & Quota Guardrail)",
    options=[15, 25, 50, 100, 250, 500],
    value=50,
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
if len(df) > max_reviews:
    st.info(
        f"Uploaded {len(df)} reviews. Auditing the first {max_reviews} reviews to keep demo latency low "
        "(increase the limit up to 500 in the left sidebar)."
    )
audit_df = df.head(max_reviews).copy()
st.dataframe(audit_df, width="stretch")

if st.button("Run AI Triage & Safety Audit"):
    if not api_key:
        st.error("Please configure GEMINI_API_KEY in Secrets or enter it in the sidebar.")
    else:
        client = genai.Client(api_key=api_key)
        results = []
        batch_size = 25
        total_rows = len(audit_df)
        progress_bar = st.progress(0, text="Starting batch telemetry audit...")
        quota_notice = None

        for start_idx in range(0, total_rows, batch_size):
            batch = audit_df.iloc[start_idx : start_idx + batch_size]
            payload = [
                {
                    "Review_ID": str(r["Review_ID"]),
                    "User_Review": str(r["User_Review"])[:500],
                }
                for _, r in batch.iterrows()
            ]
            prompt = (
                "Analyze each mobile app review in the JSON list below. For every item, return its exact "
                "Review_ID, category, engineering severity (P0=critical security/payment/auth, "
                "P1=major bug/privacy, P2=minor bug/feature), and a 1-sentence root_cause.\n\n"
                f"Reviews:\n{json.dumps(payload)}"
            )

            batch_parsed = None
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=list[ReviewAudit],
                            temperature=0.0,
                        ),
                    )
                    batch_parsed = json.loads(response.text)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt == 0:
                        progress_bar.progress(
                            min(start_idx / total_rows, 0.99),
                            text="Rate limit pause — retrying batch in 15 seconds...",
                        )
                        time.sleep(15)
                    else:
                        quota_notice = (
                            f"Stopped after {len(results)}/{total_rows} reviews due to API limit ({e}). "
                            "Showing completed results below."
                        )
                        break

            if batch_parsed is None:
                break

            results.extend(batch_parsed)
            processed_so_far = min(start_idx + batch_size, total_rows)
            progress_bar.progress(
                processed_so_far / total_rows,
                text=f"Audited {processed_so_far} of {total_rows} reviews...",
            )

            if processed_so_far < total_rows:
                time.sleep(4)

        progress_bar.empty()
        if quota_notice:
            st.warning(quota_notice)

        if results:
            res_df = pd.DataFrame(results)
            cols = [c for c in ["Review_ID", "category", "severity", "root_cause"] if c in res_df.columns]
            st.session_state["results"] = res_df[cols]
        else:
            st.error("Could not process reviews right now. Please check your API key or quota.")

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
            client = genai.Client(api_key=api_key)
            priority_sample = res_df[res_df["severity"].isin(["P0", "P1"])].head(25)
            if priority_sample.empty:
                priority_sample = res_df.head(25)

            prd_prompt = f"""
            Act as a Senior Product Manager. Based on these triaged app reviews (out of {len(res_df)} total audited):
            {priority_sample.to_dict(orient='records')}

            Write a concise 1-page Product Requirement Document (PRD) addressing the highest-priority user pain point.
            Include:
            1. Problem Statement
            2. Target User Story
            3. Functional Acceptance Criteria
            4. North Star & Guardrail Metrics
            """

            prd_text = None
            last_err = None
            with st.spinner("Drafting PRD specification..."):
                for candidate_model in ["gemini-3.8-flash", "gemini-3.5-flash-lite"]:
                    try:
                        prd_response = client.models.generate_content(
                            model=candidate_model,
                            contents=prd_prompt,
                            config=types.GenerateContentConfig(temperature=0.2),
                        )
                        prd_text = prd_response.text
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(2)

            if prd_text:
                st.markdown(prd_text)
            else:
                st.error(f"PRD generation temporarily busy: {last_err}")

