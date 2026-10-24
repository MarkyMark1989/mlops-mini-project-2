"""Streamlit front end for the churn model deployed on an Azure ML online endpoint.

    .venv/bin/streamlit run frontend/streamlit_app.py

The endpoint comes from .streamlit/secrets.toml (written by `deployment/deploy.py secrets`)
or the SCORING_URI / ENDPOINT_KEY environment variables.
"""
import json
import os

import requests
import streamlit as st

# (column, label, kind, min, max, default, step); kind "binary" renders a Yes/No choice.
# Defaults are roughly the dataset medians.
FIELDS = [
    ("AccountWeeks", "Weeks with an active account", "int", 0, 300, 101, 1),
    ("ContractRenewal", "Recently renewed contract?", "binary", 0, 1, 1, 1),
    ("DataPlan", "Has a data plan?", "binary", 0, 1, 0, 1),
    ("DataUsage", "Monthly data usage (GB)", "float", 0.0, 10.0, 0.0, 0.1),
    ("CustServCalls", "Calls to customer service", "int", 0, 15, 1, 1),
    ("DayMins", "Average daytime minutes per month", "float", 0.0, 400.0, 179.4, 1.0),
    ("DayCalls", "Average daytime calls per month", "int", 0, 200, 101, 1),
    ("MonthlyCharge", "Average monthly bill ($)", "float", 0.0, 150.0, 53.5, 0.5),
    ("OverageFee", "Largest overage fee, last 12 months ($)", "float", 0.0, 25.0, 10.07, 0.01),
    ("RoamMins", "Roaming minutes per month", "float", 0.0, 30.0, 10.3, 0.1),
]


def setting(env_name, secret_name):
    if os.environ.get(env_name):
        return os.environ[env_name]
    try:
        return st.secrets.get(secret_name)
    except Exception:  # no secrets.toml
        return None


def call_endpoint(scoring_uri, key, deployment, values):
    body = {"input_data": {
        "columns": [f[0] for f in FIELDS],
        "index": [0],
        "data": [[values[f[0]] for f in FIELDS]],
    }}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    if deployment:
        headers["azureml-model-deployment"] = deployment
    response = requests.post(scoring_uri, json=body, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()
    # Custom scoring scripts may return their result as a JSON string.
    if isinstance(result, str):
        result = json.loads(result)
    # The endpoint names the deployment that answered, which shows the traffic split at work.
    return body, result, response.headers.get("azureml-model-deployment")


def parse_result(result):
    """Return (prediction, probability or None) for the first row of a response."""
    first = result[0] if isinstance(result, list) else result
    if isinstance(first, dict):
        return int(first["prediction"]), first.get("churn_probability")
    return int(first), None  # no-code MLflow deployments return bare predictions


st.set_page_config(page_title="Churn Predictor")
st.title("Telecom Customer Churn Predictor")
st.caption("Enter a customer's account details to predict whether they will cancel their service.")

scoring_uri = setting("SCORING_URI", "scoring_uri")
key = setting("ENDPOINT_KEY", "endpoint_key")

with st.sidebar:
    st.header("Endpoint")
    if scoring_uri:
        st.code(scoring_uri, language=None)
    else:
        st.warning("No endpoint configured. Run `deployment/deploy.py secrets` or set SCORING_URI and ENDPOINT_KEY.")
    deployment = st.selectbox(
        "Deployment",
        ["", "blue", "green"],
        format_func=lambda d: d or "Endpoint traffic split",
        help="Target a specific deployment, or let the endpoint route by its traffic split.",
    )

with st.form("customer"):
    values = {}
    left, right = st.columns(2)
    for i, (col, label, kind, lo, hi, default, step) in enumerate(FIELDS):
        container = left if i % 2 == 0 else right
        if kind == "binary":
            choice = container.radio(label, ["Yes", "No"], index=0 if default == 1 else 1, horizontal=True, key=col)
            values[col] = 1 if choice == "Yes" else 0
        else:
            values[col] = container.number_input(label, min_value=lo, max_value=hi, value=default, step=step, key=col)
    submitted = st.form_submit_button("Predict churn", type="primary", disabled=not (scoring_uri and key))

if submitted:
    try:
        with st.spinner("Calling the Azure ML endpoint..."):
            request_body, result, served_by = call_endpoint(scoring_uri, key, deployment, values)
        prediction, probability = parse_result(result)
    except requests.HTTPError as e:
        st.error(f"The endpoint returned {e.response.status_code}: {e.response.text[:500]}")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        st.error(f"Could not get a prediction: {e}")
    else:
        if prediction == 1:
            st.error("Prediction: this customer is **likely to churn**.")
        else:
            st.success("Prediction: this customer is **likely to stay**.")
        if probability is not None:
            st.metric("Churn probability", f"{probability:.0%}")
        if served_by:
            st.caption(f"Answered by deployment: {served_by}")
        with st.expander("Request and response"):
            st.json({"request": request_body, "response": result})
