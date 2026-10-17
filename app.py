"""CustomerIQ - Streamlit dashboard.   Run:  python -m streamlit run app.py"""
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from customeriq.config import CATEGORICAL_FEATURES
from customeriq.data import load_dataset
from customeriq.predict import ModelBundle, ModelNotTrainedError
from customeriq.theme import CSS, PALETTE, RISK_COLORS, kpi, result_card, style_fig

st.set_page_config(page_title="CustomerIQ", page_icon=":material/signal_cellular_alt:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return ModelBundle()


@st.cache_data
def get_scored_data():
    df, _, _ = load_dataset()
    return get_bundle().score_dataframe(df)


def words(feature):
    """SeniorCitizen -> Senior Citizen, StreamingTV -> Streaming TV."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", feature)


def fmt(value):
    return int(value) if isinstance(value, float) and value.is_integer() else value


def show(fig, key):
    st.plotly_chart(fig, key=key, theme=None)


try:
    bundle = get_bundle()
except ModelNotTrainedError:
    st.markdown('<p class="ciq-title">CustomerIQ</p>', unsafe_allow_html=True)
    st.error("No trained model found. Open a terminal in the project folder and run "
             "`python -m customeriq.train`, then reload this page.")
    st.stop()

meta = bundle.meta
tm = meta["test_metrics"]
schema = meta["schema"]["categorical"]
data = get_scored_data()

# ------------------------------------------------------------------ sidebar: model card
with st.sidebar:
    st.markdown("### CustomerIQ")
    st.caption("Customer retention intelligence")
    st.markdown("---")
    st.markdown(f"**Model**  \n{meta['best_model']}")
    st.markdown(f"**Trained on**  \n{meta['n_customers']:,} customers ({meta['data_source']})")
    st.markdown(f"**Flag customers above**  \n{meta['threshold']:.0%} churn probability")
    ca = meta["cost_assumptions"]
    st.markdown(f"**Cost assumption**  \nA missed churner costs {ca['false_negative']:g}x "
                f"a retention offer sent to someone who would have stayed")

# ------------------------------------------------------------------ header
st.markdown('<p class="ciq-title">CustomerIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="ciq-sub">Find the customers most likely to leave, and see why.</p>',
            unsafe_allow_html=True)
if meta["synthetic_data"]:
    st.warning("This model was trained on synthetic demo data, so the numbers are not real results. "
               "Add the Telco CSV to the data folder and run `python -m customeriq.train`.")

tab_overview, tab_predict, tab_model, tab_risk = st.tabs(
    ["Overview", "Predict churn", "Model performance", "At-risk customers"])

# ------------------------------------------------------------------ overview
with tab_overview:
    by_contract = data.groupby("Contract")["Churn"].mean().mul(100).sort_values(ascending=False)
    top, bottom = by_contract.index[0], by_contract.index[-1]
    st.markdown(f'<div class="ciq-lead">{top} customers churn at {by_contract.iloc[0]:.0f}%, '
                f'compared with {by_contract.iloc[-1]:.0f}% on {bottom} contracts.</div>',
                unsafe_allow_html=True)

    high = data[data["risk_level"] == "High"]
    k = st.columns(4)
    k[0].markdown(kpi("Customers", f"{len(data):,}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Churn rate", f"{data['Churn'].mean():.1%}", "Share who have left"),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("High-risk customers", f"{len(high):,}", f"{len(high) / len(data):.0%} of all customers",
                      "high"), unsafe_allow_html=True)
    k[3].markdown(kpi("Monthly charges at risk", f"{high['MonthlyCharges'].sum():,.0f}",
                      "From high-risk customers"), unsafe_allow_html=True)
    st.write("")

    def bar(series, title, colors=None, key=""):
        colors = colors or [PALETTE["coral"] if i == 0 else PALETTE["mist"] for i in range(len(series))]
        fig = go.Figure(go.Bar(x=series.index.astype(str), y=series.values, marker_color=colors,
                               text=[f"{v:.0f}%" for v in series.values], textposition="outside",
                               cliponaxis=False, hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig.update_layout(title=title, yaxis_ticksuffix="%", yaxis_range=[0, series.max() * 1.2])
        show(style_fig(fig), key)

    c1, c2 = st.columns(2)
    with c1:
        bar(by_contract, "Churn rate by contract type", key="ov_contract")
    with c2:
        bands = pd.cut(data["tenure"], [-1, 11, 23, 47, 100], labels=["0-11", "12-23", "24-47", "48+"])
        by_tenure = data.groupby(bands, observed=True)["Churn"].mean().mul(100)
        bar(by_tenure, "Churn rate by months as a customer",
            [PALETTE["coral"], PALETTE["mist"], PALETTE["mist"], PALETTE["mist"]], "ov_tenure")

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for outcome, color in (("Stayed", PALETTE["mist"]), ("Churned", PALETTE["coral"])):
            subset = data[data["Churn"] == (outcome == "Churned")]
            fig.add_trace(go.Histogram(x=subset["MonthlyCharges"], name=outcome, marker_color=color,
                                       opacity=0.8, nbinsx=30))
        fig.update_layout(title="Monthly charges: customers who left vs stayed", barmode="overlay")
        show(style_fig(fig), "ov_charges")
    with c4:
        counts = data["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[RISK_COLORS[r] for r in counts.index],
                               text=[f"{int(v):,}" for v in counts.values], textposition="outside",
                               cliponaxis=False, hovertemplate="%{x}: %{y:,} customers<extra></extra>"))
        fig.update_layout(title="Customers by predicted risk level", yaxis_range=[0, counts.max() * 1.2])
        show(style_fig(fig), "ov_risk")
    st.caption("Risk levels here score every customer, including those the model learned from, so counts "
               "are slightly optimistic. The Model performance tab uses unseen customers only.")

# ------------------------------------------------------------------ predict
def pick(feature, *preferred):
    choices = schema[feature]["choices"]
    return next((p for p in preferred if p in choices), schema[feature]["default"])


def apply_preset(name):
    values = {f: schema[f]["default"] for f in CATEGORICAL_FEATURES}
    if name == "risky":
        values.update(Contract=pick("Contract", "Month-to-month"), InternetService=pick("InternetService", "Fiber optic"),
                      OnlineSecurity=pick("OnlineSecurity", "No"), OnlineBackup=pick("OnlineBackup", "No"),
                      DeviceProtection=pick("DeviceProtection", "No"), TechSupport=pick("TechSupport", "No"),
                      StreamingTV=pick("StreamingTV", "Yes"), StreamingMovies=pick("StreamingMovies", "Yes"),
                      PaymentMethod=pick("PaymentMethod", "Electronic check", "Mailed Check", "Mailed check"),
                      PaperlessBilling=pick("PaperlessBilling", "Yes"), Partner=pick("Partner", "No"),
                      Dependents=pick("Dependents", "No"), PhoneService=pick("PhoneService", "Yes"),
                      MultipleLines=pick("MultipleLines", "No"))
        numbers = dict(tenure=2, monthly=89.0, senior=False)
    else:
        values.update(Contract=pick("Contract", "Two year"), InternetService=pick("InternetService", "DSL"),
                      OnlineSecurity=pick("OnlineSecurity", "Yes"), OnlineBackup=pick("OnlineBackup", "Yes"),
                      DeviceProtection=pick("DeviceProtection", "Yes"), TechSupport=pick("TechSupport", "Yes"),
                      StreamingTV=pick("StreamingTV", "No"), StreamingMovies=pick("StreamingMovies", "No"),
                      PaymentMethod=pick("PaymentMethod", "Credit card (automatic)", "Credit Card",
                                         "Bank transfer (automatic)", "Bank Withdrawal"),
                      PaperlessBilling=pick("PaperlessBilling", "No"), Partner=pick("Partner", "Yes"),
                      Dependents=pick("Dependents", "Yes"), PhoneService=pick("PhoneService", "Yes"),
                      MultipleLines=pick("MultipleLines", "No"))
        numbers = dict(tenure=60, monthly=58.0, senior=False)
    for feature, value in values.items():
        st.session_state[f"f_{feature}"] = value
    st.session_state["f_tenure"] = numbers["tenure"]
    st.session_state["f_monthly"] = numbers["monthly"]
    st.session_state["f_senior"] = numbers["senior"]


def choice(feature):
    options = schema[feature]["choices"]
    if len(options) <= 3:
        return st.radio(words(feature), options, key=f"f_{feature}", horizontal=True)
    return st.selectbox(words(feature), options, key=f"f_{feature}")


with tab_predict:
    if "f_tenure" not in st.session_state:
        apply_preset("risky")

    left, right = st.columns([1.2, 1], gap="large")
    customer = {}
    with left:
        p1, p2, _ = st.columns([1, 1, 0.6])
        p1.button("Example: likely to leave", on_click=apply_preset, args=("risky",))
        p2.button("Example: loyal customer", on_click=apply_preset, args=("loyal",))

        st.markdown("##### Account")
        a1, a2 = st.columns(2)
        with a1:
            tenure = st.number_input("Months as a customer", min_value=0, max_value=100, step=1, key="f_tenure")
            customer["Partner"] = choice("Partner")
        with a2:
            customer["gender"] = choice("gender")
            customer["Dependents"] = choice("Dependents")
        senior = st.toggle("Senior citizen", key="f_senior")

        st.markdown("##### Services")
        s1, s2 = st.columns(2)
        with s1:
            customer["PhoneService"] = choice("PhoneService")
        with s2:
            if customer["PhoneService"] == "No":
                customer["MultipleLines"] = pick("MultipleLines", "No phone service")
            else:
                customer["MultipleLines"] = choice("MultipleLines")
        customer["InternetService"] = choice("InternetService")
        has_internet = customer["InternetService"] != "No"
        addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                  "StreamingMovies"]
        if has_internet:
            for column, feature in zip(st.columns(3) * 2, addons):
                with column:
                    customer[feature] = choice(feature)
        else:
            for feature in addons:
                customer[feature] = pick(feature, "No internet service")
            st.caption("No internet service, so internet add-ons do not apply.")

        st.markdown("##### Billing")
        b1, b2 = st.columns(2)
        with b1:
            customer["Contract"] = choice("Contract")
            customer["PaperlessBilling"] = choice("PaperlessBilling")
        with b2:
            customer["PaymentMethod"] = choice("PaymentMethod")
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, step=0.5, key="f_monthly")

    customer.update(tenure=tenure, SeniorCitizen=int(senior), MonthlyCharges=monthly,
                    TotalCharges=round(tenure * monthly, 2))

    with right:
        try:
            result = bundle.predict_one(customer)
        except ValueError as err:
            st.error(f"Some details are not valid: {err}")
        else:
            st.markdown(result_card(result["churn_probability"], result["risk_level"], meta["threshold"]),
                        unsafe_allow_html=True)

            factors = pd.DataFrame(result["top_factors"])
            if not factors.empty:
                factors["label"] = [f"{words(f)}: {fmt(v)}" for f, v in zip(factors["feature"], factors["value"])]
                factors["points"] = factors["effect"] * 100
                factors = factors.iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=factors["points"], y=factors["label"], orientation="h", cliponaxis=False,
                    marker_color=[PALETTE["coral"] if p > 0 else PALETTE["teal"] for p in factors["points"]],
                    text=[f"{p:+.1f}" for p in factors["points"]], textposition="outside",
                    hovertemplate="%{y}: %{x:+.1f} points<extra></extra>"))
                fig.update_layout(title="What moves this score (percentage points)")
                fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
                low, high = factors["points"].min(), factors["points"].max()
                fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor=PALETTE["line"],
                                 range=[min(0, low) * 1.3 - 1, max(0, high) * 1.3 + 5])
                show(style_fig(fig, 300), "pred_factors")
                st.caption("Red raises the churn estimate, teal lowers it. These are patterns the model "
                           "learned, not proven causes.")

            st.markdown("##### Suggested actions")
            for tip in result["suggestions"]:
                if tip.startswith("These suggestions"):
                    st.caption(tip)
                else:
                    st.markdown(f"- {tip}")

# ------------------------------------------------------------------ model performance
with tab_model:
    n_test = tm["tp"] + tm["fp"] + tm["tn"] + tm["fn"]
    st.markdown(
        f'<div class="ciq-lead">On {n_test:,} customers the model had never seen, it found '
        f'{tm["recall"]:.0%} of the people who left. About {tm["precision"]:.0%} of the customers it '
        f'flagged really did leave.</div>', unsafe_allow_html=True)

    k = st.columns(4)
    k[0].markdown(kpi("Precision", f"{tm['precision']:.2f}", "Of flagged customers, share who left"),
                  unsafe_allow_html=True)
    k[1].markdown(kpi("Recall", f"{tm['recall']:.2f}", "Of customers who left, share we flagged"),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("F1-score", f"{tm['f1']:.2f}", "Balance of precision and recall"),
                  unsafe_allow_html=True)
    k[3].markdown(kpi("ROC-AUC", f"{tm['roc_auc']:.2f}", "Ranking quality (0.50 is guessing)"),
                  unsafe_allow_html=True)
    st.write("")
    st.info(f"Accuracy is {tm['accuracy']:.1%}, but accuracy alone misleads: predicting that nobody leaves "
            f"would score {tm['majority_class_accuracy']:.1%} while finding no churners at all.")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("##### Models compared on validation data")
        comp = pd.DataFrame(meta["comparison"]).rename(columns={
            "model": "Model", "val_roc_auc": "ROC-AUC", "val_pr_auc": "PR-AUC", "val_f1_at_0.5": "F1 at 50%"})
        st.dataframe(comp, hide_index=True, column_config={
            c: st.column_config.NumberColumn(c, format="%.3f") for c in ["ROC-AUC", "PR-AUC", "F1 at 50%"]})
        st.caption(f"Selected: {meta['best_model']}, the highest ROC-AUC on validation data.")
    with m2:
        cm = [[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]]
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted stay", "Predicted leave"], y=["Actually stayed", "Actually left"],
                                   text=[[f"{v:,}" for v in row] for row in cm], texttemplate="%{text}",
                                   colorscale=[[0, "#EEF1F6"], [1, PALETTE["teal"]]], showscale=False,
                                   hovertemplate="%{y}, %{x}: %{z:,}<extra></extra>"))
        fig.update_layout(title="Test results (confusion matrix)")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        show(style_fig(fig, 300), "mp_cm")

    m3, m4 = st.columns(2)
    with m3:
        roc = meta["roc_curve"]
        fig = go.Figure([
            go.Scatter(x=roc["fpr"], y=roc["tpr"], name="Model", line=dict(color=PALETTE["teal"], width=3)),
            go.Scatter(x=[0, 1], y=[0, 1], name="Guessing", line=dict(color=PALETTE["mist"], dash="dash"))])
        fig.update_layout(title="ROC curve: the higher above the dashed line, the better")
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        show(style_fig(fig), "mp_roc")
    with m4:
        imp = pd.DataFrame(meta["feature_importance"]).head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importance"], y=[words(f) for f in imp["feature"]], orientation="h",
                               marker_color=PALETTE["ink"], hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Most influential features (drop in ROC-AUC when shuffled)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        show(style_fig(fig), "mp_imp")

    st.markdown("##### Performance by customer group")
    segments = pd.DataFrame(meta.get("segments", []))
    if segments.empty:
        st.info("Retrain the model (`python -m customeriq.train`) to see results by customer group.")
    else:
        group_by = st.radio("Group customers by", list(segments["group_by"].unique()), horizontal=True,
                            key="seg_by")
        seg = segments[segments["group_by"] == group_by]
        label = lambda v: "" if pd.isna(v) else f"{v:.0f}%"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=seg["group"], y=seg["recall"] * 100, name="Recall: churners found",
                             marker_color=PALETTE["teal"], text=[label(v) for v in seg["recall"] * 100],
                             textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(x=seg["group"], y=seg["precision"] * 100, name="Precision: flags that were right",
                             marker_color=PALETTE["mist"], text=[label(v) for v in seg["precision"] * 100],
                             textposition="outside", cliponaxis=False))
        fig.update_layout(barmode="group", yaxis_ticksuffix="%", yaxis_range=[0, 118],
                          title=f"Recall and precision by {group_by.lower()}")
        show(style_fig(fig), "mp_seg")
        table = seg.assign(**{"Churn rate": seg["churn_rate"] * 100, "Flagged": seg["flagged_rate"] * 100,
                              "Recall": seg["recall"] * 100, "Precision": seg["precision"] * 100})
        table = table[["group", "customers", "Churn rate", "Flagged", "Recall", "Precision"]].rename(
            columns={"group": "Group", "customers": "Customers"})
        st.dataframe(table, hide_index=True, column_config={
            c: st.column_config.NumberColumn(c, format="%.0f%%") for c in ["Churn rate", "Flagged", "Recall", "Precision"]})
        st.caption("Use this to spot groups where the model finds fewer churners. Groups with fewer than 30 "
                   "test customers are hidden, and small groups give noisy results.")

    with st.expander("Limits to keep in mind"):
        st.markdown(
            "- Influence shows association, not cause. Changing a feature may not change whether someone leaves.\n"
            "- The model learned from past customers and can get worse if behaviour changes.\n"
            "- Probabilities are estimates. Review the list before contacting anyone.\n"
            "- Check performance across customer groups before using it in a real business.")

# ------------------------------------------------------------------ at-risk customers
with tab_risk:
    f1, f2, f3 = st.columns(3)
    levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    contract_options = sorted(data["Contract"].unique())
    contracts = f2.multiselect("Contract", contract_options, default=contract_options)
    min_prob = f3.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05, format="%.2f")

    view = data[data["risk_level"].isin(levels) & data["Contract"].isin(contracts)
                & (data["churn_probability"] >= min_prob)].sort_values("churn_probability", ascending=False)

    if view.empty:
        st.info("No customers match these filters. Add a risk level or lower the minimum probability.")
    else:
        st.markdown(f'<div class="ciq-lead">{len(view):,} customers match, with '
                    f'{view["MonthlyCharges"].sum():,.0f} in monthly charges.</div>', unsafe_allow_html=True)
        table = view.assign(**{"Churn risk": view["churn_probability"] * 100})
        columns = ["customerID", "Churn risk", "risk_level", "Contract", "tenure", "MonthlyCharges",
                   "InternetService", "TechSupport", "PaymentMethod"]
        table = table[[c for c in columns if c in table.columns]].rename(columns={
            "customerID": "Customer", "risk_level": "Risk level", "tenure": "Months",
            "MonthlyCharges": "Monthly charges", "InternetService": "Internet", "TechSupport": "Tech support",
            "PaymentMethod": "Payment method"})
        st.dataframe(table, hide_index=True, column_config={
            "Churn risk": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=100, format="%.0f%%"),
            "Monthly charges": st.column_config.NumberColumn("Monthly charges", format="%.2f")})
        st.download_button("Download this list (CSV)", view.to_csv(index=False).encode(),
                           "at_risk_customers.csv", "text/csv")
