import os
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Page config 
st.set_page_config(
    page_title="Kenya Presidential Dashboard",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Colors for the different presidents 
COLORS = {
    "Moi":      "#534AB7",
    "Kibaki":   "#1D9E75",
    "Kenyatta": "#D85A30",
    "Ruto":     "#378ADD",
}

PILLARS = ["Growth", "Stability", "External Balance", "Inclusion", "Sustainability"]

# Indicators where lower value = better performance
INVERT_FOR_SCORE = {
    "FP.CPI.TOTL.ZG", "NY.GDP.DEFL.KD.ZG",
    "GC.DOD.TOTL.GD.ZS", "DT.DOD.DECT.GD.ZS",
    "SL.UEM.TOTL.ZS", "SI.POV.DDAY", "SI.POV.GAPS",
    "SI.POV.GINI", "NE.IMP.GNFS.ZS", "FR.INR.LNDP",
}


# Supabase client 
@st.cache_resource
def get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


#  Data loaders (cached) 
@st.cache_data(ttl=3600)
def load_timeseries() -> pd.DataFrame:
    sb = get_supabase()
    rows = sb.table("timeseries").select("*").execute().data
    df = pd.DataFrame(rows)
    if not df.empty:
        df["year"]  = df["year"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_averages() -> pd.DataFrame:
    sb = get_supabase()
    rows = sb.table("president_averages").select("*").execute().data
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_meta() -> pd.DataFrame:
    sb = get_supabase()
    rows = sb.table("indicator_meta").select("*").execute().data
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_scorecard() -> pd.DataFrame:
    sb = get_supabase()
    rows = sb.table("scorecard").select("*").execute().data
    return pd.DataFrame(rows)


#  Gemini chatbot 
@st.cache_resource
def get_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(

        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def build_data_context(df_avg: pd.DataFrame, df_meta: pd.DataFrame) -> str:
    """Build a concise data summary to inject into Gemini's context."""
    if df_avg.empty or df_meta.empty:
        return "No data available."
    merged = df_avg.merge(df_meta[["indicator", "name", "unit"]], on="indicator", how="left")
    lines = []
    for pillar in PILLARS:
        lines.append(f"\n=== {pillar} ===")
        sub = merged[merged["pillar"] == pillar]
        for _, row in sub.iterrows():
            for president in ["Moi", "Kibaki", "Kenyatta", "Ruto"]:
                p_row = sub[(sub["indicator"] == row["indicator"]) & (sub["president"] == president)]
                if not p_row.empty:
                    val = p_row.iloc[0]["avg_value"]
                    lines.append(f"{president} | {row['name']} ({row['unit']}): {val}")
    return "\n".join(lines)


def ask_gemini(question: str, data_context: str, history: list) -> str:
    model = get_gemini()
    system = f"""You are an expert analyst of Kenya's economic and social history.
You answer questions about presidential performance using ONLY the data provided below.
Be specific, cite numbers, and be balanced and factual.
Always mention which president did better or worse on specific metrics.
If asked something not in the data, say so clearly.

DATA (World Bank indicators, tenure averages):
{data_context}

Presidents and tenures:
- Moi: 1998–2002 (last years of his presidency shown)
- Kibaki: 2003–2013
- Kenyatta (Uhuru): 2013–2022
- Ruto: 2022–present (limited data)
"""
    # Build history for multi-turn
    chat_history = []
    for msg in history:
        chat_history.append({"role": msg["role"], "parts": [msg["content"]]})

    chat = model.start_chat(history=chat_history)
    response = chat.send_message(
        f"System context:\n{system}\n\nUser question: {question}"
    )
    return response.text


# Sidebar 
def render_sidebar():
    with st.sidebar:
        st.image("https://www.freepik.com/free-photo/flag-kenya_1179386.htm#fromView=search&page=1&position=0&uuid=278129c8-ab3d-4482-8f32-21082dcf910d&query=kenya", width=120)
        st.title("🇰🇪 Kenya Dashboard")
        st.caption("Presidential performance 1998–2024")
        st.divider()

        selected_presidents = st.multiselect(
            "Presidents",
            options=["Moi", "Kibaki", "Kenyatta", "Ruto"],
            default=["Moi", "Kibaki", "Kenyatta", "Ruto"],
        )
        selected_pillars = st.multiselect(
            "Pillars",
            options=PILLARS,
            default=PILLARS,
        )
        year_range = st.slider("Year range", 1998, 2024, (1998, 2024))

        st.divider()
        page = st.radio(
            "View",
            ["Overview", "Time Series", "Scorecard", "AI Chatbot"],
            index=0,
        )
        st.divider()
        st.caption("Data: World Bank Open Data\nBuilt with Streamlit + Supabase")

    return selected_presidents, selected_pillars, year_range, page


#  Overview page 
def render_overview(df_avg: pd.DataFrame, presidents: list, year_range: tuple):
    st.header("Overview — Key Economic Indicators")

    # KPI cards — one row per president
    key_indicators = {
        "Real GDP growth (%)":          "NY.GDP.MKTP.KD.ZG",
        "Inflation (%)":                "FP.CPI.TOTL.ZG",
        "Public debt (% GDP)":          "GC.DOD.TOTL.GD.ZS",
        "Unemployment (%)":             "SL.UEM.TOTL.ZS",
        "Life expectancy (yrs)":        "SP.DYN.LE00.IN",
        "Access to electricity (%)":    "EG.ELC.ACCS.ZS",
    }

    for president in presidents:
        st.subheader(f"{president}", divider=False)
        cols = st.columns(len(key_indicators))
        p_avg = df_avg[df_avg["president"] == president]
        for col, (label, code) in zip(cols, key_indicators.items()):
            row = p_avg[p_avg["indicator"] == code]
            val = f"{row.iloc[0]['avg_value']:.1f}" if not row.empty else "—"
            col.metric(label, val)

    st.divider()

    # GDP growth comparison bar chart
    st.subheader("GDP growth by president — annual average")
    gdp_data = df_avg[
        (df_avg["indicator"] == "NY.GDP.MKTP.KD.ZG") &
        (df_avg["president"].isin(presidents))
    ].copy()
    if not gdp_data.empty:
        fig = px.bar(
            gdp_data,
            x="president", y="avg_value",
            color="president",
            color_discrete_map=COLORS,
            text="avg_value",
            labels={"avg_value": "Avg GDP growth (%)", "president": ""},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Inflation vs GDP growth scatter
    st.subheader("Growth vs. Inflation trade-off")
    gdp  = df_avg[df_avg["indicator"] == "NY.GDP.MKTP.KD.ZG"][["president", "avg_value"]].rename(columns={"avg_value": "gdp_growth"})
    infl = df_avg[df_avg["indicator"] == "FP.CPI.TOTL.ZG"][["president", "avg_value"]].rename(columns={"avg_value": "inflation"})
    scatter_df = gdp.merge(infl, on="president")
    scatter_df = scatter_df[scatter_df["president"].isin(presidents)]
    if not scatter_df.empty:
        fig2 = px.scatter(
            scatter_df,
            x="inflation", y="gdp_growth",
            color="president",
            color_discrete_map=COLORS,
            text="president",
            size_max=20,
            labels={"inflation": "Avg inflation (%)", "gdp_growth": "Avg GDP growth (%)"},
        )
        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        fig2.update_traces(textposition="top center", marker_size=14)
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


#  Time Series page
def render_timeseries(df_ts: pd.DataFrame, df_meta: pd.DataFrame,
                      presidents: list, pillars: list, year_range: tuple):
    st.header("Time Series Explorer")

    pillar_sel = st.selectbox("Pillar", pillars)
    indicators = df_meta[df_meta["pillar"] == pillar_sel]["name"].tolist()
    if not indicators:
        st.warning("No indicators for this pillar.")
        return

    indicator_sel = st.selectbox("Indicator", indicators)
    code = df_meta[df_meta["name"] == indicator_sel]["indicator"].iloc[0]
    unit = df_meta[df_meta["name"] == indicator_sel]["unit"].iloc[0]

    filtered = df_ts[
        (df_ts["indicator"] == code) &
        (df_ts["president"].isin(presidents)) &
        (df_ts["year"] >= year_range[0]) &
        (df_ts["year"] <= year_range[1])
    ].sort_values("year")

    if filtered.empty:
        st.info("No data for selected filters.")
        return

    fig = px.line(
        filtered,
        x="year", y="value",
        color="president",
        color_discrete_map=COLORS,
        markers=True,
        labels={"value": f"{indicator_sel} ({unit})", "year": "Year"},
    )

    eras = {"Moi": (1998,2002), "Kibaki": (2003,2013), "Kenyatta": (2013,2022), "Ruto": (2022,2024)}
    for p, (s, e) in eras.items():
        if p in presidents:
            fig.add_vrect(x0=s, x1=e, fillcolor=COLORS[p], opacity=0.07, layer="below", line_width=0)
            fig.add_annotation(x=(s+e)/2, y=0.97, yref="paper", text=p, showarrow=False,
                               font=dict(size=10, color=COLORS[p]))

    fig.update_layout(height=480, legend_title="President")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(
            filtered[["year", "president", "value"]].rename(
                columns={"value": f"{indicator_sel} ({unit})"}
            ),
            use_container_width=True,
        )


#  Scorecard page 
def render_scorecard(df_score: pd.DataFrame, df_avg: pd.DataFrame,
                     presidents: list, pillars: list):
    st.header("Presidential Scorecard")
    st.caption("Scores are normalised 0–100 within each indicator. For negative indicators (inflation, debt etc.) the score is inverted so higher always means better.")

    if df_score.empty:
        st.warning("Scorecard data not available. Check the Supabase view.")
        return

    # Invert scores for negative indicators
    df_score = df_score.copy()
    df_score.loc[df_score["indicator"].isin(INVERT_FOR_SCORE), "score_raw"] = (
        100 - pd.to_numeric(df_score.loc[df_score["indicator"].isin(INVERT_FOR_SCORE), "score_raw"], errors="coerce")
    )

    df_score = df_score[df_score["president"].isin(presidents)]

    # Overall score per president per pillar
    pillar_scores = (
        df_score[df_score["pillar"].isin(pillars)]
        .groupby(["president", "pillar"])["score_raw"]
        .mean()
        .reset_index()
        .rename(columns={"score_raw": "score"})
    )

    # Radar chart
    st.subheader("Pillar radar")
    fig = go.Figure()
    for president in presidents:
        p_data = pillar_scores[pillar_scores["president"] == president]
        scores  = [p_data[p_data["pillar"] == pl]["score"].values[0]
                   if pl in p_data["pillar"].values else 0 for pl in pillars]
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=pillars + [pillars[0]],
            name=president,
            line_color=COLORS[president],
            fill="toself",
            opacity=0.4,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Overall ranking
    st.subheader("Overall ranking")
    overall = (
        df_score[df_score["pillar"].isin(pillars)]
        .groupby("president")["score_raw"]
        .mean()
        .reset_index()
        .rename(columns={"score_raw": "overall_score"})
        .sort_values("overall_score", ascending=False)
    )
    overall = overall[overall["president"].isin(presidents)]
    fig2 = px.bar(
        overall,
        x="overall_score", y="president",
        orientation="h",
        color="president",
        color_discrete_map=COLORS,
        text="overall_score",
        labels={"overall_score": "Score (0–100)", "president": ""},
    )
    fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig2.update_layout(showlegend=False, height=300, xaxis_range=[0, 105])
    st.plotly_chart(fig2, use_container_width=True)


#  AI Chatbot page 
def render_chatbot(df_avg: pd.DataFrame, df_meta: pd.DataFrame):
    st.header("AI Analyst — Ask about presidential performance")
    st.caption("Powered by Gemini. Answers are grounded in World Bank data loaded into this dashboard.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    data_context = build_data_context(df_avg, df_meta)

    # Starter questions
    starters = [
        "Which president had the best GDP growth?",
        "How did inflation compare across presidents?",
        "Which era saw the sharpest rise in public debt?",
        "Who performed best on poverty reduction?",
        "Compare Kibaki and Kenyatta overall.",
    ]
    st.write("**Try asking:**")
    cols = st.columns(len(starters))
    for col, q in zip(cols, starters):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about any president or indicator...")

    # Handle starter button OR typed input
    question = None
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
    elif user_input:
        question = user_input

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analysing data..."):
                try:
                    answer = ask_gemini(
                        question,
                        data_context,
                        st.session_state.chat_history[:-1],
                    )
                except Exception as e:
                    answer = f"Error calling Gemini API: {e}. Check your GEMINI_API_KEY."
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


#  Main
def main():
    presidents, pillars, year_range, page = render_sidebar()

    # Load data
    with st.spinner("Loading data from Supabase..."):
        df_ts    = load_timeseries()
        df_avg   = load_averages()
        df_meta  = load_meta()
        df_score = load_scorecard()

    if df_avg.empty or df_meta.empty:
        st.error("Data not found. Run the ETL script first and ensure the Supabase tables/views exist.")
        st.stop()

    if page in ["Overview", "Time Series"] and df_ts.empty:
        st.error("No timeseries data found. Run the ETL script first and ensure the timeseries table exists.")
        st.stop()

    # Route to page
    if page == "Overview":
        render_overview(df_avg, presidents, year_range)
    elif page == "Time Series":
        render_timeseries(df_ts, df_meta, presidents, pillars, year_range)
    elif page == "Scorecard":
        render_scorecard(df_score, df_avg, presidents, pillars)
    elif page == "AI Chatbot":
        render_chatbot(df_avg, df_meta)


if __name__ == "__main__":
    main()
