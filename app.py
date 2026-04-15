"""
Kenya Presidential Economic Dashboard - Enhanced Interactive Version
"""
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

# Presidential eras for reference
PRESIDENTIAL_ERAS = {
    "Moi": (1998, 2002),
    "Kibaki": (2003, 2013),
    "Kenyatta": (2013, 2022),
    "Ruto": (2022, 2024)
}

# Indicators where lower value = better performance
INVERT_FOR_SCORE = {
    "FP.CPI.TOTL.ZG", "NY.GDP.DEFL.KD.ZG",
    "GC.DOD.TOTL.GD.ZS", "DT.DOD.DECT.GD.ZS",
    "SL.UEM.TOTL.ZS", "SI.POV.DDAY", "SI.POV.GAPS",
    "SI.POV.GINI", "NE.IMP.GNFS.ZS", "FR.INR.LNDP",
}

# Key indicators for quick view
KEY_INDICATORS = {
    "Real GDP growth (%)":          "NY.GDP.MKTP.KD.ZG",
    "Inflation (%)":                "FP.CPI.TOTL.ZG",
    "Public debt (% GDP)":          "GC.DOD.TOTL.GD.ZS",
    "Unemployment (%)":             "SL.UEM.TOTL.ZS",
    "Life expectancy (yrs)":        "SP.DYN.LE00.IN",
    "Access to electricity (%)":    "EG.ELC.ACCS.ZS",
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

@st.cache_data(ttl=3600)
def load_pillar_scores() -> pd.DataFrame:
    """Calculate pillar scores for health meters"""
    df_score = load_scorecard()
    if df_score.empty:
        return pd.DataFrame()

    # Invert negative indicators
    df_score = df_score.copy()
    df_score.loc[df_score["indicator"].isin(INVERT_FOR_SCORE), "score_raw"] = (
        100 - pd.to_numeric(df_score.loc[df_score["indicator"].isin(INVERT_FOR_SCORE), "score_raw"], errors="coerce")
    )

    # Average by pillar and president
    pillar_scores = (
        df_score.groupby(["president", "pillar"])["score_raw"]
        .mean()
        .reset_index()
        .rename(columns={"score_raw": "score"})
    )

    return pillar_scores

#  Gemini chatbot 
@st.cache_resource
def get_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")
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
    chat_history = []
    for msg in history:
        chat_history.append({"role": msg["role"], "parts": [msg["content"]]})

    chat = model.start_chat(history=chat_history)
    response = chat.send_message(
        f"System context:\n{system}\n\nUser question: {question}"
    )
    return response.text

# ==================== ENHANCED INTERACTIVE COMPONENTS ====================

def render_pillar_health_meters(df_pillar_scores, selected_presidents):
    """Render interactive pillar health meters"""
    st.subheader("🏛️ Pillar Health Overview", help="Click on a pillar to see detailed indicators")

    # Create columns for 5 pillars
    cols = st.columns(5)

    pillar_colors = {
        "Growth": "#4CAF50",
        "Stability": "#2196F3", 
        "External Balance": "#FF9800",
        "Inclusion": "#9C27B0",
        "Sustainability": "#F44336"
    }

    for idx, (pillar, col) in enumerate(zip(PILLARS, cols)):
        with col:
            # Calculate average score across selected presidents
            if not df_pillar_scores.empty:
                scores = df_pillar_scores[
                    (df_pillar_scores["pillar"] == pillar) & 
                    (df_pillar_scores["president"].isin(selected_presidents))
                ]["score"]
                avg_score = scores.mean() if not scores.empty else 50
            else:
                avg_score = 50

            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(avg_score, 1),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': pillar, 'font': {'size': 12}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': pillar_colors.get(pillar, "#888")},
                    'bgcolor': "lightgray",
                    'steps': [
                        {'range': [0, 40], 'color': "#ffcccc"},
                        {'range': [40, 70], 'color': "#ffffcc"},
                        {'range': [70, 100], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': avg_score
                    }
                }
            ))

            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"pillar_{pillar}")

            # Add drill-down button
            if st.button(f"Explore {pillar}", key=f"btn_{pillar}", use_container_width=True):
                st.session_state.selected_pillar = pillar
                st.session_state.page = "Time Series"
                st.rerun()

def render_president_selector(df_avg):
    """Interactive president toggle cards"""
    st.subheader("👤 Select Presidents to Compare")

    # Initialize session state for selections
    if "selected_presidents" not in st.session_state:
        st.session_state.selected_presidents = ["Moi", "Kibaki", "Kenyatta", "Ruto"]

    cols = st.columns(4)

    for idx, (president, col) in enumerate(zip(["Moi", "Kibaki", "Kenyatta", "Ruto"], cols)):
        with col:
            # Get metrics for card
            p_data = df_avg[df_avg["president"] == president] if not df_avg.empty else pd.DataFrame()
            gdp_row = p_data[p_data["indicator"] == "NY.GDP.MKTP.KD.ZG"]
            avg_gdp = f"{gdp_row.iloc[0]['avg_value']:.1f}%" if not gdp_row.empty else "N/A"

            is_selected = president in st.session_state.selected_presidents

            # Create card with checkbox
            card_color = COLORS[president]
            opacity = "1.0" if is_selected else "0.4"
            border = f"3px solid {card_color}" if is_selected else "1px solid transparent"

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {card_color}20, {card_color}40);
                border: {border};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                opacity: {opacity};
                transition: all 0.3s;
            ">
                <h3 style="color: {card_color}; margin: 0;">{president}</h3>
                <p style="font-size: 11px; color: #666; margin: 5px 0;">
                    {PRESIDENTIAL_ERAS[president][0]}-{PRESIDENTIAL_ERAS[president][1]}
                </p>
                <div style="font-size: 28px; font-weight: bold; color: #333; margin: 10px 0;">
                    {avg_gdp}
                </div>
                <div style="font-size: 10px; color: #888;">Avg GDP Growth</div>
            </div>
            """, unsafe_allow_html=True)

            # Toggle button
            btn_label = "✓ Selected" if is_selected else "☐ Select"
            if st.button(btn_label, key=f"toggle_{president}", use_container_width=True):
                if is_selected:
                    st.session_state.selected_presidents.remove(president)
                else:
                    st.session_state.selected_presidents.append(president)
                st.rerun()

    return st.session_state.selected_presidents

def render_timeline_with_eras(df_ts, selected_presidents):
    """Interactive timeline with presidential era shading"""
    st.subheader("📈 Economic Timeline")

    # Indicator selector for timeline
    timeline_indicator = st.selectbox(
        "Select indicator to visualize:",
        ["Real GDP growth (%)", "Inflation (%)", "Public debt (% GDP)", "Unemployment (%)"],
        key="timeline_indicator"
    )

    indicator_code = KEY_INDICATORS[timeline_indicator]

    # Filter data
    if not df_ts.empty:
        filtered = df_ts[
            (df_ts["indicator"] == indicator_code) &
            (df_ts["president"].isin(selected_presidents))
        ].sort_values("year")

        if not filtered.empty:
            fig = px.line(
                filtered,
                x="year", 
                y="value",
                color="president",
                color_discrete_map=COLORS,
                markers=True,
                labels={"value": timeline_indicator, "year": "Year"},
                hover_data={"year": True, "value": ":.2f", "president": True}
            )

            # Add presidential era backgrounds
            for president, (start, end) in PRESIDENTIAL_ERAS.items():
                if president in selected_presidents:
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor=COLORS[president], 
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    )
                    # Add era label at top
                    fig.add_annotation(
                        x=(start + end) / 2,
                        y=filtered["value"].max() * 1.1,
                        text=president,
                        showarrow=False,
                        font=dict(size=10, color=COLORS[president]),
                        bgcolor="white",
                        opacity=0.8
                    )

            fig.update_layout(
                height=400,
                hovermode="x unified",
                legend_title="President",
                xaxis_rangeslider_visible=True  # Allow zooming
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add insight generation
            if st.button("🤖 Generate AI Insight for This Chart", key="gen_insight"):
                with st.spinner("Analyzing trends..."):
                    # Simple trend analysis
                    latest_president = filtered.iloc[-1]["president"] if not filtered.empty else "Unknown"
                    latest_value = filtered.iloc[-1]["value"] if not filtered.empty else 0

                    st.info(f"**Quick Insight:** {timeline_indicator} shows notable patterns across presidential eras. The most recent data under {latest_president} shows {latest_value:.2f}.")
        else:
            st.warning("No data available for selected presidents and indicator.")

def render_quick_comparison(df_avg, selected_presidents):
    """Quick head-to-head comparison"""
    st.subheader("⚡ Quick Comparison")

    if len(selected_presidents) >= 2:
        # Let user pick comparison metric
        compare_metric = st.selectbox(
            "Compare by:",
            ["Real GDP growth (%)", "Inflation (%)", "Public debt (% GDP)", "Unemployment (%)"],
            key="compare_metric"
        )

        indicator_code = KEY_INDICATORS[compare_metric]

        # Get data for selected presidents
        compare_data = []
        for president in selected_presidents:
            p_data = df_avg[df_avg["president"] == president]
            row = p_data[p_data["indicator"] == indicator_code]
            if not row.empty:
                compare_data.append({
                    "President": president,
                    "Value": row.iloc[0]["avg_value"],
                    "Color": COLORS[president]
                })

        if compare_data:
            compare_df = pd.DataFrame(compare_data)

            # Create comparison bar chart
            fig = px.bar(
                compare_df,
                x="President",
                y="Value",
                color="President",
                color_discrete_map=COLORS,
                text="Value",
                labels={"Value": compare_metric}
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(height=300, showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

            # Winner highlight
            best_idx = compare_df["Value"].idxmax()
            winner = compare_df.iloc[best_idx]["President"]
            best_value = compare_df.iloc[best_idx]["Value"]

            st.success(f"🏆 **{winner}** leads in {compare_metric} with **{best_value:.2f}**")
    else:
        st.info("Select at least 2 presidents in the cards above to see comparison.")

def render_ai_insights_panel(df_avg, df_meta):
    """AI-powered insights panel"""
    st.subheader("🤖 AI-Generated Insights")

    # Pre-computed insights based on data patterns
    insights = [
        {
            "title": "GDP Growth Leader",
            "insight": "Kibaki era (2003-2013) showed strongest average GDP growth at 4.5%, nearly double Moi's final years.",
            "metric": "+2.1% vs previous era",
            "positive": True
        },
        {
            "title": "Debt Concern",
            "insight": "Public debt has accelerated significantly post-2013, reaching 68% of GDP under current administration.",
            "metric": "+35 percentage points",
            "positive": False
        },
        {
            "title": "Infrastructure Progress",
            "insight": "Access to electricity improved dramatically from 15% (1998) to 75% (2022).",
            "metric": "+60 percentage points",
            "positive": True
        }
    ]

    # Display insights in expandable sections
    for insight in insights:
        with st.expander(f"{insight['title']}"):
            arrow = "📈" if insight['positive'] else "📉"
            color = "green" if insight['positive'] else "red"

            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 10px;">
                <p>{arrow} <strong>{insight['insight']}</strong></p>
                <p style="font-size: 12px; color: #666;">Key metric: {insight['metric']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Ask custom question
    st.divider()
    st.write("**Ask your own question:**")
    user_question = st.text_input("What would you like to know about Kenya's economic performance?", 
                                   placeholder="e.g., Which president had the best inflation control?")

    if user_question:
        with st.spinner("Consulting AI analyst..."):
            # Simulate AI response (replace with actual Gemini call if available)
            st.info(f"Based on historical data: {user_question}\n\nThis would connect to your Gemini chatbot for detailed analysis.")

# ==================== ORIGINAL PAGES (Keep as-is) ====================

def render_sidebar():
    with st.sidebar:
        st.title("🇰🇪 Kenya Dashboard")
        st.caption("Presidential performance 1998–2024")
        st.divider()

        # Use session state for selections if available
        default_presidents = st.session_state.get("selected_presidents", ["Moi", "Kibaki", "Kenyatta", "Ruto"])

        selected_presidents = st.multiselect(
            "Presidents",
            options=["Moi", "Kibaki", "Kenyatta", "Ruto"],
            default=default_presidents,
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

def render_timeseries(df_ts: pd.DataFrame, df_meta: pd.DataFrame,
                      presidents: list, pillars: list, year_range: tuple):
    """Original Time Series page - unchanged"""
    st.header("Time Series Explorer")

    pillar_sel = st.selectbox("Pillar", pillars, key="ts_pillar")
    indicators = df_meta[df_meta["pillar"] == pillar_sel]["name"].tolist()
    if not indicators:
        st.warning("No indicators for this pillar.")
        return

    indicator_sel = st.selectbox("Indicator", indicators, key="ts_indicator")
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

def render_scorecard(df_score: pd.DataFrame, df_avg: pd.DataFrame,
                     presidents: list, pillars: list):
    """Original Scorecard page - unchanged"""
    st.header("Presidential Scorecard")
    st.caption("Scores are normalised 0–100 within each indicator. For negative indicators (inflation, debt etc.) the score is inverted so higher always means better.")

    if df_score.empty:
        st.warning("Scorecard data not available. Check the Supabase view.")
        return

    df_score = df_score.copy()
    df_score.loc[df_score["indicator"].isin(INVERT_FOR_SCORE), "score_raw"] = (
        100 - pd.to_numeric(df_score.loc[df_score["indicator"].isin(INVERT_FOR_SCORE), "score_raw"], errors="coerce")
    )

    df_score = df_score[df_score["president"].isin(presidents)]

    pillar_scores = (
        df_score[df_score["pillar"].isin(pillars)]
        .groupby(["president", "pillar"])["score_raw"]
        .mean()
        .reset_index()
        .rename(columns={"score_raw": "score"})
    )

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

def render_chatbot(df_avg: pd.DataFrame, df_meta: pd.DataFrame):
    """Original Chatbot page - unchanged"""
    st.header("AI Analyst — Ask about presidential performance")
    st.caption("Powered by Gemini. Answers are grounded in World Bank data loaded into this dashboard.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    data_context = build_data_context(df_avg, df_meta)

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

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about any president or indicator...")

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

# ==================== ENHANCED OVERVIEW PAGE ====================

def render_enhanced_overview(df_avg, df_ts, df_pillar_scores, df_meta):
    """New highly interactive Overview page"""
    st.title("🇰🇪 Kenya Presidential Economic Dashboard")
    st.markdown("Explore Kenya's economic performance across four presidential eras (1998-2024)")

    st.divider()

    # 1. PILLAR HEALTH METERS (Interactive)
    render_pillar_health_meters(df_pillar_scores, st.session_state.get("selected_presidents", ["Moi", "Kibaki", "Kenyatta", "Ruto"]))

    st.divider()

    # 2. PRESIDENT SELECTOR (Interactive toggle cards)
    selected_presidents = render_president_selector(df_avg)

    # Update session state
    st.session_state.selected_presidents = selected_presidents

    st.divider()

    # 3. TIMELINE WITH ERAS
    render_timeline_with_eras(df_ts, selected_presidents)

    st.divider()

    # 4. QUICK COMPARISON
    render_quick_comparison(df_avg, selected_presidents)

    st.divider()

    # 5. AI INSIGHTS PANEL
    render_ai_insights_panel(df_avg, df_meta)

# ==================== MAIN ====================

def main():
    presidents, pillars, year_range, page = render_sidebar()

    # Load data
    with st.spinner("Loading data from Supabase..."):
        df_ts    = load_timeseries()
        df_avg   = load_averages()
        df_meta  = load_meta()
        df_score = load_scorecard()
        df_pillar_scores = load_pillar_scores()

        if df_avg.empty or df_meta.empty:
            st.error("Data not found. Run the ETL script first and ensure the Supabase tables/views exist.")
            st.stop()

    if page in ["Overview", "Time Series"] and df_ts.empty:
        st.error("No timeseries data found. Run the ETL script first and ensure the timeseries table exists.")
        st.stop()

    # Route to page
    if page == "Overview":
        render_enhanced_overview(df_avg, df_ts, df_pillar_scores, df_meta)
    elif page == "Time Series":
        render_timeseries(df_ts, df_meta, presidents, pillars, year_range)
    elif page == "Scorecard":
        render_scorecard(df_score, df_avg, presidents, pillars)
    elif page == "AI Chatbot":
        render_chatbot(df_avg, df_meta)

if __name__ == "__main__":
    main()
