"""Kenya Presidential Economic Dashboard"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import google.generativeai as genai


# ── Configuration ──────────────────────────────────────────────────────────────

def get_config() -> dict:
    """Load config from Streamlit secrets (Cloud) or local .env."""
    config = {}
    required_keys = ["GEMINI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]

    use_streamlit_secrets = False
    try:
        if all(k in st.secrets for k in required_keys):
            use_streamlit_secrets = True
    except Exception:
        pass

    if use_streamlit_secrets:
        for k in required_keys:
            config[k] = st.secrets[k]
        st.sidebar.success("✅ Using Streamlit Cloud secrets")
    else:
        load_dotenv()
        for k in required_keys:
            config[k] = os.environ.get(k)
        st.sidebar.info("ℹ️ Using local .env file")

    missing = [k for k, v in config.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required config values: {', '.join(missing)}")

    return config


# ── Constants ──────────────────────────────────────────────────────────────────

COLORS = {
    "Moi":      "#534AB7",
    "Kibaki":   "#1D9E75",
    "Kenyatta": "#D85A30",
    "Ruto":     "#378ADD",
}

# rgba() equivalents for each president — safe in both light and dark mode
COLORS_RGBA = {
    "Moi":      (83,  74,  183),
    "Kibaki":   (29,  158, 117),
    "Kenyatta": (216, 90,  48),
    "Ruto":     (55,  138, 221),
}

PILLARS = ["Growth", "Stability", "External Balance", "Inclusion", "Sustainability"]

PRESIDENTIAL_ERAS = {
    "Moi":      (1998, 2002),
    "Kibaki":   (2003, 2013),
    "Kenyatta": (2013, 2022),
    "Ruto":     (2022, 2024),
}

# FIX 2: indicators where lower value = better performance
INVERT_FOR_SCORE = {
    "FP.CPI.TOTL.ZG", "NY.GDP.DEFL.KD.ZG",
    "GC.DOD.TOTL.GD.ZS", "DT.DOD.DECT.GD.ZS",
    "SL.UEM.TOTL.ZS", "SI.POV.DDAY", "SI.POV.GAPS",
    "SI.POV.GINI", "NE.IMP.GNFS.ZS", "FR.INR.LNDP",
}

KEY_INDICATORS = {
    "Real GDP growth (%)":       "NY.GDP.MKTP.KD.ZG",
    "Inflation (%)":             "FP.CPI.TOTL.ZG",
    "Public debt (% GDP)":       "GC.DOD.TOTL.GD.ZS",
    "Unemployment (%)":          "SL.UEM.TOTL.ZS",
    "Life expectancy (yrs)":     "SP.DYN.LE00.IN",
    "Access to electricity (%)": "EG.ELC.ACCS.ZS",
}

# FIX 2: which display names are "lower is better"
LOWER_IS_BETTER_LABELS = {"Inflation (%)", "Public debt (% GDP)", "Unemployment (%)"}

PRESIDENT_ORDER = ["Moi", "Kibaki", "Kenyatta", "Ruto"]


# ── Supabase client ────────────────────────────────────────────────────────────

@st.cache_resource
def get_supabase() -> Client:
    cfg = get_config()
    return create_client(cfg["SUPABASE_URL"], cfg["SUPABASE_KEY"])


# ── Data loaders (cached) ──────────────────────────────────────────────────────

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
    """Compute pillar scores (0–100, higher = better)."""
    df_score = load_scorecard()
    if df_score.empty:
        return pd.DataFrame()

    df_score = df_score.copy()
    df_score["score_raw"] = pd.to_numeric(df_score["score_raw"], errors="coerce")
    mask = df_score["indicator"].isin(INVERT_FOR_SCORE)
    df_score.loc[mask, "score_raw"] = 100 - df_score.loc[mask, "score_raw"]

    return (
        df_score
        .groupby(["president", "pillar"])["score_raw"]
        .mean()
        .reset_index()
        .rename(columns={"score_raw": "score"})
    )


# ── FIX 6: load all data once, store in session state ─────────────────────────

def ensure_data_loaded():
    """Load all tables exactly once per session; show spinner only on first load."""
    if "app_data" not in st.session_state:
        with st.spinner("Loading data from Supabase…"):
            ts    = load_timeseries()
            avg   = load_averages()
            meta  = load_meta()
            score = load_scorecard()
            pillar = load_pillar_scores()

        if avg.empty or meta.empty:
            st.error("Data not found. Run the ETL script first and ensure Supabase tables exist.")
            st.stop()

        st.session_state["app_data"] = {
            "ts":     ts,
            "avg":    avg,
            "meta":   meta,
            "score":  score,
            "pillar": pillar,
        }


def get_data() -> dict:
    return st.session_state["app_data"]


# ── FIX 5: single source of truth for selected presidents ─────────────────────

def init_president_state():
    if "selected_presidents" not in st.session_state:
        st.session_state["selected_presidents"] = list(PRESIDENT_ORDER)


# ── Gemini chatbot ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_gemini():
    cfg = get_config()
    genai.configure(api_key=cfg["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-2.5-flash")


def build_data_context(df_avg: pd.DataFrame, df_meta: pd.DataFrame) -> str:
    if df_avg.empty or df_meta.empty:
        return "No data available."
    merged = df_avg.merge(df_meta[["indicator", "name", "unit"]], on="indicator", how="left")
    lines = []
    for pillar in PILLARS:
        lines.append(f"\n=== {pillar} ===")
        sub = merged[merged["pillar"] == pillar]
        for ind in sub["indicator"].unique():
            ind_rows = sub[sub["indicator"] == ind]
            ind_name = ind_rows.iloc[0]["name"] if not ind_rows.empty else ind
            ind_unit = ind_rows.iloc[0]["unit"] if not ind_rows.empty else ""
            for president in PRESIDENT_ORDER:
                p_row = ind_rows[ind_rows["president"] == president]
                if not p_row.empty:
                    val = p_row.iloc[0]["avg_value"]
                    lines.append(f"  {president} | {ind_name} ({ind_unit}): {val:.2f}")
    return "\n".join(lines)


def ask_gemini(question: str, data_context: str, history: list) -> str:
    model = get_gemini()
    system = f"""You are an expert analyst of Kenya's economic and social history.
Answer questions about presidential performance using ONLY the data provided.
Be specific, cite numbers, and remain balanced and factual.
Always mention which president did better or worse on specific metrics.
If asked something not in the data, say so clearly.

DATA (World Bank indicators, tenure averages):
{data_context}

Presidents and tenures:
- Moi: 1998–2002 (final years of his presidency)
- Kibaki: 2003–2013
- Kenyatta (Uhuru): 2013–2022
- Ruto: 2022–present (limited data)
"""

    def normalise_role(role: str) -> str:
        r = role.lower()
        if r == "user":     return "USER"
        if r in {"assistant", "model"}: return "MODEL"
        if r == "system":   return "SYSTEM"
        return r.upper()

    chat_history = [
        {"role": normalise_role(m.get("role", "user")), "parts": [m.get("content", "")]}
        for m in history
    ]
    chat = model.start_chat(history=chat_history)
    response = chat.send_message(f"{system}\n\nUser question: {question}")
    return response.text


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    """
    FIX 5: Sidebar multiselect writes directly to st.session_state["selected_presidents"].
    Returns (selected_presidents, selected_pillars, year_range, page).
    """
    with st.sidebar:
        st.title("🇰🇪 Kenya Dashboard")
        st.caption("Presidential performance 1998–2024")
        st.divider()

        # Keep sidebar selection in sync with session state
        current = st.session_state.get("selected_presidents", PRESIDENT_ORDER)

        new_selection = st.multiselect(
            "Presidents",
            options=PRESIDENT_ORDER,
            default=current,
            key="sidebar_presidents",
        )

        # Enforce minimum 1 selection
        if len(new_selection) == 0:
            st.warning("Select at least one president.")
            new_selection = current

        # Write back to shared session state
        st.session_state["selected_presidents"] = new_selection

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

    return new_selection, selected_pillars, year_range, page


# ── Overview: pillar health meters ────────────────────────────────────────────

def render_pillar_health_meters(df_pillar: pd.DataFrame, selected_presidents: list):
    st.subheader("🏛️ Pillar Scores by President")

    if df_pillar.empty:
        st.warning("No pillar scores available.")
        return

    df_sel = df_pillar[df_pillar["president"].isin(selected_presidents)]
    if df_sel.empty:
        st.warning("No pillar scores for the selected presidents.")
        return

    pivot = (
        df_sel
        .pivot(index="pillar", columns="president", values="score")
        .reindex(PILLARS)
    )
    st.dataframe(pivot.round(1).fillna("N/A"), use_container_width=True)

    fig = px.bar(
        df_sel,
        x="pillar",
        y="score",
        color="president",
        color_discrete_map=COLORS,
        barmode="group",
        category_orders={"pillar": PILLARS, "president": PRESIDENT_ORDER},
        labels={"score": "Score (0–100)", "pillar": "Pillar"},
    )
    fig.update_layout(height=360, legend_title="President",
                      xaxis_title="Pillar", yaxis_title="Score (0–100)")
    st.plotly_chart(fig, use_container_width=True)


# ── Overview: president selector cards ────────────────────────────────────────

def render_president_selector(df_avg: pd.DataFrame) -> list:
    """
    FIX 5 + FIX 7 + FIX 12:
    - Reads/writes a single shared session state key.
    - Buttons labeled "Remove" / "+ Compare".
    - Minimum 1 president enforced with user feedback.
    - Card colors use rgba() — safe in dark mode.
    - Text colors use CSS variables via st.markdown inline style.
    """
    st.subheader("👤 Select Presidents to Compare")
    cols = st.columns(4)

    current = st.session_state.get("selected_presidents", list(PRESIDENT_ORDER))

    for president, col in zip(PRESIDENT_ORDER, cols):
        with col:
            r, g, b = COLORS_RGBA[president]
            is_selected = president in current

            p_data  = df_avg[df_avg["president"] == president] if not df_avg.empty else pd.DataFrame()
            gdp_row = p_data[p_data["indicator"] == "NY.GDP.MKTP.KD.ZG"]
            avg_gdp = f"{gdp_row.iloc[0]['avg_value']:.1f}%" if not gdp_row.empty else "N/A"

            border_width = "2px" if is_selected else "0.5px"
            bg_opacity   = "0.15" if is_selected else "0.05"
            hex_color    = COLORS[president]

            # FIX 12: rgba background, no hardcoded text colors
            st.markdown(f"""
            <div style="
                background: rgba({r},{g},{b},{bg_opacity});
                border: {border_width} solid {hex_color};
                border-radius: 12px;
                padding: 18px;
                text-align: center;
                margin-bottom: 8px;
            ">
                <div style="font-size:16px; font-weight:500; color:{hex_color};">{president}</div>
                <div style="font-size:11px; color:var(--text-color, #888); margin:4px 0 8px;">
                    {PRESIDENTIAL_ERAS[president][0]}–{PRESIDENTIAL_ERAS[president][1]}
                </div>
                <div style="font-size:26px; font-weight:500; color:var(--text-color, inherit);">
                    {avg_gdp}
                </div>
                <div style="font-size:10px; color:var(--text-color, #888);">Avg GDP Growth</div>
            </div>
            """, unsafe_allow_html=True)

            # FIX 7: clear action labels
            if is_selected:
                btn_label = "Remove"
                btn_type  = "secondary"
            else:
                btn_label = "+ Compare"
                btn_type  = "primary"

            if st.button(btn_label, key=f"toggle_{president}",
                         use_container_width=True, type=btn_type):
                if is_selected:
                    # FIX 7: enforce minimum 1 selection
                    if len(current) <= 1:
                        st.toast("At least one president must stay selected.", icon="⚠️")
                    else:
                        new_list = [p for p in current if p != president]
                        st.session_state["selected_presidents"] = new_list
                        st.rerun()
                else:
                    st.session_state["selected_presidents"] = current + [president]
                    st.rerun()

    return st.session_state["selected_presidents"]


# ── Overview: timeline ────────────────────────────────────────────────────────

def render_timeline_with_eras(df_ts: pd.DataFrame, selected_presidents: list):
    """
    FIX 3: era annotation uses yref="paper" so negative data values don't break it.
    FIX 10: era shading is clipped to the actual data range for the chosen indicator.
    """
    st.subheader("📈 Economic Timeline")

    timeline_indicator = st.selectbox(
        "Select indicator to visualize:",
        list(KEY_INDICATORS.keys()),
        key="timeline_indicator",
    )
    indicator_code = KEY_INDICATORS[timeline_indicator]

    if df_ts.empty:
        st.info("No time-series data available.")
        return

    filtered = df_ts[
        (df_ts["indicator"] == indicator_code) &
        (df_ts["president"].isin(selected_presidents))
    ].sort_values("year")

    if filtered.empty:
        st.warning("No data for selected presidents and indicator.")
        return

    fig = px.line(
        filtered,
        x="year",
        y="value",
        color="president",
        color_discrete_map=COLORS,
        markers=True,
        labels={"value": timeline_indicator, "year": "Year"},
        hover_data={"year": True, "value": ":.2f", "president": True},
        category_orders={"president": PRESIDENT_ORDER},
    )

    data_min_year = int(filtered["year"].min())
    data_max_year = int(filtered["year"].max())

    for president, (era_start, era_end) in PRESIDENTIAL_ERAS.items():
        if president not in selected_presidents:
            continue

        # FIX 10: clip shading to actual data years
        clipped_start = max(era_start, data_min_year)
        clipped_end   = min(era_end, data_max_year)

        if clipped_start >= clipped_end:
            continue

        fig.add_vrect(
            x0=clipped_start, x1=clipped_end,
            fillcolor=COLORS[president],
            opacity=0.08,
            layer="below",
            line_width=0,
        )

        # FIX 3: use paper coordinates (0–1) so annotations never depend on data values
        fig.add_annotation(
            x=(clipped_start + clipped_end) / 2,
            y=1.05,
            yref="paper",
            text=president,
            showarrow=False,
            font=dict(size=10, color=COLORS[president]),
            bgcolor="rgba(255,255,255,0.75)",
        )

    fig.update_layout(
        height=420,
        hovermode="x unified",
        legend_title="President",
        xaxis_rangeslider_visible=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🤖 Generate AI Insight for This Chart", key="gen_insight"):
        with st.spinner("Analysing trends…"):
            data   = get_data()
            ctx    = build_data_context(data["avg"], data["meta"])
            prompt = (
                f"Summarise the trend in '{timeline_indicator}' across the selected "
                f"presidential eras ({', '.join(selected_presidents)}). "
                "Cite the key turning points and which president performed best and worst."
            )
            try:
                insight = ask_gemini(prompt, ctx, [])
                st.info(insight)
            except Exception as e:
                st.error(f"Could not generate insight: {e}")


# ── Overview: quick comparison ────────────────────────────────────────────────

def render_quick_comparison(df_avg: pd.DataFrame, selected_presidents: list):
    """FIX 2: use idxmin() for lower-is-better indicators and adjust trophy text."""
    st.subheader("⚡ Quick Comparison")

    if len(selected_presidents) < 2:
        st.info("Select at least 2 presidents above to see a head-to-head comparison.")
        return

    compare_metric = st.selectbox(
        "Compare by:",
        list(KEY_INDICATORS.keys()),
        key="compare_metric",
    )
    indicator_code = KEY_INDICATORS[compare_metric]

    compare_data = []
    for president in selected_presidents:
        row = df_avg[(df_avg["president"] == president) &
                     (df_avg["indicator"] == indicator_code)]
        if not row.empty:
            compare_data.append({
                "President": president,
                "Value":     round(row.iloc[0]["avg_value"], 2),
            })

    if not compare_data:
        st.warning("No comparison data available for the selected metric.")
        return

    compare_df = pd.DataFrame(compare_data)

    fig = px.bar(
        compare_df,
        x="President",
        y="Value",
        color="President",
        color_discrete_map=COLORS,
        text="Value",
        labels={"Value": compare_metric},
        category_orders={"President": PRESIDENT_ORDER},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # FIX 2: pick winner correctly based on indicator direction
    lower_is_better = compare_metric in LOWER_IS_BETTER_LABELS
    if lower_is_better:
        best_idx = compare_df["Value"].idxmin()
        direction_word = "lowest"
    else:
        best_idx = compare_df["Value"].idxmax()
        direction_word = "highest"

    winner = compare_df.iloc[best_idx]["President"]
    best_value = compare_df.iloc[best_idx]["Value"]
    st.success(
        f"🏆 **{winner}** leads with the {direction_word} {compare_metric}: **{best_value:.2f}**"
    )


# ── Overview: AI insights panel ───────────────────────────────────────────────

def render_ai_insights_panel(df_avg: pd.DataFrame, df_meta: pd.DataFrame):
    """
    FIX 4: custom question calls ask_gemini() for real.
    FIX 8: insight numbers computed from df_avg, not hardcoded strings.
    """
    st.subheader("🤖 AI-Generated Insights")

    # ── FIX 8: compute insight values dynamically ──────────────────────────────
    insights = []

    # Insight 1: GDP growth leader
    gdp_code = "NY.GDP.MKTP.KD.ZG"
    gdp_rows = df_avg[df_avg["indicator"] == gdp_code]
    if not gdp_rows.empty:
        best_gdp = gdp_rows.loc[gdp_rows["avg_value"].idxmax()]
        worst_gdp = gdp_rows.loc[gdp_rows["avg_value"].idxmin()]
        insights.append({
            "title":    "GDP Growth Leader",
            "insight":  (
                f"{best_gdp['president']} had the highest average GDP growth at "
                f"{best_gdp['avg_value']:.1f}%, compared to {worst_gdp['president']}'s "
                f"{worst_gdp['avg_value']:.1f}%."
            ),
            "metric":   f"+{best_gdp['avg_value'] - worst_gdp['avg_value']:.1f} pp vs lowest era",
            "positive": True,
        })

    # Insight 2: Public debt trajectory
    debt_code = "GC.DOD.TOTL.GD.ZS"
    debt_rows = df_avg[df_avg["indicator"] == debt_code]
    if not debt_rows.empty:
        max_debt = debt_rows.loc[debt_rows["avg_value"].idxmax()]
        min_debt = debt_rows.loc[debt_rows["avg_value"].idxmin()]
        swing    = max_debt["avg_value"] - min_debt["avg_value"]
        insights.append({
            "title":    "Public Debt Concern",
            "insight":  (
                f"Public debt peaked under {max_debt['president']} at "
                f"{max_debt['avg_value']:.0f}% of GDP, versus a low of "
                f"{min_debt['avg_value']:.0f}% under {min_debt['president']}."
            ),
            "metric":   f"+{swing:.0f} percentage points peak-to-trough",
            "positive": False,
        })

    # Insight 3: Electricity access
    elec_code = "EG.ELC.ACCS.ZS"
    elec_rows = df_avg[df_avg["indicator"] == elec_code].sort_values("avg_value")
    if len(elec_rows) >= 2:
        earliest = elec_rows.iloc[0]
        latest   = elec_rows.iloc[-1]
        gain     = latest["avg_value"] - earliest["avg_value"]
        insights.append({
            "title":    "Infrastructure Progress",
            "insight":  (
                f"Access to electricity improved from {earliest['avg_value']:.0f}% "
                f"(under {earliest['president']}) to {latest['avg_value']:.0f}% "
                f"(under {latest['president']})."
            ),
            "metric":   f"+{gain:.0f} percentage points",
            "positive": True,
        })

    for item in insights:
        with st.expander(item["title"]):
            color  = "green" if item["positive"] else "red"
            arrow  = "📈"    if item["positive"] else "📉"
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 12px;">
                <p>{arrow} {item['insight']}</p>
                <p style="font-size:12px; color:var(--text-color,#666);">Key metric: {item['metric']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # FIX 4: wire up the real Gemini call
    st.write("**Ask your own question about Kenya's presidential performance:**")
    user_question = st.text_input(
        "",
        placeholder="e.g. Which president had the best inflation control?",
        key="ai_panel_question",
    )

    if st.button("Ask", key="ai_panel_ask") and user_question.strip():
        with st.spinner("Consulting AI analyst…"):
            ctx = build_data_context(df_avg, df_meta)
            try:
                answer = ask_gemini(user_question.strip(), ctx, [])
                st.info(answer)
            except Exception as e:
                st.error(f"Gemini error: {e}. Check your GEMINI_API_KEY.")


# ── Overview (full page) ───────────────────────────────────────────────────────

def render_overview(df_avg, df_ts, df_pillar, df_meta):
    """
    FIX 1: selector runs first so meters and charts always reflect the current choice.
    """
    st.title("🇰🇪 Kenya Presidential Economic Dashboard")
    st.markdown("Explore Kenya's economic performance across four presidential eras (1998–2024).")
    st.divider()

    # FIX 1: selector FIRST — everything below reads from session state
    selected = render_president_selector(df_avg)

    st.divider()

    render_pillar_health_meters(df_pillar, selected)

    st.divider()

    render_timeline_with_eras(df_ts, selected)

    st.divider()

    render_quick_comparison(df_avg, selected)

    st.divider()

    render_ai_insights_panel(df_avg, df_meta)


# ── Time Series page ───────────────────────────────────────────────────────────

def render_timeseries(df_ts, df_meta, presidents, pillars, year_range):
    """Original Time Series page — era shading fix (FIX 10) also applied here."""
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
        st.info("No data for the selected filters.")
        return

    fig = px.line(
        filtered,
        x="year", y="value",
        color="president",
        color_discrete_map=COLORS,
        markers=True,
        labels={"value": f"{indicator_sel} ({unit})", "year": "Year"},
        category_orders={"president": PRESIDENT_ORDER},
    )

    data_min = int(filtered["year"].min())
    data_max = int(filtered["year"].max())

    for p, (s, e) in PRESIDENTIAL_ERAS.items():
        if p not in presidents:
            continue
        cs = max(s, data_min)
        ce = min(e, data_max)
        if cs >= ce:
            continue
        fig.add_vrect(x0=cs, x1=ce, fillcolor=COLORS[p],
                      opacity=0.07, layer="below", line_width=0)
        # FIX 3: paper-coordinate annotation
        fig.add_annotation(
            x=(cs + ce) / 2,
            y=1.04,
            yref="paper",
            text=p,
            showarrow=False,
            font=dict(size=10, color=COLORS[p]),
            bgcolor="rgba(255,255,255,0.75)",
        )

    fig.update_layout(height=480, legend_title="President")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(
            filtered[["year", "president", "value"]].rename(
                columns={"value": f"{indicator_sel} ({unit})"}
            ),
            use_container_width=True,
        )


# ── Scorecard page ────────────────────────────────────────────────────────────

def render_scorecard(df_score, df_avg, presidents, pillars):
    """
    FIX 9: bar chart uses dynamic x-axis range and a midpoint reference line.
    """
    st.header("Presidential Scorecard")
    st.caption(
        "Scores are normalised 0–100 within each indicator. "
        "For negative indicators (inflation, debt, etc.) the score is inverted "
        "so higher always means better."
    )

    if df_score.empty:
        st.warning("Scorecard data not available. Check the Supabase view.")
        return

    df_score = df_score.copy()
    df_score["score_raw"] = pd.to_numeric(df_score["score_raw"], errors="coerce")
    mask = df_score["indicator"].isin(INVERT_FOR_SCORE)
    df_score.loc[mask, "score_raw"] = 100 - df_score.loc[mask, "score_raw"]
    df_score = df_score[df_score["president"].isin(presidents)]

    pillar_scores = (
        df_score[df_score["pillar"].isin(pillars)]
        .groupby(["president", "pillar"])["score_raw"]
        .mean()
        .reset_index()
        .rename(columns={"score_raw": "score"})
    )

    # Radar chart
    st.subheader("Pillar radar")
    fig_radar = go.Figure()
    for president in presidents:
        p_data = pillar_scores[pillar_scores["president"] == president]
        scores = [
            float(p_data[p_data["pillar"] == pl]["score"].values[0])
            if pl in p_data["pillar"].values else 0.0
            for pl in pillars
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=pillars + [pillars[0]],
            name=president,
            line_color=COLORS[president],
            fill="toself",
            opacity=0.4,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # FIX 9: overall ranking bar chart
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

    min_score = overall["overall_score"].min()
    x_min     = max(0, min_score - 10)  # don't go below 0

    fig_bar = px.bar(
        overall,
        x="overall_score", y="president",
        orientation="h",
        color="president",
        color_discrete_map=COLORS,
        text="overall_score",
        labels={"overall_score": "Score (0–100)", "president": ""},
        category_orders={"president": PRESIDENT_ORDER},
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_bar.update_layout(
        showlegend=False,
        height=300,
        xaxis_range=[x_min, 105],
    )
    # FIX 9: midpoint reference line
    fig_bar.add_vline(
        x=50,
        line_dash="dot",
        line_color="gray",
        opacity=0.4,
        annotation_text="50",
        annotation_position="top",
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── AI Chatbot page ────────────────────────────────────────────────────────────

def render_chatbot(df_avg, df_meta):
    """
    FIX 11: starter question buttons handle the question directly in their own
    block — no pending_question session-state hack and no double-render.
    """
    st.header("AI Analyst — Ask about presidential performance")
    st.caption("Powered by Gemini. Answers are grounded in World Bank data.")

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

    # FIX 11: handle question directly inside each button block
    for col, question in zip(cols, starters):
        if col.button(question, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Analysing data…"):
                try:
                    answer = ask_gemini(
                        question,
                        data_context,
                        st.session_state.chat_history[:-1],
                    )
                except Exception as e:
                    answer = f"Error calling Gemini API: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about any president or indicator…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analysing data…"):
                try:
                    answer = ask_gemini(
                        user_input,
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Kenya Presidential Dashboard",
        page_icon="🇰🇪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # FIX 5: initialise shared president state before any render
    init_president_state()

    # FIX 6: load all data once; subsequent calls read from session state
    ensure_data_loaded()
    data = get_data()

    # FIX 6: timeseries check done once, cleanly
    if data["ts"].empty:
        st.error("No timeseries data found. Run the ETL script first.")
        st.stop()

    presidents, pillars, year_range, page = render_sidebar()

    if page == "Overview":
        render_overview(data["avg"], data["ts"], data["pillar"], data["meta"])

    elif page == "Time Series":
        render_timeseries(data["ts"], data["meta"], presidents, pillars, year_range)

    elif page == "Scorecard":
        render_scorecard(data["score"], data["avg"], presidents, pillars)

    elif page == "AI Chatbot":
        render_chatbot(data["avg"], data["meta"])


if __name__ == "__main__":
    main()