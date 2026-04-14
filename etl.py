import os
import time
import pandas as pd
import wbgapi as wb
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
# Config
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
COUNTRY      = "KEN"
START_YEAR   = 1998
END_YEAR     = 2026


def validate_supabase_config(url: str, key: str) -> None:
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your environment or .env file."
        )
    if "supabase.com/dashboard" in url:
        raise RuntimeError(
            "SUPABASE_URL is set to the Supabase dashboard URL. "
            "Use your project API URL instead, for example: https://<project>.supabase.co"
        )
    if not url.startswith("https://") or ".supabase.co" not in url:
        raise RuntimeError(
            "SUPABASE_URL must be the Supabase API endpoint, such as https://<project>.supabase.co. "
            "Do not use the dashboard link."
        )

# President mapping
def get_president(year: int) -> str:
    if year <= 2002:
        return "Moi"
    elif year <= 2013:
        return "Kibaki"
    elif year <= 2022:
        return "Kenyatta"
    else:
        return "Ruto"

# All indicators 
INDICATORS = {
    # GROWTH
    "NY.GDP.MKTP.CD":    {"name": "GDP (current US$)",                          "pillar": "Growth",           "unit": "current US$",      "description": "Size of the economy in current dollars"},
    "NY.GDP.MKTP.KD.ZG": {"name": "Real GDP growth (%)",                        "pillar": "Growth",           "unit": "%",                 "description": "Actual expansion adjusted for inflation"},
    "NY.GDP.PCAP.CD":    {"name": "GDP per capita (current US$)",               "pillar": "Growth",           "unit": "current US$",      "description": "Living standards proxy"},
    "NY.GDP.PCAP.KD":    {"name": "GDP per capita (constant 2015 US$)",         "pillar": "Growth",           "unit": "constant 2015 US$","description": "Living standards proxy (inflation-adjusted)"},
    "NY.GNP.MKTP.CD":    {"name": "GNI (current US$)",                          "pillar": "Growth",           "unit": "current US$",      "description": "Income earned by residents"},
    "NY.GNP.PCAP.CD":    {"name": "GNI per capita (current US$)",               "pillar": "Growth",           "unit": "current US$",      "description": "Income per resident"},
    "NE.GDI.TOTL.ZS":    {"name": "Gross capital formation (% of GDP)",         "pillar": "Growth",           "unit": "% of GDP",         "description": "Investment as share of economy"},

    # STABILITY
    "FP.CPI.TOTL.ZG":    {"name": "Inflation, consumer prices (annual %)",      "pillar": "Stability",        "unit": "%",                 "description": "Consumer price inflation"},
    "NY.GDP.DEFL.KD.ZG": {"name": "Inflation, GDP deflator (annual %)",         "pillar": "Stability",        "unit": "%",                 "description": "Overall price level changes"},
    "FR.INR.RINR":        {"name": "Real interest rate (%)",                     "pillar": "Stability",        "unit": "%",                 "description": "Inflation-adjusted interest rate"},
    "FM.LBL.BMNY.GD.ZS": {"name": "Broad money (% of GDP)",                    "pillar": "Stability",        "unit": "% of GDP",         "description": "Money supply M2 as share of GDP"},
    "GC.NLD.TOTL.GD.ZS": {"name": "Net lending/net borrowing (% of GDP)",      "pillar": "Stability",        "unit": "% of GDP",         "description": "Government balance (deficit/surplus)"},
    "GC.TAX.TOTL.GD.ZS": {"name": "Tax revenue (% of GDP)",                    "pillar": "Stability",        "unit": "% of GDP",         "description": "Tax burden"},
    "NE.CON.GOVT.ZS":    {"name": "Government final consumption (% of GDP)",   "pillar": "Stability",        "unit": "% of GDP",         "description": "Government spending"},
    "FR.INR.LNDP":        {"name": "Interest rate spread (%)",                  "pillar": "Stability",        "unit": "%",                 "description": "Banking sector efficiency indicator"},

    # EXTERNAL BALANCE
    "BN.CAB.XOKA.GD.ZS": {"name": "Current account balance (% of GDP)",        "pillar": "External Balance", "unit": "% of GDP",         "description": "Trade and income balance"},
    "NE.EXP.GNFS.ZS":    {"name": "Exports of goods and services (% of GDP)",  "pillar": "External Balance", "unit": "% of GDP",         "description": "Export orientation"},
    "NE.IMP.GNFS.ZS":    {"name": "Imports of goods and services (% of GDP)",  "pillar": "External Balance", "unit": "% of GDP",         "description": "Import dependence"},
    "PA.NUS.FCRF":        {"name": "Official exchange rate (LCU per US$)",      "pillar": "External Balance", "unit": "LCU per US$",      "description": "Currency value"},
    "BX.KLT.DINV.CD.WD": {"name": "FDI net inflows (current US$)",             "pillar": "External Balance", "unit": "current US$",      "description": "FDI inflows"},
    "BX.KLT.DINV.WD.GD.ZS": {"name": "FDI net inflows (% of GDP)",            "pillar": "External Balance", "unit": "% of GDP",         "description": "FDI as share of economy"},
    "NE.TRD.GNFS.ZS":    {"name": "Trade (% of GDP)",                          "pillar": "External Balance", "unit": "% of GDP",         "description": "Trade openness"},

    # INCLUSION
    "SL.UEM.TOTL.ZS":    {"name": "Unemployment, total (%)",                   "pillar": "Inclusion",        "unit": "%",                 "description": "Total unemployment rate"},
    "SL.TLF.CACT.ZS":    {"name": "Labor force participation rate (%)",        "pillar": "Inclusion",        "unit": "%",                 "description": "Labor force participation"},
    "SL.EMP.TOTL.SP.ZS": {"name": "Employment to population ratio (%)",        "pillar": "Inclusion",        "unit": "%",                 "description": "Employment rate"},
    "SI.POV.DDAY":        {"name": "Poverty headcount at $2.15/day (%)",        "pillar": "Inclusion",        "unit": "%",                 "description": "Extreme poverty rate"},
    "SI.POV.GAPS":        {"name": "Poverty gap at $3.65/day (%)",              "pillar": "Inclusion",        "unit": "%",                 "description": "Poverty depth"},
    "SI.POV.GINI":        {"name": "Gini index",                                "pillar": "Inclusion",        "unit": "index 0-100",      "description": "Income inequality"},
    "NE.CON.PRVT.ZS":    {"name": "Household consumption (% of GDP)",          "pillar": "Inclusion",        "unit": "% of GDP",         "description": "Household consumption share"},
    "HD.HCI.OVRL":        {"name": "Human Capital Index (0-1)",                 "pillar": "Inclusion",        "unit": "0-1 scale",        "description": "Human capital development"},

    # SUSTAINABILITY
    "GC.DOD.TOTL.GD.ZS": {"name": "Central government debt (% of GDP)",        "pillar": "Sustainability",   "unit": "% of GDP",         "description": "Public debt burden"},
    "DT.DOD.DECT.GD.ZS": {"name": "External debt stocks (% of GNI)",           "pillar": "Sustainability",   "unit": "% of GNI",         "description": "External debt burden"},
    "FS.AST.PRVT.GD.ZS": {"name": "Domestic credit to private sector (% GDP)", "pillar": "Sustainability",   "unit": "% of GDP",         "description": "Private sector credit access"},
    "EG.ELC.ACCS.ZS":    {"name": "Access to electricity (% of population)",   "pillar": "Sustainability",   "unit": "%",                 "description": "Infrastructure access"},
    "NV.AGR.TOTL.ZS":    {"name": "Agriculture, value added (% of GDP)",       "pillar": "Sustainability",   "unit": "% of GDP",         "description": "Agricultural share"},
    "NV.IND.TOTL.ZS":    {"name": "Industry, value added (% of GDP)",          "pillar": "Sustainability",   "unit": "% of GDP",         "description": "Industrial share"},
    "NV.SRV.TOTL.ZS":    {"name": "Services, value added (% of GDP)",          "pillar": "Sustainability",   "unit": "% of GDP",         "description": "Services share"},
    "SP.URB.TOTL.IN.ZS": {"name": "Urban population (% of total)",             "pillar": "Sustainability",   "unit": "%",                 "description": "Urbanization rate"},
}

# Lower-is-better indicators (inverted when scoring) 
INVERT_FOR_SCORE = {
    "FP.CPI.TOTL.ZG", "NY.GDP.DEFL.KD.ZG",
    "GC.DOD.TOTL.GD.ZS", "DT.DOD.DECT.GD.ZS",
    "SL.UEM.TOTL.ZS", "SI.POV.DDAY", "SI.POV.GAPS",
    "SI.POV.GINI", "NE.IMP.GNFS.ZS", "FR.INR.LNDP",
    "BN.CAB.XOKA.GD.ZS",  # current account deficit is bad
}


def fetch_world_bank_data() -> pd.DataFrame:
    """Pull all indicators for Kenya from World Bank API."""
    codes = list(INDICATORS.keys())
    print(f"Fetching {len(codes)} indicators for Kenya ({START_YEAR}–{END_YEAR})...")

    all_rows = []
    failed  = []

    for i, code in enumerate(codes, 1):
        meta = INDICATORS[code]
        try:
            df = wb.data.DataFrame(
                code,
                economy=COUNTRY,
                time=range(START_YEAR, END_YEAR + 1),
                skipBlanks=True,
                numericTimeKeys=True,
            )
            if df.empty:
                print(f"  [{i}/{len(codes)}] {code} — no data")
                continue

            # wbgapi returns MultiIndex (economy, time) or just time
            df = df.reset_index()
            if "economy" in df.columns:
                df = df[df["economy"] == COUNTRY]

            # Melt wide → long if needed
            year_cols = [c for c in df.columns if str(c).isdigit()]
            if year_cols:
                df = df.melt(id_vars=[c for c in df.columns if not str(c).isdigit()],
                             value_vars=year_cols,
                             var_name="year", value_name="value")
                df["year"] = df["year"].astype(int)
            elif "time" in df.columns:
                df = df.rename(columns={"time": "year", df.columns[-1]: "value"})

            df = df[["year", "value"]].dropna()
            df["indicator"]  = code
            df["pillar"]     = meta["pillar"]
            df["president"]  = df["year"].apply(get_president)
            df["source"]     = "World Bank"
            all_rows.append(df)
            print(f"  [{i}/{len(codes)}] {code} — {len(df)} rows")

        except Exception as e:
            print(f"  [{i}/{len(codes)}] {code} — FAILED: {e}")
            failed.append(code)
        time.sleep(0.3)  # be polite to the API

    if failed:
        print(f"\nFailed indicators ({len(failed)}): {failed}")

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def upsert_metadata(supabase: Client):
    """Load indicator metadata into indicator_meta table."""
    print("\nUpserting indicator metadata...")
    rows = [
        {
            "indicator":   code,
            "name":        meta["name"],
            "pillar":      meta["pillar"],
            "unit":        meta["unit"],
            "description": meta["description"],
        }
        for code, meta in INDICATORS.items()
    ]
    result = supabase.table("indicator_meta").upsert(rows, on_conflict="indicator").execute()
    print(f"  Metadata rows upserted: {len(rows)}")


def upsert_indicators(supabase: Client, df: pd.DataFrame):
    """Batch upsert indicator rows into Supabase."""
    print(f"\nUpserting {len(df)} data rows into Supabase...")
    records = df.to_dict("records")

    BATCH = 500
    for i in range(0, len(records), BATCH):
        batch = records[i:i + BATCH]
        # Convert numpy types to native Python
        for row in batch:
            row["year"]  = int(row["year"])
            row["value"] = float(row["value"]) if row["value"] is not None else None
        supabase.table("indicators").upsert(
            batch, on_conflict="year,indicator"
        ).execute()
        print(f"  Upserted rows {i+1}–{min(i+BATCH, len(records))}")

    print("Done.")


def main():
    print("=" * 55)
    print("Kenya Presidential Dashboard — ETL")
    print("=" * 55)

    validate_supabase_config(SUPABASE_URL, SUPABASE_KEY)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Load metadata
    upsert_metadata(supabase)

    # 2. Fetch from World Bank
    df = fetch_world_bank_data()
    if df.empty:
        print("No data fetched. Check your internet connection.")
        return

    print(f"\nTotal rows fetched: {len(df)}")
    print(df.groupby("pillar")["indicator"].nunique().to_string())

    # 3. Push to Supabase
    upsert_indicators(supabase, df)

    print("\nAll done! Check Supabase table editor to verify data.")
    print("Run: SELECT president, COUNT(*) FROM indicators GROUP BY president;")


if __name__ == "__main__":
    main()


