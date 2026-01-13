import streamlit as st
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
def pace_to_sec(p):
    """Convert 'mm:ss' (or 'hh:mm:ss') pace string to seconds.
    Returns NaN if not parseable.
    """
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return float("nan")
    if isinstance(p, (int, float)):
        return float(p)

    s = str(p).strip()
    if ":" not in s:
        return float("nan")

    parts = s.split(":")
    try:
        parts = [int(x) for x in parts]
    except ValueError:
        return float("nan")

    if len(parts) == 2:  # mm:ss
        m, sec = parts
        return m * 60 + sec
    if len(parts) == 3:  # hh:mm:ss
        h, m, sec = parts
        return h * 3600 + m * 60 + sec

    return float("nan")


def sec_to_pace(sec):
    """Convert seconds to 'm:ss' string."""
    if sec is None or (isinstance(sec, float) and pd.isna(sec)):
        return "—"
    try:
        sec = int(round(float(sec)))
    except Exception:
        return "—"
    if sec < 0:
        # show negative pace deltas as -m:ss
        sec_abs = abs(sec)
        m = sec_abs // 60
        s = sec_abs % 60
        return f"-{m}:{s:02d}"
    m = sec // 60
    s = sec % 60
    return f"{m}:{s:02d}"


def pick_col(df, candidates):
    """Return the first column name that exists in df from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="RunLab Prototype", layout="wide")
st.title("🏃 RunLab — Race Analysis Prototype")
st.write("Upload a Garmin/Strava CSV and see lap-based insights.")

uploaded_file = st.file_uploader("Upload run CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Read CSV
df = pd.read_csv(uploaded_file)

st.subheader("Raw Data Preview")
st.dataframe(df.head(25), use_container_width=True)

# Identify likely columns (handles minor name differences)
col_laps = pick_col(df, ["Laps", "Lap", "lap", "laps"])
col_dist = pick_col(df, ["Distancemi", "Distance", "Distance (mi)", "DistanceMi", "Distance mi"])
col_pace = pick_col(df, ["Avg Pacemin/mi", "Avg Pace", "Pace", "Avg Pace (min/mi)"])
col_gap  = pick_col(df, ["Avg GAPmin/mi", "Avg GAP", "GAP", "Avg GAP (min/mi)"])
col_hr   = pick_col(df, ["Avg HRbpm", "Avg HR", "HR", "AvgHR", "Heart Rate"])
col_ascent = pick_col(df, ["Total Ascentft", "Total Ascent", "Ascent", "Elevation Gain", "Total Ascent (ft)"])

missing = []
for label, c in [("Laps", col_laps), ("Distance", col_dist), ("Avg Pace", col_pace)]:
    if c is None:
        missing.append(label)

if missing:
    st.error(
        "Your CSV is missing required columns for this prototype: "
        + ", ".join(missing)
        + ".\n\nColumns found:\n"
        + ", ".join(list(df.columns))
    )
    st.stop()

# Build analysis dataframe (start from required cols, then optional)
keep_cols = [col_laps, col_dist, col_pace]
if col_gap: keep_cols.append(col_gap)
if col_hr: keep_cols.append(col_hr)
if col_ascent: keep_cols.append(col_ascent)

analysis = df[keep_cols].copy()

# Normalize column names to a consistent internal schema
rename_map = {
    col_laps: "Laps",
    col_dist: "Distancemi",
    col_pace: "Avg Pacemin/mi",
}
if col_gap:
    rename_map[col_gap] = "Avg GAPmin/mi"
if col_hr:
    rename_map[col_hr] = "Avg HRbpm"
if col_ascent:
    rename_map[col_ascent] = "Total Ascentft"

analysis = analysis.rename(columns=rename_map)

# Remove "Summary" row if present
if "Laps" in analysis.columns:
    analysis = analysis[analysis["Laps"].astype(str).str.lower() != "summary"].copy()

# Ensure Distance numeric
analysis["Distancemi"] = pd.to_numeric(analysis["Distancemi"], errors="coerce")

# Pace seconds (math columns)
analysis["Avg Pace sec"] = analysis["Avg Pacemin/mi"].apply(pace_to_sec)

if "Avg GAPmin/mi" in analysis.columns:
    analysis["Avg GAP sec"] = analysis["Avg GAPmin/mi"].apply(pace_to_sec)
else:
    analysis["Avg GAP sec"] = float("nan")

# Display pace columns (human columns)
analysis["Avg Pace (min/mi)"] = analysis["Avg Pace sec"].apply(sec_to_pace)
analysis["Avg GAP (min/mi)"] = analysis["Avg GAP sec"].apply(sec_to_pace)

# Pace delta by lap (sec)
analysis["Pace Δ (sec)"] = analysis["Avg Pace sec"].diff()

# Terrain penalty (pace - GAP): positive means terrain slowed you vs GAP
analysis["Terrain Penalty (sec)"] = analysis["Avg Pace sec"] - analysis["Avg GAP sec"]

st.subheader("Normalized Lap Analysis")
show_cols = ["Laps", "Distancemi", "Avg Pace (min/mi)"]
if "Avg GAPmin/mi" in analysis.columns:
    show_cols += ["Avg GAP (min/mi)"]
if "Avg HRbpm" in analysis.columns:
    show_cols += ["Avg HRbpm"]
if "Total Ascentft" in analysis.columns:
    show_cols += ["Total Ascentft"]
show_cols += ["Pace Δ (sec)"]
if "Avg GAPmin/mi" in analysis.columns:
    show_cols += ["Terrain Penalty (sec)"]

st.dataframe(analysis[show_cols], use_container_width=True)

# ----------------------------
# Charts
# ----------------------------
st.subheader("📉 Pace Drift Over Laps")

# Use full-mile laps only for the chart by default
full_laps = analysis[(analysis["Distancemi"] >= 0.95) & (analysis["Distancemi"] <= 1.05)].copy()
if len(full_laps) >= 2:
    chart_df = full_laps[["Laps", "Avg Pace sec"]].copy()
    chart_df["Laps"] = chart_df["Laps"].astype(str)
    st.line_chart(chart_df.set_index("Laps"))
else:
    st.info("Not enough full-mile laps to chart pace drift.")

# ----------------------------
# Key Insights (Fatigue / pace fade)
# ----------------------------
st.subheader("Key Insights (Fatigue / Pace Fade)")

if len(full_laps) >= 2:
    best_idx = full_laps["Avg Pace sec"].idxmin()
    worst_idx = full_laps["Avg Pace sec"].idxmax()

    best_pace = full_laps.loc[best_idx, "Avg Pace sec"]
    worst_pace = full_laps.loc[worst_idx, "Avg Pace sec"]
    pace_fade_sec = worst_pace - best_pace

    avg_pace_sec = full_laps["Avg Pace sec"].mean()

    best_lap_label = str(full_laps.loc[best_idx, "Laps"])
    worst_lap_label = str(full_laps.loc[worst_idx, "Laps"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pace Fade (min/mi)", f"+{sec_to_pace(pace_fade_sec)}")
    c2.metric("Avg Pace (full laps)", f"{sec_to_pace(avg_pace_sec)}")
    c3.metric("Best Lap (pace)", f"Lap {best_lap_label}")
    c4.metric("Worst Lap (pace)", f"Lap {worst_lap_label}")

    st.caption("Full-lap splits used for fatigue metrics:")
    st.dataframe(
        full_laps[["Laps", "Distancemi", "Avg Pacemin/mi", "Avg Pace sec"]]
        .assign(**{"Avg Pace (min/mi)": full_laps["Avg Pace sec"].apply(sec_to_pace)})[
            ["Laps", "Distancemi", "Avg Pace (min/mi)", "Avg Pace sec"]
        ],
        use_container_width=True
    )
else:
    st.info("Need at least 2 full-mile laps to compute pace fade.")

# Other quick metrics
st.subheader("Key Insights")

k1, k2 = st.columns(2)

if "Avg HRbpm" in analysis.columns:
    avg_hr = pd.to_numeric(analysis["Avg HRbpm"], errors="coerce").mean()
    k1.metric("Average HR", f"{int(round(avg_hr))}" if not pd.isna(avg_hr) else "—")
else:
    k1.metric("Average HR", "—")

if "Total Ascentft" in analysis.columns and len(full_laps) > 0:
    ascent_vals = pd.to_numeric(full_laps["Total Ascentft"], errors="coerce")
    avg_ascent = ascent_vals.mean()
    k2.metric("Avg Ascent per Lap (ft)", f"{int(round(avg_ascent))}" if not pd.isna(avg_ascent) else "—")
else:
    k2.metric("Avg Ascent per Lap (ft)", "—")

st.markdown("### 🧠 What this means")
if len(full_laps) >= 2:
    if pace_fade_sec >= 60:
        st.warning("Significant pace fade detected. Early pacing may have been too aggressive or fatigue built late.")
    elif pace_fade_sec >= 25:
        st.info("Moderate pace fade. Some fatigue or terrain effects likely influenced later laps.")
    else:
        st.success("Pacing looks fairly consistent across full-mile laps.")
else:
    st.info("Upload a run with at least 2 full-mile laps for pace-fade interpretation.")
