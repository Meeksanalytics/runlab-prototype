import io
import math
from datetime import datetime

import pandas as pd
import streamlit as st

# Optional libs for GPX/TCX
try:
    import gpxpy
except Exception:
    gpxpy = None

try:
    from tcxreader.tcxreader import TCXReader
except Exception:
    TCXReader = None


# -----------------------------
# Helpers
# -----------------------------
def pace_str_from_seconds(sec_per_mile: float) -> str:
    """Convert seconds per mile to M:SS string."""
    if sec_per_mile is None or (isinstance(sec_per_mile, float) and math.isnan(sec_per_mile)):
        return ""
    sec_per_mile = float(sec_per_mile)
    if sec_per_mile <= 0:
        return ""
    m = int(sec_per_mile // 60)
    s = int(round(sec_per_mile - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Distance in miles between two lat/lon points."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 0.0
    R = 3958.7613  # Earth radius (miles)
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def build_splits_from_points(points_df: pd.DataFrame, split_miles: float = 1.0) -> pd.DataFrame:
    """
    Create mile splits from point stream with columns:
    time (datetime), lat, lon, ele_ft (optional), hr (optional)
    """
    if points_df.empty:
        return pd.DataFrame()

    points_df = points_df.sort_values("time").reset_index(drop=True)

    # Compute cumulative distance
    cum = [0.0]
    for i in range(1, len(points_df)):
        d = haversine_miles(
            points_df.loc[i - 1, "lat"],
            points_df.loc[i - 1, "lon"],
            points_df.loc[i, "lat"],
            points_df.loc[i, "lon"],
        )
        cum.append(cum[-1] + d)
    points_df["dist_mi"] = cum

    total_dist = points_df["dist_mi"].iloc[-1]
    if total_dist < 0.05:
        return pd.DataFrame()

    # Split boundaries at 1.0, 2.0, ...
    split_targets = []
    k = 1
    while k * split_miles <= total_dist + 1e-9:
        split_targets.append(k * split_miles)
        k += 1

    # Walk through points and cut splits
    rows = []
    start_time = points_df["time"].iloc[0]
    prev_target_time = start_time
    prev_target_idx = 0
    prev_target_dist = 0.0

    for split_idx, target in enumerate(split_targets, start=1):
        # Find first point where dist >= target
        idx = points_df.index[points_df["dist_mi"] >= target]
        if len(idx) == 0:
            break
        i = int(idx[0])

        # Linear interpolate time at exactly 'target'
        d0 = points_df.loc[i - 1, "dist_mi"] if i > 0 else 0.0
        d1 = points_df.loc[i, "dist_mi"]
        t0 = points_df.loc[i - 1, "time"] if i > 0 else points_df.loc[i, "time"]
        t1 = points_df.loc[i, "time"]

        if d1 <= d0:
            frac = 0.0
        else:
            frac = (target - d0) / (d1 - d0)

        # Interpolated time
        interp_time = t0 + (t1 - t0) * frac

        split_seconds = (interp_time - prev_target_time).total_seconds()
        sec_per_mile = split_seconds / split_miles if split_miles > 0 else None

        # HR average inside the split (if available)
        hr_avg = None
        if "hr" in points_df.columns and points_df["hr"].notna().any():
            seg = points_df.iloc[prev_target_idx : i + 1]
            if not seg.empty:
                hr_avg = float(pd.to_numeric(seg["hr"], errors="coerce").dropna().mean()) if seg["hr"].notna().any() else None

        # Elevation gain inside split (if available)
        ascent_ft = None
        if "ele_ft" in points_df.columns and points_df["ele_ft"].notna().any():
            seg = points_df.iloc[prev_target_idx : i + 1].copy()
            seg["ele_ft"] = pd.to_numeric(seg["ele_ft"], errors="coerce")
            diffs = seg["ele_ft"].diff()
            ascent_ft = float(diffs[diffs > 0].sum()) if diffs.notna().any() else None

        rows.append(
            {
                "Laps": split_idx,
                "Distancemi": float(split_miles),
                "Avg Pace sec": float(sec_per_mile) if sec_per_mile is not None else None,
                "Avg Pacemin/mi": pace_str_from_seconds(sec_per_mile),
                "Avg HRbpm": hr_avg,
                "Total Ascentft": ascent_ft,
            }
        )

        prev_target_time = interp_time
        prev_target_idx = i
        prev_target_dist = target

    # Add remainder split (last partial)
    last_dist = total_dist - prev_target_dist
    if last_dist >= 0.10:  # only if meaningful
        end_time = points_df["time"].iloc[-1]
        split_seconds = (end_time - prev_target_time).total_seconds()
        sec_per_mile = split_seconds / last_dist if last_dist > 0 else None

        hr_avg = None
        if "hr" in points_df.columns and points_df["hr"].notna().any():
            seg = points_df.iloc[prev_target_idx:].copy()
            seg["hr"] = pd.to_numeric(seg["hr"], errors="coerce")
            hr_avg = float(seg["hr"].dropna().mean()) if seg["hr"].notna().any() else None

        ascent_ft = None
        if "ele_ft" in points_df.columns and points_df["ele_ft"].notna().any():
            seg = points_df.iloc[prev_target_idx:].copy()
            seg["ele_ft"] = pd.to_numeric(seg["ele_ft"], errors="coerce")
            diffs = seg["ele_ft"].diff()
            ascent_ft = float(diffs[diffs > 0].sum()) if diffs.notna().any() else None

        rows.append(
            {
                "Laps": len(rows) + 1,
                "Distancemi": float(last_dist),
                "Avg Pace sec": float(sec_per_mile) if sec_per_mile is not None else None,
                "Avg Pacemin/mi": pace_str_from_seconds(sec_per_mile),
                "Avg HRbpm": hr_avg,
                "Total Ascentft": ascent_ft,
            }
        )

    return pd.DataFrame(rows)


def parse_gpx(uploaded_bytes: bytes) -> pd.DataFrame:
    if gpxpy is None:
        raise RuntimeError("Missing dependency: gpxpy. Run: python -m pip install gpxpy")

    gpx = gpxpy.parse(io.StringIO(uploaded_bytes.decode("utf-8", errors="ignore")))

    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                if not p.time or p.latitude is None or p.longitude is None:
                    continue
                rows.append(
                    {
                        "time": p.time.replace(tzinfo=None),
                        "lat": float(p.latitude),
                        "lon": float(p.longitude),
                        "ele_ft": float(p.elevation) * 3.28084 if p.elevation is not None else None,
                        "hr": None,  # GPX HR varies by device/extensions; leaving blank for now
                    }
                )
    return pd.DataFrame(rows)


def parse_tcx(uploaded_bytes: bytes) -> pd.DataFrame:
    if TCXReader is None:
        raise RuntimeError("Missing dependency: tcxreader. Run: python -m pip install tcxreader")

    # tcxreader wants a file path, so we use a temp file in memory-like approach:
    # easiest: write to a temp file via Streamlit (safe in local / cloud env)
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tcx") as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = tmp.name

    tcx = TCXReader().read(tmp_path)

    rows = []
    # tcx.trackpoints: list of TrackPoint objects
    # Depending on file, might be tcx.trackpoints or tcx.trackpoints_list; tcxreader uses trackpoints
    trackpoints = getattr(tcx, "trackpoints", [])
    for tp in trackpoints:
        # timestamp is datetime
        t = getattr(tp, "time", None) or getattr(tp, "timestamp", None)
        lat = getattr(tp, "latitude", None)
        lon = getattr(tp, "longitude", None)
        ele = getattr(tp, "altitude", None)
        hr = getattr(tp, "hr_value", None) or getattr(tp, "heart_rate", None)

        if t is None or lat is None or lon is None:
            continue
        rows.append(
            {
                "time": t.replace(tzinfo=None) if hasattr(t, "replace") else t,
                "lat": float(lat),
                "lon": float(lon),
                "ele_ft": float(ele) * 3.28084 if ele is not None else None,
                "hr": float(hr) if hr is not None else None,
            }
        )

    return pd.DataFrame(rows)


def normalize_lap_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to normalize a lap-based CSV (Garmin/Strava style).
    Returns a dataframe with at least:
    Laps, Distancemi, Avg Pacemin/mi, Avg Pace sec, Avg HRbpm, Total Ascentft
    """
    # Make a copy and strip summary rows if present
    work = df.copy()
    if "Laps" in work.columns:
        work = work[work["Laps"].astype(str).str.lower() != "summary"].copy()

    # Attempt to map likely column names
    # These match what you’ve been using already.
    col_map = {
        "Laps": ["Laps", "Lap", "lap"],
        "Distancemi": ["Distancemi", "Distance", "Distance mi", "Distance (mi)"],
        "Avg Pacemin/mi": ["Avg Pacemin/mi", "Avg Pace", "Avg Pace (min/mi)", "Pace"],
        "Avg GAPmin/mi": ["Avg GAPmin/mi", "Avg GAP", "GAP"],
        "Avg HRbpm": ["Avg HRbpm", "Avg HR", "HR", "Average Heart Rate"],
        "Total Ascentft": ["Total Ascentft", "Total Ascent", "Ascent", "Elevation Gain"],
    }

    def find_col(candidates):
        for c in candidates:
            if c in work.columns:
                return c
        return None

    laps_col = find_col(col_map["Laps"])
    dist_col = find_col(col_map["Distancemi"])
    pace_col = find_col(col_map["Avg Pacemin/mi"])
    hr_col = find_col(col_map["Avg HRbpm"])
    ascent_col = find_col(col_map["Total Ascentft"])
    gap_col = find_col(col_map.get("Avg GAPmin/mi", []))

    if laps_col is None or dist_col is None or pace_col is None:
        # Not a lap csv we recognize
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Laps"] = pd.to_numeric(work[laps_col], errors="coerce")
    out["Distancemi"] = pd.to_numeric(work[dist_col], errors="coerce")

    # Pace is usually like "9:27" -> convert to sec and also keep as string
    pace_str = work[pace_col].astype(str)
    out["Avg Pacemin/mi"] = pace_str

    def pace_to_sec(p):
        p = str(p).strip()
        if ":" not in p:
            return None
        parts = p.split(":")
        if len(parts) != 2:
            return None
        m, s = parts
        try:
            return int(m) * 60 + int(s)
        except Exception:
            return None

    out["Avg Pace sec"] = out["Avg Pacemin/mi"].apply(pace_to_sec)

    if gap_col is not None:
        out["Avg GAPmin/mi"] = work[gap_col].astype(str)
        out["Avg GAP sec"] = out["Avg GAPmin/mi"].apply(pace_to_sec)
    else:
        out["Avg GAPmin/mi"] = ""
        out["Avg GAP sec"] = None

    out["Avg HRbpm"] = pd.to_numeric(work[hr_col], errors="coerce") if hr_col else None
    out["Total Ascentft"] = pd.to_numeric(work[ascent_col], errors="coerce") if ascent_col else None

    out = out.dropna(subset=["Laps"]).copy()
    out["Laps"] = out["Laps"].astype(int)
    return out.reset_index(drop=True)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RunLab Prototype", layout="wide")
st.title("🏃 RunLab — Upload CSV / GPX / TCX")
st.write("Upload a lap CSV (Garmin/Strava) or a GPX/TCX file. GPX/TCX will be converted into mile splits automatically.")

uploaded = st.file_uploader("Upload file", type=["csv", "gpx", "tcx"])

if uploaded is None:
    st.info("Upload a CSV / GPX / TCX to begin.")
    st.stop()

file_name = uploaded.name.lower()
raw_bytes = uploaded.getvalue()

# Show raw preview (for CSV)
if file_name.endswith(".csv"):
    df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    st.subheader("Raw CSV Preview")
    st.dataframe(df_raw.head(20))

    analysis = normalize_lap_csv(df_raw)

    if analysis.empty:
        st.warning("This CSV doesn't look like a lap-based export I recognize. Try GPX/TCX or export laps from Garmin/Strava.")
        st.stop()

elif file_name.endswith(".gpx"):
    points = parse_gpx(raw_bytes)
    st.subheader("Parsed GPX Points Preview")
    st.dataframe(points.head(20))

    analysis = build_splits_from_points(points, split_miles=1.0)

elif file_name.endswith(".tcx"):
    points = parse_tcx(raw_bytes)
    st.subheader("Parsed TCX Points Preview")
    st.dataframe(points.head(20))

    analysis = build_splits_from_points(points, split_miles=1.0)

else:
    st.error("Unsupported file type.")
    st.stop()

# -----------------------------
# Analysis / Display
# -----------------------------
st.subheader("Normalized Lap / Split Analysis")

# Ensure pace columns exist
if "Avg Pace sec" not in analysis.columns and "Avg Pacemin/mi" in analysis.columns:
    # Build Avg Pace sec from string
    def _p2s(p):
        p = str(p).strip()
        if ":" not in p:
            return None
        m, s = p.split(":")
        try:
            return int(m) * 60 + int(s)
        except Exception:
            return None

    analysis["Avg Pace sec"] = analysis["Avg Pacemin/mi"].apply(_p2s)

# Add helpful columns
analysis["Pace (min/mi)"] = analysis["Avg Pace sec"].apply(pace_str_from_seconds)
analysis["Pace Δ (sec)"] = analysis["Avg Pace sec"].diff()

# Best/Worst lap (ignore partial last split if under 0.75 mi)
full_laps = analysis.copy()
if "Distancemi" in full_laps.columns:
    full_laps = full_laps[full_laps["Distancemi"] >= 0.75].copy()

best_lap_num = None
worst_lap_num = None
pace_fade_sec = None

if not full_laps.empty and full_laps["Avg Pace sec"].notna().any():
    best_idx = full_laps["Avg Pace sec"].idxmin()
    worst_idx = full_laps["Avg Pace sec"].idxmax()
    best_lap_num = int(full_laps.loc[best_idx, "Laps"])
    worst_lap_num = int(full_laps.loc[worst_idx, "Laps"])
    pace_fade_sec = float(full_laps.loc[worst_idx, "Avg Pace sec"] - full_laps.loc[best_idx, "Avg Pace sec"])

st.dataframe(
    analysis[
        [c for c in ["Laps", "Distancemi", "Pace (min/mi)", "Avg HRbpm", "Total Ascentft", "Pace Δ (sec)"] if c in analysis.columns]
    ]
)

# Charts
st.subheader("Pace Drift Over Laps")
chart_df = analysis[["Laps", "Avg Pace sec"]].dropna().copy()
chart_df["Pace (min/mi)"] = chart_df["Avg Pace sec"].apply(pace_str_from_seconds)
st.line_chart(chart_df.set_index("Laps")["Avg Pace sec"])

# Key insights cards
st.subheader("Key Insights")

c1, c2, c3, c4 = st.columns(4)

avg_hr = None
if "Avg HRbpm" in analysis.columns and analysis["Avg HRbpm"].notna().any():
    avg_hr = float(pd.to_numeric(analysis["Avg HRbpm"], errors="coerce").dropna().mean())

avg_ascent = None
if "Total Ascentft" in analysis.columns and analysis["Total Ascentft"].notna().any():
    # average per lap/split (ignoring partial last split if tiny)
    asc_df = analysis.copy()
    if "Distancemi" in asc_df.columns:
        asc_df = asc_df[asc_df["Distancemi"] >= 0.75].copy()
    if asc_df["Total Ascentft"].notna().any():
        avg_ascent = float(pd.to_numeric(asc_df["Total Ascentft"], errors="coerce").dropna().mean())

avg_pace_full = None
if not full_laps.empty and full_laps["Avg Pace sec"].notna().any():
    avg_pace_full = float(full_laps["Avg Pace sec"].dropna().mean())

with c1:
    st.metric("Average HR", f"{avg_hr:.0f} bpm" if avg_hr is not None else "—")
with c2:
    st.metric("Avg Pace (full laps)", pace_str_from_seconds(avg_pace_full) if avg_pace_full is not None else "—")
with c3:
    st.metric("Best Lap (pace)", f"Lap {best_lap_num}" if best_lap_num is not None else "—")
with c4:
    st.metric("Worst Lap (pace)", f"Lap {worst_lap_num}" if worst_lap_num is not None else "—")

st.markdown("### What this means")
if pace_fade_sec is not None:
    st.write(
        f"**Pace fade:** {int(round(pace_fade_sec))} sec/mi from best to worst full lap. "
        "Big fades usually mean early pacing was too hot, terrain hit late, or fueling/hydration wasn’t matched to effort."
    )
else:
    st.write("Upload a lap-based CSV or a GPX/TCX with enough data to compute splits.")
