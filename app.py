"""
Maritime Risk Decision-Support Dashboard

Streamlit interface for the maritime risk engine.
Run with: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from maritime_risk_engine import process

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Maritime Risk Decision Support",
    page_icon="⚓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load and process data
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_process(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return process(raw)


result = load_and_process("sample_input.json")
insights = result["insights"]
dq = result["data_quality"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("⚓ Maritime Risk Decision Support")
st.caption(insights["key_summary"])

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

gri = insights["global_risk_index"]
col1.metric("Global Risk Index", f"{gri}/100", delta_color="inverse")
col2.metric("Validated Risk Zones", dq["risk_map_valid"])
col3.metric("Validated Incidents", dq["incidents_valid"])

route_info = result["enhanced_route_analysis"]
route_label = route_info.get("best_route", "N/A")
if route_info.get("override"):
    col4.metric("Recommended Route", route_label, delta="OVERRIDDEN", delta_color="inverse")
else:
    col4.metric("Recommended Route", route_label)

# ---------------------------------------------------------------------------
# Global Risk Index gauge
# ---------------------------------------------------------------------------

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Global Risk Index")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gri,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred" if gri > 70 else "orange" if gri > 40 else "green"},
            "steps": [
                {"range": [0, 30], "color": "#d4edda"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 80], "color": "#ffeeba"},
                {"range": [80, 100], "color": "#f8d7da"},
            ],
        },
        title={"text": "Composite Score"},
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=20, l=40, r=40))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------------------------------------------------------
# Risk map
# ---------------------------------------------------------------------------

with right:
    st.subheader("Risk Map")
    risk_df = pd.DataFrame(result["validated_risk_map"])
    color_map = {"critical": "#dc3545", "high": "#fd7e14", "medium": "#ffc107", "low": "#28a745"}
    risk_df["color"] = risk_df["risk_level"].map(color_map)

    fig_map = px.scatter_mapbox(
        risk_df,
        lat="lat",
        lon="lon",
        size="risk_score",
        color="risk_level",
        color_discrete_map=color_map,
        hover_name="location",
        hover_data=["risk_score", "risk_level"],
        size_max=25,
        zoom=1.5,
        mapbox_style="open-street-map",
    )
    fig_map.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------------------------------
# Route comparison
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Route Comparison")

route_df = pd.DataFrame(route_info["routes"])
route_cols = ["name", "composite_score", "risk_score", "delay_risk", "cost_impact"]
available_cols = [c for c in route_cols if c in route_df.columns]
route_display = route_df[available_cols].copy()
route_display = route_display.sort_values("composite_score")

col_chart, col_detail = st.columns([2, 1])

with col_chart:
    fig_routes = px.bar(
        route_display,
        x="composite_score",
        y="name",
        orientation="h",
        color="composite_score",
        color_continuous_scale=["green", "orange", "red"],
        labels={"composite_score": "Composite Score (lower = better)", "name": "Route"},
    )
    fig_routes.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig_routes, use_container_width=True)

with col_detail:
    if route_info.get("override"):
        st.warning(
            f"**Route Override:** {route_info['override_justification']}",
            icon="⚠️",
        )
    st.metric("Comparative Advantage", f"{route_info.get('comparative_advantage', 0)} pts")
    st.dataframe(route_display.set_index("name"), use_container_width=True)

# ---------------------------------------------------------------------------
# Incidents & clusters
# ---------------------------------------------------------------------------

st.divider()

inc_col, cluster_col = st.columns(2)

with inc_col:
    st.subheader("Critical Incidents")
    critical = insights["critical_incidents"]
    if critical:
        crit_df = pd.DataFrame(critical)
        display_cols = [c for c in ["id", "location", "type", "date", "severity"] if c in crit_df.columns]
        st.dataframe(crit_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.success("No critical incidents recorded.")

with cluster_col:
    st.subheader("Incident Clusters")
    clusters = insights["incident_clusters"]
    if clusters:
        for i, cl in enumerate(clusters):
            with st.expander(f"Cluster {i+1}: {', '.join(cl['locations'])} ({cl['incident_count']} incidents)"):
                st.json(cl["severity_breakdown"])
                st.caption(f"Centroid: {cl['centroid_lat']}, {cl['centroid_lon']}")
    else:
        st.info("No incident clusters detected.")

# ---------------------------------------------------------------------------
# Regional risk breakdown
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Regional Risk Breakdown")

region_data = insights["region_risk_aggregate"]
region_df = pd.DataFrame([
    {"Region": k, "Mean Risk": v["mean_risk"], "Max Risk": v["max_risk"], "Zones": v["count"]}
    for k, v in region_data.items()
]).sort_values("Mean Risk", ascending=False)

reg_chart, reg_table = st.columns([2, 1])

with reg_chart:
    fig_region = px.bar(
        region_df,
        x="Region",
        y="Mean Risk",
        color="Mean Risk",
        color_continuous_scale=["green", "orange", "red"],
        text="Mean Risk",
    )
    fig_region.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_region, use_container_width=True)

with reg_table:
    st.dataframe(region_df.set_index("Region"), use_container_width=True)

# ---------------------------------------------------------------------------
# Highest risk chokepoints
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Top Chokepoints")

cp_df = pd.DataFrame(insights["highest_risk_locations"])
st.dataframe(cp_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

st.divider()
with st.expander("Data Quality Report"):
    dq_col1, dq_col2 = st.columns(2)
    dq_col1.metric("Risk Map — Valid", dq["risk_map_valid"])
    dq_col1.metric("Risk Map — Flagged", dq["risk_map_flagged"])
    dq_col2.metric("Incidents — Valid", dq["incidents_valid"])
    dq_col2.metric("Incidents — Flagged", dq["incidents_flagged"])

    if dq["flagged_risk_details"]:
        st.warning("Flagged risk map entries:")
        st.json(dq["flagged_risk_details"])
    if dq["flagged_incident_details"]:
        st.warning("Flagged incidents:")
        st.json(dq["flagged_incident_details"])

# ---------------------------------------------------------------------------
# Raw JSON output
# ---------------------------------------------------------------------------

with st.expander("Raw Engine Output (JSON)"):
    st.json(result)
