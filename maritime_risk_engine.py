"""
Maritime Risk Decision-Support System

Transforms structured tanker intelligence data (risk_map, incidents, route_analysis)
into validated, enriched, decision-ready output for operational use.
"""

import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# 1. CONSTANTS & CONFIGURATION
# ---------------------------------------------------------------------------

REQUIRED_RISK_MAP_FIELDS = {"location", "lat", "lon", "risk_level", "risk_score"}
REQUIRED_INCIDENT_FIELDS = {"id", "location", "type", "date", "severity"}
VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}
VALID_SEVERITIES = {"minor", "moderate", "major", "critical"}
RISK_LEVEL_SCORES = {"low": 15, "medium": 40, "high": 70, "critical": 95}
SEVERITY_WEIGHTS = {"minor": 1, "moderate": 2, "major": 3, "critical": 4}
CLUSTER_RADIUS_DEG = 1.5  # ~165 km at equator
LOW_CONFIDENCE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# 2. VALIDATION
# ---------------------------------------------------------------------------

def _normalise_risk_level(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in VALID_RISK_LEVELS:
        return v
    aliases = {"med": "medium", "hi": "high", "crit": "critical", "lo": "low"}
    return aliases.get(v)


def _normalise_severity(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in VALID_SEVERITIES:
        return v
    aliases = {"crit": "critical", "maj": "major", "mod": "moderate", "min": "minor"}
    return aliases.get(v)


def _valid_coord(lat: Any, lon: Any) -> bool:
    try:
        return -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180
    except (TypeError, ValueError):
        return False


def _dedup_key_risk(entry: dict) -> str:
    return f"{entry.get('location','').strip().lower()}|{entry.get('lat')}|{entry.get('lon')}"


def _dedup_key_incident(entry: dict) -> str:
    return str(entry.get("id", "")).strip().lower()


def validate_risk_map(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (valid_entries, flagged_entries)."""
    valid, flagged = [], []
    seen_keys: set[str] = set()

    for entry in raw:
        issues: list[str] = []
        cleaned = deepcopy(entry)

        # Missing fields
        missing = REQUIRED_RISK_MAP_FIELDS - set(entry.keys())
        if missing:
            issues.append(f"missing fields: {sorted(missing)}")

        # Coordinates
        if not _valid_coord(entry.get("lat"), entry.get("lon")):
            issues.append("invalid coordinates")
        else:
            cleaned["lat"] = float(entry["lat"])
            cleaned["lon"] = float(entry["lon"])

        # Risk level normalisation
        rl = _normalise_risk_level(entry.get("risk_level", ""))
        if rl is None:
            issues.append(f"unrecognised risk_level: {entry.get('risk_level')}")
        else:
            cleaned["risk_level"] = rl

        # Risk score normalisation
        try:
            rs = float(entry.get("risk_score", -1))
            if not 0 <= rs <= 100:
                issues.append(f"risk_score out of range: {rs}")
                rs = max(0.0, min(100.0, rs))
            cleaned["risk_score"] = round(rs, 2)
        except (TypeError, ValueError):
            issues.append(f"non-numeric risk_score: {entry.get('risk_score')}")
            if rl:
                cleaned["risk_score"] = RISK_LEVEL_SCORES[rl]

        # Confidence check
        conf = entry.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
                cleaned["confidence"] = conf
                if conf < LOW_CONFIDENCE_THRESHOLD:
                    issues.append(f"low confidence ({conf})")
            except (TypeError, ValueError):
                pass

        # Deduplicate
        key = _dedup_key_risk(cleaned)
        if key in seen_keys:
            issues.append("duplicate entry")
            flagged.append({**cleaned, "_issues": issues})
            continue
        seen_keys.add(key)

        if issues:
            cleaned["_issues"] = issues
            flagged.append(cleaned)
        else:
            valid.append(cleaned)

    return valid, flagged


def validate_incidents(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (valid_entries, flagged_entries)."""
    valid, flagged = [], []
    seen_ids: set[str] = set()

    for entry in raw:
        issues: list[str] = []
        cleaned = deepcopy(entry)

        missing = REQUIRED_INCIDENT_FIELDS - set(entry.keys())
        if missing:
            issues.append(f"missing fields: {sorted(missing)}")

        # Severity
        sev = _normalise_severity(entry.get("severity", ""))
        if sev is None:
            issues.append(f"unrecognised severity: {entry.get('severity')}")
        else:
            cleaned["severity"] = sev

        # Date
        raw_date = entry.get("date", "")
        date_parsed = None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                date_parsed = datetime.strptime(str(raw_date), fmt)
                cleaned["date"] = date_parsed.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        if date_parsed is None:
            issues.append(f"unparseable date: {raw_date}")

        # Coordinates (optional for incidents)
        if "lat" in entry and "lon" in entry:
            if _valid_coord(entry["lat"], entry["lon"]):
                cleaned["lat"] = float(entry["lat"])
                cleaned["lon"] = float(entry["lon"])
            else:
                issues.append("invalid coordinates")

        # Dedup by id
        key = _dedup_key_incident(cleaned)
        if key in seen_ids:
            issues.append("duplicate id")
            flagged.append({**cleaned, "_issues": issues})
            continue
        seen_ids.add(key)

        if issues:
            cleaned["_issues"] = issues
            flagged.append(cleaned)
        else:
            valid.append(cleaned)

    return valid, flagged


# ---------------------------------------------------------------------------
# 3. ENRICHMENT
# ---------------------------------------------------------------------------

def _haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in degrees (cheap clustering metric)."""
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def aggregate_risk_by_region(risk_map: list[dict]) -> dict[str, dict]:
    regions: dict[str, list[float]] = defaultdict(list)
    for entry in risk_map:
        loc = entry.get("region") or entry.get("location", "unknown")
        regions[loc].append(entry.get("risk_score", 0))
    return {
        region: {
            "mean_risk": round(sum(scores) / len(scores), 2),
            "max_risk": max(scores),
            "count": len(scores),
        }
        for region, scores in regions.items()
    }


def identify_chokepoints(risk_map: list[dict], top_n: int = 5) -> list[dict]:
    scored = sorted(risk_map, key=lambda e: e.get("risk_score", 0), reverse=True)
    return [
        {
            "location": e.get("location"),
            "risk_score": e.get("risk_score"),
            "risk_level": e.get("risk_level"),
            "lat": e.get("lat"),
            "lon": e.get("lon"),
        }
        for e in scored[:top_n]
    ]


def detect_incident_clusters(incidents: list[dict]) -> list[dict]:
    """Simple radius-based clustering on geo-located incidents."""
    geo = [i for i in incidents if "lat" in i and "lon" in i]
    clusters: list[list[dict]] = []
    assigned: set[int] = set()

    for i, a in enumerate(geo):
        if i in assigned:
            continue
        cluster = [a]
        assigned.add(i)
        for j, b in enumerate(geo):
            if j in assigned:
                continue
            if _haversine_deg(a["lat"], a["lon"], b["lat"], b["lon"]) <= CLUSTER_RADIUS_DEG:
                cluster.append(b)
                assigned.add(j)
        if len(cluster) >= 2:
            clusters.append(cluster)

    result = []
    for cl in clusters:
        lats = [c["lat"] for c in cl]
        lons = [c["lon"] for c in cl]
        locs = list({c.get("location", "unknown") for c in cl})
        result.append({
            "centroid_lat": round(sum(lats) / len(lats), 4),
            "centroid_lon": round(sum(lons) / len(lons), 4),
            "incident_count": len(cl),
            "locations": locs,
            "severity_breakdown": dict(Counter(c.get("severity", "unknown") for c in cl)),
        })
    return sorted(result, key=lambda c: c["incident_count"], reverse=True)


def compute_global_risk_index(
    risk_map: list[dict],
    incidents: list[dict],
) -> float:
    """
    Weighted composite: 60 % avg risk_score from map, 40 % incident severity pressure.
    Clamped to 0-100.
    """
    if not risk_map:
        map_component = 0.0
    else:
        map_component = sum(e.get("risk_score", 0) for e in risk_map) / len(risk_map)

    if not incidents:
        incident_component = 0.0
    else:
        total_weight = sum(SEVERITY_WEIGHTS.get(i.get("severity", "minor"), 1) for i in incidents)
        max_weight = len(incidents) * 4
        incident_component = (total_weight / max_weight) * 100 if max_weight else 0

    index = 0.6 * map_component + 0.4 * incident_component
    return round(max(0, min(100, index)), 1)


# ---------------------------------------------------------------------------
# 4. DECISION LOGIC — ROUTE EVALUATION
# ---------------------------------------------------------------------------

def _score_route(route: dict) -> float:
    """
    Lower is better. Composite of risk_score, delay_risk, cost_impact.
    Weights: risk 0.5, delay 0.3, cost 0.2.
    """
    risk = float(route.get("risk_score", 50))
    delay = float(route.get("delay_risk", 50))
    cost = float(route.get("cost_impact", 50))
    return 0.5 * risk + 0.3 * delay + 0.2 * cost


def evaluate_routes(route_analysis: dict) -> dict:
    """
    Re-rank routes and potentially override the stated best_route.
    Returns the enhanced route_analysis block.
    """
    enhanced = deepcopy(route_analysis)
    routes = enhanced.get("routes", [])
    if not routes:
        return enhanced

    for r in routes:
        r["composite_score"] = round(_score_route(r), 2)

    ranked = sorted(routes, key=lambda r: r["composite_score"])
    best = ranked[0]
    enhanced["routes"] = routes  # preserve original order, scores added

    original_best = enhanced.get("best_route")
    new_best = best.get("name") or best.get("route_name") or best.get("id")

    if original_best and original_best != new_best:
        enhanced["best_route"] = new_best
        enhanced["override"] = True
        enhanced["override_justification"] = (
            f"Route '{new_best}' (composite {best['composite_score']}) "
            f"outperforms original recommendation '{original_best}' on "
            f"weighted risk/delay/cost analysis."
        )
    else:
        enhanced["best_route"] = new_best
        enhanced["override"] = False

    enhanced["ranking"] = [
        {"route": r.get("name") or r.get("route_name") or r.get("id"), "composite_score": r["composite_score"]}
        for r in ranked
    ]

    # Route comparative advantage
    if len(ranked) >= 2:
        advantage = round(ranked[1]["composite_score"] - ranked[0]["composite_score"], 2)
        enhanced["comparative_advantage"] = advantage
    else:
        enhanced["comparative_advantage"] = 0.0

    return enhanced


# ---------------------------------------------------------------------------
# 5. KEY SUMMARY GENERATION
# ---------------------------------------------------------------------------

def generate_summary(
    risk_map: list[dict],
    incidents: list[dict],
    enhanced_routes: dict,
    global_risk_index: float,
    chokepoints: list[dict],
) -> str:
    parts: list[str] = []
    parts.append(f"Global risk index stands at {global_risk_index}/100.")

    if chokepoints:
        top = chokepoints[0]
        parts.append(
            f"Highest-risk chokepoint is {top['location']} "
            f"(score {top['risk_score']})."
        )

    critical = [i for i in incidents if i.get("severity") == "critical"]
    if critical:
        parts.append(f"{len(critical)} critical incident(s) recorded.")

    best = enhanced_routes.get("best_route")
    if best:
        if enhanced_routes.get("override"):
            parts.append(
                f"Recommended route overridden to '{best}' based on "
                f"composite risk-delay-cost scoring."
            )
        else:
            parts.append(f"Recommended route: '{best}'.")

    return " ".join(parts[:4])


# ---------------------------------------------------------------------------
# 6. STREAMLIT / PANDAS SUPPORT
# ---------------------------------------------------------------------------

def pandas_metadata() -> dict:
    return {
        "risk_map_table": {
            "columns": ["location", "lat", "lon", "risk_level", "risk_score", "confidence", "region"],
            "index": None,
            "notes": "Use for scatter-mapbox plot; color by risk_level, size by risk_score.",
        },
        "incidents_table": {
            "columns": ["id", "location", "type", "date", "severity", "lat", "lon"],
            "index": "id",
            "notes": "Time-series bar chart by date+severity; map overlay with risk_map.",
        },
        "route_ranking_table": {
            "columns": ["route", "composite_score", "risk_score", "delay_risk", "cost_impact"],
            "index": None,
            "notes": "Horizontal bar chart sorted by composite_score; highlight best route.",
        },
        "visualisation_suggestions": [
            "Folium/Plotly choropleth map colored by regional mean_risk",
            "Incident cluster markers with popup severity breakdowns",
            "Gauge chart for global_risk_index",
            "Alert banner for critical incidents in last 30 days",
        ],
    }


# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------------------------

def process(data: dict) -> dict:
    """Main entry point. Accepts raw input dict, returns decision-ready output."""
    raw_risk_map = data.get("risk_map", [])
    raw_incidents = data.get("incidents", [])
    raw_routes = data.get("route_analysis", {})

    # --- Validation ---
    valid_risk, flagged_risk = validate_risk_map(raw_risk_map)
    valid_incidents, flagged_incidents = validate_incidents(raw_incidents)

    # --- Enrichment ---
    region_agg = aggregate_risk_by_region(valid_risk)
    chokepoints = identify_chokepoints(valid_risk)
    clusters = detect_incident_clusters(valid_incidents)
    gri = compute_global_risk_index(valid_risk, valid_incidents)

    # --- Decision logic ---
    enhanced_routes = evaluate_routes(raw_routes)

    # --- Summary ---
    summary = generate_summary(valid_risk, valid_incidents, enhanced_routes, gri, chokepoints)

    # --- Critical incidents ---
    critical_incidents = [
        {k: v for k, v in i.items() if k != "_issues"}
        for i in valid_incidents
        if i.get("severity") in ("critical", "major")
    ]

    return {
        "validated_risk_map": valid_risk,
        "validated_incidents": valid_incidents,
        "enhanced_route_analysis": enhanced_routes,
        "insights": {
            "highest_risk_locations": chokepoints,
            "critical_incidents": critical_incidents,
            "incident_clusters": clusters,
            "region_risk_aggregate": region_agg,
            "recommended_route": enhanced_routes.get("best_route", ""),
            "global_risk_index": gri,
            "key_summary": summary,
        },
        "data_quality": {
            "risk_map_valid": len(valid_risk),
            "risk_map_flagged": len(flagged_risk),
            "incidents_valid": len(valid_incidents),
            "incidents_flagged": len(flagged_incidents),
            "flagged_risk_details": flagged_risk,
            "flagged_incident_details": flagged_incidents,
        },
        "streamlit_support": pandas_metadata(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    result = process(data)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
