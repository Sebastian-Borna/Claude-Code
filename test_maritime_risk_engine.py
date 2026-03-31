"""Tests for the Maritime Risk Decision-Support System."""

import json
import math
from copy import deepcopy

from maritime_risk_engine import (
    validate_risk_map,
    validate_incidents,
    aggregate_risk_by_region,
    identify_chokepoints,
    detect_incident_clusters,
    compute_global_risk_index,
    evaluate_routes,
    generate_summary,
    process,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_RISK_MAP = [
    {"location": "Strait of Hormuz", "lat": 26.56, "lon": 56.25, "risk_level": "critical", "risk_score": 92, "region": "Persian Gulf"},
    {"location": "Bab el-Mandeb", "lat": 12.58, "lon": 43.33, "risk_level": "high", "risk_score": 78, "region": "Red Sea"},
    {"location": "Strait of Malacca", "lat": 1.43, "lon": 103.5, "risk_level": "medium", "risk_score": 45, "region": "Southeast Asia"},
    {"location": "Gulf of Guinea", "lat": 4.0, "lon": 2.0, "risk_level": "high", "risk_score": 80, "region": "West Africa"},
    {"location": "Singapore Strait", "lat": 1.25, "lon": 103.8, "risk_level": "low", "risk_score": 20, "region": "Southeast Asia"},
]

SAMPLE_INCIDENTS = [
    {"id": "INC-001", "location": "Strait of Hormuz", "type": "drone_attack", "date": "2025-12-01", "severity": "critical", "lat": 26.6, "lon": 56.3},
    {"id": "INC-002", "location": "Strait of Hormuz", "type": "missile_threat", "date": "2025-12-05", "severity": "major", "lat": 26.5, "lon": 56.2},
    {"id": "INC-003", "location": "Bab el-Mandeb", "type": "piracy_attempt", "date": "2025-11-20", "severity": "moderate", "lat": 12.6, "lon": 43.4},
    {"id": "INC-004", "location": "Gulf of Guinea", "type": "armed_robbery", "date": "2025-10-15", "severity": "major", "lat": 4.1, "lon": 2.1},
    {"id": "INC-005", "location": "Strait of Malacca", "type": "collision_risk", "date": "2025-11-28", "severity": "minor", "lat": 1.4, "lon": 103.5},
]

SAMPLE_ROUTE_ANALYSIS = {
    "best_route": "Suez Canal Route",
    "routes": [
        {"name": "Suez Canal Route", "risk_score": 70, "delay_risk": 60, "cost_impact": 40},
        {"name": "Cape of Good Hope Route", "risk_score": 30, "delay_risk": 50, "cost_impact": 70},
        {"name": "Northern Sea Route", "risk_score": 55, "delay_risk": 80, "cost_impact": 60},
    ],
}

FULL_INPUT = {
    "risk_map": SAMPLE_RISK_MAP,
    "incidents": SAMPLE_INCIDENTS,
    "route_analysis": SAMPLE_ROUTE_ANALYSIS,
}


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateRiskMap:
    def test_all_valid_entries_pass(self):
        valid, flagged = validate_risk_map(SAMPLE_RISK_MAP)
        assert len(valid) == 5
        assert len(flagged) == 0

    def test_missing_fields_flagged(self):
        bad = [{"location": "Nowhere"}]
        valid, flagged = validate_risk_map(bad)
        assert len(valid) == 0
        assert len(flagged) == 1
        assert "missing fields" in flagged[0]["_issues"][0]

    def test_invalid_coordinates_flagged(self):
        bad = [{"location": "X", "lat": 999, "lon": 0, "risk_level": "low", "risk_score": 10}]
        valid, flagged = validate_risk_map(bad)
        assert len(flagged) == 1

    def test_risk_level_normalised(self):
        data = [{"location": "Y", "lat": 10, "lon": 20, "risk_level": "HI", "risk_score": 50}]
        valid, flagged = validate_risk_map(data)
        assert len(valid) == 1
        assert valid[0]["risk_level"] == "high"

    def test_duplicates_removed(self):
        dup = SAMPLE_RISK_MAP[:1] * 3
        valid, flagged = validate_risk_map(dup)
        assert len(valid) == 1
        assert len(flagged) == 2

    def test_risk_score_clamped(self):
        data = [{"location": "Z", "lat": 0, "lon": 0, "risk_level": "low", "risk_score": 150}]
        valid, flagged = validate_risk_map(data)
        assert flagged[0]["risk_score"] == 100.0

    def test_low_confidence_flagged(self):
        data = [{"location": "W", "lat": 5, "lon": 5, "risk_level": "low", "risk_score": 10, "confidence": 0.2}]
        valid, flagged = validate_risk_map(data)
        assert len(flagged) == 1
        assert any("low confidence" in i for i in flagged[0]["_issues"])


class TestValidateIncidents:
    def test_all_valid(self):
        valid, flagged = validate_incidents(SAMPLE_INCIDENTS)
        assert len(valid) == 5
        assert len(flagged) == 0

    def test_duplicate_id(self):
        duped = SAMPLE_INCIDENTS[:1] * 2
        valid, flagged = validate_incidents(duped)
        assert len(valid) == 1
        assert len(flagged) == 1

    def test_bad_date_flagged(self):
        bad = [{"id": "X", "location": "A", "type": "t", "date": "not-a-date", "severity": "minor"}]
        valid, flagged = validate_incidents(bad)
        assert len(flagged) == 1

    def test_severity_normalised(self):
        data = [{"id": "Y", "location": "A", "type": "t", "date": "2025-01-01", "severity": "CRIT"}]
        valid, flagged = validate_incidents(data)
        assert len(valid) == 1
        assert valid[0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# Enrichment tests
# ---------------------------------------------------------------------------

class TestEnrichment:
    def test_aggregate_risk_by_region(self):
        valid, _ = validate_risk_map(SAMPLE_RISK_MAP)
        agg = aggregate_risk_by_region(valid)
        assert "Persian Gulf" in agg
        assert "Southeast Asia" in agg
        assert agg["Southeast Asia"]["count"] == 2

    def test_chokepoints_ordered(self):
        valid, _ = validate_risk_map(SAMPLE_RISK_MAP)
        cp = identify_chokepoints(valid, top_n=3)
        assert len(cp) == 3
        assert cp[0]["risk_score"] >= cp[1]["risk_score"] >= cp[2]["risk_score"]

    def test_incident_clusters(self):
        valid, _ = validate_incidents(SAMPLE_INCIDENTS)
        clusters = detect_incident_clusters(valid)
        hormuz = [c for c in clusters if any("Hormuz" in loc for loc in c["locations"])]
        assert len(hormuz) >= 1
        assert hormuz[0]["incident_count"] >= 2

    def test_global_risk_index_range(self):
        valid_risk, _ = validate_risk_map(SAMPLE_RISK_MAP)
        valid_inc, _ = validate_incidents(SAMPLE_INCIDENTS)
        gri = compute_global_risk_index(valid_risk, valid_inc)
        assert 0 <= gri <= 100

    def test_global_risk_index_empty(self):
        assert compute_global_risk_index([], []) == 0.0


# ---------------------------------------------------------------------------
# Decision logic tests
# ---------------------------------------------------------------------------

class TestDecisionLogic:
    def test_route_scoring(self):
        result = evaluate_routes(SAMPLE_ROUTE_ANALYSIS)
        for r in result["routes"]:
            assert "composite_score" in r

    def test_override_when_better_route(self):
        result = evaluate_routes(SAMPLE_ROUTE_ANALYSIS)
        assert result["best_route"] == "Cape of Good Hope Route"
        assert result["override"] is True
        assert "override_justification" in result

    def test_ranking_order(self):
        result = evaluate_routes(SAMPLE_ROUTE_ANALYSIS)
        scores = [r["composite_score"] for r in result["ranking"]]
        assert scores == sorted(scores)

    def test_comparative_advantage(self):
        result = evaluate_routes(SAMPLE_ROUTE_ANALYSIS)
        assert result["comparative_advantage"] > 0

    def test_no_override_when_same(self):
        routes = deepcopy(SAMPLE_ROUTE_ANALYSIS)
        routes["best_route"] = "Cape of Good Hope Route"
        result = evaluate_routes(routes)
        assert result["override"] is False


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_process_output_structure(self):
        out = process(FULL_INPUT)
        assert "validated_risk_map" in out
        assert "validated_incidents" in out
        assert "enhanced_route_analysis" in out
        assert "insights" in out
        assert "data_quality" in out
        assert "streamlit_support" in out

    def test_insights_fields(self):
        out = process(FULL_INPUT)
        ins = out["insights"]
        assert isinstance(ins["highest_risk_locations"], list)
        assert isinstance(ins["critical_incidents"], list)
        assert isinstance(ins["recommended_route"], str)
        assert 0 <= ins["global_risk_index"] <= 100
        assert len(ins["key_summary"]) > 0

    def test_output_is_json_serialisable(self):
        out = process(FULL_INPUT)
        serialised = json.dumps(out, default=str)
        assert isinstance(json.loads(serialised), dict)

    def test_empty_input(self):
        out = process({})
        assert out["insights"]["global_risk_index"] == 0.0
        assert out["insights"]["recommended_route"] == ""

    def test_data_quality_counts(self):
        out = process(FULL_INPUT)
        dq = out["data_quality"]
        assert dq["risk_map_valid"] == 5
        assert dq["incidents_valid"] == 5
        assert dq["risk_map_flagged"] == 0
        assert dq["incidents_flagged"] == 0
