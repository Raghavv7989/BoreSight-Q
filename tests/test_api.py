"""
Comprehensive integration test for BoreSight-Q FastAPI endpoints.
Tests:
  1. GET /api/health
  2. POST /api/analyze-parcel
  3. POST /api/export-kml (via analysis response and via ParcelRequest)
"""

import sys
from fastapi.testclient import TestClient
from backend.main import app

def test_full_pipeline():
    client = TestClient(app)

    # 1. Test Health Endpoint
    print("\n--- 1. Testing GET /api/health ---")
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200, f"Health check failed: {resp_health.text}"
    health_data = resp_health.json()
    print("Health response:", health_data)
    assert health_data["status"] == "healthy"

    # Sample parcel polygon: ~1.5km x 1.5km agricultural tract near Bangalore, Karnataka
    sample_parcel = {
        "parcel_geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [77.5800, 12.9600],
                    [77.6000, 12.9600],
                    [77.6000, 12.9800],
                    [77.5800, 12.9800],
                    [77.5800, 12.9600],
                ]
            ],
        },
        "buffer_m": 500.0,
    }

    # 2. Test POST /api/analyze-parcel
    print("\n--- 2. Testing POST /api/analyze-parcel ---")
    resp_analyze = client.post("/api/analyze-parcel", json=sample_parcel)
    assert resp_analyze.status_code == 200, f"Analyze-parcel failed ({resp_analyze.status_code}): {resp_analyze.text}"
    data = resp_analyze.json()

    print("Analyze response keys:", list(data.keys()))
    assert "borewell_points" in data
    assert "drainage_geojson" in data
    assert "lineaments_geojson" in data
    assert "intersections_geojson" in data
    assert "gwpi_heatmap_base64" in data
    assert "insar_heatmap_base64" in data
    assert "sar_moisture_base64" in data
    assert "bounds" in data
    assert "aquifer_summary" in data

    # Verify borewell points
    points = data["borewell_points"]
    print(f"Extracted {len(points)} borewell points:")
    assert len(points) >= 3, f"Expected at least 3 points, got {len(points)}"
    for pt in points:
        print(f"  [{pt['id']}] Lat: {pt['lat']:.5f}, Lon: {pt['lon']:.5f}, GWPI: {pt['gwpi_score']:.1f}, "
              f"Depth: {pt['recommended_depth_m']:.1f}m, Yield: {pt['estimated_yield_lph']} LPH, "
              f"Azimuth: {pt['fracture_azimuth_deg']:.1f}°, Stream Order: {pt['stream_order']}")
        assert pt["id"] in ["P1", "P2", "P3"]
        assert 0.0 <= pt["gwpi_score"] <= 100.0
        assert 0.0 <= pt["confidence_pct"] <= 100.0
        assert pt["recommended_depth_m"] > 0
        assert pt["estimated_yield_lph"] > 0
        assert len(pt["reasoning"]) > 10

    # Verify Base64 images
    for b64_field in ["gwpi_heatmap_base64", "insar_heatmap_base64", "sar_moisture_base64"]:
        val = data[b64_field]
        assert val.startswith("data:image/png;base64,"), f"{b64_field} does not have valid data URI prefix"
        assert len(val) > 100, f"{b64_field} is suspiciously short"
    print("Base64 overlays verified successfully.")

    # Verify bounds [west, south, east, north]
    bounds = data["bounds"]
    assert len(bounds) == 4
    w, s, e, n = bounds
    assert w < e and s < n, f"Invalid bounds: {bounds}"
    print(f"Bounds: west={w:.4f}, south={s:.4f}, east={e:.4f}, north={n:.4f}")

    # Verify drainage & lineaments
    drainage_features = data["drainage_geojson"].get("features", [])
    lineament_features = data["lineaments_geojson"].get("features", [])
    inter_features = data["intersections_geojson"].get("features", [])
    print(f"Drainage streams: {len(drainage_features)}, Lineaments: {len(lineament_features)}, Intersections: {len(inter_features)}")
    assert len(drainage_features) > 0
    assert len(lineament_features) > 0

    # 3. Test POST /api/export-kml (with HydroAnalysisResponse payload)
    print("\n--- 3. Testing POST /api/export-kml (Direct Export) ---")
    resp_kml = client.post("/api/export-kml", json=data)
    assert resp_kml.status_code == 200, f"Export-kml failed ({resp_kml.status_code}): {resp_kml.text}"
    assert "application/vnd.google-earth.kml+xml" in resp_kml.headers["content-type"]
    kml_text = resp_kml.text
    print(f"KML generated successfully ({len(kml_text)} bytes).")
    assert "<kml" in kml_text
    assert "<Document>" in kml_text
    assert "Target P1: Precision Borewell Location" in kml_text
    assert "Target P2: Precision Borewell Location" in kml_text
    assert "Target P3: Precision Borewell Location" in kml_text
    assert "ff00d7ff" in kml_text  # Yellow parcel line
    assert "ffffe500" in kml_text  # Cyan drainage line
    assert "ffffb000" in kml_text  # Electric-blue lineament line

    # 4. Test POST /api/export-kml (with ParcelRequest payload)
    print("\n--- 4. Testing POST /api/export-kml (On-the-fly Analysis & Export) ---")
    resp_kml2 = client.post("/api/export-kml", json=sample_parcel)
    assert resp_kml2.status_code == 200, f"Export-kml (on-the-fly) failed: {resp_kml2.text}"
    assert "application/vnd.google-earth.kml+xml" in resp_kml2.headers["content-type"]
    assert "<kml" in resp_kml2.text

    print("\n=======================================================")
    print("ALL TESTS PASSED WITH 200 OK AND VERIFIED INTEGRITY!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_full_pipeline()
