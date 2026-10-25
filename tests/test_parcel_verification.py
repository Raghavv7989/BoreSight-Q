"""
Verification test simulating a small user-drawn parcel around (13.4350, 77.7275).
Verifies:
1. len(resp['borewell_points']) == 3 (P1, P2, P3) and every point's (lon, lat) lies strictly inside the user's polygon.
2. gwpi_heatmap_base64, sar_moisture_base64, and insar_heatmap_base64 are non-empty PNGs with non-zero alpha pixels.
"""

import base64
import io
import math
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app

def point_in_poly(lon, lat, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0][0], poly[0][1]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n][0], poly[i % n][1]
        if min(p1y, p2y) < lat <= max(p1y, p2y):
            if lon <= max(p1x, p2x):
                xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x if p1y != p2y else p1x
                if p1x == p2x or lon <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def test_small_user_parcel():
    client = TestClient(app)

    # User draws a compact ~400m x 400m parcel around (13.4350, 77.7275)
    small_parcel_coords = [
        [77.7255, 13.4330],
        [77.7295, 13.4330],
        [77.7295, 13.4370],
        [77.7255, 13.4370],
        [77.7255, 13.4330],
    ]

    payload = {
        "parcel_geometry": {
            "type": "Polygon",
            "coordinates": [small_parcel_coords]
        },
        "buffer_m": 500.0
    }

    print("\n--- Sending POST /api/analyze-parcel for small user parcel ---")
    resp = client.post("/api/analyze-parcel", json=payload)
    assert resp.status_code == 200, f"Analysis failed: {resp.text}"
    data = resp.json()

    # 1. Verify borewell_points: exactly 3 points (P1, P2, P3) strictly inside the parcel
    points = data.get("borewell_points", [])
    print(f"\n[1] Borewell Points count: {len(points)}")
    assert len(points) == 3, f"Expected exactly 3 points, got {len(points)}"

    expected_ids = ["P1", "P2", "P3"]
    for idx, pt in enumerate(points):
        pid = pt["id"]
        assert pid == expected_ids[idx], f"Expected point id {expected_ids[idx]}, got {pid}"
        lon = pt["lon"]
        lat = pt["lat"]
        score = pt["gwpi_score"]
        depth = pt["recommended_depth_m"]
        yield_lph = pt["estimated_yield_lph"]

        # Check strictly inside polygon
        is_inside = point_in_poly(lon, lat, small_parcel_coords)
        print(f"  [{pid}] Lon: {lon:.6f}, Lat: {lat:.6f}, GWPI: {score}, Depth: {depth:.1f}m, Yield: {yield_lph:.0f} LPH | Inside: {is_inside}")
        assert is_inside, f"Point {pid} ({lon}, {lat}) is NOT inside the user polygon!"
        assert score > 0, f"Point {pid} has invalid score: {score}"
        assert depth > 0, f"Point {pid} has invalid depth: {depth}"
        assert yield_lph > 0, f"Point {pid} has invalid yield: {yield_lph}"

    # 2. Verify gwpi_heatmap_base64, sar_moisture_base64, insar_heatmap_base64
    print("\n[2] Verifying Base64 PNGs and non-zero alpha pixels...")
    for key in ["gwpi_heatmap_base64", "sar_moisture_base64", "insar_heatmap_base64"]:
        b64_uri = data.get(key, "")
        assert b64_uri.startswith("data:image/png;base64,"), f"{key} missing data:image/png;base64, prefix"
        raw_b64 = b64_uri.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        assert len(img_bytes) > 500, f"{key} byte size {len(img_bytes)} is too small"

        img = Image.open(io.BytesIO(img_bytes))
        assert img.format == "PNG", f"{key} is not PNG format"
        assert img.mode == "RGBA", f"{key} mode is {img.mode}, expected RGBA"

        # Check alpha channel has non-zero pixels
        r, g, b, a = img.split()
        alpha_arr = list(a.getdata())
        max_alpha = max(alpha_arr)
        avg_alpha = sum(alpha_arr) / len(alpha_arr)
        print(f"  {key}: shape={img.size}, bytes={len(img_bytes)}, max_alpha={max_alpha}, avg_alpha={avg_alpha:.1f}")
        assert max_alpha >= 200, f"{key} max alpha {max_alpha} is unexpectedly low"
        assert avg_alpha > 50, f"{key} average alpha {avg_alpha} is unexpectedly low (mostly transparent)"

    print("\n=======================================================")
    print("ALL END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_small_user_parcel()
