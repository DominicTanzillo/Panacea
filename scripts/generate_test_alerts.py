"""Generate test latest_alerts.json with synthetic orbital trajectories.

Creates realistic-looking ECI trajectory data for each alert pair using
parametric circular orbits. This is for frontend testing; the real pipeline
uses SGP4 propagation.

Two data tiers:
  - trajectory: 20-min steps for full 5-day range (chart + slider)
  - trail: 15-sec steps for ±30 min around TCA (globe orbital path lines)
"""
import json
import math
import random
from datetime import datetime

EARTH_RADIUS_KM = 6371.0
MU = 398600.4418  # km^3/s^2

# Alert pairs from the existing static data
PAIRS = [
    {"name_1": "INTELSAT 10-02", "name_2": "MEV-2", "norad_1": 28358, "norad_2": 46113, "risk_score": 0.8712, "altitude_km": 35786.3, "miss_estimate_km": 2.8, "tca_hours": 18.3, "tca_min_distance_km": 0.4},
    {"name_1": "COSMOS 2251 DEB", "name_2": "FENGYUN 1C DEB", "norad_1": 34454, "norad_2": 31141, "risk_score": 0.7234, "altitude_km": 782.4, "miss_estimate_km": 5.1, "tca_hours": 42.7, "tca_min_distance_km": 1.8},
    {"name_1": "ONEWEB-0017", "name_2": "ONEWEB-0032", "norad_1": 45132, "norad_2": 45141, "risk_score": 0.5391, "altitude_km": 1221.7, "miss_estimate_km": 8.4, "tca_hours": 67.2, "tca_min_distance_km": 4.3},
    {"name_1": "STARLINK-1007", "name_2": "COSMOS 2251 DEB", "norad_1": 44713, "norad_2": 34501, "risk_score": 0.4875, "altitude_km": 550.1, "miss_estimate_km": 12.3, "tca_hours": 31.5, "tca_min_distance_km": 6.7},
    {"name_1": "ONEWEB-0033", "name_2": "ONEWEB-0040", "norad_1": 45142, "norad_2": 45147, "risk_score": 0.4102, "altitude_km": 1221.7, "miss_estimate_km": 15.8, "tca_hours": 88.4, "tca_min_distance_km": 8.9},
    {"name_1": "STARLINK-1130", "name_2": "IRIDIUM 33 DEB", "norad_1": 44914, "norad_2": 33776, "risk_score": 0.3847, "altitude_km": 548.7, "miss_estimate_km": 18.2, "tca_hours": 55.1, "tca_min_distance_km": 11.4},
    {"name_1": "ONEWEB-0035", "name_2": "ONEWEB-0049", "norad_1": 45143, "norad_2": 45154, "risk_score": 0.3215, "altitude_km": 1221.7, "miss_estimate_km": 22.7, "tca_hours": 96.3, "tca_min_distance_km": 15.2},
    {"name_1": "FENGYUN 1C DEB", "name_2": "SL-8 R/B", "norad_1": 30847, "norad_2": 12138, "risk_score": 0.2893, "altitude_km": 847.2, "miss_estimate_km": 28.5, "tca_hours": 72.8, "tca_min_distance_km": 18.6},
    {"name_1": "ONEWEB-0040", "name_2": "ONEWEB-0059", "norad_1": 45147, "norad_2": 45162, "risk_score": 0.2456, "altitude_km": 1221.7, "miss_estimate_km": 35.1, "tca_hours": 108.6, "tca_min_distance_km": 22.3},
    {"name_1": "STARLINK-1329", "name_2": "STARLINK-1437", "norad_1": 45360, "norad_2": 45539, "risk_score": 0.2108, "altitude_km": 550.3, "miss_estimate_km": 41.8, "tca_hours": 45.9, "tca_min_distance_km": 28.7},
    {"name_1": "ONEWEB-0043", "name_2": "ONEWEB-0706", "norad_1": 45149, "norad_2": 61611, "risk_score": 0.1834, "altitude_km": 1221.7, "miss_estimate_km": 48.3, "tca_hours": 115.2, "tca_min_distance_km": 34.1},
    {"name_1": "COSMOS 2251 DEB", "name_2": "SL-16 R/B", "norad_1": 34128, "norad_2": 22285, "risk_score": 0.1527, "altitude_km": 810.6, "miss_estimate_km": 55.7, "tca_hours": 82.4, "tca_min_distance_km": 39.5},
    {"name_1": "ONEWEB-0044", "name_2": "ONEWEB-0054", "norad_1": 45150, "norad_2": 45158, "risk_score": 0.1293, "altitude_km": 1221.7, "miss_estimate_km": 62.4, "tca_hours": 102.7, "tca_min_distance_km": 45.8},
    {"name_1": "STARLINK-1008", "name_2": "FENGYUN 1C DEB", "norad_1": 44714, "norad_2": 30960, "risk_score": 0.1085, "altitude_km": 551.2, "miss_estimate_km": 71.9, "tca_hours": 38.6, "tca_min_distance_km": 52.3},
    {"name_1": "ONEWEB-0047", "name_2": "ONEWEB-0415", "norad_1": 45152, "norad_2": 51624, "risk_score": 0.0912, "altitude_km": 1221.7, "miss_estimate_km": 85.3, "tca_hours": 91.8, "tca_min_distance_km": 61.2},
    {"name_1": "IRIDIUM 33 DEB", "name_2": "CZ-2C R/B", "norad_1": 33775, "norad_2": 26960, "risk_score": 0.0741, "altitude_km": 775.8, "miss_estimate_km": 98.6, "tca_hours": 64.3, "tca_min_distance_km": 72.8},
    {"name_1": "ONEWEB-0056", "name_2": "ONEWEB-0431", "norad_1": 45159, "norad_2": 51630, "risk_score": 0.0583, "altitude_km": 1221.7, "miss_estimate_km": 112.4, "tca_hours": 118.5, "tca_min_distance_km": 85.4},
    {"name_1": "STARLINK-1049", "name_2": "STARLINK-1176", "norad_1": 44757, "norad_2": 44961, "risk_score": 0.0428, "altitude_km": 549.8, "miss_estimate_km": 128.7, "tca_hours": 52.1, "tca_min_distance_km": 94.6},
    {"name_1": "ONEWEB-0062", "name_2": "ONEWEB-0701", "norad_1": 45163, "norad_2": 61607, "risk_score": 0.0315, "altitude_km": 1221.7, "miss_estimate_km": 145.2, "tca_hours": 105.9, "tca_min_distance_km": 108.3},
    {"name_1": "ONEWEB-0065", "name_2": "ONEWEB-0439", "norad_1": 45164, "norad_2": 51635, "risk_score": 0.0218, "altitude_km": 1221.7, "miss_estimate_km": 162.8, "tca_hours": 112.4, "tca_min_distance_km": 125.7},
]


def orbit_position(radius_km, inc_rad, raan_rad, ma_rad):
    """Compute ECI position from orbital elements (circular orbit)."""
    x_orb = radius_km * math.cos(ma_rad)
    y_orb = radius_km * math.sin(ma_rad)

    ci, si = math.cos(inc_rad), math.sin(inc_rad)
    cr, sr = math.cos(raan_rad), math.sin(raan_rad)

    x_eci = cr * x_orb - sr * ci * y_orb
    y_eci = sr * x_orb + cr * ci * y_orb
    z_eci = si * y_orb

    return (round(x_eci, 1), round(y_eci, 1), round(z_eci, 1))


def make_orbit_params(pair):
    """Generate deterministic orbital parameters for a pair."""
    alt = pair["altitude_km"]
    R = EARTH_RADIUS_KM + alt
    tca_hours = pair["tca_hours"]
    min_dist = pair["tca_min_distance_km"]

    random.seed(pair["norad_1"] + pair["norad_2"])

    # Inclination/RAAN based on satellite type
    if alt > 30000:
        inc1 = math.radians(random.uniform(0.01, 0.1))
        inc2 = math.radians(random.uniform(0.01, 0.1))
        raan1 = math.radians(random.uniform(0, 360))
        raan2 = raan1 + math.radians(random.uniform(-0.5, 0.5))
    elif "ONEWEB" in pair["name_1"]:
        inc1 = math.radians(87.9 + random.uniform(-0.5, 0.5))
        inc2 = math.radians(87.9 + random.uniform(-0.5, 0.5))
        raan1 = math.radians(random.uniform(0, 360))
        raan2 = raan1 + math.radians(random.uniform(-5, 5))
    elif "STARLINK" in pair["name_1"]:
        inc1 = math.radians(53.0 + random.uniform(-0.5, 0.5))
        inc2 = math.radians(random.uniform(50, 100))
        raan1 = math.radians(random.uniform(0, 360))
        raan2 = raan1 + math.radians(random.uniform(-15, 15))
    else:
        inc1 = math.radians(random.uniform(65, 100))
        inc2 = math.radians(random.uniform(65, 100))
        raan1 = math.radians(random.uniform(0, 360))
        raan2 = raan1 + math.radians(random.uniform(-20, 20))

    R1 = R + random.uniform(-2, 2)
    R2 = R + random.uniform(-2, 2)
    omega1 = 2 * math.pi / (2 * math.pi * math.sqrt(R1**3 / MU) / 60.0)
    omega2 = 2 * math.pi / (2 * math.pi * math.sqrt(R2**3 / MU) / 60.0)

    tca_min = tca_hours * 60.0
    ma1_at_tca = math.radians(random.uniform(0, 360))
    angular_offset = min_dist / R
    ma2_at_tca = ma1_at_tca + angular_offset

    ma1_start = ma1_at_tca - omega1 * tca_min
    ma2_start = ma2_at_tca - omega2 * tca_min

    return {
        "R1": R1, "R2": R2,
        "inc1": inc1, "inc2": inc2,
        "raan1": raan1, "raan2": raan2,
        "omega1": omega1, "omega2": omega2,
        "ma1_start": ma1_start, "ma2_start": ma2_start,
    }


def generate_sparse_trajectory(params, hours_forward=120.0, step_minutes=20.0):
    """Sparse trajectory for chart/slider: 20-min steps, 5 days."""
    n_steps = int(hours_forward * 60 / step_minutes) + 1
    points = []
    for i in range(n_steps):
        mins = i * step_minutes
        h = mins / 60.0

        ma1 = params["ma1_start"] + params["omega1"] * mins
        ma2 = params["ma2_start"] + params["omega2"] * mins

        p1 = orbit_position(params["R1"], params["inc1"], params["raan1"], ma1)
        p2 = orbit_position(params["R2"], params["inc2"], params["raan2"], ma2)

        dx, dy, dz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        points.append({
            "h": round(h, 2),
            "d": round(dist, 1),
            "s1": list(p1),
            "s2": list(p2),
        })
    return points


def generate_dense_trail(params, tca_hours, half_window_min=30.0, step_minutes=0.25):
    """Dense trail around TCA for globe orbital path lines: 15-sec steps, ±30 min."""
    tca_min = tca_hours * 60.0
    start_min = tca_min - half_window_min
    end_min = tca_min + half_window_min
    n_steps = int((end_min - start_min) / step_minutes) + 1

    trail = []
    for i in range(n_steps):
        mins = start_min + i * step_minutes

        ma1 = params["ma1_start"] + params["omega1"] * mins
        ma2 = params["ma2_start"] + params["omega2"] * mins

        p1 = orbit_position(params["R1"], params["inc1"], params["raan1"], ma1)
        p2 = orbit_position(params["R2"], params["inc2"], params["raan2"], ma2)

        dx, dy, dz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        trail.append({
            "h": round(mins / 60.0, 3),
            "d": round(dist, 1),
            "s1": list(p1),
            "s2": list(p2),
        })
    return trail


def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    pairs_out = []
    total_traj_pts = 0
    total_trail_pts = 0

    for pair in PAIRS:
        params = make_orbit_params(pair)
        trajectory = generate_sparse_trajectory(params)
        trail = generate_dense_trail(params, pair["tca_hours"])

        # Recompute TCA from the actual trajectory
        min_idx = min(range(len(trajectory)), key=lambda i: trajectory[i]["d"])
        actual_tca_hours = trajectory[min_idx]["h"]
        actual_min_dist = trajectory[min_idx]["d"]

        total_traj_pts += len(trajectory)
        total_trail_pts += len(trail)

        pair_out = {
            "name_1": pair["name_1"],
            "name_2": pair["name_2"],
            "norad_1": pair["norad_1"],
            "norad_2": pair["norad_2"],
            "risk_score": pair["risk_score"],
            "altitude_km": pair["altitude_km"],
            "miss_estimate_km": pair["miss_estimate_km"],
            "prediction_date": today_str,
            "tca_hours": round(actual_tca_hours, 1),
            "tca_min_distance_km": round(actual_min_dist, 1),
            "trajectory": trajectory,
            "trail": trail,
        }
        pairs_out.append(pair_out)
        print(f"  {pair['name_1']:20s} vs {pair['name_2']:20s} — TCA {actual_tca_hours:.1f}h, min {actual_min_dist:.1f} km, {len(trajectory)} traj + {len(trail)} trail pts")

    alerts = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prediction_date": today_str,
        "pairs": pairs_out,
    }

    out_path = "webapp-react/public/latest_alerts.json"
    with open(out_path, "w") as f:
        json.dump(alerts, f)  # No indent to save space

    size_kb = len(json.dumps(alerts)) / 1024
    print(f"\n  Total: {total_traj_pts} trajectory + {total_trail_pts} trail points")
    print(f"  Wrote {len(pairs_out)} alerts to {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
