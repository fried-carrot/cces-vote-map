# Virginia tract-level augmentation: fetch real ACS 2023 5-year data for all VA
# census tracts (Census Reporter mirror of the Census Bureau API, no key needed),
# score each tract through the CCES-trained logistic model in model_export.js via
# post-stratification, build the tract adjacency graph for routing, and export
# everything to va_data.js for the 3D dashboard (virginia.html).
#
# Run: python3 va_census_export.py     (needs model_export.js + cd118_shp/ present)

import json
import math
import time

import numpy as np
import requests
import shapefile  # pyshp

STATE = "04000US51"
TABLES = ["B01001", "B01002", "B03002", "B05001", "B11001", "B11005",
          "B12001", "B15003", "B19001", "B19013", "B23025", "B25003"]
CR = "https://api.censusreporter.org/1.0"

# ---- load the trained CCES model ----------------------------------------------
src = open("model_export.js").read()
M = json.loads(src[src.index("{"): src.rindex("}") + 1])
COEF, VOCAB = M["coef"], M["vocab"]

def fetch(url, tries=4):
    for a in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": "cces-vote-map/1.0"}, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if a == tries - 1:
                raise
            print(f"  retry {a+1} ({e})", flush=True)
            time.sleep(3 * (a + 1))

# ---- 1. ACS tables for every VA tract -----------------------------------------
acs = {}
for t in TABLES:
    print(f"Fetching {t} ...", flush=True)
    d = fetch(f"{CR}/data/show/latest?table_ids={t}&geo_ids=140|{STATE}")
    for geoid, tbl in d["data"].items():
        acs.setdefault(geoid, {}).update(tbl[t]["estimate"])
print(f"ACS rows: {len(acs):,} tracts", flush=True)

# ---- 2. tract geometry ---------------------------------------------------------
print("Fetching tract geometry ...", flush=True)
geo = fetch(f"{CR}/geo/show/tiger2023?geo_ids=140|{STATE}")
feats = [f for f in geo["features"] if f["properties"]["geoid"] in acs]
print(f"Geometry: {len(feats):,} tracts", flush=True)

# ---- 3. congressional district per tract (centroid in cd118 polygon) ----------
sf = shapefile.Reader("cd118_shp/cb_2023_us_cd118_5m")
fld = [f[0] for f in sf.fields[1:]]
cds = []
for sr in sf.iterShapeRecords():
    rec = dict(zip(fld, sr.record))
    if rec["STATEFP"] == "51":
        pts = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]
        rings = [pts[parts[k]:parts[k + 1]] for k in range(len(parts) - 1)]
        cds.append((f"VA-{int(rec['CD118FP']):02d}", rings))
print(f"VA districts in shapefile: {len(cds)}", flush=True)

def in_ring(x, y, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]; xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside

def cd_of(x, y):
    for cd, rings in cds:
        # even-odd across all rings handles holes
        if sum(in_ring(x, y, r) for r in rings) % 2 == 1:
            return cd
    return None

def rep_pt(geom):
    ring = geom["coordinates"][0] if geom["type"] == "Polygon" else \
        max(geom["coordinates"], key=lambda p: len(p[0]))[0]
    xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# ---- 4. per-tract feature shares from ACS --------------------------------------
def g(row, tid, col):  # e.g. g(row,"B15003",22) -> row["B15003022"]
    v = row.get(f"{tid}{col:03d}")
    return 0.0 if v is None else float(v)

def norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else {}

# adult age bands from B01001 (male cols 007-025, female +24), representative ages
AGE_BANDS = [("18-29", 24, [7, 8, 9, 10, 11]), ("30-44", 37, [12, 13, 14]),
             ("45-64", 55, [15, 16, 17, 18, 19]), ("65+", 73, [20, 21, 22, 23, 24, 25])]
# B19001 bins 002..017 -> CCES faminc ordinal 0..11
INC_ORD = {2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 4,
           11: 5, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 17: 11}
# coarse income cells: (label, representative ordinal, B19001 cols)
INC_CELLS = [("lo", 1.5, [2, 3, 4, 5, 6]), ("mid", 5.0, [7, 8, 9, 10, 11]),
             ("up", 8.5, [12, 13, 14]), ("hi", 11.0, [15, 16, 17])]

def tract_features(row):
    f = {}
    f["age"] = norm({b: sum(g(row, "B01001", c) + g(row, "B01001", c + 24) for c in cols)
                     for b, _, cols in AGE_BANDS})
    tot = g(row, "B01001", 1)
    f["gender"] = norm({"Male": g(row, "B01001", 2), "Female": g(row, "B01001", 26)})
    f["race"] = norm({
        "White": g(row, "B03002", 3), "Black": g(row, "B03002", 4),
        "Native American": g(row, "B03002", 5), "Asian": g(row, "B03002", 6),
        "Other": g(row, "B03002", 7) + g(row, "B03002", 8),
        "Mixed": g(row, "B03002", 9), "Hispanic": g(row, "B03002", 12)})
    f["educ"] = norm({
        "No HS": sum(g(row, "B15003", c) for c in range(2, 17)),
        "High School Graduate": g(row, "B15003", 17) + g(row, "B15003", 18),
        "Some College": g(row, "B15003", 19) + g(row, "B15003", 20),
        "2-Year": g(row, "B15003", 21), "4-Year": g(row, "B15003", 22),
        "Post-Grad": g(row, "B15003", 23) + g(row, "B15003", 24) + g(row, "B15003", 25)})
    f["inc"] = norm({lbl: sum(g(row, "B19001", c) for c in cols)
                     for lbl, _, cols in INC_CELLS})
    f["ownhome"] = norm({"Own": g(row, "B25003", 2), "Rent": g(row, "B25003", 3)})
    f["marstat"] = norm({
        "Single / Never Married": g(row, "B12001", 3) + g(row, "B12001", 12),
        "Married": g(row, "B12001", 4) + g(row, "B12001", 13),
        "Widowed": g(row, "B12001", 9) + g(row, "B12001", 18),
        "Divorced": g(row, "B12001", 10) + g(row, "B12001", 19)})
    lf = g(row, "B23025", 1)
    f["employ"] = norm({"Full-Time": g(row, "B23025", 4),
                        "Unemployed": g(row, "B23025", 5),
                        "Retired": g(row, "B23025", 7)}) if lf else {}
    f["citizen"] = norm({"Citizen": g(row, "B05001", 1) - g(row, "B05001", 6),
                         "Non-Citizen": g(row, "B05001", 6)})
    f["has_child"] = norm({"Yes": g(row, "B11005", 2),
                           "No": g(row, "B11005", 1) - g(row, "B11005", 2)})
    return f, tot

# ---- 5. post-stratified P(Republican) through the CCES model --------------------
# Cells over the strongest interacting dims (race x educ x age x income x gender),
# independence assumption within tract. Remaining observable features enter E[z]
# linearly as share-weighted coefficients (exact for the linear index). Features
# the census cannot observe (religion, union, born-again) use the model's own
# "Unknown" levels rather than invented values.
UNKNOWNS = ["religion=Unknown", "union=Unknown",
            "relig_imp=Unknown", "relig_bornagain=Unknown"]

def soft(feature, shares):
    return sum(p * COEF.get(f"{feature}={lvl}", 0.0) for lvl, p in shares.items())

def tract_pR(f, lean):
    z0 = M["intercept"]
    z0 += COEF["dist_lean"] * (lean - M["place_mean"]) / M["place_std"]
    z0 += sum(COEF.get(k, 0.0) for k in UNKNOWNS)
    for feat in ["ownhome", "marstat", "employ", "citizen", "has_child"]:
        z0 += soft(feat, f[feat])
    num = 0.0
    den = 0.0
    for race, pr in f["race"].items():
        zr = z0 + COEF.get(f"race={race}", 0.0) \
            + COEF.get("hispanic=Yes" if race == "Hispanic" else "hispanic=No", 0.0)
        for educ, pe in f["educ"].items():
            ze = zr + COEF.get(f"educ={educ}", 0.0)
            for (band, age_rep, _), pa in zip(AGE_BANDS, [f["age"].get(b, 0) for b, _, _ in AGE_BANDS]):
                za = ze + COEF["age"] * (age_rep - M["num_mean"]["age"]) / M["num_std"]["age"]
                for (lbl, ord_rep, _), pi in zip(INC_CELLS, [f["inc"].get(l, 0) for l, _, _ in INC_CELLS]):
                    zi = za + COEF["faminc_ord"] * \
                        (ord_rep - M["num_mean"]["faminc_ord"]) / M["num_std"]["faminc_ord"]
                    for gender, pg in f["gender"].items():
                        w = pr * pe * pa * pi * pg
                        if w > 0:
                            num += w / (1 + math.exp(-(zi + COEF.get(f"gender={gender}", 0.0))))
                            den += w
    return 1 - (num / den) if den > 0 else None  # P(Republican two-party)

# ---- 6. assemble tracts ---------------------------------------------------------
tracts = []
skipped = 0
for ft in feats:
    p = ft["properties"]
    geoid = p["geoid"]
    row = acs[geoid]
    f, pop = tract_features(row)
    hh = g(row, "B11001", 1)
    if pop < 50 or hh < 20 or not f["race"] or not f["educ"]:
        skipped += 1
        continue
    cx, cy = rep_pt(ft["geometry"])
    cd = cd_of(cx, cy)
    if cd is None:
        skipped += 1
        continue
    lean = M["dist_lean"].get(cd, M["state_lean"].get("Virginia", M["national_dem"]))
    pR = tract_pR(f, lean)
    if pR is None:
        skipped += 1
        continue
    name_parts = p["name"].split(", ")
    vap = sum(g(row, "B01001", c) + g(row, "B01001", c + 24) for c in range(7, 26))
    aland = float(p["aland"] or 1)
    tracts.append({
        "id": geoid[7:],                      # 51XXXYYYYYY fips
        "name": name_parts[0],
        "county": name_parts[1] if len(name_parts) > 1 else "",
        "cd": cd, "c": [round(cx, 5), round(cy, 5)],
        "pop": int(pop), "vap": int(vap), "hh": int(hh),
        "mi": row.get("B19013001"), "ma": row.get("B01002001"),
        "wh": round(f["race"].get("White", 0), 3),
        "ba": round(f["educ"].get("4-Year", 0) + f["educ"].get("Post-Grad", 0), 3),
        "own": round(f["ownhome"].get("Own", 0), 3),
        "dens": round(pop / (aland / 2.59e6), 1),   # people per sq mile
        "pR": round(pR, 4),
        "geom": ft["geometry"],
    })
print(f"Tracts kept: {len(tracts):,}  skipped: {skipped}", flush=True)
pRs = np.array([t["pR"] for t in tracts])
print(f"pR mean {pRs.mean():.3f}  min {pRs.min():.3f}  max {pRs.max():.3f}", flush=True)
for cd in sorted({t['cd'] for t in tracts}):
    sub = [t["pR"] for t in tracts if t["cd"] == cd]
    print(f"  {cd}: n={len(sub):4d}  mean pR={np.mean(sub):.3f}")

# ---- 7. adjacency graph (shared TIGER vertices) ---------------------------------
def rings_of(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    return [r for poly in geom["coordinates"] for r in poly]

vert = {}
for i, t in enumerate(tracts):
    for ring in rings_of(t["geom"]):
        for x, y in ring:
            vert.setdefault((round(x, 6), round(y, 6)), set()).add(i)

pair_count = {}
for members in vert.values():
    ms = sorted(members)
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            k = (ms[a], ms[b])
            pair_count[k] = pair_count.get(k, 0) + 1

def hav_mi(a, b):
    R = 3958.8
    la1, la2 = math.radians(a[1]), math.radians(b[1])
    dla = la2 - la1
    dlo = math.radians(b[0] - a[0])
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

edges = [[a, b, round(hav_mi(tracts[a]["c"], tracts[b]["c"]), 3)]
         for (a, b), n in pair_count.items() if n >= 2]
print(f"Adjacency edges: {len(edges):,}", flush=True)

# connect stray components (islands) to nearest neighbor so routing never strands
parent = list(range(len(tracts)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
for a, b, _ in edges:
    parent[find(a)] = find(b)
from collections import defaultdict
comps = defaultdict(list)
for i in range(len(tracts)):
    comps[find(i)].append(i)
comps = sorted(comps.values(), key=len, reverse=True)
print(f"Components: {len(comps)} (largest {len(comps[0])})", flush=True)
main = comps[0]
for comp in comps[1:]:
    best = None
    for i in comp:
        for j in main:
            d = hav_mi(tracts[i]["c"], tracts[j]["c"])
            if best is None or d < best[2]:
                best = [i, j, d]
    edges.append([best[0], best[1], round(best[2], 3)])
    main = main + comp

# ---- 8. export ------------------------------------------------------------------
def quant(geom):
    def q(ring):
        return [[round(x, 5), round(y, 5)] for x, y in ring]
    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [q(r) for r in geom["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[q(r) for r in poly] for poly in geom["coordinates"]]}

out = {
    "meta": {
        "source": "ACS 2023 5-year (Census Bureau via Census Reporter), TIGER 2023 tracts",
        "model": "CCES 2006-2019 logistic (model_export.js), post-stratified over "
                 "race x educ x age x income x gender cells per tract",
        "tables": TABLES,
        "state_lean_va": M["state_lean"].get("Virginia"),
    },
    "tracts": [{**{k: v for k, v in t.items() if k != "geom"}, "poly": quant(t["geom"])}
               for t in tracts],
    "adj": edges,
}
with open("va_data.js", "w") as fjs:
    fjs.write("window.VA = " + json.dumps(out, separators=(",", ":")) + ";\n")
import os
print(f"Wrote va_data.js ({os.path.getsize('va_data.js')/1e6:.1f} MB)")
