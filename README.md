# cces-vote-map

Predicts individual two-party US presidential vote from demographics and congressional-district political lean, visualized as an interactive district-level choropleth (436 districts, 118th Congress).

**Live:** https://fried-carrot.github.io/cces-vote-map/

**Data:** CCES Cumulative Common Content 2006–2024 (701k respondents, Harvard Dataverse). District boundaries from Census cartographic files.

**Model:** L2 logistic regression on demographics + district lean, hierarchically shrunk district → state → national (party ID excluded to avoid leakage; post-2022 districts fall back to state lean). Coefficients exported to JS and evaluated client-side. Temporal holdout (train ≤2019 / test 2020–2024): 74.7% accuracy, AUC 0.822.

**Two map modes:**
- *Actual respondents* — model-predicted D share averaged over each district's real 2024 respondents
- *Custom persona* — adjust 15 demographic sliders/dropdowns and see predictions recompute live across all districts

## Virginia Field Ops (`virginia.html`)

Tract-level Republican canvassing planner for Virginia, built on the same trained model:

- **Data augmentation** — `va_census_export.py` pulls 12 real ACS 2023 5-year tables
  (B01001, B03002, B15003, B19001, B25003, ...) for all 2,198 VA census tracts plus
  TIGER 2023 tract geometry (Census Bureau data via the Census Reporter API, no key
  required), assigns each tract to its congressional district by point-in-polygon
  against the cd118 shapefile, and writes `va_data.js`.
- **Scoring** — the CCES logistic model is post-stratified over
  race × education × age × income × sex cells per tract (independence within tract;
  features the census cannot observe use the model's own "Unknown" levels). Output:
  tract-level P(Republican two-party vote).
- **Platform conservatism index** — a 0–100 gauge. Responsivity =
  P(R) × Gaussian ideological match (σ=25) between the platform and the tract.
  Moving the slider re-scores all tracts and re-optimizes live routes. The kernel is
  a stylized spatial-voting assumption, not measured ad response.
- **Routes** — team-orienteering door-knocking routes on the tract adjacency graph
  (shared TIGER boundary vertices): Dijkstra shortest paths, round-robin greedy
  yield-per-mile insertion, 2-opt improvement. Configurable walkers, per-walker
  mileage budget, and click-to-set HQ.
- **Canvass feedback loop (the moat)** — logged door results update tract scores as a
  beta-binomial posterior over the model prior (N0=30 pseudo-doors), with calibration
  readout, export/import, and localStorage persistence. The public data is only the
  prior; the calibrated posterior state accrues inside the tool. In a production SaaS
  this state plus the scoring itself would live server-side — client-side data can
  always be copied.

Rebuild the data: `python3 va_census_export.py` (needs `cd118_shp/` unzipped from
`cd118.zip` and `model_export.js` present). Ecological-inference caveat applies:
tract scores say nothing about any individual household.

## Run locally

```bash
python3 train_export.py   # writes model_export.js
python3 -m http.server 8777
```

Then open `http://localhost:8777`.

## Deploy

Static site, no build step. GitHub Pages serves `index.html` + `model_export.js` +
`districts.js` from the `main` branch root (`.nojekyll` present). Push to `main` and
Pages redeploys. To regenerate the model artifact, rerun `train_export.py` and commit
the updated `model_export.js`.

> `.dta` file not included (675 MB). Download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/II2DB6).
