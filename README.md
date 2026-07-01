# cces-vote-map

Predicts individual two-party US presidential vote from demographics and congressional-district political lean, visualized as an interactive district-level choropleth (436 districts, 118th Congress).

**Live:** https://fried-carrot.github.io/cces-vote-map/

**Data:** CCES Cumulative Common Content 2006–2024 (701k respondents, Harvard Dataverse). District boundaries from Census cartographic files.

**Model:** L2 logistic regression on demographics + district lean, hierarchically shrunk district → state → national (party ID excluded to avoid leakage; post-2022 districts fall back to state lean). Coefficients exported to JS and evaluated client-side. Temporal holdout (train ≤2019 / test 2020–2024): 74.7% accuracy, AUC 0.822.

**Two map modes:**
- *Actual respondents* — model-predicted D share averaged over each district's real 2024 respondents
- *Custom persona* — adjust 15 demographic sliders/dropdowns and see predictions recompute live across all districts

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
