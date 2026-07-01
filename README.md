# cces-vote-map

Predicts individual two-party US presidential vote from demographics and state political lean, visualized as an interactive choropleth.

**Data:** CCES Cumulative Common Content 2006–2024 (701k respondents, Harvard Dataverse).

**Model:** L2 logistic regression trained on demographics + lagged state lean (party ID excluded to avoid leakage). Coefficients exported to JS and evaluated client-side. Temporal holdout (train ≤2019 / test 2020–2024): 74.5% accuracy, AUC 0.819.

**Two map modes:**
- *Actual respondents* — model-predicted D share averaged over each state's real 2024 respondents
- *Custom persona* — adjust 15 demographic sliders/dropdowns and see predictions recompute live across all states

## Run

```bash
python3 train_export.py   # writes model_export.js
python3 -m http.server 8777
```

Then open `http://localhost:8777`.

> `.dta` file not included (675 MB). Download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/II2DB6).
