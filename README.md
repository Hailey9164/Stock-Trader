# Stock Trader
CS 4200 final project

# Project Checklist

Check each box when the task is done, then initial and date it. Each step ends with a sign-off line that a second teammate initials after reviewing the work.

## Team

| Name | Initials |
|---|---|
| Hailey Campebll | HC |
| Matthew Sychareun| MS |
| | |

## Module layout

| File | Contents | Origin |
|---|---|---|
| `data.py` | `load_prices`, `load_news`, `data_info` | A2 plus new |
| `features.py` | features, forward-return label, 3-way split with embargo, normalization | A2 modified |
| `models.py` | baseline, RF regressor, PyTorch ANN with per-epoch loss callback | new |
| `sentiment.py` | LLM scorer with the JSON-or-fallback-to-0 logic | new |
| `portfolio.py` | rank, combine score, allocate | A2 modified |
| `planner.py` | BFS port | A1 ported |
| `explain.py` | prompt builder, LLM call, template fallback | new |
| `pipeline.py` | runs stages in order and reports progress via callbacks | new |
| `gui.py` | Tkinter app | new |

Keep `pipeline.py` free of Tkinter code, so you can test the whole thing from the command line before the GUI exists.

---

## Step 1: Port and sanity-check (day 1-2, parallel)

- [X] Port the Java planner to `planner.py` — Initials: HC Date: 11/22
- [X] Test the planner on a small case, such as the one in the walkthrough — Initials: HC Date: 11/22
- [X] Copy `core.py` into `data.py` and `features.py` — Initials: HC Date: 11/22
- [X] Switch to the continuous forward-return label — Initials: MS Date: 11/24
- [X] Add the 3-way split with embargo — Initials: MS Date: 11/24
- [X] Add training-only normalization — Initials: MS Date: 11/24
- [X] Print the date ranges and the means and stds — Initials: HC Date: 11/24
- [X] Write the sample news file — Initials: HC Date: 11/24
- [X] Write the LLM scorer using the Qwen setup from the lab notebook — Initials: HC Date: 11/24
- [X] Return `(0.0, "offline")` on any failure — Initials: HC Date: 11/24

**Step 1 sign-off (reviewer):** Initials: HC Date: 11/24

---

## Step 2: Models (day 3-4)

- [X] Build the baseline — Initials: HC Date: 11/25
- [X] Build the forest regressor — Initials: HC Date: 11/25
- [X] Build the ANN (32,16 hidden, ReLU, linear output, MSE loss, Adam at lr 0.001, early stopping on validation) — Initials: HC Date: 11/25
- [ ] Emit per-epoch train and validation loss — Initials: ____ Date: ______
- [ ] Evaluate once on test: baseline, forest, and ANN with MAE/RMSE and direction accuracy — Initials: ____ Date: ______
- [ ] Run with and without sentiment (zeroed) and keep both results for the write-up — Initials: ____ Date: ______

**Step 2 sign-off (reviewer):** Initials: ____ Date: ______

---

## Step 3: Rank, allocate, plan (day 4-5)

- [ ] Compute the final score as predicted return plus weight × sentiment — Initials: ____ Date: ______
- [ ] Pick the top N — Initials: ____ Date: ______
- [ ] Allocate with equal dollars and whole shares — Initials: ____ Date: ______
- [ ] Build the goal map with zeros for unpicked holdings — Initials: ____ Date: ______
- [ ] Call the planner and compute the running cash balance for each trade — Initials: ____ Date: ______
- [ ] Write the explainer prompt from computed values only — Initials: ____ Date: ______
- [ ] Write the template fallback for LLM-off mode — Initials: ____ Date: ______

**Step 3 sign-off (reviewer):** Initials: ____ Date: ______

---

## Step 4: `pipeline.py` (day 5)

- [ ] Chain every stage in order, with progress callbacks — Initials: ____ Date: ______
- [ ] Confirm `pipeline.py` has no Tkinter code — Initials: ____ Date: ______
- [ ] Test from the command line with the LLM on — Initials: ____ Date: ______
- [ ] Test from the command line with the LLM off — Initials: ____ Date: ______

**Step 4 sign-off (reviewer):** Initials: ____ Date: ______

---

## Step 5: Tkinter GUI (day 5-7)

**Layout**
- [ ] `ttk.Notebook` tabs or panels for inputs, data and split info, training, model comparison, per-stock predictions, portfolio and trades, and explanation — Initials: ____ Date: ______
- [ ] `ttk.Treeview` for the tables — Initials: ____ Date: ______
- [ ] matplotlib `FigureCanvasTkAgg` for the loss curves — Initials: ____ Date: ______
- [ ] `ScrolledText` for the prompt and explanation side by side — Initials: ____ Date: ______

**Threading**
- [ ] Run the pipeline in a background thread — Initials: ____ Date: ______
- [ ] Worker puts progress and results on a `queue.Queue` — Initials: ____ Date: ______
- [ ] Main thread polls with `root.after(100, ...)` — Initials: ____ Date: ______
- [ ] No widgets are touched from the worker thread — Initials: ____ Date: ______

**Required inputs**
- [ ] File loaders (price history and news) — Initials: ____ Date: ______
- [ ] Split dates — Initials: ____ Date: ______
- [ ] Hidden sizes — Initials: ____ Date: ______
- [ ] Learning rate — Initials: ____ Date: ______
- [ ] Epochs — Initials: ____ Date: ______
- [ ] Sentiment weight — Initials: ____ Date: ______
- [ ] Number of stocks to pick (N) — Initials: ____ Date: ______
- [ ] Cash — Initials: ____ Date: ______
- [ ] Trading fee — Initials: ____ Date: ______
- [ ] LLM on/off toggle — Initials: ____ Date: ______

**Required outputs**
- [ ] Model comparison table — Initials: ____ Date: ______
- [ ] Per-stock predictions with sentiment scores and reasons — Initials: ____ Date: ______
- [ ] Goal portfolio — Initials: ____ Date: ______
- [ ] Planned trade sequence with total fee — Initials: ____ Date: ______
- [ ] Written explanation next to its raw prompt — Initials: ____ Date: ______

**Step 5 sign-off (reviewer):** Initials: ____ Date: ______

---

## Step 6: Verification and write-up (day 7+)

**Leakage checklist**
- [ ] Normalization uses training rows only — Initials: ____ Date: ______
- [ ] Split is by date, not random — Initials: ____ Date: ______
- [ ] Embargo applied on both boundaries — Initials: ____ Date: ______
- [ ] No future headlines (asserted in code) — Initials: ____ Date: ______
- [ ] Test set touched only once — Initials: ____ Date: ______

**Other checks**
- [ ] Every number in the explanation appears in the displayed prompt — Initials: ____ Date: ______
- [ ] App runs end to end with the LLM off — Initials: ____ Date: ______

**Write-up**
- [ ] Model comparison — Initials: ____ Date: ______
- [ ] Sentiment ablation (with vs. zeroed) — Initials: ____ Date: ______
- [ ] Any numbers the LLM invented — Initials: ____ Date: ______
- [ ] README with run instructions — Initials: ____ Date: ______

**Step 6 sign-off (reviewer):** Initials: ____ Date: ______

---

## Final sign-off

All steps complete and reviewed.

| Name | Initials | Date |
|---|---|---|
| | | |
| | | |
| | | |
