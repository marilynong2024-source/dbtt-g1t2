# DBTT-G1T2

IS215-style analytics for **Giant Singapore**: food vertical metrics in `food/cleaned_analysis.py`, household search/purchase summaries in `household_items/`, and a static **HTML dashboard** in `giant_dashboard_readable/`.

---

## Requirements

**Household** (`household_items/household_analysis.py`):

```bash
pip install pandas
```

**Food** (`food/cleaned_analysis.py`):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Python 3. Use `python3` on macOS if `python` is not on your PATH.

---

## How to run

### Food analytics

From the **`food/`** directory (paths are relative to there):

```bash
cd food
python3 cleaned_analysis.py
```

The script loads **`both_test/`**, prints data checks, then:

- Weekly demand vs rolling forecast (top 5 products) → `output/2_demand_forecast.png`
- Promotion vs no promotion (avg order total) → `output/3a_promo_overall.png`
- Top searches and searches by day → `output/4a_top_searches.png`, `output/4b_search_by_day.png`
- Recipe popularity and inferred orders by cuisine → `output/5a_recipe_popularity.png`, `output/5b_orders_by_cuisine.png`
- Console-only **most searched recipe** / **best recipe for bundle** (50% ingredient rules; see below)

After saving, an interactive **chart browser** opens (small **← / →** buttons and keyboard arrows) unless you disable it:

```bash
DBTT_NO_CHART_BROWSER=1 python3 cleaned_analysis.py
```

---

### Household analytics

From the **repository root**:

```bash
python3 household_items/household_analysis.py
```

Prints **top 3 searched** and **top 3 purchased** products (searches limited to known product names).

---

### HTML dashboard

Open in a browser:

`giant_dashboard_readable/giant_dashboard_readable.html`

Or serve the **repo root** and open (note the full filename):

```bash
cd /path/to/dbtt-g1t2
python3 -m http.server 8080
```

Then visit: **http://localhost:8080/giant_dashboard_readable/giant_dashboard_readable.html**

Most figures in the dashboard use **`giant_dashboard_readable/images/chart_*.png`**. Regenerate **`food/output/*.png`** with `cleaned_analysis.py` if you wire those paths in HTML yourself.

---

## Repository layout

```
DBTT-G1T2/
├── food/
│   ├── cleaned_analysis.py    # Demand, promos, search, recipes + chart browser
│   ├── output/                # PNG charts (created on run)
│   └── both_test/             # Fixture CSVs (orders, search, recipes, …)
├── household_items/
│   ├── household_analysis.py
│   └── *.csv
└── giant_dashboard_readable/
    ├── giant_dashboard.html
    └── images/                # Dashboard chart assets
```

---

## Recipe matching rules (`food/cleaned_analysis.py`)

Used by **`most_searched_recipe`** and **`most_purchased_recipe`** at the end of the script:

| Match type | Rule |
|------------|------|
| **Search** | Count a user for a recipe if they searched the recipe name, or if ≥50% of that recipe’s ingredients appear in their search terms. |
| **Order** | Count an order for a recipe if ≥50% of the recipe’s ingredients appear as `product_name` in that order’s line items. |
