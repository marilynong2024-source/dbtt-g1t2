# DBTT-G1T2

Analytics for the Giant Singapore digital transformation project (IS215): **recipe and customer analytics** in `food/`, and **household product** search vs. purchase summaries in `household_items/`.

---

## Requirements

**Household script** (`household_items/household_analysis.py`):

```bash
pip install pandas
```

**Food analytics** (`food/cleaned_analysis.py`) — Customer segmentation, Demand forecasting (rolling average), Promotion effectiveness analysis, Search trend analysis, Recipe popularity ranking



```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Python 3.

---

## How to run

### Food — full pipeline (`food/cleaned_analysis.py`)

Run from the **`food/`** directory (CSV paths are relative to there):

```bash
cd food
python cleaned_analysis.py
```

The script loads **`both_test/`** and runs:

- Data quality checks  
- Customer segmentation (K-Means)  
- Demand forecasting (rolling average)  
- Promotion effectiveness  
- Search trend analysis  
- Recipe popularity ranking  

Figures are written under **`food/output/`** (folder is created if missing).

**Closing block (three scenarios):** The file ends by looping `mains_test`, `dessert_test`, and `both_test` for “most searched recipe” and “best recipe for bundle” (50% ingredient coverage; direct recipe-name search counts for the search side). This repository currently includes only **`both_test/`**. To run without errors, either add `mains_test/` and `dessert_test/` with the same CSV layout, or change the `testcases` list near the bottom of `cleaned_analysis.py` to e.g. `["both_test"]` only.

---

### Household — top products

From the **repository root**:

```bash
python household_items/household_analysis.py
```

Prints **top 3 most searched** and **top 3 most purchased** products (search terms restricted to known product names).

---

## Repository layout

```
DBTT-G1T2/
├── food/
│   ├── cleaned_analysis.py   # Full IS215 analytics + recipe match loop
│   ├── output/               # PNG charts (created on run)
│   └── both_test/            # users, orders, order_items, search_history,
│                             # recipes, recipe_ingredients, recipe_ratings.csv
└── household_items/
    ├── household_analysis.py
    ├── users.csv
    ├── products.csv
    ├── orders.csv
    ├── order_items.csv
    └── search_history.csv
```

---

## Recipe matching rules (tail of `food/cleaned_analysis.py`)

| Match type | Rule |
|------------|------|
| **Search** | A user counts for a recipe if they searched the recipe name, or if ≥50% of that recipe’s ingredients appear among their search terms. |
| **Order** | An order counts for a recipe if ≥50% of the recipe’s ingredients appear as `product_name` in that order’s line items. |
