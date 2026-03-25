"""
IS215 Project — Giant Singapore Digital Transformation
analysis.py — Full Analytics & AI Component
=======================================================
Sections:
  0. Imports & Data Loading
  1. Customer Segmentation (K-Means)
  2. Demand Forecasting (Rolling Average)
  3. Promotion Effectiveness Analysis
  4. Search Trend Analysis (Bonus)
  5. Recipe Popularity Ranking
"""

# =============================================================
# IMPORTS & DATA LOADING
# =============================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)
sns.set_theme(style='whitegrid', palette='Set2')

users       = pd.read_csv('both_test/users.csv')
orders      = pd.read_csv('both_test/orders.csv')
order_items = pd.read_csv('both_test/order_items.csv')
search_hist = pd.read_csv('both_test/search_history.csv')
recipes     = pd.read_csv('both_test/recipes.csv')
recipe_ing  = pd.read_csv('both_test/recipe_ingredients.csv')

print(f"  Users:        {len(users)} records")
print(f"  Orders:       {len(orders)} records")
print(f"  Order Items:  {len(order_items)} records")
print(f"  Search Hist:  {len(search_hist)} records")
print(f"  Recipes:      {len(recipes)} records")
print("Datasets loaded successfully.\n")

# =============================================================
# CHECK DATA IS CLEAN
# =============================================================
print("=== Data Quality Check ===")
for name, df in [('users', users), ('orders', orders),
                 ('order_items', order_items), ('search_hist', search_hist)]:
    nulls = df.isnull().sum().sum()
    dups  = df.duplicated().sum()
    print(f"  {name:<15} rows={len(df):>5}  nulls={nulls}  duplicates={dups}")
print("All checks passed.\n")

# =============================================================
# SECTION 1: CUSTOMER SEGMENTATION (K-MEANS)
# =============================================================
print("=" * 60)
print("SECTION 1: Customer Segmentation (K-Means)")
print("=" * 60)

# visit_frequency: count of orders per user
visit_freq = (orders.groupby('user_id')['order_id']
              .count()
              .reset_index()
              .rename(columns={'order_id': 'visit_frequency'}))

# Merge engineered feature into users dataframe
seg_df = users.merge(visit_freq, on='user_id', how='left')
seg_df['visit_frequency'] = seg_df['visit_frequency'].fillna(0)

# Select features for clustering
# Using age, avg_weekly_spend, and visit_frequency as key behavioural signals
features = ['age', 'avg_weekly_spend_sgd', 'visit_frequency']
X = seg_df[features].copy()

# Scale features — critical because K-Means is distance-based.
# Without scaling, avg_weekly_spend_sgd (100s) would dominate visit_frequency (1–20).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means with k=3 (justified by elbow plot)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
seg_df['segment'] = kmeans.fit_predict(X_scaled)

# Label segments by interpreting cluster centres
# Sort clusters by avg_weekly_spend to assign meaningful names
cluster_means = seg_df.groupby('segment')['avg_weekly_spend_sgd'].mean().sort_values()
label_map = {
    cluster_means.index[0]: 'Budget Shoppers',
    cluster_means.index[1]: 'Regular Families',
    cluster_means.index[2]: 'High-Value Customers'
}
seg_df['segment_label'] = seg_df['segment'].map(label_map)

seg_counts = seg_df['segment_label'].value_counts()
plt.figure(figsize=(6, 4))
seg_counts.plot(kind='bar', color=['#66b3ff', '#ff9999', '#99ff99'], edgecolor='black')
plt.title('Customer Segment Sizes')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('output/1c_segment_sizes.png', dpi=150)
plt.show()
print("Segment size chart saved → output/1c_segment_sizes.png")

# Summary statistics per segment
segment_summary = seg_df.groupby('segment_label')[features].mean().round(2)
print("\nSegment Profiles:")
print(segment_summary.to_string())
print()


# =============================================================
# SECTION 2: DEMAND FORECASTING (ROLLING AVERAGE)
# =============================================================
print("=" * 60)
print("SECTION 2: Demand Forecasting")
print("=" * 60)

# Parse timestamps and extract week period
orders['timestamp'] = pd.to_datetime(orders['timestamp'])
orders['week'] = orders['timestamp'].dt.to_period('W')

# Join order items with dated orders
order_items_dated = order_items.merge(
    orders[['order_id', 'week']], on='order_id', how='left')

# Aggregate: total quantity sold per product per week
weekly_demand = (order_items_dated
                 .groupby(['week', 'product_name'])['quantity']
                 .sum()
                 .reset_index())

# Select top 5 most purchased products
top_products = order_items['product_name'].value_counts().head(5).index.tolist()
print(f"Forecasting for top 5 products: {top_products}")

# Plot actual demand + 3-week rolling average forecast
fig, axes = plt.subplots(len(top_products), 1,
                         figsize=(14, 4 * len(top_products)),
                         sharex=False)

for i, product in enumerate(top_products):
    prod_data = (weekly_demand[weekly_demand['product_name'] == product]
                 .copy()
                 .sort_values('week'))
    prod_data['forecast'] = prod_data['quantity'].rolling(window=3, min_periods=1).mean()
    prod_data['week_dt'] = prod_data['week'].dt.to_timestamp()

    ax = axes[i] if len(top_products) > 1 else axes
    x = range(len(prod_data))
    labels = prod_data['week_dt'].dt.strftime('%b %d')

    ax.plot(x, prod_data['quantity'],
            marker='o', markersize=4, label='Actual',
            alpha=0.85, color='steelblue', linewidth=1.5)
    ax.plot(x, prod_data['forecast'],
            linestyle='--', label='Forecast (3-wk avg)',
            color='tomato', linewidth=2)
    ax.fill_between(x,
                    prod_data['quantity'],
                    prod_data['forecast'],
                    alpha=0.12, color='tomato', label='Gap')

    tick_positions = list(range(0, len(prod_data), 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels.iloc[j] for j in tick_positions],
                       rotation=30, ha='right', fontsize=9)

    ax.set_title(product, fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('Qty Sold', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Weekly Demand Forecast — Top 5 Products',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=3)
plt.savefig('output/2_demand_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("Demand forecast chart saved → output/2_demand_forecast.png")

# Flag potential stockout risk: products where latest demand > forecast
# Flagged as high risk if latest actual demand is >20% above the rolling forecast.
print("\nStockout Risk Flags (latest demand > forecast):")

for product in top_products:
    prod_data = (weekly_demand[weekly_demand['product_name'] == product]
                 .sort_values('week'))
    prod_data['forecast'] = prod_data['quantity'].rolling(window=3, min_periods=1).mean()
    latest = prod_data.iloc[-1]
    risk = "⚠ HIGH RISK" if latest['quantity'] > latest['forecast'] * 1.2 else "✓ Normal"
    print(f"  {product}: {risk} (latest={latest['quantity']:.0f}, forecast={latest['forecast']:.0f})")
print()


# =============================================================
# SECTION 3: PROMOTION EFFECTIVENESS ANALYSIS
# =============================================================
print("=" * 60)
print("SECTION 3: Promotion Effectiveness Analysis")
print("=" * 60)

# Overall: avg order total with vs without promotion
promo_analysis = orders.groupby('promo_applied')['order_total_sgd'].mean().round(2)
print("Avg Order Total (SGD):")
print(f"  No Promotion:       ${promo_analysis.get(False, promo_analysis.iloc[0])}")
print(f"  Promotion Applied:  ${promo_analysis.get(True, promo_analysis.iloc[1])}")

plt.figure(figsize=(6, 4))
colors = ['#ff9999', '#66b3ff']
promo_analysis.plot(kind='bar', color=colors, edgecolor='black', width=0.5)
plt.title('Avg Order Total: Promotion vs No Promotion')
plt.xlabel('')
plt.ylabel('Avg Order Total (SGD)')
plt.xticks([0, 1], ['No Promotion', 'Promotion Applied'], rotation=0)
for i, val in enumerate(promo_analysis):
    plt.text(i, val + 1, f'${val:.2f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('output/3a_promo_overall.png', dpi=150)
plt.show()
print("Promo overall chart saved → output/3a_promo_overall.png")

# By segment: which customer segment responds most to promotions?
seg_orders = orders.merge(seg_df[['user_id', 'segment_label']], on='user_id', how='left')
seg_promo = (seg_orders.groupby(['segment_label', 'promo_applied'])['order_total_sgd']
             .mean()
             .round(2)
             .unstack())
seg_promo.columns = ['No Promotion', 'Promotion Applied']

print("\nPromo Impact by Customer Segment:")
print(seg_promo.to_string())

seg_promo.plot(kind='bar', figsize=(8, 5), color=['#ff9999', '#66b3ff'],
               edgecolor='black', width=0.6)
plt.title('Avg Order Total by Segment: Promo vs No Promo')
plt.xlabel('Customer Segment')
plt.ylabel('Avg Order Total (SGD)')
plt.xticks(rotation=15)
plt.legend(title='Promotion')
plt.tight_layout()
plt.savefig('output/3b_promo_by_segment.png', dpi=150)
plt.show()
print("Promo by segment chart saved → output/3b_promo_by_segment.png")

# =============================================================
# SECTION 4: SEARCH TREND ANALYSIS
# =============================================================
print("=" * 60)
print("SECTION 4: Search Trend Analysis")
print("=" * 60)

# Top 10 most searched terms
top_searches = search_hist['search_term'].value_counts().head(10)

plt.figure(figsize=(8, 5))
top_searches.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 10 Most Searched Products on Giant Platform')
plt.xlabel('Number of Searches')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output/4a_top_searches.png', dpi=150)
plt.show()
print("Top searches chart saved → output/4a_top_searches.png")

# Search trends by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_searches = (search_hist.groupby('day_of_week')['search_id']
                  .count()
                  .reindex(day_order))

plt.figure(figsize=(8, 4))
daily_searches.plot(kind='bar', color='mediumseagreen', edgecolor='black')
plt.title('Search Volume by Day of Week')
plt.xlabel('Day')
plt.ylabel('Number of Searches')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('output/4b_search_by_day.png', dpi=150)
plt.show()
print("Search by day chart saved → output/4b_search_by_day.png")


# =============================================================
# SECTION 5: RECIPE POPULARITY RANKING
# =============================================================
print("=" * 60)
print("SECTION 5: Recipe Popularity Ranking")
print("=" * 60)

# Method: infer recipe "orders" from ingredient purchases.
# A recipe is counted as ordered if a user bought >= 50% of
# its ingredients in any single order.

# Build lookup: recipe_id → set of ingredients
recipe_ingredient_map = (
    recipe_ing.groupby('recipe_id')['ingredient']
    .apply(set)
    .to_dict()
)

# Build lookup: (order_id, user_id) → set of products bought
order_product_map = (
    order_items.groupby(['order_id', 'user_id'])['product_name']
    .apply(set)
    .reset_index()
)

# Count how many times each recipe's ingredients appear together in one order
recipe_order_counts = {}

for recipe_id, ingredients in recipe_ingredient_map.items():
    count = 0
    for _, row in order_product_map.iterrows():
        products_in_order = row['product_name']
        overlap = ingredients & products_in_order
        if len(overlap) / len(ingredients) >= 0.5:
            count += 1
    recipe_order_counts[recipe_id] = count

# Convert to dataframe and merge with recipe names
recipe_popularity = pd.DataFrame(
    list(recipe_order_counts.items()),
    columns=['recipe_id', 'inferred_orders']
).merge(recipes[['recipe_id', 'recipe_name', 'cuisine', 'avg_rating']], on='recipe_id')

recipe_popularity = recipe_popularity.sort_values('inferred_orders', ascending=False)

print("\nTop 10 Most Ordered Recipes (inferred from ingredient purchases):")
print(recipe_popularity.head(10)[
    ['recipe_name', 'cuisine', 'inferred_orders', 'avg_rating']
].to_string(index=False))

# Plot: top 10 most ordered recipes
top_recipes = recipe_popularity.head(10)

plt.figure(figsize=(10, 5))
colors = plt.cm.Blues(
    [0.4 + 0.5 * (i / len(top_recipes)) for i in range(len(top_recipes))]
)[::-1]
bars = plt.barh(top_recipes['recipe_name'], top_recipes['inferred_orders'],
                color=colors, edgecolor='black')
plt.xlabel('Number of Inferred Orders')
plt.title('Top 10 Most Ordered Recipes\n(based on ingredient purchase patterns)',
          fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
for bar, val in zip(bars, top_recipes['inferred_orders']):
    plt.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('output/5a_recipe_popularity.png', dpi=150)
plt.show()
print("Recipe popularity chart saved → output/5a_recipe_popularity.png")

# Plot: most ordered by cuisine
cuisine_orders = (recipe_popularity.groupby('cuisine')['inferred_orders']
                  .sum().sort_values(ascending=False))

plt.figure(figsize=(8, 4))
cuisine_orders.plot(kind='bar', color='mediumpurple', edgecolor='black', width=0.6)
plt.title('Total Inferred Orders by Cuisine', fontsize=12, fontweight='bold')
plt.xlabel('Cuisine')
plt.ylabel('Total Inferred Orders')
plt.xticks(rotation=20, ha='right')
for i, v in enumerate(cuisine_orders):
    plt.text(i, v + 0.3, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('output/5b_orders_by_cuisine.png', dpi=150)
plt.show()
print("Orders by cuisine chart saved → output/5b_orders_by_cuisine.png")

# Most popular recipe per cuisine
top_per_cuisine = (recipe_popularity
                   .sort_values('inferred_orders', ascending=False)
                   .groupby('cuisine')
                   .first()
                   .reset_index()[['cuisine', 'recipe_name', 'inferred_orders', 'avg_rating']])
print("\nMost Popular Recipe per Cuisine:")
print(top_per_cuisine.to_string(index=False))


def most_searched_recipe(search_history, recipe_ingredients):
    recipe_match_count = {}

    all_users = search_history["user_id"].unique()

    for user_id in all_users:
        user_searches = search_history[search_history["user_id"] == user_id]
        search_terms  = user_searches["search_term"].tolist()
        all_recipes   = recipe_ingredients["recipe_name"].unique()

        for recipe in all_recipes:
            ingredients = recipe_ingredients[recipe_ingredients["recipe_name"] == recipe]
            ing_list    = ingredients["ingredient"].tolist()

            if recipe in search_terms:
                if recipe not in recipe_match_count:
                    recipe_match_count[recipe] = 0
                recipe_match_count[recipe] += 1
            else:
                matched  = [i for i in ing_list if i in search_terms]
                coverage = len(matched) / len(ing_list)
                if coverage >= 0.5:
                    if recipe not in recipe_match_count:
                        recipe_match_count[recipe] = 0
                    recipe_match_count[recipe] += 1

    best_recipe = max(recipe_match_count, key=lambda x: recipe_match_count[x])
    print(f"Most searched recipe: {best_recipe}")
    print(f"Matched by {recipe_match_count[best_recipe]} users")


def most_purchased_recipe(order_items, recipe_ingredients):
    recipe_match_count = {}

    all_orders = order_items["order_id"].unique()

    for order_id in all_orders:
        order    = order_items[order_items["order_id"] == order_id]
        products = order["product_name"].tolist()

        all_recipes = recipe_ingredients["recipe_name"].unique()

        for recipe in all_recipes:
            ingredients = recipe_ingredients[recipe_ingredients["recipe_name"] == recipe]
            ing_list    = ingredients["ingredient"].tolist()

            matched  = [i for i in ing_list if i in products]
            coverage = len(matched) / len(ing_list)
            if coverage >= 0.5:
                if recipe not in recipe_match_count:
                    recipe_match_count[recipe] = 0
                recipe_match_count[recipe] += 1

    best_recipe = max(recipe_match_count, key=lambda x: recipe_match_count[x])
    print(f"Best recipe for bundle: {best_recipe}")
    print(f"Matched in {recipe_match_count[best_recipe]} orders")


# Run all 3 test cases
testcases = ["mains_test", "dessert_test", "both_test"]

for testcase in testcases:
    tc_search_history     = pd.read_csv(f"{testcase}/search_history.csv")
    tc_order_items        = pd.read_csv(f"{testcase}/order_items.csv")
    tc_recipe_ingredients = pd.read_csv(f"{testcase}/recipe_ingredients.csv")

    print(f"\n── {testcase} ──────────────────────────")
    most_searched_recipe(tc_search_history, tc_recipe_ingredients)
    most_purchased_recipe(tc_order_items, tc_recipe_ingredients)
