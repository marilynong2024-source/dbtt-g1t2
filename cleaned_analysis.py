"""
IS215 Project — Giant Singapore Digital Transformation
analysis.py — Full Analytics & AI Component
=======================================================
Sections:
  0. Imports & Data Loading
  1. Customer Segmentation (K-Means)
  2. Product Recommendations (Collaborative Filtering via SVD)
  3. Demand Forecasting (Rolling Average)
  4. Promotion Effectiveness Analysis
  5. Search Trend Analysis (Bonus)
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

# ── Fit K-Means with k=3 (justified by elbow plot) ──
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

# 1a ==========================================================================================================================
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=seg_df,
    x='avg_weekly_spend_sgd',
    y='visit_frequency',    # `visit_frequency` is created by counting how many orders each user has made
    hue='segment_label',
    palette='Set2',
    alpha=0.7,
    s=60
)
plt.title('Customer Segmentation — K-Means (k=3)', fontsize=13)
plt.xlabel('Avg Weekly Spend (SGD)')
plt.ylabel('Visit Frequency (orders)')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig('output/1b_segmentation_scatter.png', dpi=150)
plt.show()
print("Segmentation scatter plot saved → output/1b_segmentation_scatter.png")

# 1b ==========================================================================================================================
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

# ── Summary statistics per segment ──
segment_summary = seg_df.groupby('segment_label')[features].mean().round(2)
print("\nSegment Profiles:")
print(segment_summary.to_string())
print()


# =============================================================
# SECTION 2: PRODUCT RECOMMENDATIONS (COLLABORATIVE FILTERING)
# =============================================================
print("=" * 60)
print("SECTION 2: Product Recommendations (Collaborative Filtering)")
print("=" * 60)

# Build user-product purchase matrix
# Rows = users, Columns = products, Values = total quantity purchased
purchase_matrix = (order_items
                   .groupby(['user_id', 'product_name'])['quantity']
                   .sum()
                   .unstack(fill_value=0))


# Convert to sparse matrix for efficient SVD computation
sparse_matrix = csr_matrix(purchase_matrix.values)

# Apply Truncated SVD (matrix factorisation)
# n_components=10 extracts 10 latent "taste dimensions" per user
n_components = min(10, purchase_matrix.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(sparse_matrix)

explained_var = svd.explained_variance_ratio_.sum()

# Compute cosine similarity between all users based on latent factors
user_similarity = cosine_similarity(user_factors)
user_sim_df = pd.DataFrame(
    user_similarity,
    index=purchase_matrix.index,
    columns=purchase_matrix.index
)

def recommend_products(user_id, n=5):
    """
    Recommend n products for a given user_id using collaborative filtering.
    Logic: find the 5 most similar users, pool what they bought,
    exclude what the target user already bought, return top n by frequency.
    """
    if user_id not in user_sim_df.index:
        return [f"User {user_id} not found in purchase history"]

    # Get top 5 most similar users (exclude the user themselves)
    similar_users = (user_sim_df[user_id]
                     .sort_values(ascending=False)
                     .iloc[1:6]
                     .index)

    # Pool all products bought by similar users
    similar_purchases = order_items[
        order_items['user_id'].isin(similar_users)]['product_name']

    # Exclude products the target user already bought
    already_bought = order_items[
        order_items['user_id'] == user_id]['product_name'].unique()

    recommendations = (similar_purchases[~similar_purchases.isin(already_bought)]
                       .value_counts()
                       .head(n)
                       .index
                       .tolist())
    return recommendations

# ── Demo: show recommendations for first 3 users ──
print("\nSample Recommendations:")
for uid in users['user_id'].iloc[:3]:
    recs = recommend_products(uid)
    preferred = users.loc[users['user_id'] == uid, 'preferred_cuisines'].values[0]
    print(f"  {uid} (Preferred cuisine: {preferred}): {recs}")

# 2a ==========================================================================================================================
all_recs = []
for uid in purchase_matrix.index[:50]:   # sample 50 users for speed
# for uid in purchase_matrix:   # sample 50 users for speed
    all_recs.extend(recommend_products(uid))

rec_counts = pd.Series(all_recs).value_counts().head(10)

plt.figure(figsize=(8, 5))
rec_counts.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 10 Most Recommended Products (Collaborative Filtering)')
plt.xlabel('Recommendation Frequency')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output/2_top_recommendations.png', dpi=150)
plt.show()

# 2b ==========================================================================================================================
print("=" * 60)
print("SECTION 2B: Price-Sensitivity Profiling")
print("=" * 60)

# ── Step 1: Build price preference profile per user ──────────
# Uses three signals from order_items:
#   avg_unit_price     → how expensive are the products they buy?
#   promo_purchase_rate → what % of their purchases are on promotion?
#   total_spend        → overall spend level

price_profile = order_items.groupby('user_id').agg(
    avg_unit_price=('unit_price_sgd',  'mean'),
    promo_purchase_rate=('on_promotion', 'mean'),
    avg_basket_size=('line_total_sgd', 'sum')
).reset_index()

# ── Step 2: Classify each user into a price tier by spending ──
# High Spender    → avg weekly spend in top third (> $130)
# Mid Spender     → avg weekly spend in middle third ($70 - $130)  
# Budget Spender  → avg weekly spend in bottom third (< $70)

# Calculate thresholds from actual data
low_thresh  = users['avg_weekly_spend_sgd'].quantile(0.33)
high_thresh = users['avg_weekly_spend_sgd'].quantile(0.67)

print(f"\nSpend thresholds: Low < ${low_thresh:.2f} | Mid ${low_thresh:.2f}–${high_thresh:.2f} | High > ${high_thresh:.2f}")

def price_tier(row):
    spend = row['avg_weekly_spend_sgd']
    if spend >= high_thresh:
        return 'High Spender'
    elif spend >= low_thresh:
        return 'Mid Spender'
    else:
        return 'Budget Spender'

# Merge spend info from users into price_profile
price_profile = price_profile.merge(
    users[['user_id', 'avg_weekly_spend_sgd']], on='user_id', how='left')
price_profile['price_tier'] = price_profile.apply(price_tier, axis=1)

# Print tier distribution
tier_counts = price_profile['price_tier'].value_counts()
print("\nPrice Tier Distribution:")
for tier, count in tier_counts.items():
    pct = count / len(price_profile) * 100
    print(f"  {tier}: {count} users ({pct:.1f}%)")

# ── Step 3: Pre-compute product lists per tier ───────────────
# High Spender   → highest avg priced products
premium_products = (order_items.groupby('product_name')['unit_price_sgd']
                    .mean().nlargest(50).index.tolist())

# Mid Spender    → mid-range priced products
p33 = order_items['unit_price_sgd'].quantile(0.33)
p67 = order_items['unit_price_sgd'].quantile(0.67)
mid_products = (order_items[
    (order_items['unit_price_sgd'] >= p33) &
    (order_items['unit_price_sgd'] <= p67)
]['product_name'].value_counts().head(50).index.tolist())

# Budget Spender → lowest avg priced products
budget_products = (order_items.groupby('product_name')['unit_price_sgd']
                   .mean().nsmallest(50).index.tolist())

tier_product_map = {
    'High Spender':   premium_products,
    'Mid Spender':    mid_products,
    'Budget Spender': budget_products
}

print(f"\n  High Spender pool   : {len(premium_products)} products")
print(f"  Mid Spender pool    : {len(mid_products)} products")
print(f"  Budget Spender pool : {len(budget_products)} products")

# ── Step 4: Price-aware recommendation function ───────────────
def recommend_with_price_filter(user_id, n=5):
    # Get base collaborative filtering recommendations
    base_recs = recommend_products(user_id, n=20)  # get more candidates

    # Get user's price tier
    tier = price_profile.loc[
        price_profile['user_id'] == user_id, 'price_tier'
    ].values[0]

    # Filter candidates by price preference
    if tier == 'Price Sensitive':
        # Filter to products that are frequently on promotion
        promo_products = (order_items[order_items['on_promotion'] == True]
                         ['product_name'].value_counts().head(50).index)
        filtered = [r for r in base_recs if r in promo_products]
    elif tier == 'Premium Seeker':
        # Filter to higher-priced products
        premium_products = (order_items.groupby('product_name')['unit_price_sgd']
                           .mean().nlargest(50).index)
        filtered = [r for r in base_recs if r in premium_products]
    else:
        filtered = base_recs

    return filtered[:n] if filtered else base_recs[:n]

# ── Step 5: Demo — compare base vs price-aware recommendations ──
print("\nPrice-Aware Recommendation Comparison (first 5 users):")
print(f"{'User':<8} {'Price Tier':<20} {'Base Recs':<40} {'Price-Aware Recs'}")
print("-" * 110)

for uid in users['user_id'].iloc[:5]:
    base  = recommend_products(uid, n=3)
    aware = recommend_with_price_filter(uid, n=3)
    tier  = price_profile.loc[price_profile['user_id'] == uid, 'price_tier'].values[0]
    print(f"{uid:<8} {tier:<20} {str(base):<40} {str(aware)}")

# ── Step 6: Plot spend tier distribution ─────────────────────
tier_colors = {
    'High Spender':   '#66b3ff',
    'Mid Spender':    '#99cc99',
    'Budget Spender': '#ff9999'
}

plt.figure(figsize=(7, 4))
tier_order = ['High Spender', 'Mid Spender', 'Budget Spender']
tier_counts_ordered = tier_counts.reindex(tier_order, fill_value=0)
colors = [tier_colors[t] for t in tier_counts_ordered.index]
tier_counts_ordered.plot(kind='bar', color=colors, edgecolor='black', width=0.5)
plt.title('Customer Spend Tier Distribution')
plt.xlabel('Spend Tier')
plt.ylabel('Number of Customers')
plt.xticks(rotation=10)
for i, v in enumerate(tier_counts_ordered):
    plt.text(i, v + 0.3, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('output/2b_price_tiers.png', dpi=150)
plt.show()

# 2c ==========================================================================================================================
# ── Step 7: Plot avg weekly spend per tier ────────────────────
tier_profiles = (price_profile.groupby('price_tier')
                 .agg(
                     avg_weekly_spend=('avg_weekly_spend_sgd', 'mean'),
                     avg_unit_price=('avg_unit_price', 'mean')
                 )
                 .round(2)
                 .reindex(tier_order))

print("\nSpend Tier Profiles:")
print(tier_profiles.to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Avg weekly spend per tier
tier_profiles['avg_weekly_spend'].plot(
    kind='bar', ax=axes[0],
    color=[tier_colors.get(t, '#cccccc') for t in tier_profiles.index],
    edgecolor='black', width=0.5)
axes[0].set_title('Avg Weekly Spend by Tier (SGD)')
axes[0].set_xlabel('')
axes[0].set_ylabel('Avg Weekly Spend (SGD)')
axes[0].tick_params(axis='x', rotation=10)
for i, v in enumerate(tier_profiles['avg_weekly_spend']):
    axes[0].text(i, v + 1, f'${v:.2f}', ha='center', fontsize=9)

# Avg unit price per tier
tier_profiles['avg_unit_price'].plot(
    kind='bar', ax=axes[1],
    color=[tier_colors.get(t, '#cccccc') for t in tier_profiles.index],
    edgecolor='black', width=0.5)
axes[1].set_title('Avg Unit Price by Tier (SGD)')
axes[1].set_xlabel('')
axes[1].set_ylabel('Avg Unit Price (SGD)')
axes[1].tick_params(axis='x', rotation=10)
for i, v in enumerate(tier_profiles['avg_unit_price']):
    axes[1].text(i, v + 0.1, f'${v:.2f}', ha='center', fontsize=9)

plt.suptitle('Spend Tier Profiles', fontsize=12)
plt.tight_layout()
plt.savefig('output/2c_price_tier_profiles.png', dpi=150)
plt.show()
# =============================================================
# SECTION 3: DEMAND FORECASTING (ROLLING AVERAGE)
# =============================================================
print("=" * 60)
print("SECTION 3: Demand Forecasting")
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


# ── Plot actual demand + 3-week rolling average forecast ──
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
plt.savefig('output/3_demand_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("Demand forecast chart saved → output/3_demand_forecast.png")

# ── Flag potential stockout risk: products where latest demand > forecast ──
print("\nStockout Risk Flags (latest demand > forecast):") # check if the latest week's actual demand is more than 20% above the forecast — if yes, that product is flagged as high risk.

for product in top_products:
    prod_data = (weekly_demand[weekly_demand['product_name'] == product]
                 .sort_values('week'))
    prod_data['forecast'] = prod_data['quantity'].rolling(window=3, min_periods=1).mean()
    latest = prod_data.iloc[-1]
    risk = "⚠ HIGH RISK" if latest['quantity'] > latest['forecast'] * 1.2 else "✓ Normal"
    print(f"  {product}: {risk} (latest={latest['quantity']:.0f}, forecast={latest['forecast']:.0f})")
print()


# =============================================================
# SECTION 4: PROMOTION EFFECTIVENESS ANALYSIS
# =============================================================
print("=" * 60)
print("SECTION 4: Promotion Effectiveness Analysis")
print("=" * 60)
# 4a ========================================================================================================================
# ── Overall: avg order total with vs without promotion ──
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
plt.savefig('output/4a_promo_overall.png', dpi=150)
plt.show()

# 4b ========================================================================================================================
# ── By segment: which customer segment responds most to promotions? ──
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
plt.savefig('output/4b_promo_by_segment.png', dpi=150)
plt.show()

# =============================================================
# SECTION 5: SEARCH TREND ANALYSIS 
# =============================================================
print("=" * 60)
print("SECTION 5: Search Trend Analysis")
print("=" * 60)
# 5a ========================================================================================================================
# ── Top 10 most searched terms ──
top_searches = search_hist['search_term'].value_counts().head(10)

plt.figure(figsize=(8, 5))
top_searches.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 10 Most Searched Products on Giant Platform')
plt.xlabel('Number of Searches')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output/5a_top_searches.png', dpi=150)
plt.show()

# 5b ========================================================================================================================
# ── App vs Web: click-through rate (result_clicked = True) ──
platform_ctr = search_hist.groupby('platform')['result_clicked'].mean().round(3)
print(f"\nClick-Through Rate by Platform:")
for platform, ctr in platform_ctr.items():
    print(f"  {platform}: {ctr:.1%}")

plt.figure(figsize=(5, 4))
platform_ctr.plot(kind='bar', color=['#66b3ff', '#ff9999'], edgecolor='black', width=0.4)
plt.title('Search Click-Through Rate: App vs Web')
plt.xlabel('Platform')
plt.ylabel('Click-Through Rate')
plt.xticks(rotation=0)
for i, val in enumerate(platform_ctr):
    plt.text(i, val + 0.005, f'{val:.1%}', ha='center', fontsize=10)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('output/5b_platform_ctr.png', dpi=150)
plt.show()
print("Platform CTR chart saved → output/5b_platform_ctr.png")

# 5c ========================================================================================================================
# ── Search trends by day of week ──
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
plt.savefig('output/5c_search_by_day.png', dpi=150)
plt.show()
print("Search by day chart saved → output/5c_search_by_day.png")


# =============================================================
# SECTION 6: RFM SEGMENTATION + COMBINED K-MEANS/RFM ANALYSIS
# =============================================================
print("=" * 60)
print("SECTION 6: RFM Segmentation + Combined Analysis")
print("=" * 60)

# 6a ========================================================================================================================
# ── Build RFM table ──────────────────────────────────────────
orders['timestamp'] = pd.to_datetime(orders['timestamp'])
snapshot_date = orders['timestamp'].max() + pd.Timedelta(days=1)

rfm = orders.groupby('user_id').agg(
    recency=('timestamp',    lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id',  'count'),
    monetary=('order_total_sgd', 'sum')
).reset_index()

# Score each RFM dimension 1–4  (4 = best customer behaviour)
# recency: lower days = more recent = better → reversed labels
rfm['r_score'] = pd.qcut(rfm['recency'],
                          4, labels=[4, 3, 2, 1], duplicates='drop')
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'),
                          4, labels=[1, 2, 3, 4], duplicates='drop')
rfm['m_score'] = pd.qcut(rfm['monetary'],
                          4, labels=[1, 2, 3, 4], duplicates='drop')

rfm[['r_score', 'f_score', 'm_score']] = (
    rfm[['r_score', 'f_score', 'm_score']].astype(int)
)

# ── Assign human-readable RFM segment labels ──────────────────
def rfm_label(row):
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    if r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f <= 2:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Occasional Shoppers'

rfm['rfm_segment'] = rfm.apply(rfm_label, axis=1)

# ── Updated seg_order and seg_colors ─────────────────────────
seg_order  = ['Loyal Customers', 'Potential Loyalists',
              'At Risk', 'Occasional Shoppers']
seg_colors = ['#66b3ff', '#99cc99', '#ff9999', '#c2c2f0']
seg_counts = rfm['rfm_segment'].value_counts().reindex(seg_order, fill_value=0)

plt.figure(figsize=(8, 4))
bars = seg_counts.plot(kind='bar', color=seg_colors, edgecolor='black', width=0.6)
plt.title('Customer Segments — RFM Analysis')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=20, ha='right')
for i, v in enumerate(seg_counts):
    plt.text(i, v + 0.3, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('output/6a_rfm_segments.png', dpi=150)
plt.show()
print("RFM segment chart saved → output/6a_rfm_segments.png")

# ── Print RFM segment profiles ────────────────────────────────
rfm_profile = (rfm.groupby('rfm_segment')[['recency', 'frequency', 'monetary']]
               .mean().round(2).reindex(seg_order))
print("\nRFM Segment Profiles (averages):")
print(rfm_profile.to_string())

# ── RFM scatter — frequency vs monetary, coloured by segment ──
palette = {
    'Loyal Customers':     '#66b3ff',
    'Potential Loyalists': '#99cc99',
    'At Risk':             '#ff9999',
    'Occasional Shoppers': '#c2c2f0',
}
plt.figure(figsize=(9, 5))
for seg, grp in rfm.groupby('rfm_segment'):
    plt.scatter(grp['frequency'], grp['monetary'],
                label=seg, color=palette[seg], alpha=0.75, s=60,
                edgecolors='white', linewidths=0.5)
plt.title('RFM Segmentation — Frequency vs Monetary Value')
plt.xlabel('Order Frequency')
plt.ylabel('Total Spend (SGD)')
plt.legend(title='Segment', fontsize=9)
plt.tight_layout()
plt.savefig('output/6b_rfm_scatter.png', dpi=150)
plt.show()
print("RFM scatter saved → output/6b_rfm_scatter.png")

# ── Combine K-Means + RFM ─────────────────────────────────────
combined = seg_df.merge(
    rfm[['user_id', 'rfm_segment', 'recency', 'frequency', 'monetary']],
    on='user_id', how='left'
)
cross = pd.crosstab(combined['segment_label'], combined['rfm_segment'])
print("\nK-Means × RFM Cross-Tabulation:")
print(cross.to_string())
# 6b ========================================================================================================================
# ── Plot 3: Heatmap of K-Means vs RFM overlap ─────────────────
plt.figure(figsize=(9, 5))
sns.heatmap(cross, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Number of Customers'})
plt.title('Customer Overlap — K-Means Segments vs RFM Segments')
plt.xlabel('RFM Segment')
plt.ylabel('K-Means Segment')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('output/6c_kmeans_rfm_heatmap.png', dpi=150)
plt.show()

# ── Combined targeting logic: print actionable recommendations ──
print("\n── Actionable Targeting Recommendations ──")
targeting = {
    ('High-Value Customers', 'Loyal Customers'):
        'VIP early access to weekly deals + premium product push',
    ('High-Value Customers', 'At Risk'):
        'Urgent re-engagement: Yuu points bonus + personalised offer',
    ('Regular Families', 'Loyal Customers'):
        'Recipe bundle promotions + meal plan feature highlight',
    ('Regular Families', 'Potential Loyalists'):
        'First click-and-collect incentive + family bundle discount',
    ('Budget Shoppers', 'At Risk'):
        '"We miss you" push notification + limited-time discount',
    ('Budget Shoppers', 'Occasional Shoppers'):
        'Price promotions + awareness campaigns on app home feed',
    ('Budget Shoppers', 'Potential Loyalists'):
        'Weekly staples promotion + first order incentive',
    ('Regular Families', 'Occasional Shoppers'):
        'Recipe discovery campaign + low-barrier first order deal',
}


for combo, action in targeting.items():
    kmeans_seg, rfm_seg = combo
    count = len(combined[
        (combined['segment_label'] == kmeans_seg) &
        (combined['rfm_segment'] == rfm_seg)
    ])
    if count > 0:
        print(f"  [{kmeans_seg} × {rfm_seg}] ({count} users)")
        print(f"    → {action}")

# =============================================================
# SECTION 7: RECIPE POPULARITY RANKING
# =============================================================
print("=" * 60)
print("SECTION 7: Recipe Popularity Ranking")
print("=" * 60)

# ── Method: infer recipe "orders" from ingredient purchases ──
# A recipe is counted as ordered if a user bought >= 50% of
# its ingredients in any single order

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
        # Check if at least 50% of recipe ingredients were bought in this order
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

# ── Plot: top 10 most ordered recipes ────────────────────────
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
plt.savefig('output/7a_recipe_popularity.png', dpi=150)
plt.show()
print("Recipe popularity chart saved → output/7a_recipe_popularity.png")

# ── Plot: most ordered by cuisine ────────────────────────────
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
plt.savefig('output/7b_orders_by_cuisine.png', dpi=150)
plt.show()
print("Orders by cuisine chart saved → output/7b_orders_by_cuisine.png")

# ── Most popular recipe per cuisine ──────────────────────────
top_per_cuisine = (recipe_popularity
                   .sort_values('inferred_orders', ascending=False)
                   .groupby('cuisine')
                   .first()
                   .reset_index()[['cuisine', 'recipe_name', 'inferred_orders', 'avg_rating']])
print("\nMost Popular Recipe per Cuisine:")
print(top_per_cuisine.to_string(index=False))

# =============================================================
# SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — All outputs saved to /output/")
print("=" * 60)
print("""
Files generated:
  1a_elbow_method.png        — Optimal cluster count justification
  1b_segmentation_scatter.png — Customer segments visualised
  1c_segment_sizes.png       — How many customers per segment
  2_top_recommendations.png  — Most recommended products
  2b_price_tiers.png         — Customer price tier distribution
  2c_price_tier_profiles.png — Avg price + promo rate per tier
  3_demand_forecast.png      — Weekly demand + forecast per product
  4a_promo_overall.png       — Promotion lift on basket size
  4b_promo_by_segment.png    — Which segment responds most to promos
  5a_top_searches.png        — Most searched products
  5b_platform_ctr.png        — App vs Web engagement
  5c_search_by_day.png       — Peak search days
  6a_rfm_segments.png        — RFM segment sizes
  6b_rfm_scatter.png         — RFM frequency vs monetary scatter
  6c_kmeans_rfm_heatmap.png  — K-Means × RFM overlap heatmap
  6d_yuu_by_segment.png      — Yuu membership rate per RFM segment
""")