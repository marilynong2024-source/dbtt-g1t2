import pandas as pd

search_history = pd.read_csv("household_items/search_history.csv")
order_items    = pd.read_csv("household_items/order_items.csv")
products       = pd.read_csv("household_items/products.csv")

# ── Function 1: top 3 most searched products ─────────────────
def top_searched(search_history, products):
    # get the list of all valid product names from products table
    valid_products = products["product_name"].tolist()

    # filter search_history to only rows where search_term is a product
    # hint: use .isin() to check if search_term is in valid_products
    product_searches = search_history[search_history["search_term"].isin(valid_products)]

    # count how many times each product was searched
    # hint: use .value_counts() on the search_term column
    counts = product_searches["search_term"].value_counts()
    # get top 3
    # hint: use .head(3)
    top3 = counts.head(3)

    print("Top 3 most searched products:")
    i = 1
    for product, count in top3.items():
        print(f"  {i}. {product} — {count} searches")
        i += 1


# ── Function 2: top 3 most purchased products ────────────────
def top_purchased(order_items):
    # count how many times each product was purchased
    # hint: use .value_counts() on product_name column
    counts = order_items["product_name"].value_counts()

    # get top 3
    top3 = counts.head(3)

    print("Top 3 most purchased products:")
    i = 1
    for product, count in top3.items():
        print(f"  {i}. {product} — {count} purchases")
        i += 1

print("\n" + "=" * 35 + "\n")
top_searched(search_history, products)
print("\n" + "─" * 35 + "\n")
top_purchased(order_items)
print("\n" + "=" * 35 + "\n")