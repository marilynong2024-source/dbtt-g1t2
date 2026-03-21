import pandas as pd

# ── Function 1: most searched recipe ─────────────────────────
# The problem is when you loop through 3 test cases, Python loads new data each loop but the function doesn't know that — it just keeps using whatever search_history was last set to. So you'd get the same result 3 times.
# Now the functions accept the data as inputs when there are parameters in ()

def most_searched_recipe(search_history, recipe_ingredients): # takes inputs
    recipe_match_count = {} # store how many users matched each recipe

    all_users = search_history["user_id"].unique()

    for user_id in all_users:
        # filter search_history to only this user's rows
        user_searches = search_history[search_history["user_id"] == user_id]

        # get just the search_term column as a list
        search_terms  = user_searches["search_term"].tolist()
        all_recipes   = recipe_ingredients["recipe_name"].unique()

        for recipe in all_recipes:
            # filter recipe_ingredients to only this recipe's rows
            ingredients = recipe_ingredients[recipe_ingredients["recipe_name"] == recipe]

            # get just the search_term column as a list
            ing_list    = ingredients["ingredient"].tolist()

            # check if user searched the recipe name directly
            if recipe in search_terms:

                # add this recipe to recipe_match_count
                # if it's not in the dict yet, set it to 0 first
                if recipe not in recipe_match_count:
                    recipe_match_count[recipe] = 0
                recipe_match_count[recipe] += 1
            
            else:
                # count how many ingredients the user searched
                matched = []
                for i in ing_list:
                    if i in search_terms:
                        matched.append(i)

                # calculate coverage (matched out of total ingredients)
                coverage = len(matched) / len(ing_list)

                # if coverage is 50% or more, count it as a match
                if coverage >= 0.5:
                    if recipe not in recipe_match_count:
                        recipe_match_count[recipe] = 0
                    recipe_match_count[recipe] += 1

    # find the recipe with the highest count
    # recipe_match_count is a dictionary that looks like this after the loop:
    # {"Chicken Rice": 12, "Laksa": 8, "Nasi Lemak": 15, "Fried Rice": 6}
    # max() normally finds the biggest value in a list:
    # max([3, 8, 5])  # returns 8
    # But for a dictionary, Python doesn't know if you want the biggest key or the biggest value — so you need to tell it.
    # lambda x is just a mini function that takes each key x (e.g. "Chicken Rice") and returns its value (e.g. 12). So max compares the numbers, not the name

    # # given this dictionary
    # recipe_match_count = {"Chicken Rice": 12, "Laksa": 8, "Nasi Lemak": 15}
    # best_recipe = max(recipe_match_count, key=lambda x: recipe_match_count[x])
    # print(best_recipe)  # "Nasi Lemak" — because 15 is the highest

    best_recipe = max(recipe_match_count, key=lambda x: recipe_match_count[x])
    print(f"Most searched recipe: {best_recipe}")
    print(f"Matched by {recipe_match_count[best_recipe]} users")


def most_purchased_recipe(order_items, recipe_ingredients):
    recipe_match_count = {} # store how many orders matched each recipe

    all_orders = order_items["order_id"].unique() # get list of all orders

    for order_id in all_orders:
        # filter order_items to only this order's rows
        order    = order_items[order_items["order_id"] == order_id]

        # get just the product_name column as a list
        products = order["product_name"].tolist()

        all_recipes = recipe_ingredients["recipe_name"].unique()

        for recipe in all_recipes:
            # filter recipe_ingredients to only this recipe's rows
            ingredients = recipe_ingredients[recipe_ingredients["recipe_name"] == recipe]
            ing_list    = ingredients["ingredient"].tolist()

            # count how many ingredients were in this order
            matched = []
            for i in ing_list:
                if i in products:
                    matched.append(i)
            coverage = len(matched) / len(ing_list)
            if coverage >= 0.5:
                if recipe not in recipe_match_count:
                    recipe_match_count[recipe] = 0
                recipe_match_count[recipe] += 1

    best_recipe = max(recipe_match_count, key=lambda x: recipe_match_count[x])
    print(f"Best recipe for bundle: {best_recipe}")
    print(f"Matched in {recipe_match_count[best_recipe]} orders")


# ── run all 3 test cases ──────────────────────────────────────
testcases = ["mains_test", "dessert_test", "both_test"]

for testcase in testcases:
    search_history     = pd.read_csv(f"{testcase}/search_history.csv")
    order_items        = pd.read_csv(f"{testcase}/order_items.csv")
    recipe_ingredients = pd.read_csv(f"{testcase}/recipe_ingredients.csv")

    print(f"\n── {testcase} ──────────────────────────")
    most_searched_recipe(search_history, recipe_ingredients)
    most_purchased_recipe(order_items, recipe_ingredients)