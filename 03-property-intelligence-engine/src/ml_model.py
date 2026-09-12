from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.sql_analytics import run_query


REFERENCE_YEAR = 2026
RANDOM_STATE = 1
VALIDATION_SIZE = 0.2
TREE_LEAF_OPTIONS = [5, 10, 25, 50, 100, 200]


def load_modeling_dataset():
    query = """
        WITH ranked_sales AS (
            SELECT
                s.property_id,
                s.sale_price,
                ROW_NUMBER() OVER (
                    PARTITION BY s.property_id
                    ORDER BY
                        s.sale_date DESC,
                        s.sale_id DESC
                ) AS sale_rank
            FROM sales AS s
        )
        SELECT
            p.property_id,
            p.bedrooms,
            p.bathrooms,
            p.area_sqm,
            p.year_built,
            p.parking_spaces,
            p.floor,
            p.furnished,
            n.distance_to_center_km,
            n.school_score,
            n.transit_score,
            n.safety_score,
            rs.sale_price
        FROM properties AS p
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        INNER JOIN ranked_sales AS rs
            ON p.property_id = rs.property_id
        WHERE rs.sale_rank = 1
        ORDER BY p.property_id
    """

    return run_query(query)


def prepare_features_target(data):
    prepared = data.copy()
    prepared["property_age"] = REFERENCE_YEAR - prepared["year_built"]

    features = [
        "bedrooms",
        "bathrooms",
        "area_sqm",
        "property_age",
        "parking_spaces",
        "floor",
        "furnished",
        "distance_to_center_km",
        "school_score",
        "transit_score",
        "safety_score",
    ]

    X = prepared[features]
    y = prepared["sale_price"]

    return X, y


def split_modeling_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    return mean_absolute_error(y, predictions)


def get_tree_validation_mae(
    max_leaf_nodes,
    train_X,
    val_X,
    train_y,
    val_y,
):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=RANDOM_STATE,
    )
    model.fit(train_X, train_y)

    return evaluate_model(model, val_X, val_y)


def compare_tree_sizes(
    train_X,
    val_X,
    train_y,
    val_y,
):
    results = {}

    for max_leaf_nodes in TREE_LEAF_OPTIONS:
        results[max_leaf_nodes] = get_tree_validation_mae(
            max_leaf_nodes,
            train_X,
            val_X,
            train_y,
            val_y,
        )

    return results


def train_tuned_tree(best_leaf_nodes, train_X, train_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=best_leaf_nodes,
        random_state=RANDOM_STATE,
    )
    model.fit(train_X, train_y)

    return model


def train_random_forest(train_X, train_y):
    model = RandomForestRegressor(
        random_state=RANDOM_STATE
    )
    model.fit(train_X, train_y)

    return model


def better_model(tree_mae, forest_mae):
    if tree_mae < forest_mae:
        return "Decision Tree"
    if forest_mae < tree_mae:
        return "Random Forest"

    return "Tie"


def forest_improvement(tree_mae, forest_mae):
    return (tree_mae - forest_mae) / tree_mae * 100