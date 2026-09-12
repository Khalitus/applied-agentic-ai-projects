from src.sql_analytics import run_query
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


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

    prepared["property_age"] = (
        2026 - prepared["year_built"]
    )

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

def train_baseline_model():
    data = load_modeling_dataset()
    X, y = prepare_features_target(data)

    model = DecisionTreeRegressor(
        random_state=1
    )

    model.fit(X,y)

    return model, X, y

def make_sample_predictions(model, X, count=5):
    sample = X.head(count)

    predictions = model.predict(sample)

    return sample, predictions

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)

    return mae

def split_modeling_data(X, y):
    train_X, val_X, train_y, val_y = train_test_split(
        X,
        y,
        test_size=50,
        random_state=1,
    )

    return train_X, val_X, train_y, val_y

def evaluate_baseline_split(
    train_X,
    val_X,
    train_y,
    val_y,
):
    model = DecisionTreeRegressor(
        random_state = 1
    )

    model.fit(
        train_X,
        train_y,
    )

    train_predictions = model.predict(
        train_X
    )

    val_predictions = model.predict(
        val_X
    )

    train_mae = mean_absolute_error(
        train_y,
        train_predictions,
    )

    val_mae = mean_absolute_error(
        val_y,
        val_predictions,
    )

    return train_mae, val_mae

def get_tree_validation_mae(
    max_leaf_nodes,
    train_X,
    val_X,
    train_y,
    val_y,
):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=1,
    )

    model.fit(train_X, train_y)

    predictions = model.predict(val_X)

    mae  = mean_absolute_error(val_y, predictions)

    return mae

def compare_tree_sizes(
    train_X,
    val_X,
    train_y,
    val_y,
):
    leaf_options = [5, 10, 25, 50, 100, 200]
    results = {}

    for max_leaf_nodes in leaf_options:
        results[max_leaf_nodes] = get_tree_validation_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    return results

def train_tuned_tree(
    best_leaf_nodes,
    train_X,
    train_y,
):
    model = DecisionTreeRegressor(
        max_leaf_nodes = best_leaf_nodes,
        random_state = 1 
    )

    model.fit(train_X, train_y)

    return model

def train_random_forest(train_X, train_y):
    model = RandomForestRegressor(
        random_state=1
    )

    model.fit(train_X,train_y)

    return model

