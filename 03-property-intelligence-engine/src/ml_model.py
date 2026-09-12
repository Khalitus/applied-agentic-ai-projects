from src.sql_analytics import run_query
from sklearn.tree import DecisionTreeRegressor

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