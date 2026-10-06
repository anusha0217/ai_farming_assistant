import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Dataset location
DATASET_PATH = "ai_smart_farmer_assistant/Crop_recommendation.csv"


def train_crop_model():
    """
    Train a Random Forest model using the crop recommendation dataset.
    """

    if not os.path.exists(DATASET_PATH):

        return None, None, (
            "Crop recommendation dataset was not found. "
            "Please add Crop_recommendation.csv inside "
            "the ai_smart_farmer_assistant folder."
        )

    data = pd.read_csv(DATASET_PATH)

    # Remove unnecessary spaces from column names
    data.columns = data.columns.str.strip()

    # Expected input columns
    feature_columns = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    # Check whether required columns exist
    missing_columns = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_columns:

        return None, None, (
            "Dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

    # Features
    X = data[feature_columns]

    # Target
    y = data["label"]

    # Convert crop names to numbers
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Calculate accuracy
    accuracy = model.score(
        X_test,
        y_test
    )

    return model, label_encoder, accuracy


def recommend_crop(
    model,
    label_encoder,
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):
    """
    Predict the most suitable crop.
    """

    input_data = pd.DataFrame(
        [[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    prediction = model.predict(input_data)

    crop = label_encoder.inverse_transform(
        prediction
    )[0]

    return crop