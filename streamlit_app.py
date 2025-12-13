import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Note: model training/creation removed. The app expects a provided `model.pkl` (uploaded or placed in repo root).
MODEL_PATH = "model.pkl"

FEATURE_COLUMNS = [
    "num_lanes",
    "curvature",
    "speed_limit",
    "road_signs_present",
    "public_road",
    "holiday",
    "school_season",
    "num_reported_accidents",
    "road_type_highway",
    "road_type_rural",
    "road_type_urban",
    "lighting_daylight",
    "lighting_dim",
    "lighting_night",
    "weather_clear",
    "weather_foggy",
    "weather_rainy",
    "time_of_day_afternoon",
    "time_of_day_evening",
    "time_of_day_morning",
]


def build_feature_vector(inputs: dict) -> pd.DataFrame:
    # Initialize zeros for all expected columns
    vec = {c: 0 for c in FEATURE_COLUMNS}

    # numeric
    vec["num_lanes"] = int(inputs.get("num_lanes", 0))
    vec["curvature"] = float(inputs.get("curvature", 0.0))
    vec["speed_limit"] = int(inputs.get("speed_limit", 0))
    vec["num_reported_accidents"] = int(inputs.get("num_reported_accidents", 0))

    # booleans
    vec["road_signs_present"] = 1 if inputs.get("road_signs_present") else 0
    vec["public_road"] = 1 if inputs.get("public_road") else 0
    vec["holiday"] = 1 if inputs.get("holiday") else 0
    vec["school_season"] = 1 if inputs.get("school_season") else 0

    # categorical one-hot
    road_type = inputs.get("road_type")
    if road_type == "highway":
        vec["road_type_highway"] = 1
    elif road_type == "rural":
        vec["road_type_rural"] = 1
    elif road_type == "urban":
        vec["road_type_urban"] = 1

    lighting = inputs.get("lighting")
    if lighting == "daylight":
        vec["lighting_daylight"] = 1
    elif lighting == "dim":
        vec["lighting_dim"] = 1
    elif lighting == "night":
        vec["lighting_night"] = 1

    weather = inputs.get("weather")
    if weather == "clear":
        vec["weather_clear"] = 1
    elif weather == "foggy":
        vec["weather_foggy"] = 1
    elif weather == "rainy":
        vec["weather_rainy"] = 1

    tod = inputs.get("time_of_day")
    if tod == "afternoon":
        vec["time_of_day_afternoon"] = 1
    elif tod == "evening":
        vec["time_of_day_evening"] = 1
    elif tod == "morning":
        vec["time_of_day_morning"] = 1

    return pd.DataFrame([vec], columns=FEATURE_COLUMNS)


def load_model():
    """Attempt to load `MODEL_PATH`. Returns the loaded model or None if not available/failed."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def main():
    st.set_page_config(page_title="Accident Risk Predictor", layout="centered")
    st.title("Road Accident Risk Prediction")

    st.markdown("Provide road and environmental inputs to predict accident risk (regression output).")

    # Allow user to upload a trained model file which will be saved as MODEL_PATH
    uploaded_file = st.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            with open(MODEL_PATH, "wb") as f:
                f.write(bytes_data)
            st.success(f"Uploaded model saved to `{MODEL_PATH}` and will be used for predictions.")
        except Exception as e:
            st.error(f"Failed to save uploaded model: {e}")

    if os.path.exists(MODEL_PATH):
        st.info(f"Model file found at `{MODEL_PATH}` — it will be used for predictions.")
    else:
        st.info("No `model.pkl` found — a fallback synthetic model will be trained and used.")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            road_type = st.selectbox("Road type", ["urban", "rural", "highway"], index=0)
            num_lanes = st.number_input("Number of lanes", min_value=1, max_value=10, value=2)
            curvature = st.number_input("Curvature (0-10)", min_value=0.0, max_value=100.0, value=1.0, format="%.2f")
            speed_limit = st.number_input("Speed limit (km/h)", min_value=0, max_value=300, value=50)
            road_signs_present = st.checkbox("Road signs present", value=True)

        with col2:
            public_road = st.checkbox("Public road", value=True)
            time_of_day = st.selectbox("Time of day", ["morning", "afternoon", "evening"], index=0)
            holiday = st.checkbox("Holiday", value=False)
            school_season = st.checkbox("School season", value=False)
            num_reported_accidents = st.number_input("Number of reported accidents (historic)", min_value=0, max_value=1000, value=0)

        weather = st.selectbox("Weather", ["clear", "foggy", "rainy"], index=0)
        lighting = st.selectbox("Lighting", ["daylight", "dim", "night"], index=0)

        submitted = st.form_submit_button("Predict")

    inputs = {
        "road_type": road_type,
        "num_lanes": num_lanes,
        "curvature": curvature,
        "speed_limit": speed_limit,
        "road_signs_present": road_signs_present,
        "public_road": public_road,
        "time_of_day": time_of_day,
        "holiday": holiday,
        "school_season": school_season,
        "num_reported_accidents": num_reported_accidents,
        "weather": weather,
        "lighting": lighting,
    }

    if submitted:
        features_df = build_feature_vector(inputs)
        model = load_model()
        if model is None:
            st.error("No trained model available. Please upload a `model.pkl` or place one in the app directory.")
        else:
            try:
                pred = model.predict(features_df.values)[0]
                pred = float(pred)
                if pred < 0:
                    pred = 0.0
                st.metric("Predicted accident risk (regression output)", f"{pred:.3f}")

                with st.expander("Processed features (for model)"):
                    st.write(features_df.transpose())

                with st.expander("Download features as CSV"):
                    csv = features_df.to_csv(index=False)
                    st.download_button("Download CSV", csv, file_name="features.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Model prediction failed: {e}")

    st.caption("If you have a trained model file `model.pkl` in the app directory it will be used.")


if __name__ == "__main__":
    main()
