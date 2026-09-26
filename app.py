import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Score Predictor", page_icon="📚", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

data = load_data()

features = ["study_hours", "attendance", "previous_score"]
X = data[features]
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.title("📚 Student Score Predictor")
st.caption("A small machine learning project for exploring factors that may relate to academic performance.")

st.write(
    "Enter a few details about a student and the model will estimate their final score. "
    "The dataset used here is a small synthetic dataset made for this project."
)

st.subheader("Enter student details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study hours per day", min_value=0.0, max_value=12.0, value=5.0, step=0.5
    )
    attendance = st.slider("Attendance (%)", 0, 100, 75)

with col2:
    previous_score = st.slider("Previous score (%)", 0, 100, 70)

if st.button("Predict final score", use_container_width=True):
    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score
    }])

    score = float(model.predict(student)[0])
    score = max(0, min(100, score))

    st.subheader("Prediction")
    st.metric("Estimated final score", f"{score:.1f}%")

    if score >= 75:
        st.success("The model estimates a relatively strong score.")
    elif score >= 50:
        st.info("The model estimates a moderate score.")
    else:
        st.warning("The model estimates a lower score.")

st.divider()

st.subheader("Model information")

c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error", f"{mae:.2f}")
c2.metric("R² score", f"{r2:.2f}")

st.write(
    "The model uses Random Forest Regression. The metrics above are calculated on "
    "the held-out test data, so they give a basic idea of how the model performed "
    "on data it did not train on."
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.subheader("Feature importance")
st.bar_chart(importance.set_index("Feature"))

with st.expander("View sample data"):
    st.dataframe(data.head(10), use_container_width=True)

st.caption("Note: This is an educational project. Predictions should not be used for real academic decisions.")
