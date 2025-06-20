import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ML models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

st.set_page_config(page_title="Ensemble Menstrual Cycle Predictor", layout="centered")
st.title("🤖 Menstrual Cycle Predictor with Ensemble Learning")

st.write("Upload a CSV file with at least one column: `start_date` in `YYYY-MM-DD` format.")

uploaded_file = st.file_uploader("Upload your menstrual history CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["start_date"])
        df = df.sort_values("start_date").reset_index(drop=True)

        if len(df) < 4:
            st.warning("Please provide at least 4 entries for meaningful prediction.")
        else:
            # Calculate cycle lengths
            df["cycle_length"] = df["start_date"].diff().dt.days
            df_clean = df.dropna().copy()
            last_date = df["start_date"].iloc[-1]

            st.subheader("📈 Cycle Length Trend")
            fig, ax = plt.subplots()
            sns.lineplot(x=df_clean["start_date"], y=df_clean["cycle_length"], marker="o", ax=ax)
            ax.set_xlabel("Start Date")
            ax.set_ylabel("Cycle Length (days)")
            ax.set_title("Cycle History")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ML features
            X = np.arange(len(df_clean)).reshape(-1, 1)
            y = df_clean["cycle_length"].values.reshape(-1, 1)

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=0),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
                "SVM": SVR(kernel="rbf")
            }

            predictions = {}

            for name, model in models.items():
                model.fit(X, y.ravel())
                next_idx = np.array([[len(X)]])
                predicted_length = model.predict(next_idx)[0]
                predicted_date = last_date + timedelta(days=int(round(predicted_length)))
                predictions[name] = (predicted_length, predicted_date)

            # Show all predictions
            st.subheader("🔍 Individual Model Predictions")
            for name, (length, date) in predictions.items():
                st.write(f"**{name}** predicts:")
                st.markdown(f"- Cycle length: `{length:.2f}` days")
                st.markdown(f"- Start date: **{date.strftime('%Y-%m-%d')}**")

            # DataFrame
            st.subheader("📊 Consolidated Prediction Table")
            pred_df = pd.DataFrame({
                "Model": predictions.keys(),
                "Predicted Cycle Length (days)": [round(p[0], 2) for p in predictions.values()],
                "Predicted Start Date": [p[1].strftime("%Y-%m-%d") for p in predictions.values()]
            }).set_index("Model")
            st.dataframe(pred_df)

            # 🎯 1. Hard Voting (Most Common Date)
            date_series = pd.Series([p[1].strftime("%Y-%m-%d") for p in predictions.values()])
            most_common_date = date_series.value_counts().idxmax()
            st.success(f"🗳️ Most voted date: **{most_common_date}** (Hard Voting)")

            # 🎯 2. Ensemble (Average Predicted Length)
            weights = [1] * len(predictions)  # Equal weight for now
            weighted_avg_length = np.average([p[0] for p in predictions.values()], weights=weights)
            ensemble_date = last_date + timedelta(days=int(round(weighted_avg_length)))
            st.success(f"📊 Ensemble predicted date: **{ensemble_date.strftime('%Y-%m-%d')}** (Avg. Length: {weighted_avg_length:.2f} days)")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("Upload your cycle history CSV to begin.")

st.markdown("---")
st.markdown("📄 Example CSV format:")
st.code("start_date\n2024-01-01\n2024-01-29\n2024-02-27\n2024-03-28\n2024-04-25")
