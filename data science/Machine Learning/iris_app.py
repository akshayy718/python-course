
# iris_app.py
# KNN on Iris Dataset with Streamlit Interface
# Run with: streamlit run iris_app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ── Load and train model ──────────────────────
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, knn.predict(X_test_scaled))

# ── Streamlit UI ──────────────────────────────
st.title("🌸 Iris Flower Classifier")
st.subheader("Using K-Nearest Neighbors (KNN)")

st.markdown(f"**Model Accuracy: {round(accuracy * 100, 2)}%**")

st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider(
    "Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider(
    "Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
input_data = np.array([[sepal_length, sepal_width,
                         petal_length, petal_width]])
input_scaled = scaler.transform(input_data)
prediction = knn.predict(input_scaled)[0]
flower_name = iris.target_names[prediction]

# Show prediction
st.markdown("---")
st.header("Prediction Result")

if prediction == 0:
    st.success(f"🌸 Predicted Flower: **{flower_name.upper()}**")
elif prediction == 1:
    st.info(f"🌼 Predicted Flower: **{flower_name.upper()}**")
else:
    st.warning(f"🌺 Predicted Flower: **{flower_name.upper()}**")

# Show input values
st.markdown("---")
st.subheader("Your Input Values")
input_df = pd.DataFrame({
    'Feature': ['Sepal Length', 'Sepal Width',
                'Petal Length', 'Petal Width'],
    'Value (cm)': [sepal_length, sepal_width,
                   petal_length, petal_width]
})
st.table(input_df)

# Dataset overview
st.markdown("---")
st.subheader("Dataset Overview")
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = [iris.target_names[i] for i in iris.target]
st.dataframe(df.head(10))

# Bar chart of species distribution
st.markdown("---")
st.subheader("Species Distribution")
species_count = df['Species'].value_counts()
fig, ax = plt.subplots()
ax.bar(species_count.index, species_count.values,
       color=['lightblue', 'lightgreen', 'salmon'])
ax.set_xlabel('Species')
ax.set_ylabel('Count')
ax.set_title('Iris Species Count')
st.pyplot(fig)

# About section
st.markdown("---")
st.subheader("About This App")
st.write("""
- **Dataset**: Iris dataset from scikit-learn
- **Algorithm**: K-Nearest Neighbors (KNN) with k=5
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Classes**: Setosa, Versicolor, Virginica
- **Train/Test Split**: 80% / 20%
""")

# %%



