import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import streamlit as st

knn_model = joblib.load('C:\Hoc_May\Full_LinhTinh\DoAnCN\knn_model.joblib')
scaler = joblib.load('C:\Hoc_May\Full_LinhTinh\DoAnCN\scaler.pkl')

feature_columns_order = ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean',
                          'symmetry_mean', 'fractal_dimension_mean', 'area_se', 'texture_se',
                          'smoothness_se', 'concavity_se', 'symmetry_se', 'fractal_dimension_se',
                          'smoothness_worst', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst']

image_path = 'CT.jpg'

# Hiển thị ảnh
st.image(image_path, width=700)

st.title(":blue[Phân loại ung thư vú]")

st.sidebar.title("Chỉ số xét nghiệm:")  
texture_mean = st.sidebar.slider("Texture Mean", min_value=0.0, max_value=39.28, value=0.0, step=0.01, format="%.2f")
area_mean = st.sidebar.slider("Area Mean", min_value=0.0, max_value=2501.0, value=0.0, step=0.1, format="%.1f")
smoothness_mean = st.sidebar.slider("Smoothness Mean", min_value=0.0, max_value=0.1634, value=0.0, step=0.0001, format="%.4f")
concavity_mean = st.sidebar.slider("Concavity Mean", min_value=0.0, max_value=0.4268, value=0.0, step=0.0001, format="%.4f")
area_se = st.sidebar.slider("area_se", min_value=0.0, max_value=542.2, value=0.0, step=0.1, format="%.1f")
concavity_se = st.sidebar.slider("concavity_se", min_value=0.0, max_value=0.396, value=0.0, step=0.001, format="%.3f")
symmetry_mean = st.sidebar.slider("Symmetry Mean", min_value=0.0, max_value=0.304, value=0.0, step=0.001, format="%.3f")
fractal_dimension_mean = st.sidebar.slider("Fractal Dimension Mean", min_value=0.0, max_value=0.15, value=0.0, step=0.01, format="%.2f")
texture_se = st.sidebar.slider("Texture SE", min_value=0.0, max_value=4.885, value=0.0, step=0.001, format="%.3f")
smoothness_se = st.sidebar.slider("Smoothness SE", min_value=0.0, max_value=0.031130, value=0.0, step=0.000001, format="%.6f")
symmetry_se = st.sidebar.slider("Symmetry SE", min_value=0.0, max_value=0.7895, value=0.0, step=0.0001, format="%.4f")
fractal_dimension_se = st.sidebar.slider("Fractal Dimension SE", min_value=0.0, max_value=0.029840, value=0.0, step=0.000001, format="%.6f")
smoothness_worst = st.sidebar.slider("Smoothness Worst", min_value=0.0, max_value=0.2226, value=0.0, step=0.0001, format="%.4f")
concavity_worst = st.sidebar.slider("Concavity Worst", min_value=0.0, max_value=1.252, value=0.0, step=0.001, format="%.3f")
symmetry_worst = st.sidebar.slider("Symmetry Worst", min_value=0.0, max_value=0.6638, value=0.0, step=0.0001, format="%.4f")
fractal_dimension_worst = st.sidebar.slider("Fractal Dimension Worst", min_value=0.0, max_value=0.2075, value=0.0, step=0.0001, format="%.4f")


new_data_for_prediction = pd.DataFrame({
    'texture_mean': [texture_mean],
    'area_se': [area_se],  # Keep the slider value for 'area_se'
    'concavity_se': [concavity_se],
    'area_mean': [area_mean],
    'smoothness_mean': [smoothness_mean],
    'concavity_mean': [concavity_mean],
    'symmetry_mean': [symmetry_mean],
    'fractal_dimension_mean': [fractal_dimension_mean],
    'texture_se': [texture_se],
    'smoothness_se': [smoothness_se],
    'symmetry_se': [symmetry_se],
    'fractal_dimension_se': [fractal_dimension_se],
    'smoothness_worst': [smoothness_worst],
    'concavity_worst': [concavity_worst],
    'symmetry_worst': [symmetry_worst],
    'fractal_dimension_worst': [fractal_dimension_worst]
})

new_data_for_prediction = pd.DataFrame({
    'texture_mean': [texture_mean],
    'area_se': [area_se],
    'concavity_se': [concavity_se],
    'area_mean': [area_mean],
    'smoothness_mean': [smoothness_mean],
    'concavity_mean': [concavity_mean],
    'symmetry_mean': [symmetry_mean],
    'fractal_dimension_mean': [fractal_dimension_mean],
    'texture_se': [texture_se],
    'smoothness_se': [smoothness_se],
    'symmetry_se': [symmetry_se],
    'fractal_dimension_se': [fractal_dimension_se],
    'smoothness_worst': [smoothness_worst],
    'concavity_worst': [concavity_worst],
    'symmetry_worst': [symmetry_worst],
    'fractal_dimension_worst': [fractal_dimension_worst]
}, columns=feature_columns_order)  # Set the column order explicitly

new_data_scaled = scaler.transform(new_data_for_prediction)

prediction = knn_model.predict(new_data_scaled)

st.info("Chú thích: \n- Benign (B) : Lành tính\n - Malignant (M) : Ác tính \n\n \t :red[Người dùng kéo thanh chỉ số của kết quả xét nghiệm để hiển thị kết quả chẩn đoán !]")

st.subheader("Kết quả chẩn đoán:")
if prediction[0] == 0:
    st.warning("Bệnh nhân mắc ung thư vú có khả là: **:green[Lành tính (Benign)]**.")
else:
    st.warning("Bệnh nhân mắc ung thư vú có khả là: **:red[Ác tính (Malignant)]**.")