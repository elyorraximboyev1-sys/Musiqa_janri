import streamlit as st
import joblib 
import numpy as np

# 1. Modelni yuklash (Siz ko'rsatgan pkl fayli)
bundle = joblib.load(r"C:\Users\user\Desktop\Elyor\knn_oqitildi.pkl")
model = bundle['model']
best_cols = bundle['ustunlar']
label_map = bundle['label_map']
scaler = bundle['scaler']

# Janrlar xaritasini teskarisiga o'giramiz (Raqam -> Nom)
res_map = {v: k for k, v in label_map.items()}

st.title("Music Genre Prediction with KNN model")
st.write(f"Ishlatilayotgan ustunlar: {best_cols}")

# O'zbekcha nomlar lug'ati
ozbekcha = {
    'tempo': 'Musiqa sur\'ati (Tempo)',
    'energy': 'Energiya darajasi (Energy)'
}

inputs = []
for col in best_cols:
    if col == 'tempo':
        qiymat = st.number_input(
            f"{ozbekcha[col]} ni kiriting:",
            min_value=0.0, max_value=250.0, value=120.0, step=1.0
        )
    elif col == 'energy':
        qiymat = st.number_input(
            f"{ozbekcha[col]} ni kiriting (0.0 dan 1.0 gacha):",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f"
        )
    else:
        qiymat = st.number_input(f"{col} ni kiriting:", value=0.0)
    
    inputs.append(qiymat)

# Natijani bashorat qilish tugmasi
if st.button("Janrni aniqlash"):
    # Kiruvchi ma'lumotni massivga o'tkazamiz
    X_raw = np.array([inputs], dtype=float)
    
    # KNN uchun normallashtirish (Scaling) shart
    X_scaled = scaler.transform(X_raw)

    # Bashorat ehtimoli
    proba_list = model.predict_proba(X_scaled)[0]
    pred_index = np.argmax(proba_list) # Eng yuqori ehtimollik indeksi
    proba = proba_list[pred_index]
    
    # Bashorat natijasi (Matn ko'rinishida)
    predicted_genre = res_map[pred_index]

    st.write('Kiruvchi ma\'lumotlar (X):', X_raw)
    st.write('Natija indeksi:', pred_index)
    st.write('Bashorat ishonchliligi:', f"{proba:.2f}")

    # Janrga qarab rangli xabarlar chiqarish
    if predicted_genre == 'Rock':
        st.error(f"Bashorat: {predicted_genre} 🎸")
    elif predicted_genre == 'Pop':
        st.info(f"Bashorat: {predicted_genre} 🎤")
    elif predicted_genre == 'Classical':
        st.success(f"Bashorat: {predicted_genre} 🎻")
    else:
        st.warning(f"Bashorat: {predicted_genre}")
