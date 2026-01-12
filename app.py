import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Judul Aplikasi
st.set_page_config(page_title="Diagnosa COVID-19", page_icon="🦠")
st.title("🦠Website Diagnosa Gejala COVID-19")
st.write("Masukkan gejala yang Anda alami untuk mengetahui hasil diagnosa.")

# Dataset
data = {
    'Demam': [1,1,0,1,0,1,0,1,0,1],
    'Batuk': [1,1,1,1,0,1,1,1,0,1],
    'Sesak': [1,0,0,1,0,0,0,1,0,0],
    'Tenggorokan': [1,1,1,0,1,1,0,1,0,1],
    'Penciuman': [1,1,0,1,0,1,0,1,0,1],
    'Diagnosa': ['Positif','Positif','Negatif','Positif','Negatif','Positif','Negatif','Positif','Negatif','Positif']
}

df = pd.DataFrame(data)

X = df.drop('Diagnosa', axis=1)
y = df['Diagnosa']

# Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Input User
st.header("Masukkan Gejala")

demam = st.selectbox("Apakah Anda mengalami Demam?", ["Tidak", "Ya"])
batuk = st.selectbox("Apakah Anda mengalami Batuk?", ["Tidak", "Ya"])
sesak = st.selectbox("Apakah Anda mengalami Sesak Nafas?", ["Tidak", "Ya"])
tenggorokan = st.selectbox("Apakah Anda mengalami Sakit Tenggorokan?", ["Tidak", "Ya"])
penciuman = st.selectbox("Apakah Anda kehilangan Penciuman?", ["Tidak", "Ya"])

# Konversi ke angka
def convert(val):
    return 1 if val == "Ya" else 0

demam = convert(demam)
batuk = convert(batuk)
sesak = convert(sesak)
tenggorokan = convert(tenggorokan)
penciuman = convert(penciuman)

# Tombol Diagnosa
if st.button("🔍 Diagnosa Sekarang"):
    sample = pd.DataFrame([[demam, batuk, sesak, tenggorokan, penciuman]], columns=X.columns)
    hasil = model.predict(sample)

    if hasil[0] == "Positif":
        st.error("⚠️ Hasil Diagnosa: POSITIF COVID-19")
    else:
        st.success("✅ Hasil Diagnosa: NEGATIF COVID-19")

    st.info("Hasil ini hanya sebagai simulasi pembelajaran yang dibuat oleh Fauzan Akbar Ansyori, bukan diagnosa medis resmi.")