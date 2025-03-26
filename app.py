
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Depurador de archivo DENUE", layout="wide")
st.title("🧠 Depurador de archivo DENUE - Detección de columnas")

archivo = st.file_uploader("📂 Sube tu archivo del DENUE (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo:
    try:
        if archivo.name.endswith("csv"):
            df = pd.read_csv(archivo, encoding="latin1")
        else:
            df = pd.read_excel(archivo)
        df.columns = df.columns.str.strip()  # Elimina espacios invisibles
        columnas = df.columns.tolist()

        st.success("✅ Archivo cargado correctamente.")
        st.write("🧾 Estas son las columnas encontradas en tu archivo:")
        st.write(columnas)

        # Validación de columnas clave esperadas
        columnas_esperadas = {
            "nom_ent": "Estado",
            "nombre_act": "Giro económico",
            "estrato": "Tamaño de empresa",
            "telefono": "Teléfono",
            "correoelec": "Correo electrónico",
            "www": "Sitio web",
            "codpos": "Código postal",
            "nom_mun": "Municipio",
            "nom_estab": "Nombre del establecimiento",
            "latitud": "Latitud",
            "longitud": "Longitud"
        }

        st.markdown("### 📋 Estado de columnas requeridas:")
        for col, desc in columnas_esperadas.items():
            if col in columnas:
                st.success(f"✔️ '{col}' encontrada ✅ ({desc})")
            else:
                st.error(f"❌ '{col}' NO encontrada 🚫 ({desc})")

    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {str(e)}")
