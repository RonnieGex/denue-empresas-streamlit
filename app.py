
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
import plotly.express as px

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados. Ahora con IA especializada de DeepSeek.")

@st.cache_data
def cargar_datos(archivo):
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo, encoding="latin1", low_memory=False)
    else:
        df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip()
    return df

def estimar_empleados(valor):
    if pd.isna(valor):
        return None
    valor = str(valor).lower()
    if "a" in valor:
        partes = valor.split("a")
        try:
            return (int(partes[0].strip()) + int(partes[1].strip().split()[0])) // 2
        except:
            return None
    elif "menos de" in valor:
        try:
            return int(valor.split("menos de")[1].strip().split()[0]) - 1
        except:
            return None
    elif "más de" in valor:
        try:
            return int(valor.split("más de")[1].strip().split()[0]) + 1
        except:
            return None
    else:
        try:
            return int(valor.strip())
        except:
            return None

def get_recommendations_from_deepseek(api_key, prompt):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.splitlines()
    except Exception as e:
        return [f"❌ Error al conectar con DeepSeek: {str(e)}"]

def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de empresas", len(df_filtrado))
    if "per_ocu_estimado" in df_filtrado.columns and not df_filtrado["per_ocu_estimado"].empty:
        col2.metric("Promedio de empleados", round(df_filtrado["per_ocu_estimado"].mean(), 2))
    else:
        col2.metric("Promedio de empleados", "N/A")
    if "telefono" in df_filtrado.columns and "correoelec" in df_filtrado.columns:
        col3.metric("Empresas con contacto", df_filtrado[["telefono", "correoelec"]].dropna(how="all").shape[0])
    else:
        col3.metric("Empresas con contacto", "N/A")
    if "nombre_act" in df_filtrado.columns:
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
        top_giros.columns = ["Giro", "Cantidad"]
        fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos en tu Filtro")
        st.plotly_chart(fig, use_container_width=True)

archivo = st.file_uploader("📂 Sube tu archivo del sistema (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo:
    try:
        df = cargar_datos(archivo)
        columnas_clave = ["nom_estab", "nombre_act", "per_ocu", "telefono", "correoelec", "www", "municipio", "localidad", "entidad", "latitud", "longitud"]
        if not all(col in df.columns for col in columnas_clave):
            st.error("❌ El archivo no contiene todas las columnas necesarias.")
        else:
            df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
            df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)

            st.subheader("🧠 Filtros avanzados")
            with st.form("formulario_filtros"):
                estados = sorted(df["entidad"].dropna().unique())
                estado = st.selectbox("📍 Estado", estados)
                df_estado = df[df["entidad"] == estado]
                municipios = sorted(df_estado["municipio"].dropna().unique())
                municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)
                tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio", "")
                api_key = st.text_input("🔐 Ingresa tu API Key de DeepSeek", type="password")
                obtener_ia = st.form_submit_button("🤖 Obtener giros sugeridos")

            if obtener_ia and tipo_negocio and api_key:
                prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
                recomendaciones = get_recommendations_from_deepseek(api_key, prompt)
                st.markdown("🎯 **Sugerencias de giros:**")
                st.write(recomendaciones)
    except Exception as e:
        st.error("❌ Error crítico: " + str(e))
