
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import openai
import anthropic

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados. Ahora con IA, métricas y visualización.")

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

def get_recommendations_from_ai(api_key, provider, prompt):
    try:
        if provider == "OpenAI (GPT-4)":
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        elif provider == "Anthropic (Claude)":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content
        elif provider == "DeepSeek":
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis de mercado."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        else:
            return "❌ Proveedor de IA no reconocido."
    except Exception as e:
        return f"❌ Error al conectar con el proveedor: {str(e)}"

def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de empresas", len(df_filtrado))
    if "per_ocu_estimado" in df_filtrado.columns:
        try:
            col2.metric("Promedio de empleados", round(df_filtrado["per_ocu_estimado"].mean(), 2))
        except:
            col2.metric("Promedio de empleados", "N/A")
    try:
        col3.metric("Empresas con contacto", df_filtrado[["telefono", "correoelec"]].dropna(how="all").shape[0])
    except:
        col3.metric("Empresas con contacto", "N/A")
    if "nombre_act" in df_filtrado.columns:
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
        top_giros.columns = ["Giro", "Cantidad"]
        fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos")
        st.plotly_chart(fig, use_container_width=True)

archivo = st.file_uploader("📂 Sube tu archivo del sistema (.csv o .xlsx)", type=["csv", "xlsx"])
if archivo:
    df = cargar_datos(archivo)
    columnas_requeridas = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 'correoelec', 'www',
                           'municipio', 'localidad', 'entidad', 'latitud', 'longitud']
    if not all(col in df.columns for col in columnas_requeridas):
        st.error("❌ El archivo no contiene todas las columnas necesarias.")
    else:
        df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
        df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)
        estado = st.selectbox("📍 Estado", sorted(df["entidad"].dropna().unique()))
        df_estado = df[df["entidad"] == estado]
        municipios = sorted(df_estado["municipio"].dropna().unique())
        municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)
        tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio")
        proveedor = st.selectbox("🔎 Selecciona proveedor de IA", ["OpenAI (GPT-4)", "Anthropic (Claude)", "DeepSeek"])
        api_key = st.text_input("🔐 Ingresa tu API Key", type="password")
        if st.button("📥 Obtener giros sugeridos") and tipo_negocio and api_key:
            prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
            recomendaciones = get_recommendations_from_ai(api_key, proveedor, prompt)
            st.markdown("🎯 **Sugerencias de giros:**")
            st.write(recomendaciones)
