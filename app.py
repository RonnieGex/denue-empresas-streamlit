
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px

try:
    import openai
    openai_version = int(openai.__version__.split(".")[0])
except:
    openai_version = 0

import anthropic

# Recomendaciones desde IA
def get_recommendations_from_ai(api_key, provider, prompt):
    if provider == "OpenAI (GPT-4)":
        if openai_version >= 1:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        else:
            return "La versión del paquete OpenAI es obsoleta. Por favor actualiza con: pip install openai --upgrade"
    elif provider == "Anthropic (Claude)":
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
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
        return "Proveedor no válido"

# Métricas
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

    if "nombre_act" in df_filtrado.columns:
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
        top_giros.columns = ["Giro", "Cantidad"]
        fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos en tu Filtro")
        st.plotly_chart(fig, use_container_width=True)

# Carga de archivo
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
    try:
        if "a" in valor:
            partes = valor.split("a")
            return (int(partes[0].strip()) + int(partes[1].strip().split()[0])) // 2
        elif "menos de" in valor:
            return int(valor.split("menos de")[1].strip().split()[0]) - 1
        elif "más de" in valor:
            return int(valor.split("más de")[1].strip().split()[0]) + 1
        else:
            return int(valor.strip())
    except:
        return None

# Streamlit config
st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados.")

archivo = st.file_uploader("📂 Sube tu archivo (.csv o .xlsx)", type=["csv", "xlsx"])
if archivo:
    df = cargar_datos(archivo)
    columnas_clave = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 'correoelec', 'www',
                      'municipio', 'localidad', 'entidad', 'latitud', 'longitud']
    if not all(col in df.columns for col in columnas_clave):
        st.error("❌ El archivo no contiene todas las columnas necesarias.")
    else:
        df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
        df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)
        estados = sorted(df["entidad"].dropna().unique())
        st.subheader("🧠 Filtros avanzados")
        estado = st.selectbox("📍 Estado", estados)
        df_estado = df[df["entidad"] == estado]
        municipios = sorted(df_estado["municipio"].dropna().unique())
        municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)
        tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio")
        proveedor = st.selectbox("🤖 Proveedor de IA", ["OpenAI (GPT-4)", "Anthropic (Claude)", "DeepSeek"])
        api_key = st.text_input("🔐 API Key", type="password")
        if st.button("🔎 Obtener recomendaciones de giros con IA"):
            if tipo_negocio and api_key:
                prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
                try:
                    recomendaciones = get_recommendations_from_ai(api_key, proveedor, prompt)
                    st.markdown("🎯 **Recomendaciones de IA para tu negocio:**")
                    st.write(recomendaciones if isinstance(recomendaciones, str) else str(recomendaciones))
                except Exception as e:
                    st.error("❌ Error al consultar IA: " + str(e))
            else:
                st.warning("Ingresa una descripción de negocio y tu API key.")

# El resto del sistema puede agregarse después
