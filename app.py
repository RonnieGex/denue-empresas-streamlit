
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
import plotly.express as px

# Configuración de cliente para DeepSeek
def get_deepseek_response(api_key, prompt):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            return "❌ No se recibió respuesta válida del modelo DeepSeek."
    except Exception as e:
        return f"❌ Error al conectar con DeepSeek: {str(e)}"

# Mostrar métricas y gráficos
def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de empresas", len(df_filtrado))
    if "per_ocu_estimado" in df_filtrado.columns and not df_filtrado["per_ocu_estimado"].empty:
        col2.metric("Promedio de empleados", round(df_filtrado["per_ocu_estimado"].mean(), 2))
    else:
        col2.metric("Promedio de empleados", "N/A")
    col3.metric("Empresas con contacto", df_filtrado[["Teléfono", "Correo"]].dropna(how="all").shape[0])
    if "Giro" in df_filtrado.columns:
        top_giros = df_filtrado["Giro"].value_counts().nlargest(5).reset_index()
        top_giros.columns = ["Giro", "Cantidad"]
        fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos en tu Filtro")
        st.plotly_chart(fig, use_container_width=True)

# Configuración de la app
st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados y sugerencias de IA.")

# Cargar datos
@st.cache_data
def cargar_datos(archivo):
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo, encoding="latin1", low_memory=False)
    else:
        df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip()
    return df

# Estimar empleados
def estimar_empleados(valor):
    try:
        valor = str(valor).lower()
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

archivo = st.file_uploader("📂 Sube tu archivo (.csv o .xlsx)", type=["csv", "xlsx"])

# Estados de sesión
if "mostrar_resultados" not in st.session_state:
    st.session_state.mostrar_resultados = False
if "df_final" not in st.session_state:
    st.session_state.df_final = pd.DataFrame()
if "filtros" not in st.session_state:
    st.session_state.filtros = {}

if archivo:
    try:
        df = cargar_datos(archivo)
        columnas_clave = ["nom_estab", "nombre_act", "per_ocu", "telefono", "correoelec", "www", "municipio", "localidad", "entidad", "latitud", "longitud"]
        if not all(col in df.columns for col in columnas_clave):
            st.error("❌ El archivo no contiene todas las columnas necesarias.")
        else:
            df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
            df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)
            estados = sorted(df["entidad"].dropna().unique())
            with st.form("filtros_form"):
                st.subheader("🧠 Filtros avanzados")
                estado = st.selectbox("📍 Estado", estados)
                df_estado = df[df["entidad"] == estado]
                municipios = sorted(df_estado["municipio"].dropna().unique())
                municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)
                tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio", "")
                api_key = st.text_input("🔐 Ingresa tu API Key de DeepSeek", type="password")
                if tipo_negocio and api_key:
                    if st.form_submit_button("🤖 Obtener giros sugeridos"):
                        prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
                        recomendaciones = get_deepseek_response(api_key, prompt)
                        st.markdown("🎯 **Sugerencias de giros:**")
                        st.write(recomendaciones.splitlines())

        # Mostrar resultados fuera del form
        if st.session_state.df_final is not None and not st.session_state.df_final.empty:
            df_final = st.session_state.df_final
            st.success(f"✅ Empresas encontradas: {len(df_final)}")
            limite = st.slider("🔢 ¿Cuántos resultados mostrar en pantalla?", 10, 500, 50)
            st.dataframe(df_final.head(limite), use_container_width=True)
            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar CSV optimizado", csv, file_name="denue_opt.csv", mime="text/csv")
            mostrar_metricas(df_final)
            if not df_final[["Latitud", "Longitud"]].dropna().empty:
                st.subheader("🗺️ Mapa interactivo")
                mapa = folium.Map(location=[
                    df_final["Latitud"].dropna().astype(float).mean(),
                    df_final["Longitud"].dropna().astype(float).mean()
                ], zoom_start=11)
                cluster = MarkerCluster().add_to(mapa)
                for _, row in df_final.dropna(subset=["Latitud", "Longitud"]).iterrows():
                    folium.Marker(
                        location=[float(row["Latitud"]), float(row["Longitud"])],
                        popup=f"{row['Nombre']}<br>{row['Giro']}"
                    ).add_to(cluster)
                st_folium(mapa, height=500, width=900)
            else:
                st.info("No hay coordenadas disponibles para mostrar el mapa.")
    except Exception as e:
        st.error("❌ Error crítico: " + str(e))
