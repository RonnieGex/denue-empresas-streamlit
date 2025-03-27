
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import openai
import plotly.express as px

# Solo se usará DeepSeek como proveedor
def get_recommendations_from_deepseek(api_key, prompt):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de mercado."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error al conectar con DeepSeek: {e}"

def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de empresas", len(df_filtrado))
    if "Personal Estimado" in df_filtrado.columns and not df_filtrado["Personal Estimado"].empty:
        col2.metric("Promedio de empleados", round(df_filtrado["Personal Estimado"].mean(), 2))
    else:
        col2.metric("Promedio de empleados", "N/A")
    try:
        col3.metric("Empresas con contacto", df_filtrado[["Teléfono", "Correo"]].dropna(how="all").shape[0])
    except KeyError:
        col3.metric("Empresas con contacto", "Error en columnas")

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados usando DeepSeek IA.")

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

archivo = st.file_uploader("📂 Sube tu archivo del sistema (.csv o .xlsx)", type=["csv", "xlsx"])

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
                if st.form_submit_button("📡 Obtener giros sugeridos"):
                    if tipo_negocio and api_key:
                        prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
                        recomendaciones = get_recommendations_from_deepseek(api_key, prompt)
                        st.markdown("🎯 **Sugerencias de giros:**")
                        st.write(recomendaciones.splitlines())
                giros = sorted(df_estado["nombre_act"].dropna().unique())
                giros_sel = st.multiselect("🏢 Giros económicos", giros)
                emp_validos = df_estado["per_ocu_estimado"].dropna().astype(int)
                min_emp = int(emp_validos.min()) if not emp_validos.empty else 0
                max_emp = int(emp_validos.max()) if not emp_validos.empty else 500
                rango_emp = st.slider("👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp))
                nombre_busqueda = st.text_input("🔎 Buscar palabra clave en nombre del negocio")
                col1, col2, col3 = st.columns(3)
                con_tel = col1.checkbox("📞 Solo con teléfono", value=False)
                con_mail = col2.checkbox("📧 Solo con correo electrónico", value=False)
                con_web = col3.checkbox("🌐 Solo con sitio web", value=False)
                if st.form_submit_button("📊 Mostrar resultados y descargar"):
                    st.session_state.filtros = {
                        "estado": estado,
                        "municipios_sel": municipios_sel,
                        "giros_sel": giros_sel,
                        "rango_emp": rango_emp,
                        "nombre_busqueda": nombre_busqueda,
                        "con_tel": con_tel,
                        "con_mail": con_mail,
                        "con_web": con_web
                    }
                    st.session_state.mostrar_resultados = True

            if st.session_state.mostrar_resultados:
                f = st.session_state.filtros
                filtrado = df[
                    (df["entidad"] == f["estado"]) &
                    (df["municipio"].isin(f["municipios_sel"])) &
                    (df["per_ocu_estimado"].between(f["rango_emp"][0], f["rango_emp"][1]))
                ]
                if f["giros_sel"]:
                    filtrado = filtrado[filtrado["nombre_act"].isin(f["giros_sel"])]
                if f["nombre_busqueda"]:
                    filtrado = filtrado[filtrado["nom_estab"].str.lower().str.contains(f["nombre_busqueda"].lower())]
                if f["con_tel"]:
                    filtrado = filtrado[filtrado["telefono"].notna()]
                if f["con_mail"]:
                    filtrado = filtrado[filtrado["correoelec"].notna()]
                if f["con_web"]:
                    filtrado = filtrado[filtrado["www"].notna()]
                columnas_exportar = [
                    "nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                    "telefono", "correoelec", "www", "municipio", "localidad",
                    "entidad", "latitud", "longitud"
                ]
                df_final = filtrado[columnas_exportar].copy()
                df_final.columns = ["Nombre", "Giro", "Personal (texto)", "Personal Estimado", "Teléfono", "Correo", "Web", "Municipio", "Localidad", "Estado", "Latitud", "Longitud"]
                st.session_state.df_final = df_final
                st.success(f"✅ Empresas encontradas: {len(df_final)}")
                st.dataframe(df_final.head(100), use_container_width=True)
                csv = df_final.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar CSV optimizado", csv, file_name="denue_opt.csv", mime="text/csv")
                mostrar_metricas(df_final)
                if not df_final[["Latitud", "Longitud"]].dropna().empty:
                    st.subheader("🗺️ Mapa interactivo")
                    mapa = folium.Map(
                        location=[df_final["Latitud"].dropna().astype(float).mean(), df_final["Longitud"].dropna().astype(float).mean()],
                        zoom_start=11
                    )
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
        st.error(f"❌ Error crítico: {str(e)}")
