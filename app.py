
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Katalis Ads DB Optimizer", layout="wide")
st.markdown("# Katalis Ads DB Optimizer")
st.markdown("Optimiza tu base del DENUE para campañas B2B en Google Ads con filtros avanzados y ejecución controlada.")

# Inicializar session_state
for key in ["mostrar_resultados", "df_final", "filtros"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key == "filtros" else False if key == "mostrar_resultados" else pd.DataFrame()

@st.cache_data
def cargar_datos(archivo):
    if archivo.name.endswith("csv"):
        df = pd.read_csv(archivo, encoding="latin1", low_memory=False)
    else:
        df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip()
    return df

def estimar_empleados(valor):
    if pd.isna(valor):
        return None
    valor = str(valor).lower().strip()
    if "1 a 5" in valor: return 3
    elif "6 a 10" in valor: return 8
    elif "11 a 30" in valor: return 20
    elif "31 a 50" in valor: return 40
    elif "51 a 100" in valor: return 75
    elif "101 a 250" in valor: return 175
    elif "251 y más" in valor or "más de 250" in valor: return 300
    elif "más de 100" in valor: return 150
    try: return int(valor)
    except: return None

archivo = st.file_uploader("📂 Sube tu archivo DENUE (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo:
    try:
        df = cargar_datos(archivo)
        columnas = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 'correoelec', 'www',
                    'municipio', 'localidad', 'entidad', 'latitud', 'longitud']
        columnas_faltantes = [col for col in columnas if col not in df.columns]
        if columnas_faltantes:
            st.error(f"❌ Columnas faltantes: {columnas_faltantes}")
        else:
            df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
            df['per_ocu_estimado'] = df['per_ocu'].apply(estimar_empleados)
            estados = sorted(df['entidad'].dropna().unique())

            with st.expander("🎛️ Filtros avanzados", expanded=True):
                st.session_state.filtros['estado'] = st.selectbox("📍 Estado", estados)

                df_estado = df[df['entidad'] == st.session_state.filtros['estado']]
                municipios = sorted(df_estado['municipio'].dropna().unique())
                st.session_state.filtros['municipios'] = st.multiselect("🏘️ Municipios", municipios, default=municipios)

                giros = sorted(df_estado['nombre_act'].dropna().unique())
                st.session_state.filtros['giros'] = st.multiselect("🏢 Giros económicos", giros)

                empleados = df_estado['per_ocu_estimado'].dropna()
                min_emp = int(empleados.min()) if not empleados.empty else 0
                max_emp = int(empleados.max()) if not empleados.empty else 500
                st.session_state.filtros['rango_emp'] = st.slider(
                    "👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp)
                )

                st.session_state.filtros['buscar_nombre'] = st.text_input("🔎 Buscar palabra clave en nombre del negocio").strip().lower()
                st.session_state.filtros['telefono'] = st.checkbox("📞 Solo empresas con teléfono", value=True)
                st.session_state.filtros['correo'] = st.checkbox("📧 Solo con correo electrónico", value=False)
                st.session_state.filtros['web'] = st.checkbox("🌐 Solo con sitio web", value=False)

            if st.button("📊 Mostrar resultados y descargar"):
                f = st.session_state.filtros
                filtrado = df[
                    (df['entidad'] == f['estado']) &
                    (df['municipio'].isin(f['municipios'])) &
                    (df['per_ocu_estimado'].between(f['rango_emp'][0], f['rango_emp'][1]))
                ]
                if f['giros']:
                    filtrado = filtrado[filtrado['nombre_act'].isin(f['giros'])]
                if f['buscar_nombre']:
                    filtrado = filtrado[filtrado['nom_estab'].str.lower().str.contains(f['buscar_nombre'])]
                if f['telefono']:
                    filtrado = filtrado[filtrado['telefono'].notna()]
                if f['correo']:
                    filtrado = filtrado[filtrado['correoelec'].notna()]
                if f['web']:
                    filtrado = filtrado[filtrado['www'].notna()]

                columnas_mostrar = ["nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                                    "telefono", "correoelec", "www", "municipio", "localidad", "entidad",
                                    "latitud", "longitud"]
                filtrado = filtrado[columnas_mostrar]
                filtrado.columns = ["Nombre", "Giro", "Personal (texto)", "Personal Estimado",
                                    "Teléfono", "Correo", "Web", "Municipio", "Localidad", "Estado",
                                    "Latitud", "Longitud"]

                st.session_state.df_final = filtrado
                st.session_state.mostrar_resultados = True

            if st.session_state.mostrar_resultados and not st.session_state.df_final.empty:
                st.success(f"✅ Empresas encontradas: {len(st.session_state.df_final)}")
                mostrar = st.slider("🔢 ¿Cuántos resultados mostrar en pantalla?", 10, 500, 50)
                st.dataframe(st.session_state.df_final.head(mostrar), use_container_width=True)

                csv = st.session_state.df_final.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar CSV optimizado", data=csv, file_name="denue_opt.csv", mime="text/csv")

                if not st.session_state.df_final[["Latitud", "Longitud"]].dropna().empty:
                    st.markdown("### 🗺️ Mapa interactivo")
                    mapa = folium.Map(location=[
                        st.session_state.df_final["Latitud"].dropna().astype(float).mean(),
                        st.session_state.df_final["Longitud"].dropna().astype(float).mean()
                    ], zoom_start=10)

                    marker_cluster = MarkerCluster().add_to(mapa)
                    for _, row in st.session_state.df_final.head(200).dropna(subset=["Latitud", "Longitud"]).iterrows():
                        folium.Marker(
                            location=[float(row["Latitud"]), float(row["Longitud"])],
                            popup=f"{row['Nombre']}<br>{row['Giro']}"
                        ).add_to(marker_cluster)

                    st_folium(mapa, width=1000, height=600)
                else:
                    st.warning("⚠️ No hay coordenadas para mostrar el mapa.")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: " + str(e))
