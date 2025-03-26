
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.markdown("# 🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos del DENUE con filtros avanzados para campañas B2B en Google Ads. ✨ Ahora con mayor velocidad, control y precisión.")

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
    if "1 a 5" in valor:
        return 3
    elif "6 a 10" in valor:
        return 8
    elif "11 a 30" in valor:
        return 20
    elif "31 a 50" in valor:
        return 40
    elif "51 a 100" in valor:
        return 75
    elif "101 a 250" in valor:
        return 175
    elif "251 y más" in valor or "más de 250" in valor:
        return 300
    elif "más de 100" in valor:
        return 150
    try:
        return int(valor)
    except:
        return None

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

            with st.expander("🎛️ Filtros avanzados"):
                estado = st.selectbox("📍 Estado", sorted(df['entidad'].dropna().unique()))
                municipios = sorted(df[df['entidad'] == estado]['municipio'].dropna().unique())
                municipios_seleccionados = st.multiselect("🏘️ Municipios", municipios, default=municipios)

                giros = sorted(df['nombre_act'].dropna().unique())
                giros_seleccionados = st.multiselect("🏢 Giros económicos", giros)

                df = df[df['per_ocu_estimado'].notna()]
                if not df.empty:
                    min_emp = int(df['per_ocu_estimado'].min())
                    max_emp = int(df['per_ocu_estimado'].max())
                    rango_emp = st.slider("👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp))
                else:
                    rango_emp = None

                buscar_nombre = st.text_input("🔎 Buscar palabra clave en nombre del negocio").strip().lower()

                mostrar_tel = st.checkbox("📞 Solo empresas con teléfono", value=True)
                mostrar_mail = st.checkbox("📧 Solo con correo electrónico", value=False)
                mostrar_web = st.checkbox("🌐 Solo con sitio web", value=False)

            if st.button("📊 Mostrar resultados y descargar"):
                filtrado = df[df['entidad'] == estado]
                filtrado = filtrado[filtrado['municipio'].isin(municipios_seleccionados)]
                if giros_seleccionados:
                    filtrado = filtrado[filtrado['nombre_act'].isin(giros_seleccionados)]
                if rango_emp:
                    filtrado = filtrado[filtrado['per_ocu_estimado'].between(rango_emp[0], rango_emp[1])]
                if buscar_nombre:
                    filtrado = filtrado[filtrado['nom_estab'].str.lower().str.contains(buscar_nombre)]
                if mostrar_tel:
                    filtrado = filtrado[filtrado['telefono'].notna()]
                if mostrar_mail:
                    filtrado = filtrado[filtrado['correoelec'].notna()]
                if mostrar_web:
                    filtrado = filtrado[filtrado['www'].notna()]

                st.success(f"✅ Empresas encontradas: {len(filtrado)}")
                mostrar = st.slider("🔢 ¿Cuántos resultados mostrar en pantalla?", 10, 500, 50)
                columnas_mostrar = ["nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                                    "telefono", "correoelec", "www", "municipio", "localidad", "entidad",
                                    "latitud", "longitud"]
                filtrado = filtrado[columnas_mostrar]
                filtrado.columns = ["Nombre", "Giro", "Personal (texto)", "Personal Estimado",
                                    "Teléfono", "Correo", "Web", "Municipio", "Localidad", "Estado",
                                    "Latitud", "Longitud"]
                st.dataframe(filtrado.head(mostrar), use_container_width=True)

                csv = filtrado.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar CSV optimizado", data=csv, file_name="denue_opt.csv", mime="text/csv")

                if not filtrado[["Latitud", "Longitud"]].dropna().empty:
                    st.markdown("### 🗺️ Mapa interactivo")
                    mapa = folium.Map(location=[
                        filtrado["Latitud"].dropna().astype(float).mean(),
                        filtrado["Longitud"].dropna().astype(float).mean()
                    ], zoom_start=10)

                    marker_cluster = MarkerCluster().add_to(mapa)
                    for _, row in filtrado.head(200).dropna(subset=["Latitud", "Longitud"]).iterrows():
                        folium.Marker(
                            location=[float(row["Latitud"]), float(row["Longitud"])],
                            popup=f"{row['Nombre']}<br>{row['Giro']}"
                        ).add_to(marker_cluster)

                    st_folium(mapa, width=1000, height=600)
                else:
                    st.warning("⚠️ No hay coordenadas para mostrar el mapa.")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
