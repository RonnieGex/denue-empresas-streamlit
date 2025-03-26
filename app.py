
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Katalis Ads Data Base Optimizer", layout="wide")
st.markdown("## 🚀 Katalis Ads Data Base Optimizer")
st.markdown("Optimiza tu base de datos del DENUE para campañas de Google Ads. Aplica filtros inteligentes por ubicación, giro, contacto, digitalización y tamaño.")

# Función para convertir texto en número estimado
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

archivo = st.file_uploader("📂 Sube tu archivo del DENUE (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo:
    try:
        if archivo.name.endswith("csv"):
            df = pd.read_csv(archivo, encoding="latin1")
        else:
            df = pd.read_excel(archivo)

        df.columns = df.columns.str.strip()

        columnas_necesarias = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 'correoelec', 'www',
                               'cod_postal', 'municipio', 'localidad', 'entidad', 'latitud', 'longitud']
        columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

        if columnas_faltantes:
            st.error(f"❌ Faltan las siguientes columnas en el archivo: {columnas_faltantes}")
        else:
            df['per_ocu_estimado'] = df['per_ocu'].apply(estimar_empleados)

            estado = st.selectbox("📍 Estado", sorted(df['entidad'].dropna().unique()))
            df_estado = df[df['entidad'] == estado]

            municipios = sorted(df_estado['municipio'].dropna().unique())
            municipios_seleccionados = st.multiselect("🏘️ Municipios", municipios, default=municipios)
            df_estado = df_estado[df_estado['municipio'].isin(municipios_seleccionados)]

            giros = sorted(df_estado['nombre_act'].dropna().unique())
            giros_seleccionados = st.multiselect("🏢 Giros económicos", giros)
            df_estado = df_estado[df_estado['nombre_act'].isin(giros_seleccionados)]

            df_estado = df_estado[df_estado['per_ocu_estimado'].notna()]
            if not df_estado.empty:
                min_emp = int(df_estado['per_ocu_estimado'].min())
                max_emp = int(df_estado['per_ocu_estimado'].max())
                rango_emp = st.slider("👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp))
                df_estado = df_estado[df_estado['per_ocu_estimado'].between(rango_emp[0], rango_emp[1])]
            else:
                st.warning("⚠️ No hay datos válidos para filtrar por número de empleados.")

            buscar_nombre = st.text_input("🔎 Buscar palabra clave en el nombre del negocio", "").strip().lower()
            if buscar_nombre:
                df_estado = df_estado[df_estado['nom_estab'].str.lower().str.contains(buscar_nombre)]

            codigos_postales = sorted(df_estado['cod_postal'].dropna().unique())
            codigos_seleccionados = st.multiselect("📮 Código Postal", codigos_postales)
            if codigos_seleccionados:
                df_estado = df_estado[df_estado['cod_postal'].isin(codigos_seleccionados)]

            if st.checkbox("📞 Solo empresas con teléfono", value=True):
                df_estado = df_estado[df_estado["telefono"].notna()]
            if st.checkbox("📧 Solo empresas con correo electrónico", value=False):
                df_estado = df_estado[df_estado["correoelec"].notna()]
            if st.checkbox("🌐 Solo empresas con sitio web", value=False):
                df_estado = df_estado[df_estado["www"].notna()]

            if st.button("📊 Mostrar resultados y descargar"):
                columnas = [
                    "nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                    "telefono", "correoelec", "www",
                    "cod_postal", "municipio", "localidad", "entidad",
                    "latitud", "longitud"
                ]

                df_final = df_estado[columnas].copy()
                df_final.columns = [
                    "Nombre", "Giro", "Personal Original", "Personal Estimado",
                    "Teléfono", "Correo", "Sitio Web",
                    "C.P.", "Municipio", "Localidad", "Estado",
                    "Latitud", "Longitud"
                ]

                st.success(f"✅ Empresas encontradas: {len(df_final)}")
                st.dataframe(df_final.head(10), use_container_width=True)

                csv = df_final.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Descargar archivo CSV optimizado",
                    data=csv,
                    file_name="empresas_filtradas_katalis.csv",
                    mime="text/csv"
                )

                if not df_final[["Latitud", "Longitud"]].dropna().empty:
                    st.markdown("### 🗺️ Mapa interactivo de empresas")
                    mapa = folium.Map(location=[
                        df_final["Latitud"].dropna().astype(float).mean(),
                        df_final["Longitud"].dropna().astype(float).mean()
                    ], zoom_start=10)

                    marker_cluster = MarkerCluster().add_to(mapa)
                    for _, row in df_final.dropna(subset=["Latitud", "Longitud"]).iterrows():
                        folium.Marker(
                            location=[float(row["Latitud"]), float(row["Longitud"])],
                            popup=f"{row['Nombre']}<br>{row['Giro']}"
                        ).add_to(marker_cluster)

                    st_folium(mapa, width=1000, height=600)
                else:
                    st.warning("⚠️ No hay coordenadas disponibles para mostrar el mapa.")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
