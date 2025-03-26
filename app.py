
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Filtrador de Empresas DENUE", layout="wide")
st.title("📊 Filtrador de Empresas DENUE para Google Ads")

st.markdown("Sube tu archivo .csv o .xlsx del DENUE original para comenzar a filtrar empresas por estado, giro económico, tamaño y disponibilidad de teléfono. El resultado podrá verse en tabla, mapa y podrás descargarlo en formato CSV optimizado para campañas publicitarias.")

archivo = st.file_uploader("Sube el archivo del DENUE", type=["csv", "xlsx"])
if archivo:
    try:
        if archivo.name.endswith("csv"):
            df = pd.read_csv(archivo, encoding="latin1")
        else:
            df = pd.read_excel(archivo)
        df.columns = df.columns.str.strip()  # Eliminar espacios en encabezados
    except:
        st.error("No se pudo leer el archivo. Asegúrate de que esté bien codificado.")

    estados = sorted(df["nom_ent"].dropna().unique())
    estado = st.selectbox("📍 Selecciona un estado", estados)

    df_estado = df[df["nom_ent"] == estado]

    giros = sorted(df_estado["nombre_act"].dropna().unique())
    giros_seleccionados = st.multiselect("🏢 Selecciona giros económicos", giros)

    tamaños = sorted(df_estado["estrato"].dropna().unique())
    tamaños_seleccionados = st.multiselect("👥 Selecciona tamaños de empresa", tamaños)

    solo_telefono = st.checkbox("📞 Solo empresas con teléfono disponible", value=True)

    if st.button("🔍 Aplicar filtros"):
        df_filtrado = df_estado[
            df_estado["nombre_act"].isin(giros_seleccionados) &
            df_estado["estrato"].isin(tamaños_seleccionados)
        ]

        if solo_telefono:
            df_filtrado = df_filtrado[df_filtrado["telefono"].notna()]

        columnas = [
            "nom_estab", "nombre_act", "estrato",
            "telefono", "correoelec", "www",
            "codpos", "nom_mun", "nom_ent",
            "latitud", "longitud"
        ]

        df_final = df_filtrado[columnas].copy()
        df_final.columns = [
            "Nombre", "Giro", "Tamaño",
            "Teléfono", "Correo", "Sitio Web",
            "C.P.", "Municipio", "Estado",
            "Latitud", "Longitud"
        ]

        st.success(f"Empresas encontradas: {len(df_final)}")
        st.dataframe(df_final.head(10), use_container_width=True)

        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar archivo CSV optimizado",
            data=csv,
            file_name="empresas_filtradas.csv",
            mime="text/csv"
        )

        if not df_final[["Latitud", "Longitud"]].dropna().empty:
            st.markdown("### 🗺️ Mapa de empresas filtradas")
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
            st.warning("No hay coordenadas disponibles para generar el mapa.")
