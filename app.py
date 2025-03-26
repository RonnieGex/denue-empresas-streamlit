
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos del DENUE con filtros avanzados para campañas B2B en Google Ads. ✨ Ahora con mayor velocidad, control y precisión.")

archivo = st.file_uploader("📂 Sube tu archivo del DENUE (.csv o .xlsx)", type=["csv", "xlsx"])

# Función robusta para convertir per_ocu textual a número estimado
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

if archivo:
    try:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo, encoding="latin1")
        else:
            df = pd.read_excel(archivo)

        df.columns = df.columns.str.strip()

        columnas_clave = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 'correoelec', 'www',
                          'municipio', 'localidad', 'entidad', 'latitud', 'longitud']
        if not all(col in df.columns for col in columnas_clave):
            st.error("❌ El archivo no contiene todas las columnas necesarias.")
        else:
            df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)

            with st.expander("🧠 Filtros avanzados", expanded=True):
                estado = st.selectbox("📍 Estado", sorted(df['entidad'].dropna().unique()))
                municipios_df = df[df['entidad'] == estado]
                municipios = sorted(municipios_df['municipio'].dropna().unique())
                municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)

                giros = sorted(municipios_df['nombre_act'].dropna().unique())
                giros_sel = st.multiselect("🏢 Giros económicos", giros)

                emp_validos = df["per_ocu_estimado"].dropna().astype(int)
                min_emp = int(emp_validos.min()) if not emp_validos.empty else 0
                max_emp = int(emp_validos.max()) if not emp_validos.empty else 500
                rango_emp = st.slider("👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp))

                nombre_busqueda = st.text_input("🔎 Buscar palabra clave en nombre del negocio")

                col1, col2, col3 = st.columns(3)
                with col1:
                    con_tel = st.checkbox("📞 Solo empresas con teléfono", value=False)
                with col2:
                    con_mail = st.checkbox("📧 Solo con correo electrónico", value=False)
                with col3:
                    con_web = st.checkbox("🌐 Solo con sitio web", value=False)

            if st.button("📊 Mostrar resultados y descargar"):
                df_filtrado = df[
                    (df["entidad"] == estado) &
                    (df["municipio"].isin(municipios_sel)) &
                    (df["nombre_act"].isin(giros_sel)) &
                    (df["per_ocu_estimado"].between(rango_emp[0], rango_emp[1]))
                ]

                if nombre_busqueda:
                    df_filtrado = df_filtrado[df_filtrado["nom_estab"].str.lower().str.contains(nombre_busqueda.lower())]

                if con_tel:
                    df_filtrado = df_filtrado[df_filtrado["telefono"].notna()]
                if con_mail:
                    df_filtrado = df_filtrado[df_filtrado["correoelec"].notna()]
                if con_web:
                    df_filtrado = df_filtrado[df_filtrado["www"].notna()]

                columnas_exportar = [
                    "nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                    "telefono", "correoelec", "www",
                    "municipio", "localidad", "entidad",
                    "latitud", "longitud"
                ]

                df_final = df_filtrado[columnas_exportar].copy()
                df_final.columns = [
                    "Nombre", "Giro", "Personal (texto)", "Personal Estimado",
                    "Teléfono", "Correo", "Web",
                    "Municipio", "Localidad", "Estado",
                    "Latitud", "Longitud"
                ]

                st.success(f"✅ Empresas encontradas: {len(df_final)}")

                limite = st.slider("📄 ¿Cuántos resultados mostrar en pantalla?", 10, 500, 50)
                st.dataframe(df_final.head(limite), use_container_width=True)

                csv = df_final.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar CSV optimizado", csv, file_name="empresas_katalis.csv", mime="text/csv")

                if not df_final[["Latitud", "Longitud"]].dropna().empty:
                    st.subheader("🗺️ Mapa interactivo")
                    mapa = folium.Map(
                        location=[
                            df_final["Latitud"].astype(float).mean(),
                            df_final["Longitud"].astype(float).mean()
                        ],
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
