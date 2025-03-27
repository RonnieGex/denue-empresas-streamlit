
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from openai import OpenAI

# Configurar cliente de DeepSeek
def get_deepseek_response(api_key, prompt):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un consultor de negocios experto en identificar giros económicos adecuados para empresas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error al conectar con DeepSeek: {str(e)}"

def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de empresas", len(df_filtrado))
    if "per_ocu_estimado" in df_filtrado.columns and not df_filtrado["per_ocu_estimado"].empty:
        col2.metric("Promedio de empleados", round(df_filtrado["per_ocu_estimado"].mean(), 2))
    else:
        col2.metric("Promedio de empleados", "N/A")
    try:
        col3.metric("Empresas con contacto", df_filtrado[["telefono", "correoelec"]].dropna(how="all").shape[0])
    except:
        col3.metric("Empresas con contacto", "N/A")
    if "nombre_act" in df_filtrado.columns:
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
        top_giros.columns = ["Giro", "Cantidad"]
        fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos en tu Filtro")
        st.plotly_chart(fig, use_container_width=True)

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados y recomendaciones de IA usando DeepSeek.")

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

archivo = st.file_uploader("📂 Sube tu archivo del sistema (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)
    columnas_clave = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 'correoelec', 'www',
                      'municipio', 'localidad', 'entidad', 'latitud', 'longitud']
    if not all(col in df.columns for col in columnas_clave):
        st.error("❌ El archivo no contiene todas las columnas necesarias.")
    else:
        df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
        df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)

        with st.form("formulario_filtros"):
            st.subheader("🧠 Filtros avanzados")
            estado = st.selectbox("📍 Estado", sorted(df["entidad"].dropna().unique()))
            municipios = sorted(df[df["entidad"] == estado]["municipio"].dropna().unique())
            municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)
            tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio")
            api_key = st.text_input("🔐 Ingresa tu API Key de DeepSeek", type="password")
            obtener_ideas = st.form_submit_button("🤖 Obtener giros sugeridos")

            if obtener_ideas and tipo_negocio and api_key:
                prompt = f"Dame solo 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}. Solo enumera los giros, sin detalles ni palabras clave."
                resultado = get_deepseek_response(api_key, prompt)
                st.markdown("🎯 **Sugerencias de giros:**")
                st.write(resultado)

        # Filtros para generar base de datos
        with st.form("formulario_base"):
            st.subheader("🎯 Generar base de datos filtrada")
            giros = sorted(df["nombre_act"].dropna().unique())
            giros_sel = st.multiselect("🏢 Giros económicos", giros)
            emp_validos = df["per_ocu_estimado"].dropna().astype(int)
            min_emp = int(emp_validos.min()) if not emp_validos.empty else 0
            max_emp = int(emp_validos.max()) if not emp_validos.empty else 500
            rango_emp = st.slider("👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp))
            nombre_busqueda = st.text_input("🔎 Buscar palabra clave en nombre del negocio")
            col1, col2, col3 = st.columns(3)
            con_tel = col1.checkbox("📞 Solo empresas con teléfono")
            con_mail = col2.checkbox("📧 Solo con correo electrónico")
            con_web = col3.checkbox("🌐 Solo con sitio web")
            submitted = st.form_submit_button("📊 Mostrar resultados y mapa")

        if submitted:
            filtrado = df[
                (df["entidad"] == estado) &
                (df["municipio"].isin(municipios_sel)) &
                (df["per_ocu_estimado"].between(rango_emp[0], rango_emp[1]))
            ]
            if giros_sel:
                filtrado = filtrado[filtrado["nombre_act"].isin(giros_sel)]
            if nombre_busqueda:
                filtrado = filtrado[filtrado["nom_estab"].str.lower().str.contains(nombre_busqueda.lower())]
            if con_tel:
                filtrado = filtrado[filtrado["telefono"].notna()]
            if con_mail:
                filtrado = filtrado[filtrado["correoelec"].notna()]
            if con_web:
                filtrado = filtrado[filtrado["www"].notna()]

            columnas_exportar = [
                "nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                "telefono", "correoelec", "www", "municipio", "localidad", "entidad",
                "latitud", "longitud"
            ]
            df_final = filtrado[columnas_exportar].copy()
            df_final.columns = ["Nombre", "Giro", "Personal (texto)", "Personal Estimado",
                                "Teléfono", "Correo", "Web", "Municipio", "Localidad", "Estado",
                                "Latitud", "Longitud"]
            st.success(f"✅ Empresas encontradas: {len(df_final)}")
            limite = st.slider("🔢 ¿Cuántos resultados mostrar?", 10, 500, 50)
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
