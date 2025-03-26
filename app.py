
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import openai
import anthropic
import plotly.express as px

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados. IA, métricas, mapa y análisis, todo en uno.")

# ---------------------- FUNCIONES ----------------------

def get_recommendations_from_ai(api_key, provider, prompt):
    try:
        if provider == "OpenAI (GPT-4)":
            openai.api_key = api_key
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
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
            return "❌ Proveedor no válido."
    except Exception as e:
        return f"❌ Error al conectar con el proveedor: {str(e)}"

def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de empresas", len(df_filtrado))
    if "per_ocu_estimado" in df_filtrado.columns and not df_filtrado["per_ocu_estimado"].empty:
        col2.metric("Promedio de empleados", round(df_filtrado["per_ocu_estimado"].mean(), 2))
    else:
        col2.metric("Promedio de empleados", "N/A")
    columnas_contacto = [col for col in df_filtrado.columns if col in ["telefono", "correoelec"]]
    if columnas_contacto:
        col3.metric("Empresas con contacto", df_filtrado[columnas_contacto].dropna(how="all").shape[0])
    else:
        col3.metric("Empresas con contacto", "N/A")

    if "nombre_act" in df_filtrado.columns:
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
        top_giros.columns = ["Giro", "Cantidad"]
        fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos")
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def cargar_datos(archivo):
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo, encoding="latin1", low_memory=False)
    else:
        df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip()
    return df

def estimar_empleados(valor):
    try:
        if pd.isna(valor): return None
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

# ---------------------- INTERFAZ ----------------------

archivo = st.file_uploader("📂 Sube tu archivo del sistema (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)
    columnas_clave = ["nom_estab", "nombre_act", "per_ocu", "telefono", "correoelec", "www", "municipio", "localidad", "entidad", "latitud", "longitud"]
    if not all(col in df.columns for col in columnas_clave):
        st.error("❌ El archivo no contiene todas las columnas necesarias.")
    else:
        df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
        df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)
        estados = sorted(df["entidad"].dropna().unique())

        with st.form("formulario_filtros"):
            st.subheader("🎯 Filtros avanzados")

            estado = st.selectbox("📍 Selecciona Estado", estados)
            df_estado = df[df["entidad"] == estado]
            municipios = sorted(df_estado["municipio"].dropna().unique())
            municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)

            tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio")
            proveedor = st.selectbox("Selecciona proveedor de IA", ["OpenAI (GPT-4)", "Anthropic (Claude)", "DeepSeek"])
            api_key = st.text_input("🔐 Ingresa tu API Key", type="password")

            obtener_recomendaciones = st.form_submit_button("📤 Obtener giros sugeridos")

        if obtener_recomendaciones and tipo_negocio and api_key:
            prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
            recomendaciones = get_recommendations_from_ai(api_key, proveedor, prompt)
            st.markdown("### 💡 Sugerencias de giros:")
            st.code(recomendaciones.splitlines(), language="json")

        with st.form("formulario_resultados"):
            giros = sorted(df_estado["nombre_act"].dropna().unique())
            giros_sel = st.multiselect("🏢 Giros económicos", giros)
            emp_validos = df_estado["per_ocu_estimado"].dropna().astype(int)
            min_emp = int(emp_validos.min()) if not emp_validos.empty else 0
            max_emp = int(emp_validos.max()) if not emp_validos.empty else 500
            rango_emp = st.slider("👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp))
            nombre_busqueda = st.text_input("🔎 Buscar palabra clave en nombre del negocio")
            col1, col2, col3 = st.columns(3)
            con_tel = col1.checkbox("📞 Solo empresas con teléfono")
            con_mail = col2.checkbox("📧 Solo con correo electrónico")
            con_web = col3.checkbox("🌐 Solo con sitio web")
            submitted = st.form_submit_button("📊 Mostrar resultados")

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

            columnas_exportar = ["nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                                 "telefono", "correoelec", "www", "municipio", "localidad",
                                 "entidad", "latitud", "longitud"]
            df_final = filtrado[columnas_exportar].copy()
            df_final.columns = ["Nombre", "Giro", "Personal (texto)", "Personal Estimado",
                                "Teléfono", "Correo", "Web", "Municipio", "Localidad", "Estado",
                                "Latitud", "Longitud"]

            st.success(f"✅ Empresas encontradas: {len(df_final)}")
            limite = st.slider("🔢 ¿Cuántos resultados mostrar en pantalla?", 10, 500, 50)
            st.dataframe(df_final.head(limite), use_container_width=True)

            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar CSV optimizado", csv, file_name="denue_opt.csv", mime="text/csv")

            if not df_final.empty:
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
                    st.info("⚠️ No hay coordenadas para el mapa.")
