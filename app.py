import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import openai
import anthropic
import plotly.express as px

# Función para obtener recomendaciones desde distintos proveedores de IA
def get_recommendations_from_ai(api_key, provider, prompt):
    if provider == "OpenAI (GPT-4)":
        openai.api_key = api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error al consultar IA: {str(e)}"
    elif provider == "Anthropic (Claude)":
        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content
        except Exception as e:
            return f"Error al consultar IA: {str(e)}"
    elif provider == "DeepSeek":
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
            return f"Error al consultar IA: {str(e)}"
    else:
        return "Proveedor no válido"

# Función para mostrar métricas y gráficos de la base filtrada
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

st.set_page_config(page_title="Katalis Ads DB Optimizer Pro", layout="wide")
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("Optimiza tu base de datos con filtros avanzados. Ahora con mayor velocidad, control y precisión.")

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

if archivo:
    df = cargar_datos(archivo)
    columnas_clave = ["nom_estab", "nombre_act", "per_ocu", "telefono", "correoelec", "www", "municipio", "localidad", "entidad", "latitud", "longitud"]
    if not all(col in df.columns for col in columnas_clave):
        st.error("❌ El archivo no contiene todas las columnas necesarias.")
    else:
        df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
        df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)
        estados = sorted(df["entidad"].dropna().unique())

        estado = st.selectbox("📍 Estado", estados)
        df_estado = df[df["entidad"] == estado]
        municipios = sorted(df_estado["municipio"].dropna().unique())
        municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)
        tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio", "")
        proveedor = st.selectbox("Proveedor de IA", ["OpenAI (GPT-4)", "Anthropic (Claude)", "DeepSeek"])
        api_key = st.text_input("🔐 Ingresa tu API Key", type="password")
        if st.button("📡 Obtener recomendaciones de giros con IA"):
            if tipo_negocio and api_key:
                prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
                respuesta_ia = get_recommendations_from_ai(api_key, proveedor, prompt)
                st.markdown("🎯 **Recomendaciones de IA para tu negocio:**")
                st.write(respuesta_ia)

        giros = sorted(df_estado["nombre_act"].dropna().unique())
        giros_sel = st.multiselect("🏢 Giros económicos", giros)
        emp_validos = df_estado["per_ocu_estimado"].dropna().astype(int)
        min_emp = int(emp_validos.min()) if not emp_validos.empty else 0
        max_emp = int(emp_validos.max()) if not emp_validos.empty else 500
        rango_emp = st.slider("👥 Rango de empleados (estimado)", min_emp, max_emp, (min_emp, max_emp))
        nombre_busqueda = st.text_input("🔎 Buscar palabra clave en nombre del negocio")
        con_tel = st.checkbox("📞 Solo con teléfono")
        con_mail = st.checkbox("📧 Solo con correo electrónico")
        con_web = st.checkbox("🌐 Solo con sitio web")

        if st.button("📊 Mostrar resultados y descargar"):
            df_filtrado = df[
                (df["entidad"] == estado) &
                (df["municipio"].isin(municipios_sel)) &
                (df["per_ocu_estimado"].between(rango_emp[0], rango_emp[1]))
            ]
            if giros_sel:
                df_filtrado = df_filtrado[df_filtrado["nombre_act"].isin(giros_sel)]
            if nombre_busqueda:
                df_filtrado = df_filtrado[df_filtrado["nom_estab"].str.lower().str.contains(nombre_busqueda.lower())]
            if con_tel:
                df_filtrado = df_filtrado[df_filtrado["telefono"].notna()]
            if con_mail:
                df_filtrado = df_filtrado[df_filtrado["correoelec"].notna()]
            if con_web:
                df_filtrado = df_filtrado[df_filtrado["www"].notna()]
            columnas_exportar = ["nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
                                 "telefono", "correoelec", "www", "municipio", "localidad",
                                 "entidad", "latitud", "longitud"]
            df_final = df_filtrado[columnas_exportar].copy()
            df_final.columns = ["Nombre", "Giro", "Personal (texto)", "Personal Estimado", "Teléfono", "Correo", "Web", "Municipio", "Localidad", "Estado", "Latitud", "Longitud"]
            st.success(f"✅ Empresas encontradas: {len(df_final)}")
            st.dataframe(df_final, use_container_width=True)
            mostrar_metricas(df_final)
            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar CSV optimizado", csv, file_name="denue_opt.csv", mime="text/csv")
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
