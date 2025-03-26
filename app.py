
import plotly.express as px

def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
col1, col2, col3 = st.columns(3)
col1.metric("Total de empresas", len(df_filtrado))
col2.metric("Promedio de empleados", round(df_filtrado["per_ocu_est"].mean(), 2))
col3.metric("Empresas con contacto", df_filtrado[["telefono", "correoelec"]].dropna(how="all").shape[0])

if "nombre_act" in df_filtrado.columns:
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
top_giros.columns = ["Giro", "Cantidad"]
fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos en tu Filtro")
st.plotly_chart(fig, use_container_width=True)



# === 🔌 Integración de IA con proveedores múltiples ===
import openai
import anthropic

def get_recommendations_from_ai(api_key, provider, prompt):
if provider == "OpenAI (GPT-4)":
openai.api_key = api_key
response = openai.ChatCompletion.create(
model="gpt-4",
messages=[
{"role": "system", "content": "Eres un experto en segmentación de clientes para campañas de Google Ads B2B."},
{"role": "user", "content": prompt}
]
)
return response['choices'][0]['message']['content']

elif provider == "Anthropic (Claude)":
client = anthropic.Anthropic(api_key=api_key)
response = client.messages.create(
model="claude-3-opus-20240229",
max_tokens=300,
messages=[
{"role": "user", "content": prompt}
]
)
return response.content[0].text

elif provider == "DeepSeek":
openai.api_key = api_key  # DeepSeek usa misma interfaz que OpenAI
response = openai.ChatCompletion.create(
model="deepseek-chat",
messages=[
{"role": "system", "content": "Eres un experto en análisis de mercado."},
{"role": "user", "content": prompt}
]
)
return response['choices'][0]['message']['content']
else:
return "Proveedor no válido"


def recomendar_giros(tipo_negocio):
recomendaciones = {
"telefonía": [
"Comercio al por mayor de aparatos de telecomunicaciones, fotografía y cinematografía",
"Servicios de instalación de equipo de telecomunicaciones",
"Otros servicios de telecomunicaciones"
],
"marketing digital": [
"Servicios de diseño gráfico",
"Servicios de publicidad",
"Agencias de medios"
],
"consultoría empresarial": [
"Servicios de consultoría en administración",
"Servicios de contabilidad y auditoría",
"Otros servicios de apoyo a los negocios"
],
"software": [
"Servicios de diseño de sistemas de cómputo y servicios relacionados",
"Otros servicios relacionados con los sistemas de información",
"Servicios de programación informática"
]
}
tipo = tipo_negocio.lower()
return recomendaciones.get(tipo, [])


import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Katalis DB Optimizer", layout="wide")
st.title("🔥 Katalis DB Optimizer")
st.markdown("Optimiza tu base de datos con filtros avanzados. Ahora con mayor velocidad, control y precisión.")

archivo = st.file_uploader("📂 Sube tu archivo del DENUE (.csv o .xlsx)", type=["csv", "xlsx"])

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

if "mostrar" not in st.session_state:
st.session_state.mostrar = False

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

with st.form("filtros_form"):
st.subheader("🧠 Filtros avanzados")

estado = st.selectbox("📍 Estado", sorted(df['entidad'].dropna().unique()))
df_estado = df[df["entidad"] == estado]

municipios = sorted(df_estado['municipio'].dropna().unique())
municipios_sel = st.multiselect("🏘️ Municipios", municipios, default=municipios)


# 🔍 Ingreso del tipo de negocio para recomendaciones
tipo_negocio = st.text_input("🧠 Describe tu tipo de negocio (Ej: telefonía, software, marketing digital)", "")
giros_recomendados = []
if tipo_negocio:
giros_recomendados = recomendar_giros(tipo_negocio)
if giros_recomendados:
st.markdown("🎯 **Sugerencias de giros para tu negocio:**")
st.write(giros_recomendados)
else:
st.info("No se encontraron sugerencias para ese tipo de negocio.")


st.markdown("### 🤖 Recomendación de Giros con IA Avanzada")
proveedor = st.selectbox("Selecciona proveedor de IA", ["OpenAI (GPT-4)", "Anthropic (Claude)", "DeepSeek"])
api_key = st.text_input("🔐 Ingresa tu API Key (no se guarda)", type="password")

tipo_negocio = st.text_input("✍️ Describe tu tipo de negocio", "")
giros_ia = []
if st.button("📡 Obtener recomendaciones con IA") and api_key and tipo_negocio:
with st.spinner("Consultando inteligencia artificial..."):
prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}."
recomendacion_raw = get_recommendations_from_ai(api_key, proveedor, prompt)
giros_ia = recomendacion_raw.split("\n")
st.success("✅ Giros recomendados:")
for g in giros_ia:
st.markdown(f"- {g}")

giros = sorted(df_estado['nombre_act'].dropna().unique())
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

submitted = st.form_submit_button("📊 Mostrar resultados y descargar")
if submitted:
st.session_state.mostrar = True
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

if st.session_state.mostrar:
f = st.session_state.filtros
df_filtrado = df[
(df["entidad"] == f["estado"]) &
(df["municipio"].isin(f["municipios_sel"])) &
(df["nombre_act"].isin(f["giros_sel"])) &
(df["per_ocu_estimado"].between(f["rango_emp"][0], f["rango_emp"][1]))
]

if f["nombre_busqueda"]:
df_filtrado = df_filtrado[df_filtrado["nom_estab"].str.lower().str.contains(f["nombre_busqueda"].lower())]

if f["con_tel"]:
df_filtrado = df_filtrado[df_filtrado["telefono"].notna()]
if f["con_mail"]:
df_filtrado = df_filtrado[df_filtrado["correoelec"].notna()]
if f["con_web"]:
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