# app.py
import streamlit as st
import pandas as pd
import folium
import re
import unicodedata
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
import plotly.express as px

# Configuración inicial
st.set_page_config(
    page_title="Katalis Ads DB Optimizer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 Katalis Ads DB Optimizer Pro")

# Constantes y mapeos
COLUMNAS_REQUERIDAS = {
    'nom_estab': ['nombre', 'establecimiento', 'empresa'],
    'nombre_act': ['giro', 'actividad', 'rubro'],
    'per_ocu': ['empleados', 'personal', 'trabajadores'],
    'telefono': ['tel', 'contacto'],
    'correoelec': ['email', 'correo'],
    'www': ['web', 'sitio'],
    'municipio': ['ciudad', 'delegacion'],
    'localidad': ['zona', 'region'],
    'entidad': ['estado'],
    'latitud': ['lat'],
    'longitud': ['lon', 'long']
}

COLUMNAS_RENOMBRAR = {
    'nom_estab': 'Nombre',
    'nombre_act': 'Giro',
    'per_ocu': 'Personal (texto)',
    'per_ocu_estimado': 'Personal Estimado',
    'telefono': 'Teléfono',
    'correoelec': 'Correo',
    'www': 'Web',
    'municipio': 'Municipio',
    'localidad': 'Localidad',
    'entidad': 'Estado',
    'latitud': 'Latitud',
    'longitud': 'Longitud'
}

# Funciones auxiliares
def normalizar_nombres(columna):
    columna = unicodedata.normalize('NFKD', columna).encode('ASCII', 'ignore').decode('utf-8')
    return columna.lower().strip().replace(' ', '_')

@st.cache_data
def cargar_datos(archivo):
    try:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo, encoding='latin1', low_memory=False)
        else:
            df = pd.read_excel(archivo)
        
        df.columns = [normalizar_nombres(col) for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        st.stop()

def mapear_columnas(df):
    st.markdown("### 🔍 Mapeo de Columnas Requeridas")
    st.write("Asigna cada columna de tu archivo a las requeridas por el sistema:")
    
    mapeo = {}
    for col_requerida, alternativas in COLUMNAS_REQUERIDAS.items():
        columnas_posibles = [c for c in df.columns if c in alternativas + [col_requerida]]
        default = columnas_posibles[0] if columnas_posibles else None
        mapeo[col_requerida] = st.selectbox(
            f"{col_requerida} (puede ser: {', '.join(alternativas)})",
            options=[''] + list(df.columns),
            index=df.columns.get_loc(default) + 1 if default else 0,
            key=f"map_{col_requerida}"
        )
    return mapeo

def estimar_empleados(valor):
    if pd.isna(valor):
        return None
    
    valor = str(valor).lower()
    patterns = {
        'range': r'(\d+)\s*a\s*(\d+)',
        'less_than': r'menos de\s*(\d+)',
        'more_than': r'más de\s*(\d+)',
        'single': r'^\d+$'
    }
    
    try:
        if match := re.search(patterns['range'], valor):
            min_val, max_val = map(int, match.groups())
            return (min_val + max_val) // 2
        if match := re.search(patterns['less_than'], valor):
            return int(match.group(1)) - 1
        if match := re.search(patterns['more_than'], valor):
            return int(match.group(1)) + 1
        if match := re.search(patterns['single'], valor):
            return int(match.group())
        return None
    except (ValueError, AttributeError):
        return None

@st.cache_data(ttl=3600)
def get_recommendations_deepseek(api_key, prompt):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto en segmentación de mercado para campañas B2B en Google Ads."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return [line.strip() for line in response.choices[0].message.content.splitlines() if line.strip()]
    except Exception as e:
        return [f"❌ Error: {str(e)}"]

def mostrar_metricas(df):
    st.markdown("## 📊 Panel de Analítica")
    cols = st.columns(4)
    
    metricas = {
        'Total Empresas': len(df),
        'Prom. Empleados': df['Personal Estimado'].mean(),
        'Con Contacto': df[['Teléfono', 'Correo']].dropna(how='all').shape[0],
        'Con Web': df['Web'].notna().sum()
    }
    
    for (nombre, valor), col in zip(metricas.items(), cols):
        col.metric(nombre, f"{valor:,.0f}" if isinstance(valor, float) else valor)

# Flujo principal
def main():
    archivo = st.file_uploader("📂 Sube tu archivo (.csv o .xlsx)", type=["csv", "xlsx"])
    
    if not archivo:
        st.info("👉 Sube un archivo para comenzar")
        return
    
    df = cargar_datos(archivo)
    
    with st.expander("🔍 Ver estructura del archivo"):
        st.write("Columnas detectadas:", df.columns.tolist())
    
    mapeo = mapear_columnas(df)
    
    if not all(mapeo.values()):
        st.error("Debes mapear todas las columnas requeridas")
        return
    
    try:
        df = df.rename(columns={v: k for k, v in mapeo.items() if v})[COLUMNAS_REQUERIDAS.keys()]
        df = df.rename(columns=COLUMNAS_RENOMBRAR)
        df['Personal Estimado'] = df['Personal (texto)'].apply(estimar_empleados)
        df['Coordenadas Validas'] = df.apply(lambda x: abs(x['Latitud']) <= 90 and abs(x['Longitud']) <= 180, axis=1)
    except Exception as e:
        st.error(f"Error procesando datos: {str(e)}")
        return
    
    # Filtros en sidebar
    with st.sidebar:
        st.header("⚙️ Filtros")
        estado = st.selectbox("Estado", sorted(df['Estado'].unique()))
        municipios = st.multiselect("Municipios", sorted(df[df['Estado'] == estado]['Municipio'].unique()))
        rango_emp = st.slider("Rango de empleados", 
                             int(df['Personal Estimado'].min()), 
                             int(df['Personal Estimado'].max()), 
                             (10, 100))
        
        st.divider()
        with st.expander("Opciones avanzadas"):
            con_tel = st.checkbox("Solo con teléfono")
            con_email = st.checkbox("Solo con email")
            con_web = st.checkbox("Solo con web")

    # Aplicar filtros
    df_filtrado = df[
        (df['Estado'] == estado) &
        (df['Municipio'].isin(municipios)) &
        (df['Personal Estimado'].between(*rango_emp)) &
        (df['Coordenadas Validas'])
    ]
    
    if con_tel:
        df_filtrado = df_filtrado[df_filtrado['Teléfono'].notna()]
    if con_email:
        df_filtrado = df_filtrado[df_filtrado['Correo'].notna()]
    if con_web:
        df_filtrado = df_filtrado[df_filtrado['Web'].notna()]

    # Visualización
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 🗺️ Mapa de {estado}")
        if not df_filtrado.empty:
            mapa = folium.Map(
                location=[df_filtrado['Latitud'].mean(), df_filtrado['Longitud'].mean()],
                zoom_start=10
            )
            MarkerCluster().add_to(mapa)
            for _, row in df_filtrado.iterrows():
                folium.Marker(
                    [row['Latitud'], row['Longitud']],
                    popup=f"{row['Nombre']}<br>{row['Giro']}"
                ).add_to(mapa)
            st_folium(mapa, width=700, height=500)
        else:
            st.warning("No hay datos para mostrar")

    with col2:
        st.markdown("### 📋 Vista previa")
        st.dataframe(df_filtrado.head(100), height=500)
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar datos filtrados", csv, "datos_filtrados.csv")

    mostrar_metricas(df_filtrado)
    
    # Sección IA
    st.divider()
    with st.expander("🤖 Asistente de Marketing con IA"):
        api_key = st.text_input("DeepSeek API Key", type="password")
        consulta = st.text_area("Describe tu campaña")
        if st.button("Generar recomendaciones") and api_key and consulta:
            with st.spinner("Analizando..."):
                recomendaciones = get_recommendations_deepseek(
                    api_key,
                    f"{consulta}. Contexto: {df_filtrado.describe()}"
                )
                for line in recomendaciones:
                    st.markdown(f"- {line}")

if __name__ == "__main__":
    main()
