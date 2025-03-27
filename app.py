import streamlit as st
import pandas as pd
import folium
import re
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
import plotly.express as px
from io import StringIO

# Configuración inicial
st.set_page_config(
    page_title="Katalis Ads DB Optimizer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 Katalis Ads DB Optimizer Pro")
st.markdown("""
    Optimiza tu base de datos con filtros avanzados. 
    **Características principales:**  
    ✅ Análisis predictivo de segmentación  
    ✅ Recomendaciones de IA integradas  
    ✅ Visualización geoespacial avanzada  
    ✅ Exportación inteligente de datos  
""")

# Constantes
COLUMNAS_REQUERIDAS = [
    'nom_estab', 'nombre_act', 'per_ocu', 'telefono', 'correoelec', 'www',
    'municipio', 'localidad', 'entidad', 'latitud', 'longitud'
]
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

@st.cache_data(show_spinner="Cargando y procesando datos...")
def cargar_datos(archivo):
    try:
        if archivo.name.endswith(".csv"):
            # Intenta primero con UTF-8
            try:
                df = pd.read_csv(archivo, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(archivo, encoding='latin1', low_memory=False)
        else:
            df = pd.read_excel(archivo)
        
        # Normalización de datos
        df.columns = df.columns.str.strip().str.lower()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        st.stop()

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
        # Rango (ej. 10 a 50)
        if match := re.search(patterns['range'], valor):
            min_val, max_val = map(int, match.groups())
            return (min_val + max_val) // 2
        
        # Menos de X
        if match := re.search(patterns['less_than'], valor):
            return int(match.group(1)) - 1
        
        # Más de X
        if match := re.search(patterns['more_than'], valor):
            return int(match.group(1)) + 1
        
        # Valor único
        if match := re.search(patterns['single'], valor):
            return int(match.group())
        
        return None
    except (ValueError, AttributeError):
        return None

@st.cache_data(ttl=3600, max_entries=10)
def get_recommendations_deepseek(api_key, prompt):
    try:
        if not api_key.startswith('ds-'):
            return [{'type': 'error', 'content': 'Formato de API Key inválido'}]
        
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """
                    Eres un experto en segmentación de mercado para campañas B2B en Google Ads. 
                    Proporciona recomendaciones prácticas y específicas basadas en los datos proporcionados.
                """},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return [
            {'type': 'success', 'content': line.strip()} 
            for line in response.choices[0].message.content.splitlines() 
            if line.strip()
        ]
    except Exception as e:
        return [{'type': 'error', 'content': f'Error en la conexión: {str(e)}'}]

def mostrar_metricas(df):
    st.markdown("## 📊 Panel de Analítica Avanzada")
    
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.markdown("### Distribución de Empleados")
        if df['Personal Estimado'].notna().any():
            fig = px.histogram(
                df, 
                x='Personal Estimado',
                nbins=20,
                title='Distribución de Tamaño de Empresas'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos suficientes de empleados para mostrar")
    
    metric_cols = cols[1:]
    metric_data = {
        'Total Empresas': len(df),
        'Contacto Completo': df[['Teléfono', 'Correo']].dropna().shape[0],
        'Promedio Empleados': df['Personal Estimado'].mean(skipna=True),
        'Presencia Web': df['Web'].notna().sum()
    }
    
    for (name, value), col in zip(metric_data.items(), metric_cols):
        col.metric(
            label=name,
            value=f"{value:,.0f}" if isinstance(value, (int, float)) else value,
            delta=None
        )

def validar_coordenadas(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except (ValueError, TypeError):
        return False

# Interfaz de usuario principal
def main():
    archivo = st.file_uploader(
        "📂 Sube tu archivo del sistema (.csv o .xlsx)",
        type=["csv", "xlsx"],
        help="El archivo debe contener columnas específicas del DENUE"
    )
    
    if not archivo:
        st.info("👉 Por favor sube un archivo para comenzar el análisis")
        return
    
    df = cargar_datos(archivo)
    
    # Validación de columnas
    missing_cols = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if missing_cols:
        st.error(f"""
            ❌ Estructura de archivo incompleta. 
            Columnas faltantes: {', '.join(missing_cols)}
        """)
        st.stop()
    
    # Preprocesamiento
    with st.spinner("Procesando datos..."):
        df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
        df["per_ocu_estimado"] = df["per_ocu"].apply(estimar_empleados)
        df = df.rename(columns=COLUMNAS_RENOMBRAR)
        df['Coordenadas Validas'] = df.apply(
            lambda x: validar_coordenadas(x['Latitud'], x['Longitud']), 
            axis=1
        )

    # Sidebar de filtros
    with st.sidebar:
        st.header("⚙️ Parámetros de Filtrado")
        estado = st.selectbox("Seleccionar Estado", sorted(df['Estado'].unique()))
        
        df_filtrado = df[df['Estado'] == estado]
        municipios = st.multiselect(
            "Seleccionar Municipios",
            options=sorted(df_filtrado['Municipio'].unique()),
            default=sorted(df_filtrado['Municipio'].unique())[:3]
        )
        
        st.divider()
        with st.expander("Filtros Avanzados"):
            rango_empleados = st.slider(
                "Rango de Empleados",
                min_value=int(df['Personal Estimado'].min()),
                max_value=int(df['Personal Estimado'].max()),
                value=(10, 100)
            )
            
            contacto_cols = st.columns(3)
            con_tel = contacto_cols[0].checkbox("Teléfono", True)
            con_mail = contacto_cols[1].checkbox("Email", True)
            con_web = contacto_cols[2].checkbox("Sitio Web")

    # Filtrado principal
    df_filtrado = df_filtrado[
        (df_filtrado['Municipio'].isin(municipios)) &
        (df_filtrado['Personal Estimado'].between(*rango_empleados)) &
        (df_filtrado['Coordenadas Validas'])
    ]
    
    if con_tel:
        df_filtrado = df_filtrado[df_filtrado['Teléfono'].notna()]
    if con_mail:
        df_filtrado = df_filtrado[df_filtrado['Correo'].notna()]
    if con_web:
        df_filtrado = df_filtrado[df_filtrado['Web'].notna()]

    # Visualización de datos
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📍 Mapa de Distribución - {estado}")
        if not df_filtrado.empty:
            mapa = folium.Map(
                location=[
                    df_filtrado['Latitud'].mean(),
                    df_filtrado['Longitud'].mean()
                ],
                zoom_start=10,
                tiles='cartodbpositron'
            )
            cluster = MarkerCluster().add_to(mapa)
            
            for _, row in df_filtrado.iterrows():
                folium.Marker(
                    location=[row['Latitud'], row['Longitud']],
                    popup=f"""
                        <b>{row['Nombre']}</b><br>
                        {row['Giro']}<br>
                        Empleados: {row['Personal Estimado']}
                    """,
                    tooltip=row['Nombre']
                ).add_to(cluster)
            
            st_folium(mapa, height=600, width=800)
        else:
            st.warning("No hay datos para mostrar en el mapa con los filtros actuales")
    
    with col2:
        st.markdown("### 🔍 Vista Previa de Datos")
        st.dataframe(
            df_filtrado.head(100),
            use_container_width=True,
            height=600,
            column_config={
                "Coordenadas Validas": st.column_config.Column(disabled=True)
            }
        )
        
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exportar Dataset Filtrado",
            data=csv,
            file_name='dataset_filtrado.csv',
            mime='text/csv'
        )

    # Sección de analítica
    mostrar_metricas(df_filtrado)
    
    # Integración con IA
    st.divider()
    with st.expander("🧠 Asistente de Segmentación con IA", expanded=True):
        api_key = st.text_input("Clave API DeepSeek", type="password")
        consulta_ia = st.text_area("Describe tu objetivo de campaña")
        
        if st.button("Generar Recomendaciones"):
            if not api_key or not consulta_ia:
                st.warning("Por favor completa ambos campos")
                return
                
            with st.spinner("Analizando con IA..."):
                recomendaciones = get_recommendations_deepseek(
                    api_key,
                    f"{consulta_ia}. Basado en el dataset: {df_filtrado.describe()}"
                )
                
                for item in recomendaciones:
                    if item['type'] == 'error':
                        st.error(item['content'])
                    else:
                        st.success(f"✅ {item['content']}")

if __name__ == "__main__":
    main()
