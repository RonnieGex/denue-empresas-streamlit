# app.py
import streamlit as st
import pandas as pd
import folium
import re
import unicodedata
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
import plotly.express as px
from io import BytesIO

# Configuración inicial
st.set_page_config(
    page_title="Katalis Ads DB Optimizer AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Constantes
REQUIRED_COLUMNS = {
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

COLUMN_NAMES_MAP = {
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

# Funciones base
def normalize_column_name(col_name):
    """Normaliza nombres de columnas para matching flexible"""
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip().replace(' ', '_')

@st.cache_data(ttl=3600, show_spinner="Optimizando carga de datos...")
def load_data(uploaded_file):
    """Carga y preprocesa datos con optimización de memoria"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Optimización de tipos de datos
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype('string')
            
        return df
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()

def map_columns_interactive(df):
    """Interfaz interactiva para mapeo de columnas"""
    st.markdown("### 🔍 Mapeo de Columnas")
    help_text = "Selecciona qué columna de tu archivo corresponde a cada campo requerido:"
    
    col_mapping = {}
    with st.expander("Configurar mapeo de columnas", expanded=True):
        st.caption(help_text)
        
        for target_col, possible_names in REQUIRED_COLUMNS.items():
            matching_cols = [col for col in df.columns if col in possible_names + [target_col]]
            default = matching_cols[0] if matching_cols else None
            
            col_mapping[target_col] = st.selectbox(
                label=f"{target_col.replace('_', ' ').title()} ({', '.join(possible_names)})",
                options=[''] + list(df.columns),
                index=df.columns.get_loc(default) + 1 if default else 0,
                key=f"colmap_{target_col}"
            )
    return col_mapping

def estimate_employees(value):
    """Estimación optimizada de empleados usando regex vectorizado"""
    if pd.isna(value): return None
    str_val = str(value).lower()
    
    patterns = {
        'range': r'(\d+)\s*a\s*(\d+)',
        'less_than': r'menos de\s*(\d+)',
        'more_than': r'más de\s*(\d+)',
        'single': r'^\d+$'
    }
    
    try:
        if re.search(patterns['range'], str_val):
            nums = list(map(int, re.findall(r'\d+', str_val)))
            return sum(nums) // len(nums)
        if re.search(patterns['less_than'], str_val):
            return int(re.search(r'\d+', str_val).group()) - 1
        if re.search(patterns['more_than'], str_val):
            return int(re.search(r'\d+', str_val).group()) + 1
        if re.search(patterns['single'], str_val):
            return int(str_val)
        return None
    except:
        return None

# Componentes UI
def show_data_summary(df):
    """Muestra métricas clave en tiempo real"""
    st.markdown("## 📈 Análisis Instantáneo")
    
    cols = st.columns(4)
    metrics = {
        '🏢 Empresas': len(df),
        '👥 Empleados (prom)': df['Personal Estimado'].mean(),
        '📞 Con teléfono': df['Teléfono'].count(),
        '🌐 Con sitio web': df['Web'].count()
    }
    
    for (label, value), col in zip(metrics.items(), cols):
        col.metric(label, f"{value:,.0f}" if isinstance(value, float) else value)

def create_interactive_map(df):
    """Crea mapa interactivo optimizado"""
    st.markdown("## 🌍 Mapa de Concentración Empresarial")
    
    if df.empty:
        st.warning("No hay datos para mostrar en el mapa")
        return
    
    map_center = [df['Latitud'].mean(), df['Longitud'].mean()]
    
    with st.spinner("Renderizando mapa..."):
        m = folium.Map(location=map_center, zoom_start=12, tiles='cartodbpositron')
        FastMarkerCluster(data=df[['Latitud', 'Longitud']].values.tolist()).add_to(m)
        st_folium(m, width=1200, height=600)

# Flujo principal
def main():
    # Header
    st.title("🚀 Katalis Ads DB Optimizer AI")
    st.markdown("""
        **Herramienta todo-en-uno para segmentación inteligente de bases empresariales**  
        Carga, filtra y optimiza tus datos para campañas B2B con tecnología AI
    """)
    
    # Paso 1: Carga de datos
    uploaded_file = st.file_uploader("Sube tu archivo DENUE", type=["csv", "xlsx"], 
                                   help="El archivo debe contener datos empresariales en formato estándar")
    
    if not uploaded_file:
        st.info("👋 ¡Comienza subiendo tu archivo!")
        return
    
    # Procesamiento inicial
    with st.spinner("Analizando estructura de datos..."):
        raw_df = load_data(uploaded_file)
        raw_df.columns = [normalize_column_name(col) for col in raw_df.columns]
    
    # Mapeo de columnas
    st.markdown("## 🔄 Configuración de Campos")
    column_mapping = map_columns_interactive(raw_df)
    
    if not all(column_mapping.values()):
        st.error("⚠️ Configura todas las columnas requeridas para continuar")
        return
    
    # Transformación de datos
    with st.spinner("Aplicando transformaciones..."):
        try:
            df = raw_df.rename(columns={v:k for k,v in column_mapping.items()})[list(REQUIRED_COLUMNS.keys())]
            df = df.rename(columns=COLUMN_NAMES_MAP)
            df['Personal Estimado'] = df['Personal (texto)'].apply(estimate_employees)
            df = df.dropna(subset=['Estado', 'Municipio', 'Giro'])
        except Exception as e:
            st.error(f"Error transformando datos: {str(e)}")
            return
    
    # Filtros interactivos
    st.markdown("## 🎯 Filtros Avanzados")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_state = st.selectbox("Estado", options=sorted(df['Estado'].unique()))
        municipalities = df[df['Estado'] == selected_state]['Municipio'].unique()
        # CORRECCIÓN APLICADA: 'municipalities' en lugar de 'municipalies'
        selected_municipalities = st.multiselect("Municipios", options=sorted(municipalities),
                                               default=sorted(municipalities)[:3])
        
    with col2:
        min_emp, max_emp = int(df['Personal Estimado'].min()), int(df['Personal Estimado'].max())
        emp_range = st.slider("Rango de empleados", min_emp, max_emp, (min_emp, max_emp))
        keyword_filter = st.text_input("Buscar en nombres de empresas")
    
    # Aplicar filtros
    filtered_df = df[
        (df['Estado'] == selected_state) &
        (df['Municipio'].isin(selected_municipalities)) &
        (df['Personal Estimado'].between(*emp_range))
    ]
    
    if keyword_filter:
        filtered_df = filtered_df[filtered_df['Nombre'].str.contains(keyword_filter, case=False)]
    
    # Resultados
    show_data_summary(filtered_df)
    create_interactive_map(filtered_df)
    
    # Exportación de datos
    st.markdown("## 📤 Exportar Datos Optimizados")
    export_format = st.radio("Formato de exportación", ["CSV", "Excel"], horizontal=True)
    
    if export_format == "CSV":
        data = filtered_df.to_csv(index=False).encode('utf-8')
    else:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, index=False)
        data = output.getvalue()
    
    st.download_button(
        label=f"Descargar {export_format}",
        data=data,
        file_name=f"katalis_export.{export_format.lower()}",
        mime='text/csv' if export_format == "CSV" else 'application/vnd.ms-excel'
    )
    
    # Asistente AI
    st.markdown("## 🤖 Asistente de Segmentación con AI")
    
    with st.expander("Obtener recomendaciones de targeting", expanded=True):
        api_key = st.text_input("Clave API DeepSeek", type="password",
                              help="Obtén tu clave en: https://platform.deepseek.com/api-keys")
        business_context = st.text_area("Describe tu negocio y objetivos",
                                      placeholder="Ej: Vendo software ERP para manufactura...")
        
        if st.button("Generar recomendaciones", type="primary") and api_key and business_context:
            with st.spinner("Analizando con DeepSeek AI..."):
                try:
                    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": f"""
                                Eres un experto en marketing B2B y segmentación de mercado. Analiza estos datos:
                                {filtered_df.describe()}
                                Sugiere estrategias de segmentación para Google Ads.
                            """},
                            {"role": "user", "content": business_context}
                        ],
                        temperature=0.5,
                        max_tokens=500
                    )
                    
                    recommendations = response.choices[0].message.content.split('\n')
                    st.markdown("### Recomendaciones de Segmentación")
                    for rec in recommendations:
                        if rec.strip():
                            st.markdown(f"- {rec.strip()}")
                except Exception as e:
                    st.error(f"Error en la consulta AI: {str(e)}")

if __name__ == "__main__":
    main()
