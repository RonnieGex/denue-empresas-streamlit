# app.py
import streamlit as st
import pandas as pd
import folium
import re
import unicodedata
import numpy as np
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración inicial
st.set_page_config(
    page_title="Katalis Ads AI Optimizer",
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
def load_and_preprocess(uploaded_file):
    """Carga y preprocesa datos con optimización de memoria"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Normalización de columnas
        df.columns = [normalize_column_name(col) for col in df.columns]
        df = df.rename(columns={v:k for k,v in REQUIRED_COLUMNS.items()})
        df = df.rename(columns=COLUMN_NAMES_MAP)
        
        # Estimación de empleados
        df['Personal Estimado'] = df['Personal (texto)'].apply(estimate_employees)
        return df.dropna(subset=['Estado', 'Municipio', 'Giro'])
    except Exception as e:
        st.error(f"Error procesando datos: {str(e)}")
        st.stop()

def estimate_employees(value):
    """Estimación optimizada de empleados usando regex"""
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
        if match := re.search(patterns['less_than'], str_val):
            return int(match.group(1)) - 1
        if match := re.search(patterns['more_than'], str_val):
            return int(match.group(1)) + 1
        if re.search(patterns['single'], str_val):
            return int(str_val)
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def analyze_with_ai(_df):
    """Análisis predictivo con clustering"""
    try:
        numeric_cols = _df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(_df[numeric_cols].fillna(0))
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        _df['cluster'] = kmeans.fit_predict(scaled_data)
        
        # Generar recomendaciones
        top_giros = _df['Giro'].value_counts().nlargest(5).index.tolist()
        emp_q75 = _df['Personal Estimado'].quantile(0.75)
        top_municipios = _df.groupby('Municipio').size().nlargest(3).index.tolist()
        
        return {
            'df': _df,
            'suggestions': {
                'giros': top_giros,
                'empleados': emp_q75,
                'municipios': top_municipios
            }
        }
    except Exception as e:
        st.error(f"Error en análisis IA: {str(e)}")
        return None

# Componentes UI
def show_ai_recommendations(suggestions):
    """Interfaz para ajustar sugerencias de IA"""
    with st.expander("🎯 Sugerencias Inteligentes de Segmentación", expanded=True):
        cols = st.columns(3)
        
        with cols[0]:
            giros = st.multiselect(
                "Giros estratégicos",
                options=suggestions['giros'],
                default=suggestions['giros'][:2]
            )
        
        with cols[1]:
            empleados = st.slider(
                "Tamaño óptimo de empresas",
                min_value=0,
                max_value=500,
                value=int(suggestions['empleados'])
            )
        
        with cols[2]:
            municipios = st.multiselect(
                "Ubicaciones clave",
                options=suggestions['municipios'],
                default=suggestions['municipios']
            )
        
        return {
            'giros': giros,
            'empleados': (empleados, 500),
            'municipios': municipios
        }

def create_interactive_map(df):
    """Genera mapa con clustering optimizado"""
    if df.empty:
        st.warning("No hay datos para mostrar en el mapa")
        return
    
    map_center = [df['Latitud'].mean(), df['Longitud'].mean()]
    
    with st.spinner("Generando mapa interactivo..."):
        m = folium.Map(location=map_center, zoom_start=10, tiles='cartodbpositron')
        FastMarkerCluster(data=df[['Latitud', 'Longitud']].values.tolist()).add_to(m)
        st_folium(m, width=1200, height=600)

# Flujo principal
def main():
    st.title("🚀 Katalis Ads DB Optimizer AI")
    st.markdown("Inteligencia avanzada para segmentación de mercados B2B")
    
    # Estado de sesión
    if 'analyzed_data' not in st.session_state:
        st.session_state.analyzed_data = None
    if 'current_filters' not in st.session_state:
        st.session_state.current_filters = None
    
    # Paso 1: Carga de datos
    uploaded_file = st.file_uploader("Sube tu base DENUE", type=["csv", "xlsx"])
    
    if uploaded_file:
        if st.session_state.analyzed_data is None:
            with st.spinner("Procesando datos..."):
                df = load_and_preprocess(uploaded_file)
                analysis = analyze_with_ai(df)
                
                if analysis:
                    st.session_state.analyzed_data = analysis['df']
                    st.session_state.suggestions = analysis['suggestions']
        
        # Paso 2: Sugerencias IA
        if st.session_state.analyzed_data is not None:
            st.markdown("## 🔍 Análisis Predictivo Automático")
            filters = show_ai_recommendations(st.session_state.suggestions)
            
            # Botón de aplicación
            if st.button("🚀 Generar Segmento Optimizado", type="primary"):
                st.session_state.current_filters = filters
                st.rerun()
    
    # Paso 3: Resultados
    if st.session_state.current_filters:
        filtered_df = st.session_state.analyzed_data[
            (st.session_state.analyzed_data['Giro'].isin(st.session_state.current_filters['giros'])) &
            (st.session_state.analyzed_data['Personal Estimado'].between(*st.session_state.current_filters['empleados'])) &
            (st.session_state.analyzed_data['Municipio'].isin(st.session_state.current_filters['municipios']))
        ]
        
        st.markdown(f"## 📊 Resultados: {len(filtered_df)} empresas seleccionadas")
        
        # Exportación
        export_format = st.radio("Formato de exportación:", ["CSV", "Excel"], horizontal=True)
        if export_format == "CSV":
            data = filtered_df.to_csv(index=False).encode('utf-8')
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False)
            data = output.getvalue()
        
        st.download_button(
            label=f"⬇️ Descargar {export_format}",
            data=data,
            file_name=f"segmento_optimizado.{export_format.lower()}",
            mime='text/csv' if export_format == "CSV" else 'application/vnd.ms-excel'
        )
        
        # Visualización
        create_interactive_map(filtered_df)
    
    # Asistente AI
    if st.session_state.analyzed_data is not None:
        with st.expander("🤖 Asistente de Campañas Inteligentes"):
            api_key = st.text_input("Clave API DeepSeek", type="password")
            prompt = st.text_area("Objetivos de tu campaña")
            
            if st.button("Generar estrategia publicitaria"):
                if not api_key or not prompt:
                    st.warning("Completa ambos campos")
                    return
                
                with st.spinner("Analizando con IA..."):
                    try:
                        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{
                                "role": "system",
                                "content": f"""
                                    Eres un experto en marketing digital B2B. 
                                    Datos clave: {st.session_state.suggestions}
                                    Sugiere estrategias para: {prompt}
                                """
                            }],
                            temperature=0.7,
                            max_tokens=500
                        )
                        st.markdown("### Recomendaciones estratégicas")
                        st.write(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error en IA: {str(e)}")

if __name__ == "__main__":
    main()
