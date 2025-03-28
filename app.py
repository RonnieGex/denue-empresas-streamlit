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
import hashlib

# Configuración inicial
st.set_page_config(
    page_title="Business Intelligence Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Constantes mejoradas (sin referencia a DENUE)
REQUIRED_COLUMNS = {
    'business_name': ['nom_estab', 'nombre_comercial'],
    'industry': ['nombre_act', 'giro_principal'],
    'employees': ['per_ocu', 'personal_ocupado'],
    'phone': ['telefono', 'contacto_telefonico'],
    'email': ['correoelec', 'correo'],
    'website': ['www', 'sitio_web'],
    'city': ['municipio', 'ciudad'],
    'state': ['entidad', 'estado'],
    'latitude': ['latitud'],
    'longitude': ['longitud']
}

COLUMN_NAMES_MAP = {
    'business_name': 'Nombre Comercial',
    'industry': 'Sector Industrial',
    'employees': 'Empleados',
    'phone': 'Teléfono',
    'email': 'Correo Electrónico',
    'website': 'Sitio Web',
    'city': 'Ciudad',
    'state': 'Estado',
    'latitude': 'Latitud',
    'longitude': 'Longitud'
}

# Función de seguridad mejorada
def encrypt_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

def normalize_column_name(col_name):
    """Normalización robusta de nombres de columnas"""
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])\
        .lower().strip().replace(' ', '_').split('[')[0]

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process(uploaded_file):
    """Carga y procesamiento optimizado con manejo de tipos"""
    progress = st.progress(0, text="Iniciando procesamiento...")
    
    try:
        progress.progress(10, "Leyendo archivo...")
        if uploaded_file.name.endswith('.csv'):
            chunks = pd.read_csv(
                uploaded_file,
                encoding='latin1',
                chunksize=50000,
                dtype={'telefono': 'string', 'correoelec': 'string'}
            )
            df = pd.concat(chunks)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        progress.progress(30, "Estandarizando datos...")
        df.columns = [normalize_column_name(col) for col in df.columns]
        
        rename_mapping = {}
        for target_col, possible_cols in REQUIRED_COLUMNS.items():
            for col in possible_cols:
                if col in df.columns:
                    rename_mapping[col] = target_col
        df = df.rename(columns=rename_mapping)
        df = df.rename(columns=COLUMN_NAMES_MAP)

        required = list(COLUMN_NAMES_MAP.values())
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Columnas requeridas faltantes: {', '.join(missing)}")
            st.stop()

        progress.progress(50, "Procesando empleados...")
        df['Empleados'] = pd.to_numeric(df['Empleados'], errors='coerce')
        df['Tamaño Empresa'] = np.select(
            [
                df['Empleados'] <= 5,
                df['Empleados'] <= 100,
                df['Empleados'] > 100
            ],
            ['PYME', 'Mediana', 'Grande'],
            default='Desconocido'
        )

        dtypes = {
            'Latitud': 'float32',
            'Longitud': 'float32',
            'Ciudad': 'category',
            'Estado': 'category'
        }
        df = df.astype(dtypes, errors='ignore')

        progress.progress(80, "Depurando datos...")
        df = df.dropna(subset=['Estado', 'Ciudad', 'Sector Industrial'])
        
        progress.progress(100, "¡Proceso completado!")
        return df

    except Exception as e:
        progress.empty()
        st.error(f"Error crítico: {str(e)}")
        st.stop()

@st.cache_data(ttl=3600)
def analyze_with_ai(_df, api_key):
    """Análisis predictivo con DeepSeek AI"""
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        numeric_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
        valid_numeric_cols = [col for col in numeric_cols if _df[col].notna().sum() > 0]

        if not valid_numeric_cols:
            raise ValueError("No hay columnas numéricas válidas para análisis de clustering.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(_df[valid_numeric_cols].fillna(0))

        if np.isnan(scaled_data).any():
            raise ValueError("El conjunto de datos contiene valores NaN después del escalado.")

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        _df['Segmento IA'] = kmeans.fit_predict(scaled_data)

        context = {
            'sectores_top': _df['Sector Industrial'].value_counts().nlargest(5).index.tolist(),
            'empleados_promedio': round(_df['Empleados'].mean(skipna=True)),
            'ciudades_clave': _df.groupby('Ciudad')['Segmento IA'].count().nlargest(3).index.tolist()
        }

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en marketing digital B2B. Genera recomendaciones basadas en:"
                },
                {
                    "role": "user",
                    "content": f"{context}\n\nSugiere estrategias segmentadas para Google Ads:"
                }
            ],
            temperature=0.6,
            max_tokens=600
        )

        return {
            'df': _df,
            'analisis': response.choices[0].message.content,
            'sugerencias': context
        }

    except Exception as e:
        st.error(f"Error en análisis IA: {str(e)}")
        return None

def create_interactive_map(df):
    """Mapa interactivo con clusters optimizados"""
    if df.empty:
        st.warning("No hay datos para mostrar")
        return
    
    map_center = [df['Latitud'].mean(), df['Longitud'].mean()]
    with st.spinner("Generando visualización geoespacial..."):
        m = folium.Map(location=map_center, zoom_start=10, tiles='cartodbpositron')
        FastMarkerCluster(data=df[['Latitud', 'Longitud']].values.tolist()).add_to(m)
        st_folium(m, width=1200, height=600)

def prepare_google_ads_data(df):
    """Transformación profesional para Google Ads"""
    return df[[
        'Nombre Comercial',
        'Sector Industrial',
        'Tamaño Empresa',
        'Teléfono',
        'Correo Electrónico',
        'Sitio Web',
        'Ciudad',
        'Estado',
        'Latitud',
        'Longitud'
    ]].rename(columns={
        'Nombre Comercial': 'Business Name',
        'Sector Industrial': 'Industry Category',
        'Tamaño Empresa': 'Company Size',
        'Teléfono': 'Phone',
        'Correo Electrónico': 'Email',
        'Sitio Web': 'Website',
        'Ciudad': 'City',
        'Estado': 'State',
        'Latitud': 'Latitude',
        'Longitud': 'Longitude'
    }).dropna()

def main():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    st.title("🚀 Business Intelligence Suite")
    st.markdown("Plataforma avanzada de análisis y optimización comercial")

    with st.expander("⚙ Configuración Avanzada", expanded=False):
        api_input = st.text_input("Clave API", type="password", 
                                help="Requerida para funciones de inteligencia artificial")
        if api_input:
            st.session_state.api_key = encrypt_data(api_input)
            st.success("Configuración de seguridad actualizada")

    uploaded_file = st.file_uploader(
        "Cargar base de empresas (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Formatos soportados: CSV, Excel (hasta 500MB)"
    )

    if uploaded_file and st.session_state.api_key:
        if st.session_state.processed_data is None or uploaded_file.file_id != st.session_state.get('file_id'):
            with st.status("Analizando datos...", expanded=True) as status:
                try:
                    st.write("🔍 Validando estructura del archivo...")
                    df = load_and_process(uploaded_file)
                    
                    st.write("🧠 Ejecutando modelos predictivos...")
                    result = analyze_with_ai(df, st.session_state.api_key)
                    
                    if result:
                        st.session_state.processed_data = result
                        st.session_state.file_id = uploaded_file.file_id
                        status.update(label="Análisis completo ✅", state="complete")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {str(e)}")
                    st.session_state.processed_data = None

    if st.session_state.processed_data:
        st.markdown("## 📈 Resultados del Análisis")
        
        with st.container():
            st.markdown("### 🎯 Recomendaciones Estratégicas")
            st.write(st.session_state.processed_data['analisis'])
        
        col1, col2 = st.columns(2)
        with col1:
            selected_sectors = st.multiselect(
                "Sectores clave",
                options=st.session_state.processed_data['sugerencias']['sectores_top'],
                default=st.session_state.processed_data['sugerencias']['sectores_top'][:2]
            )
        with col2:
            selected_cities = st.multiselect(
                "Ubicaciones estratégicas",
                options=st.session_state.processed_data['sugerencias']['ciudades_clave'],
                default=st.session_state.processed_data['sugerencias']['ciudades_clave']
            )
        
        filtered_df = st.session_state.processed_data['df'][
            (st.session_state.processed_data['df']['Sector Industrial'].isin(selected_sectors)) &
            (st.session_state.processed_data['df']['Ciudad'].isin(selected_cities))
        ]
        
        st.markdown("### 🌍 Mapa de Concentración Comercial")
        create_interactive_map(filtered_df)
        
        st.markdown("## 📤 Exportación de Datos")
        export_format = st.radio("Formato de salida:", ["CSV", "Excel"], horizontal=True)
        
        google_ads_data = prepare_google_ads_data(filtered_df)
        if export_format == "CSV":
            data = google_ads_data.to_csv(index=False).encode('utf-8')
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                google_ads_data.to_excel(writer, index=False)
            data = output.getvalue()
        
        st.download_button(
            "Descargar Dataset Optimizado",
            data=data,
            file_name=f"business_data_{pd.Timestamp.now().strftime('%Y%m%d')}.{export_format.lower()}",
            mime='text/csv' if export_format == "CSV" else 'application/vnd.ms-excel'
        )

if __name__ == "__main__":
    main()
