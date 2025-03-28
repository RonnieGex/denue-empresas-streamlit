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
    page_title="Business Optimizer Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Constantes basadas en diccionario DENUE
REQUIRED_COLUMNS = {
    'nom_estab': ['nom_estab'],
    'nombre_act': ['nombre_act'],
    'per_ocu': ['per_ocu'],
    'telefono': ['telefono'],
    'correoelec': ['correoelec'],
    'www': ['www'],
    'municipio': ['municipio'],
    'localidad': ['localidad'],
    'entidad': ['entidad'],
    'latitud': ['latitud'],
    'longitud': ['longitud']
}

COLUMN_NAMES_MAP = {
    'nom_estab': 'Nombre_Negocio',
    'nombre_act': 'Giro_Principal',
    'per_ocu': 'Personal_Ocupado',
    'telefono': 'Telefono',
    'correoelec': 'Email',
    'www': 'Sitio_Web',
    'municipio': 'Municipio',
    'localidad': 'Localidad',
    'entidad': 'Estado',
    'latitud': 'Latitud',
    'longitud': 'Longitud'
}

# Función de seguridad
def encrypt_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

def normalize_column_name(col_name):
    """Normaliza nombres de columnas según DENUE"""
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])\
        .lower().strip().replace(' ', '_').split('[')[0]

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process(uploaded_file):
    """Carga y procesamiento optimizado"""
    progress = st.progress(0, text="Iniciando procesamiento...")
    
    try:
        # Lectura segura de archivos grandes
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

        # Normalización y validación
        progress.progress(30, "Estandarizando datos...")
        df.columns = [normalize_column_name(col) for col in df.columns]
        
        # Mapeo exacto de columnas
        rename_mapping = {}
        for target_col, possible_cols in REQUIRED_COLUMNS.items():
            for col in possible_cols:
                if col in df.columns:
                    rename_mapping[col] = target_col
        df = df.rename(columns=rename_mapping)
        df = df.rename(columns=COLUMN_NAMES_MAP)

        # Validación crítica
        required = list(COLUMN_NAMES_MAP.values())
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Columnas requeridas faltantes: {', '.join(missing)}")
            st.stop()

        # Transformación de datos
        progress.progress(50, "Procesando información...")
        df['Tamaño_Empresa'] = df['Personal_Ocupado'].apply(
            lambda x: 'PYME' if x <= 5 else 'Mediana' if x <= 100 else 'Grande'
        )
        
        # Optimización de tipos
        dtypes = {
            'Latitud': 'float32',
            'Longitud': 'float32',
            'Municipio': 'category',
            'Estado': 'category'
        }
        df = df.astype(dtypes, errors='ignore')

        # Limpieza final
        progress.progress(80, "Depurando datos...")
        df = df.dropna(subset=['Estado', 'Municipio', 'Giro_Principal'])
        
        progress.progress(100, "¡Proceso completado!")
        return df

    except Exception as e:
        progress.empty()
        st.error(f"Error crítico: {str(e)}")
        st.stop()

@st.cache_data(ttl=3600)
def analyze_with_ai(_df, api_key):
    """Análisis predictivo integrado"""
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        
        # Clusterización avanzada
        numeric_cols = _df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(_df[numeric_cols].fillna(0))
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        _df['Segmento_IA'] = kmeans.fit_predict(scaled_data)
        
        # Generación de recomendaciones
        context = {
            'giros_top': _df['Giro_Principal'].value_counts().nlargest(5).index.tolist(),
            'empleados_promedio': round(_df['Personal_Ocupado'].mean()),
            'ubicaciones_clave': _df.groupby('Municipio')['Segmento_IA'].count().nlargest(3).index.tolist()
        }
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": "Eres un experto en marketing digital B2B. Genera recomendaciones basadas en:"
            },{
                "role": "user",
                "content": f"{context}\n\nSugiere estrategias segmentadas para Google Ads:"
            }],
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
    """Visualización geoespacial optimizada"""
    if df.empty:
        st.warning("No hay datos para mostrar")
        return
    
    map_center = [df['Latitud'].mean(), df['Longitud'].mean()]
    with st.spinner("Generando mapa..."):
        m = folium.Map(location=map_center, zoom_start=10, tiles='cartodbpositron')
        FastMarkerCluster(data=df[['Latitud', 'Longitud']].values.tolist()).add_to(m)
        st_folium(m, width=1200, height=600)

def prepare_google_ads_data(df):
    """Transformación para Google Ads"""
    return df[[
        'Nombre_Negocio',
        'Giro_Principal',
        'Tamaño_Empresa',
        'Telefono',
        'Email',
        'Sitio_Web',
        'Municipio',
        'Estado',
        'Latitud',
        'Longitud'
    ]].rename(columns={
        'Nombre_Negocio': 'Business Name',
        'Giro_Principal': 'Industry Category',
        'Tamaño_Empresa': 'Business Size',
        'Telefono': 'Phone',
        'Email': 'Contact Email',
        'Sitio_Web': 'Website',
        'Municipio': 'City',
        'Estado': 'State',
        'Latitud': 'Latitude',
        'Longitud': 'Longitude'
    }).dropna()

def main():
    # Estado de sesión
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # Interfaz principal
    st.title("🚀 Optimizador para Google Ads Pro")
    st.markdown("Transformación inteligente de datos empresariales")

    # Configuración de API segura
    with st.expander("🔐 Configuración Avanzada", expanded=False):
        api_input = st.text_input("Clave API", type="password", help="Requerida para análisis predictivo")
        if api_input:
            st.session_state.api_key = encrypt_data(api_input)
            st.success("Configuración guardada exitosamente")

    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Sube tu base de datos (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Tamaño máximo recomendado: 300MB"
    )

    # Procesamiento automático
    if uploaded_file and st.session_state.api_key:
        if st.session_state.processed_data is None or uploaded_file.file_id != st.session_state.get('file_id'):
            with st.status("Procesando...", expanded=True) as status:
                try:
                    st.write("🔍 Validando estructura...")
                    df = load_and_process(uploaded_file)
                    
                    st.write("🧠 Analizando con IA...")
                    result = analyze_with_ai(df, st.session_state.api_key)
                    
                    if result:
                        st.session_state.processed_data = result
                        st.session_state.file_id = uploaded_file.file_id
                        status.update(label="Análisis completo ✅", state="complete")
                except Exception as e:
                    st.error(f"Error de procesamiento: {str(e)}")
                    st.session_state.processed_data = None

    # Resultados y exportación
    if st.session_state.processed_data:
        st.markdown("## 📊 Resultados del Análisis")
        
        with st.container():
            st.markdown("### 💡 Recomendaciones Estratégicas")
            st.write(st.session_state.processed_data['analisis'])
        
        # Filtros interactivos
        col1, col2 = st.columns(2)
        with col1:
            selected_giros = st.multiselect(
                "Seleccionar giros",
                options=st.session_state.processed_data['sugerencias']['giros_top'],
                default=st.session_state.processed_data['sugerencias']['giros_top'][:2]
            )
        with col2:
            selected_ubicaciones = st.multiselect(
                "Filtrar ubicaciones",
                options=st.session_state.processed_data['sugerencias']['ubicaciones_clave'],
                default=st.session_state.processed_data['sugerencias']['ubicaciones_clave']
            )
        
        # Aplicar filtros
        filtered_df = st.session_state.processed_data['df'][
            (st.session_state.processed_data['df']['Giro_Principal'].isin(selected_giros)) &
            (st.session_state.processed_data['df']['Municipio'].isin(selected_ubicaciones))
        ]
        
        # Visualización
        st.markdown("### 🌍 Mapa de Concentración")
        create_interactive_map(filtered_df)
        
        # Exportación
        st.markdown("## 📤 Exportar para Google Ads")
        export_format = st.radio("Formato:", ["CSV", "Excel"], horizontal=True)
        
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
            file_name=f"google_ads_data_{pd.Timestamp.now().strftime('%Y%m%d')}.{export_format.lower()}",
            mime='text/csv' if export_format == "CSV" else 'application/vnd.ms-excel'
        )

if __name__ == "__main__":
    main()
