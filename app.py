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
    page_title="Business Analytics Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Constantes
REQUIRED_COLUMNS = {
    'nombre': ['establecimiento', 'empresa', 'negocio'],
    'giro': ['actividad', 'rubro', 'categoria'],
    'empleados': ['personal', 'trabajadores', 'staff'],
    'telefono': ['contacto', 'tel'],
    'correo': ['email', 'e-mail'],
    'web': ['sitio', 'pagina'],
    'municipio': ['ciudad', 'delegacion'],
    'localidad': ['zona', 'region'],
    'estado': ['entidad'],
    'latitud': ['lat'],
    'longitud': ['lon']
}

COLUMN_NAMES_MAP = {
    'nombre': 'Nombre',
    'giro': 'Giro',
    'empleados': 'Empleados (texto)',
    'empleados_estimados': 'Empleados Estimados',
    'telefono': 'Teléfono',
    'correo': 'Correo',
    'web': 'Web',
    'municipio': 'Municipio',
    'localidad': 'Localidad',
    'estado': 'Estado',
    'latitud': 'Latitud',
    'longitud': 'Longitud'
}

# Función para encriptar datos sensibles
def encrypt_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

def normalize_column_name(col_name):
    """Normaliza nombres de columnas para matching flexible"""
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    normalized = ''.join([c for c in nfkd if not unicodedata.combining(c)])
    normalized = normalized.lower().strip().replace(' ', '_').split('[')[0]
    return normalized

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process(uploaded_file):
    """Carga y procesamiento optimizado para grandes archivos"""
    progress = st.progress(0, text="Iniciando procesamiento...")
    
    try:
        # Lectura inicial
        progress.progress(10, "Leyendo archivo...")
        if uploaded_file.name.endswith('.csv'):
            chunks = pd.read_csv(
                uploaded_file,
                encoding='latin1',
                chunksize=50000,
                dtype={'telefono': 'string', 'correo': 'string'}
            )
            df = pd.concat(chunks)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Normalización
        progress.progress(30, "Estandarizando datos...")
        df.columns = [normalize_column_name(col) for col in df.columns]
        
        # Corrección del mapeo de columnas
        rename_mapping = {}
        for target_col, possible_cols in REQUIRED_COLUMNS.items():
            for col in possible_cols:
                if col in df.columns:
                    rename_mapping[col] = target_col
        df = df.rename(columns=rename_mapping)
        df = df.rename(columns=COLUMN_NAMES_MAP)

        # Validación de columnas
        missing_cols = [col for col in COLUMN_NAMES_MAP.values() if col not in df.columns]
        if missing_cols:
            st.error(f"Columnas requeridas faltantes: {', '.join(missing_cols)}")
            st.stop()

        # Estimación de empleados
        progress.progress(50, "Calculando empleados...")
        df['Empleados Estimados'] = df['Empleados (texto)'].apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else None
        ).astype('Int64')

        # Optimización de tipos de datos
        dtypes = {
            'Latitud': 'float32',
            'Longitud': 'float32',
            'Municipio': 'category',
            'Estado': 'category'
        }
        df = df.astype(dtypes, errors='ignore')

        # Limpieza final
        progress.progress(80, "Depurando datos...")
        df = df.dropna(subset=['Estado', 'Municipio', 'Giro'])
        
        progress.progress(100, "¡Procesamiento completado!")
        return df

    except Exception as e:
        progress.empty()
        st.error(f"Error crítico: {str(e)}")
        st.stop()

@st.cache_data(ttl=3600)
def analyze_data(_df, api_key):
    """Análisis predictivo con IA"""
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        
        # Análisis básico
        numeric_cols = _df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(_df[numeric_cols].fillna(0))
        
        # Segmentación
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        _df['segmento'] = kmeans.fit_predict(scaled_data)
        
        # Generar recomendaciones con IA
        context = {
            'giros_top': _df['Giro'].value_counts().nlargest(5).index.tolist(),
            'empleados_promedio': round(_df['Empleados Estimados'].mean()),
            'municipios_top': _df['Municipio'].value_counts().nlargest(3).index.tolist()
        }
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": "Eres un analista de negocios experto. Genera recomendaciones de segmentación."
            },{
                "role": "user",
                "content": f"Contexto: {context}\n\nSugiere estrategias de marketing efectivas:"
            }],
            temperature=0.5,
            max_tokens=500
        )
        
        return {
            'df': _df,
            'analysis': response.choices[0].message.content,
            'suggestions': context
        }
        
    except Exception as e:
        st.error(f"Error en análisis IA: {str(e)}")
        return None

def main():
    # Configurar estado de sesión
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # Interfaz principal
    st.title("📊 Business Intelligence Suite")
    st.markdown("Plataforma avanzada de análisis empresarial")

    # Configuración de API
    with st.expander("🔑 Configuración avanzada", expanded=False):
        api_input = st.text_input("Clave de API", type="password", help="Clave requerida para análisis avanzado")
        if api_input:
            st.session_state.api_key = encrypt_data(api_input)
            st.success("Configuración de seguridad actualizada")

    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Sube tu archivo de datos empresariales",
        type=["csv", "xlsx"],
        help="Formatos soportados: CSV, Excel (hasta 300MB)"
    )

    # Procesamiento automático
    if uploaded_file and st.session_state.api_key:
        if st.session_state.processed_data is None or uploaded_file.file_id != st.session_state.get('file_id'):
            with st.status("Analizando datos...", expanded=True) as status:
                try:
                    st.write("🔍 Validando estructura del archivo...")
                    df = load_and_process(uploaded_file)
                    
                    st.write("🧠 Ejecutando modelos predictivos...")
                    result = analyze_data(df, st.session_state.api_key)
                    
                    if result:
                        st.session_state.processed_data = result
                        st.session_state.file_id = uploaded_file.file_id
                        status.update(label="Análisis completado ✅", state="complete")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {str(e)}")
                    st.session_state.processed_data = None

    # Mostrar resultados
    if st.session_state.processed_data:
        st.markdown("## 🔍 Resultados del Análisis")
        
        # Recomendaciones estratégicas
        with st.container():
            st.markdown("### 🚀 Estrategias Recomendadas")
            st.write(st.session_state.processed_data['analysis'])
        
        # Filtros interactivos
        col1, col2 = st.columns(2)
        with col1:
            selected_giros = st.multiselect(
                "Giros comerciales",
                options=st.session_state.processed_data['suggestions']['giros_top'],
                default=st.session_state.processed_data['suggestions']['giros_top'][:2]
            )
        with col2:
            selected_municipios = st.multiselect(
                "Ubicaciones clave",
                options=st.session_state.processed_data['suggestions']['municipios_top'],
                default=st.session_state.processed_data['suggestions']['municipios_top']
            )
        
        # Filtrar datos
        filtered_df = st.session_state.processed_data['df'][
            (st.session_state.processed_data['df']['Giro'].isin(selected_giros)) &
            (st.session_state.processed_data['df']['Municipio'].isin(selected_municipios))
        ]
        
        # Visualización
        with st.container():
            st.markdown("### 🌍 Mapa de Concentración Comercial")
            if not filtered_df.empty:
                map_center = [filtered_df['Latitud'].mean(), filtered_df['Longitud'].mean()]
                m = folium.Map(location=map_center, zoom_start=10, tiles='cartodbpositron')
                FastMarkerCluster(data=filtered_df[['Latitud', 'Longitud']].values.tolist()).add_to(m)
                st_folium(m, width=1200, height=500)
            else:
                st.warning("No se encontraron resultados con los filtros actuales")

        # Exportación
        st.markdown("## 📤 Exportar Datos")
        export_format = st.radio("Formato de exportación:", ["CSV", "Excel"], horizontal=True)
        
        if export_format == "CSV":
            data = filtered_df.to_csv(index=False).encode('utf-8')
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False)
            data = output.getvalue()
        
        st.download_button(
            "Descargar segmento seleccionado",
            data=data,
            file_name=f"segmento_empresas.{export_format.lower()}",
            mime='text/csv' if export_format == "CSV" else 'application/vnd.ms-excel'
        )

if __name__ == "__main__":
    main()
