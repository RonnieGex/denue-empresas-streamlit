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
    page_title="Business Segment AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
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

# Función de normalización de columnas
def normalize_column_name(col_name):
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip().replace(' ', '_')

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
        df = df.rename(columns={v:k for k,v in REQUIRED_COLUMNS.items()})
        df = df.rename(columns=COLUMN_NAMES_MAP)

        # Estimación de empleados
        progress.progress(50, "Calculando empleados...")
        df['Empleados Estimados'] = df['Empleados (texto)'].apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else None
        )

        # Limpieza final
        progress.progress(80, "Depurando datos...")
        df = df.dropna(subset=['Estado', 'Municipio', 'Giro'])
        
        progress.progress(100, "¡Listo!")
        return df

    except Exception as e:
        progress.empty()
        st.error(f"Error crítico: {str(e)}")
        st.stop()

@st.cache_data(ttl=3600)
def analyze_data(_df, api_key):
    """Análisis predictivo con IA"""
    try:
        # Configurar cliente IA
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        
        # Análisis básico
        numeric_cols = _df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(_df[numeric_cols].fillna(0))
        
        # Segmentación
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        _df['segmento'] = kmeans.fit_predict(scaled_data)
        
        # Generar recomendaciones con IA
        context = f"""
            Datos clave:
            - Giros principales: {_df['Giro'].value_counts().nlargest(5).index.tolist()}
            - Empleados promedio: {_df['Empleados Estimados'].mean():.0f}
            - Municipios con mayor concentración: {_df['Municipio'].value_counts().nlargest(3).index.tolist()}
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": "Eres un analista de negocios experto. Genera recomendaciones de segmentación."
            },{
                "role": "user",
                "content": f"{context}\n\nSugiere estrategias de marketing efectivas:"
            }],
            temperature=0.5,
            max_tokens=500
        )
        
        return {
            'df': _df,
            'analysis': response.choices[0].message.content,
            'suggestions': {
                'giros': _df['Giro'].value_counts().nlargest(5).index.tolist(),
                'empleados': _df['Empleados Estimados'].quantile(0.75),
                'municipios': _df['Municipio'].value_counts().nlargest(3).index.tolist()
            }
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
    st.title("📈 Business Intelligence Pro")
    st.markdown("Análisis predictivo y segmentación avanzada para estrategias comerciales")

    # Configuración de API (oculta)
    with st.expander("⚙ Configuración avanzada", expanded=False):
        api_input = st.text_input("Clave de API", type="password", help="Clave requerida para análisis avanzado")
        if api_input:
            st.session_state.api_key = encrypt_data(api_input)
            st.success("Configuración guardada exitosamente")

    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Sube tu archivo de empresas (CSV o Excel)",
        type=["csv", "xlsx"],
        help="Tamaño máximo: 300MB"
    )

    # Procesamiento automático
    if uploaded_file and st.session_state.api_key:
        if st.session_state.processed_data is None or (
            uploaded_file.name != st.session_state.get('current_file') or
            uploaded_file.size != st.session_state.get('file_size')
        ):
            with st.status("Analizando datos...", expanded=True) as status:
                # Procesar archivo
                st.write("🔍 Validando estructura...")
                df = load_and_process(uploaded_file)
                
                # Análisis IA
                st.write("🧠 Ejecutando modelos predictivos...")
                result = analyze_data(df, st.session_state.api_key)
                
                if result:
                    st.session_state.processed_data = result
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.file_size = uploaded_file.size
                    status.update(label="Análisis completo", state="complete")

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
                "Seleccionar giros",
                options=st.session_state.processed_data['suggestions']['giros'],
                default=st.session_state.processed_data['suggestions']['giros'][:2]
            )
        with col2:
            selected_municipios = st.multiselect(
                "Seleccionar ubicaciones",
                options=st.session_state.processed_data['suggestions']['municipios'],
                default=st.session_state.processed_data['suggestions']['municipios']
            )
        
        # Filtrar datos
        filtered_df = st.session_state.processed_data['df'][
            (st.session_state.processed_data['df']['Giro'].isin(selected_giros)) &
            (st.session_state.processed_data['df']['Municipio'].isin(selected_municipios))
        ]
        
        # Visualización
        with st.container():
            st.markdown("### 🌍 Mapa de Concentración")
            if not filtered_df.empty:
                map_center = [filtered_df['Latitud'].mean(), filtered_df['Longitud'].mean()]
                m = folium.Map(location=map_center, zoom_start=10)
                FastMarkerCluster(data=filtered_df[['Latitud', 'Longitud']].values.tolist()).add_to(m)
                st_folium(m, width=1200, height=500)
            else:
                st.warning("No hay datos para mostrar con los filtros actuales")

        # Exportación
        st.markdown("## 📤 Exportar Segmento")
        export_format = st.selectbox("Formato de exportación", ["CSV", "Excel"])
        
        if export_format == "CSV":
            data = filtered_df.to_csv(index=False).encode('utf-8')
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False)
            data = output.getvalue()
        
        st.download_button(
            "Descargar datos",
            data=data,
            file_name=f"segmento_empresas.{export_format.lower()}",
            mime='text/csv' if export_format == "CSV" else 'application/vnd.ms-excel'
        )

if __name__ == "__main__":
    main()
