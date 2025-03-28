# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
import unicodedata
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from io import BytesIO

# Configuración inicial
st.set_page_config(
    page_title="DENUE a Google Ads",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Mapeo de columnas según diccionario DENUE
COLUMN_MAPPING = {
    'nom_estab': 'nombre_negocio',
    'nombre_act': 'giro_principal',
    'per_ocu': 'personal_ocupado',
    'telefono': 'telefono',
    'correoelec': 'email',
    'www': 'sitio_web',
    'municipio': 'municipio',
    'entidad': 'estado',
    'latitud': 'latitud',
    'longitud': 'longitud'
}

REQUIRED_COLUMNS = [
    'nombre_negocio', 'giro_principal', 'personal_ocupado',
    'telefono', 'email', 'sitio_web', 'municipio',
    'estado', 'latitud', 'longitud'
]

def clean_column_name(col_name):
    """Normaliza nombres de columnas según DENUE"""
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])\
        .lower().strip().replace(' ', '_').split('[')[0]

def convert_employee_code(code):
    """Convierte códigos DENUE a rangos de empleados"""
    try:
        code = int(str(code).strip())
        ranges = {
            1: (0, 5, 'PYME'),
            2: (6, 10, 'PYME'),
            3: (11, 30, 'PYME'),
            4: (31, 50, 'PYME'),
            5: (51, 100, 'Mediana'),
            6: (101, 250, 'Mediana'),
            7: (251, 1000, 'Grande')
        }
        return ranges.get(code, (0, 0, 'Desconocido'))
    except:
        return (0, 0, 'Desconocido')

@st.cache_data(ttl=3600)
def process_data(uploaded_file):
    """Procesamiento seguro de datos DENUE"""
    try:
        # Carga de datos
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', dtype=str)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)
        
        # Limpieza y normalización
        df.columns = [clean_column_name(col) for col in df.columns]
        df = df.rename(columns=COLUMN_MAPPING)
        
        # Conversión de códigos
        df[['empleo_min', 'empleo_max', 'tamano_empresa']] = df['personal_ocupado'].apply(
            lambda x: pd.Series(convert_employee_code(x))
        )
        
        # Filtrado y tipado
        df = df[REQUIRED_COLUMNS + ['empleo_max', 'tamano_empresa']]
        df['empleo_max'] = pd.to_numeric(df['empleo_max'], errors='coerce')
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
        
        return df.dropna(subset=['latitud', 'longitud'])
    
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()

def create_google_ads_export(df):
    """Formato final para Google Ads"""
    return df[[
        'nombre_negocio', 'giro_principal', 'tamano_empresa',
        'telefono', 'email', 'sitio_web', 'municipio',
        'estado', 'latitud', 'longitud'
    ]].rename(columns={
        'nombre_negocio': 'Business Name',
        'giro_principal': 'Industry Category',
        'tamano_empresa': 'Company Size',
        'telefono': 'Phone',
        'email': 'Email',
        'sitio_web': 'Website',
        'municipio': 'City',
        'estado': 'State',
        'latitud': 'Latitude',
        'longitud': 'Longitude'
    })

def main():
    st.title("🚀 Transformador DENUE a Google Ads")
    st.markdown("Convierte bases DENUE en datasets listos para campañas publicitarias")
    
    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Sube tu archivo DENUE (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Tamaño máximo recomendado: 500MB"
    )
    
    if uploaded_file:
        with st.status("Procesando...", expanded=True) as status:
            try:
                # Procesamiento
                st.write("🔍 Validando estructura del archivo...")
                df = process_data(uploaded_file)
                
                # Análisis rápido
                st.write("📊 Analizando datos...")
                st.session_state.processed_data = df
                status.update(label="Procesamiento completo", state="complete")
                
            except Exception as e:
                st.error(f"Error en el procesamiento: {str(e)}")
                st.stop()
        
        # Visualización
        st.markdown("## Vista previa de datos")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Mapa interactivo
        st.markdown("## 🌍 Mapa de concentración")
        if not df.empty:
            m = folium.Map(location=[df['latitud'].mean(), df['longitud'].mean()], zoom_start=10)
            FastMarkerCluster(data=df[['latitud', 'longitud']].values.tolist()).add_to(m)
            st_folium(m, width=1200, height=600)
        
        # Exportación
        st.markdown("## 📤 Exportar para Google Ads")
        export_format = st.radio("Formato de exportación:", ["CSV", "Excel"], horizontal=True)
        
        google_ads_df = create_google_ads_export(df)
        if export_format == "CSV":
            data = google_ads_df.to_csv(index=False).encode('utf-8')
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                google_ads_df.to_excel(writer, index=False)
            data = output.getvalue()
        
        st.download_button(
            "Descargar Dataset Optimizado",
            data=data,
            file_name=f"google_ads_ready.{export_format.lower()}",
            mime='text/csv' if export_format == "CSV" else 'application/vnd.ms-excel'
        )

if __name__ == "__main__":
    main()
