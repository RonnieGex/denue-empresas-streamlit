import streamlit as st
import pandas as pd
import folium
import numpy as np
import unicodedata
import requests
import time
import logging
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Configuración profesional
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Business Intelligence Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Constantes estratégicas
MAX_RESULTADOS_DENUE = 1500
CLUSTERS_OPTIMOS = range(3, 8)
UMBRAL_ALERTA_DATOS = 10000
MAPA_CENTRO_MX = [23.6345, -102.5528]

CLAE_SECTORES = {
    'Comercio': ['46'],
    'Servicios': ['81', '62', '61', '72'],
    'Manufactura': ['31', '32', '33'],
    'Tecnología': ['54', '51', '72', '62'],
    'Salud': ['62', '86'],
    'Educación': ['61', '85', '88']
}

ESTRATO_TAMANOS = {
    'Micro (1-10)': ['1', '2'],
    'Pequeña (11-50)': ['3', '4'],
    'Mediana (51-250)': ['5', '6'],
    'Grande (251+)': ['7']
}

class DenueProspector:
    """Clase optimizada para interacción con API DENUE"""
    
    def __init__(self, token):
        self.token = token
        self.base_url = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr"
        self.page_size = 150
        
    @st.cache_data(ttl=3600, show_spinner="Buscando oportunidades comerciales...")
    def buscar_prospectos(_self, filtros):
        try:
            resultados = []
            total_pages = (MAX_RESULTADOS_DENUE // _self.page_size) + 1
            
            for page in range(total_pages):
                url = f"{_self.base_url}/00/0/0/0/0/{','.join(filtros['sectores_clae']}/0/0/0/0/{(page)*_self.page_size+1}/{(page+1)*_self.page_size}/0/{','.join(filtros['estratos']}/{_self.token}"
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    batch = _self.procesar_respuesta(response.json())
                    resultados.extend(batch)
                    if len(batch) < _self.page_size:
                        break
                elif response.status_code == 429:
                    time.sleep(2)
                    continue
                
                time.sleep(0.3)
            
            df = pd.DataFrame(resultados)
            return df[df.apply(lambda x: _self.validar_coordenadas(x['Latitud'], x['Longitud']), axis=1)]
            
        except Exception as e:
            logger.error(f"Error DENUE: {str(e)}")
            st.error("Error al conectar con DENUE. Verifique filtros o intente luego.")
            return pd.DataFrame()

    def procesar_respuesta(self, data):
        return [{
            'Nombre Comercial': item[2],
            'Sector': item[4],
            'Empleados': self.mapear_estrato(item[5]),
            'Teléfono': item[14],
            'Email': item[15],
            'Sitio Web': item[16],
            'Latitud': float(item[18]),
            'Longitud': float(item[17]),
            'Origen': 'DENUE'
        } for item in data if self.validar_registro(item)]

    def validar_registro(self, item):
        return float(item[18]) != 0 and float(item[17]) != 0

    @staticmethod
    def validar_coordenadas(lat, lon):
        return -90 <= lat <= 90 and -180 <= lon <= 180

    def mapear_estrato(self, estrato):
        return {
            "1": 5, "2": 10, "3": 30, "4": 50,
            "5": 100, "6": 250, "7": 500
        }.get(estrato, 0)

def normalize_column_name(col_name):
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip().replace(' ', '_')

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            chunks = pd.read_csv(uploaded_file, encoding='latin1', chunksize=50000)
            df = pd.concat([preprocesar_chunk(chunk) for chunk in chunks])
        else:
            df = preprocesar_chunk(pd.read_excel(uploaded_file, engine='openpyxl'))
        
        df = transformar_datos_usuario(df)
        return df
    except Exception as e:
        logger.error(f"Error procesamiento: {str(e)}")
        st.error("Formato de archivo no válido")
        return pd.DataFrame()

def preprocesar_chunk(chunk):
    chunk.columns = [normalize_column_name(col) for col in chunk.columns]
    rename_mapping = {
        col: target for target, posibles in REQUIRED_COLUMNS.items()
        for col in posibles if col in chunk.columns
    }
    return chunk.rename(columns=rename_mapping).rename(columns=COLUMN_NAMES_MAP)

def transformar_datos_usuario(df):
    df['Empleados'] = pd.to_numeric(df['Empleados'], errors='coerce').fillna(0).astype(int)
    df['Tamaño Empresa'] = np.select(
        [df['Empleados'] <= 10, df['Empleados'] <= 50, df['Empleados'] <= 250, df['Empleados'] > 250],
        ['Micro', 'Pequeña', 'Mediana', 'Grande'],
        default='Desconocido'
    )
    df['Origen'] = 'Usuario'
    return df.dropna(subset=['Latitud', 'Longitud'])

def analisis_estrategico(_df, segmentacion, api_key):
    try:
        scaler = StandardScaler()
        features = _df[['Empleados', 'Latitud', 'Longitud']].fillna(0)
        scaled_data = scaler.fit_transform(features)
        
        best_n = optimizar_clusters(scaled_data)
        _df['Segmento'] = KMeans(n_clusters=best_n, random_state=42).fit_predict(scaled_data)
        
        contexto = {
            'perfil_segmentos': _df.groupby('Segmento').agg({
                'Empleados': 'mean',
                'Latitud': 'mean',
                'Longitud': 'mean'
            }).to_dict(),
            'top_sectores': _df['Sector'].value_counts().nlargest(3).index.tolist(),
            'segmentacion': segmentacion
        }
        
        return generar_recomendaciones_ia(contexto, api_key)
        
    except Exception as e:
        logger.error(f"Error análisis IA: {str(e)}")
        return None

def optimizar_clusters(data):
    best_score = -1
    best_n = 5
    for n in CLUSTERS_OPTIMOS:
        labels = KMeans(n_clusters=n, random_state=42).fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_n = n
    return best_n

def generar_recomendaciones_ia(contexto, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    prompt = f"""
    Como experto en mercados B2B en México, analiza este contexto:
    {contexto}
    
    Genera 3 estrategias de prospección que incluyan:
    1. Canales de marketing recomendados
    2. Enfoque geográfico
    3. Propuesta de valor personalizada
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error IA: {str(e)}")
        return "Recomendaciones no disponibles temporalmente"

def crear_mapa_interactivo(df):
    m = folium.Map(location=MAPA_CENTRO_MX, zoom_start=5, tiles='cartodbpositron')
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=calcular_radio(row['Empleados']),
            color='#1f77b4' if row['Origen'] == 'Usuario' else '#2ca02c',
            fill=True,
            popup=f"<b>{row['Nombre Comercial']}</b><br>{row['Sector']}<br>Empleados: {row['Empleados']}"
        ).add_to(m)
    st_folium(m, width=1200, height=600)

def calcular_radio(empleados):
    return max(3, min(empleados / 10, 15))

def main():
    st.title("🚀 Business Intelligence Pro")
    st.markdown("**Herramienta de prospección comercial inteligente para México**")
    
    # Estado de la sesión
    if 'prospectos' not in st.session_state:
        st.session_state.prospectos = None
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Segmentación Estratégica")
        
        denue_token = st.text_input("Token DENUE", type="password")
        api_key = st.text_input("DeepSeek API Key", type="password")
        
        with st.expander("Configuración Avanzada"):
            sectores = st.multiselect(
                "Sectores objetivo",
                options=list(CLAE_SECTORES.keys()),
                default=['Tecnología', 'Servicios']
            )
            
            tamanos = st.multiselect(
                "Tamaño de empresa",
                options=list(ESTRATO_TAMANOS.keys()),
                default=['Pequeña (11-50)', 'Mediana (51-250)']
            )
            
            perfil_decisionor = st.selectbox(
                "Perfil del decisionador",
                options=["Director General", "Gerente de Compras", "Director de TI"],
                index=0
            )
        
        if st.button("Buscar Prospectos"):
            with st.status("Analizando mercado..."):
                try:
                    # Obtener datos DENUE
                    denue_client = DenueProspector(denue_token)
                    filtros = {
                        'sectores_clae': [clae for s in sectores for clae in CLAE_SECTORES[s]],
                        'estratos': [e for t in tamanos for e in ESTRATO_TAMANOS[t]]
                    }
                    df_denue = denue_client.buscar_prospectos(filtros)
                    
                    # Combinar con datos usuario
                    uploaded_file = st.file_uploader("Cargar base propia (opcional)", type=["csv", "xlsx"])
                    df_usuario = load_and_process(uploaded_file) if uploaded_file else pd.DataFrame()
                    
                    st.session_state.prospectos = pd.concat([df_denue, df_usuario], ignore_index=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Resultados
    if st.session_state.prospectos is not None:
        st.markdown(f"## 📊 {len(st.session_state.prospectos)} Prospectos Identificados")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🌍 Concentración Geográfica")
            crear_mapa_interactivo(st.session_state.prospectos)
        
        with col2:
            st.markdown("### 🔍 Filtros Estratégicos")
            sector_selected = st.multiselect(
                "Sectores prioritarios",
                options=st.session_state.prospectos['Sector'].unique(),
                default=st.session_state.prospectos['Sector'].value_counts().nlargest(2).index.tolist()
            )
            
            tamano_selected = st.multiselect(
                "Tamaño objetivo",
                options=st.session_state.prospectos['Tamaño Empresa'].unique(),
                default=['Pequeña', 'Mediana']
            )
            
            if st.button("Generar Recomendaciones"):
                df_filtrado = st.session_state.prospectos[
                    (st.session_state.prospectos['Sector'].isin(sector_selected)) &
                    (st.session_state.prospectos['Tamaño Empresa'].isin(tamano_selected))
                ]
                
                with st.spinner("Analizando con IA..."):
                    recomendaciones = analisis_estrategico(df_filtrado, {
                        'sectores': sector_selected,
                        'tamanos': tamano_selected,
                        'perfil': perfil_decisionor
                    }, api_key)
                    
                    st.markdown("### 🧠 Estrategias Recomendadas")
                    st.markdown(recomendaciones)
        
        st.markdown("### 📥 Exportar Prospectos")
        export_format = st.radio("Formato", ["CSV", "Excel"], horizontal=True)
        
        if export_format == "CSV":
            data = st.session_state.prospectos.to_csv(index=False).encode('utf-8')
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state.prospectos.to_excel(writer, index=False)
            data = output.getvalue()
        
        st.download_button(
            "Descargar Dataset",
            data=data,
            file_name=f"prospectos_{pd.Timestamp.now().strftime('%Y%m%d')}.{export_format.lower()}",
            mime='text/csv' if export_format == "CSV" else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == "__main__":
    main()
