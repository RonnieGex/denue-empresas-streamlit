import streamlit as st
import pandas as pd
import folium
import numpy as np
import unicodedata
import requests
import time
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from openai import OpenAI
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración inicial
st.set_page_config(
    page_title="Business Intelligence Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Constantes y Mapeos
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

CLAE_SECTORES = {
    'Comercio': ['46'],
    'Servicios': ['81', '62', '61'],
    'Manufactura': ['31', '32'],
    'Tecnología': ['54', '51', '72'],
    'Salud': ['62'],
    'Educación': ['61', '85']
}

ESTRATO_TAMANOS = {
    'Micro (1-10)': ['1', '2'],
    'Pequeña (11-50)': ['3', '4'],
    'Mediana (51-250)': ['5', '6']
}

# Clases y Funciones principales
class DenueClient:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr"
        self.page_size = 100

    @st.cache_data(ttl=3600, show_spinner="Buscando en DENUE...")
    def buscar(_self, filtros, max_resultados=1000):
        resultados = []
        estrato = ",".join(filtros['estratos'])
        sector = ",".join(filtros['sectores_clae'])

        for page in range(0, max_resultados // _self.page_size):
            try:
                url = f"{_self.base_url}/00/0/0/0/0/{sector}/0/0/0/0/{page * _self.page_size + 1}/{(page + 1) * _self.page_size}/0/{estrato}/{_self.token}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    batch = _self.parse_response(response.json())
                    resultados.extend(batch)
                    if len(batch) < _self.page_size:
                        break
                time.sleep(0.5)
            except Exception as e:
                st.error(f"Error en página {page + 1}: {str(e)}")
                break
        return pd.DataFrame(resultados)

    def parse_response(self, data):
        return [{
            'Nombre Comercial': item[2],
            'Sector Industrial': item[4],
            'Empleados': self.mapear_estrato(item[5]),
            'Teléfono': item[14],
            'Correo Electrónico': item[15],
            'Sitio Web': item[16],
            'Latitud': float(item[18]),
            'Longitud': float(item[17]),
            'Origen': 'DENUE'
        } for item in data]

    def mapear_estrato(self, estrato):
        return {
            "1": 5, "2": 10, "3": 30, "4": 50,
            "5": 100, "6": 250, "7": 500
        }.get(estrato, 0)

def normalize_column_name(col_name):
    nfkd = unicodedata.normalize('NFKD', str(col_name))
    return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip().replace(' ', '_').split('[')[0]

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process(uploaded_file):
    progress = st.progress(0, text="Iniciando procesamiento...")
    try:
        progress.progress(10, "Leyendo archivo...")
        if uploaded_file.name.endswith('.csv'):
            chunks = pd.read_csv(
                uploaded_file,
                encoding='latin1',
                chunksize=50000,
                dtype={'telefono': 'string'}
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

        progress.progress(50, "Procesando información...")
        df['Empleados'] = pd.to_numeric(df['Empleados'], errors='coerce').fillna(0).astype(int)

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
        df = df.dropna(subset=['Estado', 'Ciudad', 'Sector Industrial', 'Latitud', 'Longitud'])
        df['Origen'] = 'Usuario'
        progress.progress(100, "¡Proceso completado!")
        return df
    except Exception as e:
        progress.empty()
        st.error(f"Error crítico: {str(e)}")
        st.stop()

@st.cache_data(ttl=3600)
def analyze_with_ai(_df, api_key, segmentacion):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        numeric_cols = _df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(_df[numeric_cols].fillna(0))

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        _df = _df.copy()
        _df['Segmento IA'] = kmeans.fit_predict(scaled_data)

        context = {
            'sectores_top': _df['Sector Industrial'].value_counts().nlargest(5).index.tolist(),
            'empleados_promedio': round(_df[_df['Empleados'] > 0]['Empleados'].mean()),
            'ciudades_clave': _df.groupby('Ciudad')['Segmento IA'].count().nlargest(3).index.tolist(),
            'segmentacion': segmentacion
        }

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"""Eres un experto en marketing digital B2B. Genera recomendaciones usando:\n- Segmentación: {segmentacion}\n- Contexto DENUE: {context}"""},
                {"role": "user", "content": "Genera 3 estrategias segmentadas con canales específicos, enfoque geográfico y propuestas de valor personalizadas:"}
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
    if df.empty:
        return st.warning("No hay datos para mostrar")

    with st.spinner("Generando visualización geoespacial..."):
        map_center = [df['Latitud'].mean(), df['Longitud'].mean()]
        m = folium.Map(location=map_center, zoom_start=10, tiles='cartodbpositron')
        FastMarkerCluster(data=df[['Latitud', 'Longitud']].values.tolist()).add_to(m)
        st_folium(m, width=1200, height=600)

def prepare_google_ads_data(df):
    return df[[
        'Nombre Comercial', 'Sector Industrial', 'Tamaño Empresa',
        'Teléfono', 'Correo Electrónico', 'Sitio Web',
        'Ciudad', 'Estado', 'Latitud', 'Longitud', 'Origen'
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
        'Longitud': 'Longitude',
        'Origen': 'Data Source'
    }).dropna()

def main():
    st.title("🚀 Business Intelligence Suite")
    st.markdown("Plataforma avanzada de análisis comercial con integración DENUE")

    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'denue_token' not in st.session_state:
        st.session_state.denue_token = None

    with st.sidebar:
        st.header("⚙ Configuración")
        st.session_state.api_key = st.text_input("Clave DeepSeek API", type="password")
        st.session_state.denue_token = st.text_input("Token DENUE", type="password")

        st.divider()
        with st.expander("🎯 Parámetros de Segmentación"):
            segmentacion = {
                'edad': st.selectbox("Edad del dueño:", ["30-50 años (Crecimiento)", "51-65 años (Consolidados)"]),
                'educacion': st.selectbox("Nivel educativo:", ["Licenciatura+", "Posgrado en negocios/tecnología"]),
                'tamanos': st.multiselect("Tamaño empresa:", list(ESTRATO_TAMANOS.keys())),
                'sectores': st.multiselect("Sectores objetivo:", list(CLAE_SECTORES.keys()))
            }

    uploaded_file = st.file_uploader("Cargar base de empresas (CSV/Excel)", type=["csv", "xlsx"])

    if uploaded_file and st.session_state.api_key and st.session_state.denue_token:
        if st.session_state.processed_data is None or uploaded_file.file_id != st.session_state.get('file_id'):
            with st.status("Analizando datos...", expanded=True) as status:
                try:
                    st.write("🔍 Procesando datos cargados...")
                    df_usuario = load_and_process(uploaded_file)

                    st.write("🌐 Consultando API DENUE...")
                    denue = DenueClient(st.session_state.denue_token)
                    filtros = {
                        'sectores_clae': [clae for s in segmentacion['sectores'] for clae in CLAE_SECTORES[s]],
                        'estratos': [e for t in segmentacion['tamanos'] for e in ESTRATO_TAMANOS[t]]
                    }
                    df_denue = denue.buscar(filtros)

                    full_df = pd.concat([df_usuario, df_denue], ignore_index=True)

                    st.write("🧐 Ejecutando modelos predictivos...")
                    result = analyze_with_ai(full_df, st.session_state.api_key, segmentacion)

                    if result:
                        st.session_state.processed_data = result
                        st.session_state.file_id = uploaded_file.file_id
                        status.update(label="Análisis completo ✅", state="complete")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {str(e)}")
                    st.session_state.processed_data = None

    if st.session_state.processed_data is not None:
        st.markdown("## 📈 Resultados del Análisis")

        st.markdown("### 🎯 Recomendaciones Estratégicas")
        st.write(st.session_state.processed_data['analisis'])

        col1, col2 = st.columns(2)
        with col1:
            selected_sectors = st.multiselect("Sectores clave", st.session_state.processed_data['sugerencias']['sectores_top'])
        with col2:
            selected_cities = st.multiselect("Ubicaciones estratégicas", st.session_state.processed_data['sugerencias']['ciudades_clave'])

        filtered_df = st.session_state.processed_data['df'][
            (st.session_state.processed_data['df']['Sector Industrial'].isin(selected_sectors)) &
            (st.session_state.processed_data['df']['Ciudad'].isin(selected_cities))
        ]

        st.markdown("### 🌍 Mapa de Concentración Comercial")
        create_interactive_map(filtered_df)

        st.markdown("## 📄 Exportación de Datos")
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
            mime="application/octet-stream"
        )

if __name__ == "__main__":
    main()

