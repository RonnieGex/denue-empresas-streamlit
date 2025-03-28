
import streamlit as st
import pandas as pd
import folium
import numpy as np
import requests
import json
import time
from streamlit_folium import st_folium
from openai import OpenAI
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Configuración inicial
st.set_page_config(
    page_title="Katalis Movistar BI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📶"
)

# Constantes
CLAE_SECTORES = {
    'Tecnología': ['51', '72'],
    'Logística': ['48', '49'],
    'Manufactura': ['31', '32', '33'],
    'Salud': ['62', '86'],
    'Educación': ['61', '85']
}

ESTRATO_TAMANOS = {
    'PYME (10-50)': ['3', '4'],
    'Mediana (51-250)': ['5', '6'],
    'Corporativo (250+)': ['7']
}

class DenueProspector:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr"
        self.page_size = 200
        
    @st.cache_data(ttl=3600, show_spinner="Buscando en DENUE...")
    def buscar_prospectos(_self, filtros):
        try:
            resultados = []
            for page in range(5):
                url = f"{_self.base_url}/00/0/0/0/0/{','.join(filtros['sectores'])}/0/0/0/0/{page*_self.page_size+1}/{(page+1)*_self.page_size}/0/{','.join(filtros['estratos'])}/{_self.token}"
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    batch = _self.procesar_respuesta(response.json())
                    resultados.extend(batch)
                    if len(batch) < _self.page_size:
                        break
                time.sleep(0.5)
            return pd.DataFrame(resultados)
        except Exception as e:
            st.error(f"Error DENUE: {str(e)}")
            return pd.DataFrame()

    def procesar_respuesta(self, data):
        return [{
            'Nombre': item[2],
            'Sector': self.clasificar_sector(item[4]),
            'Empleados': self.mapear_empleados(item[5]),
            'Teléfono': self.formatear_telefono(item[14]),
            'Email': self.validar_email(item[15]),
            'Ubicación': f"{item[7]} {item[8]}, {item[10]}",
            'Latitud': float(item[18]),
            'Longitud': float(item[17]),
            'Score': self.calcular_score(item[4], item[5])
        } for item in data if self.validar_registro(item)]

    def clasificar_sector(self, codigo):
        return {
            '51': 'Tecnología', '72': 'Tecnología',
            '48': 'Logística', '49': 'Logística',
            '62': 'Salud', '86': 'Salud',
            '61': 'Educación', '85': 'Educación'
        }.get(codigo[:2], 'Otros')

    def mapear_empleados(self, estrato):
        return {'1':'1-5','2':'6-10','3':'11-30','4':'31-50','5':'51-100','6':'101-250','7':'251+'}.get(estrato, 'N/A')

    def calcular_score(self, sector, estrato):
        return (['Tecnología','Logística','Salud'].count(self.clasificar_sector(sector)) * 0.6) + (int(estrato) * 0.4)

    def validar_email(self, email):
        return email if '@' in str(email) and '.' in str(email) else ''

    def validar_registro(self, item):
        return all([item[14], item[18], item[17]])

    def formatear_telefono(self, tel):
        return f"+52 {tel[:3]} {tel[3:6]} {tel[6:]}" if tel else ''

def mostrar_configuracion():
    with st.sidebar:
        st.header("⚙ Configuración")
        api_key = st.text_input("DeepSeek API Key", type="password", 
                              help="Obtenla en https://platform.deepseek.com/api-keys")
        
        st.divider()
        sectores = st.multiselect("Sectores Objetivo", list(CLAE_SECTORES.keys()), default=['Tecnología', 'Logística'])
        tamanos = st.multiselect("Tamaño Empresa", list(ESTRATO_TAMANOS.keys()), default=['PYME (10-50)'])
        
        if st.button("Buscar Prospectos", type="primary"):
            with st.status("Analizando...", expanded=True):
                procesar_busqueda(sectores, tamanos, api_key)

def procesar_busqueda(sectores, tamanos, api_key):
    try:
        denue = DenueProspector(st.secrets.DENUE_TOKEN)
        df = denue.buscar_prospectos({
            'sectores': [c for s in sectores for c in CLAE_SECTORES[s]],
            'estratos': [e for t in tamanos for e in ESTRATO_TAMANOS[t]]
        })
        st.session_state.prospectos = df[df['Score'] >= 3.5]
    except Exception as e:
        st.error(f"Error: {str(e)}")

def mostrar_resultados():
    st.title("📊 Resultados de Prospectación")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("Mapa de Calor", expanded=True):
            crear_mapa()
        
        with st.expander("Datos Crudos"):
            st.dataframe(st.session_state.prospectos)

    with col2:
        with st.container(border=True):
            st.download_button("📥 Exportar CSV", 
                             st.session_state.prospectos.to_csv().encode('utf-8'),
                             "prospectos.csv",
                             "text/csv")
            
            if st.button("🧠 Generar Estrategias"):
                if 'api_key' not in st.session_state:
                    st.error("Ingresa tu API Key primero")
                else:
                    generar_estrategias()

def crear_mapa():
    m = folium.Map(location=[23.6345, -102.5528], zoom_start=5)
    for _, row in st.session_state.prospectos.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"<b>{row['Nombre']}</b><br>Score: {row['Score']}",
            icon=folium.Icon(color='green' if row['Score'] > 4 else 'orange')
        ).add_to(m)
    st_folium(m, width=800, height=600)

def generar_estrategias():
    contexto = {
        'total': len(st.session_state.prospectos),
        'top_sectores': st.session_state.prospectos['Sector'].value_counts().head(3).to_dict(),
        'score_promedio': st.session_state.prospectos['Score'].mean()
    }
    
    try:
        client = OpenAI(api_key=st.session_state.api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"Genera 3 estrategias comerciales para telecomunicaciones usando: {json.dumps(contexto)}"
            }]
        )
        st.markdown(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error IA: {str(e)}")

def main():
    mostrar_configuracion()
    
    if 'prospectos' in st.session_state:
        mostrar_resultados()

if __name__ == "__main__":
    main()
