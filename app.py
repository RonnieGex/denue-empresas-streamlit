import streamlit as st
import pandas as pd
import folium
import requests
import json
from openai import OpenAI
from io import BytesIO

# Configuración inicial
st.set_page_config(
    page_title="Katalis Movistar AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📞"
)

# Clase para manejo DENUE
class DenueAI:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr"
    
    def buscar(self, parametros):
        try:
            url = f"{self.base_url}/{parametros['entidad']}/0/0/0/0/{parametros['sector']}/0/0/0/0/1/1000/0/{parametros['estrato']}/{self.token}"
            response = requests.get(url, timeout=30)
            return self.procesar_respuesta(response.json())
        except Exception as e:
            st.error(f"Error DENUE: {str(e)}")
            return pd.DataFrame()
    
    def procesar_respuesta(self, data):
        return [{
            'Nombre': item[2],
            'Sector': item[4],
            'Empleados': self.mapear_empleados(item[5]),
            'Contacto': f"{item[14]} | {item[15]}",
            'Ubicación': f"{item[7]} {item[8]}, {item[10]}",
            'Latitud': float(item[18]),
            'Longitud': float(item[17])
        } for item in data if self.validar_registro(item)]

    def mapear_empleados(self, estrato):
        return {'1':'1-5','2':'6-10','3':'11-30','4':'31-50','5':'51-100','6':'101-250','7':'251+'}.get(estrato, 'N/A')
    
    def validar_registro(self, item):
        return all([item[14], item[18], item[17]])

# Función para generación de parámetros con IA
def generar_parametros_ia(descripcion, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    prompt = f"""
    Como experto en segmentación comercial para telecomunicaciones, analiza esta descripción:
    "{descripcion}"
    
    Devuelve un JSON con:
    1. sectores_clae (códigos CLAE prioritarios)
    2. estrato (rango de empleados)
    3. ubicaciones (estados de México)
    4. keywords (palabras clave para búsqueda)
    
    Ejemplo:
    {{
        "sectores_clae": ["51", "72"],
        "estrato": ["5", "6"],
        "ubicaciones": ["Jalisco", "Nuevo León"],
        "keywords": ["tecnología", "servicios cloud"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error IA: {str(e)}")
        return None

# Interfaz principal
def main():
    st.title("💬 Chat de Segmentación Comercial")
    
    # Sidebar para credenciales
    with st.sidebar:
        st.header("🔑 Credenciales")
        denue_token = st.text_input("Token DENUE", type="password")
        deepseek_key = st.text_input("DeepSeek API Key", type="password")
    
    # Chat de segmentación
    descripcion = st.text_area("Describe tu cliente ideal:", 
                             "Ej: Empresas de tecnología en Guadalajara con 50-200 empleados que necesiten soluciones móviles integradas...")
    
    if st.button("Generar Segmentación Inteligente"):
        if not denue_token or not deepseek_key:
            st.error("Por favor ingresa ambas credenciales")
            return
            
        with st.status("Analizando con IA...", expanded=True):
            # Paso 1: Generar parámetros con IA
            st.write("🧠 Generando parámetros de búsqueda...")
            parametros = generar_parametros_ia(descripcion, deepseek_key)
            
            if not parametros:
                return
                
            # Paso 2: Buscar en DENUE
            st.write("🔍 Consultando base DENUE...")
            denue = DenueAI(denue_token)
            resultados = denue.buscar({
                'entidad': '16',  # Jalisco como ejemplo
                'sector': ','.join(parametros['sectores_clae']),
                'estrato': ','.join(parametros['estrato'])
            })
            
            if not resultados.empty:
                # Paso 3: Mostrar resultados
                st.write("📊 Procesando resultados...")
                df = pd.DataFrame(resultados)
                
                # Mapa interactivo
                with st.expander("Mapa de Prospectos", expanded=True):
                    m = folium.Map(location=[20.6597, -103.3496], zoom_start=12)
                    for _, row in df.iterrows():
                        folium.Marker(
                            location=[row['Latitud'], row['Longitud']],
                            popup=f"<b>{row['Nombre']}</b><br>{row['Sector']}",
                            icon=folium.Icon(color='green')
                        ).add_to(m)
                    st_folium(m, width=1200, height=500)
                
                # Recomendaciones
                with st.expander("Estrategias Recomendadas", expanded=True):
                    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": f"Genera 3 estrategias de acercamiento para estos prospectos: {parametros}"}]
                    )
                    st.markdown(response.choices[0].message.content)
                
                # Exportación
                st.download_button(
                    "📥 Descargar Prospectos",
                    df.to_csv().encode('utf-8'),
                    "clientes_potenciales.csv",
                    "text/csv"
                )
            else:
                st.warning("No se encontraron empresas con estos parámetros")

if __name__ == "__main__":
    main()
