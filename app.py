import streamlit as st
import pandas as pd
import folium
import requests
import json
from openai import OpenAI
from io import BytesIO

# Configuración de la aplicación
st.set_page_config(
    page_title="Katalis Movistar - Asistente Comercial",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Clase para manejo avanzado de DENUE
class DenueManager:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr"
    
    def buscar_empresas(self, parametros):
        try:
            url = f"{self.base_url}/{parametros['entidad']}/0/0/0/0/{parametros['sector']}/0/0/0/0/1/1000/0/{parametros['estrato']}/{self.token}"
            response = requests.get(url, timeout=30)
            return self.procesar_respuesta(response.json())
        except Exception as e:
            st.error(f"Error en conexión DENUE: {str(e)}")
            return pd.DataFrame()
    
    def procesar_respuesta(self, data):
        empresas = []
        for item in data:
            if self.validar_registro(item):
                empresas.append({
                    'Nombre': item[2],
                    'Sector': item[4],
                    'Empleados': self.mapear_empleados(item[5]),
                    'Contacto': f"{item[14]} | {item[15]}",
                    'Dirección': f"{item[7]} {item[8]}, {item[10]}",
                    'Latitud': float(item[18]),
                    'Longitud': float(item[17])
                })
        return pd.DataFrame(empresas)
    
    def mapear_empleados(self, estrato):
        mapeo = {'1':'1-5', '2':'6-10', '3':'11-30', 
                '4':'31-50', '5':'51-100', '6':'101-250', '7':'251+'}
        return mapeo.get(estrato, 'No especificado')
    
    def validar_registro(self, item):
        return all([item[14], item[18], item[17]])

# Función para generación de parámetros con IA
def generar_parametros_busqueda(descripcion, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    prompt = f"""
    Como experto en inteligencia comercial para telecomunicaciones, analiza esta descripción:
    "{descripcion}"
    
    Devuelve SOLO un JSON válido con:
    - sectores_clae: 2 códigos CLAE principales (usar solo números)
    - estrato: 2 códigos de rango empleados (1-7)
    - ubicaciones: 2 estados de México
    - palabras_clave: 3 términos de búsqueda
    
    Estructura requerida:
    {{
        "sectores_clae": ["51", "72"],
        "estrato": ["5", "6"],
        "ubicaciones": ["Jalisco", "Nuevo León"],
        "palabras_clave": ["tecnología", "soluciones móviles", "empresarial"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.3,
            max_tokens=500
        )
        
        # Limpieza y validación de JSON
        respuesta = response.choices[0].message.content
        respuesta_limpia = respuesta.replace("```json", "").replace("```", "").strip()
        
        return json.loads(respuesta_limpia)
        
    except json.JSONDecodeError:
        st.error("Error: Respuesta en formato incorrecto. Intenta reformular tu búsqueda.")
        return None
    except Exception as e:
        st.error(f"Error de conexión con IA: {str(e)}")
        return None

# Interfaz principal
def main():
    st.title("🕵️ Asistente Inteligente de Prospectación")
    
    # Sidebar para credenciales
    with st.sidebar:
        st.header("🔐 Configuración de Acceso")
        denue_token = st.text_input("Token DENUE", type="password")
        deepseek_key = st.text_input("DeepSeek API Key", type="password")
    
    # Entrada de descripción
    descripcion = st.text_area(
        "Describe tu cliente ideal:", 
        "Ej: Empresas de tecnología en Guadalajara con 50-200 empleados que necesiten planes corporativos de telefonía móvil...",
        height=150
    )
    
    if st.button("🚀 Iniciar Búsqueda Inteligente"):
        if not denue_token or not deepseek_key:
            st.error("Por favor ingresa ambas credenciales")
            return
            
        with st.status("Procesando...", expanded=True):
            # Generación de parámetros con IA
            st.write("🧠 Analizando descripción con IA...")
            parametros = generar_parametros_busqueda(descripcion, deepseek_key)
            
            if not parametros:
                return
                
            # Búsqueda en DENUE
            st.write("🔍 Consultando base de datos DENUE...")
            denue = DenueManager(denue_token)
            df = denue.buscar_empresas({
                'entidad': '14',  # Código para Jalisco
                'sector': ','.join(parametros['sectores_clae']),
                'estrato': ','.join(parametros['estrato'])
            })
            
            if not df.empty:
                # Mostrar resultados
                st.write("📊 Generando visualizaciones...")
                
                # Mapa interactivo
                with st.expander("🗺 Mapa de Prospectos", expanded=True):
                    mapa = folium.Map(location=[20.6597, -103.3496], zoom_start=13)
                    for _, row in df.iterrows():
                        folium.Marker(
                            location=[row['Latitud'], row['Longitud']],
                            popup=f"<b>{row['Nombre']}</b><br>{row['Sector']}",
                            icon=folium.Icon(color='blue', icon='building')
                        ).add_to(mapa)
                    st_folium(mapa, width=1200, height=500)
                
                # Datos en tabla
                with st.expander("📋 Detalles de Prospectos", expanded=True):
                    st.dataframe(df[['Nombre', 'Sector', 'Empleados', 'Contacto', 'Dirección']])
                
                # Generar estrategias
                st.write("💡 Creando recomendaciones...")
                client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{
                        "role": "user",
                        "content": f"Genera 3 estrategias comerciales para estos prospectos: {parametros}"
                    }]
                )
                
                with st.expander("📈 Estrategias Recomendadas", expanded=True):
                    st.markdown(response.choices[0].message.content)
                
                # Exportación de datos
                st.download_button(
                    "⬇️ Descargar Prospectos",
                    df.to_csv(index=False).encode('utf-8'),
                    "prospectos_movistar.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.warning("No se encontraron empresas con estos parámetros")

if __name__ == "__main__":
    main()
