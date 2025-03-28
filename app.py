import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from openai import OpenAI
import json
import re
import io
import chardet

# Configuración de la app
st.set_page_config(
    page_title="Katalis Movistar - Prospectador B2B",
    layout="wide",
    page_icon="📡"
)

# ------------------ Carga de Datos desde CSV con detección de codificación ------------------
def cargar_base_datos(uploaded_file):
    """Carga la base de datos desde un archivo CSV con detección automática de codificación"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Leer los primeros bytes para detectar la codificación
            rawdata = uploaded_file.read(10000)
            uploaded_file.seek(0)  # Volver al inicio del archivo
            
            # Detectar la codificación
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            
            # Intentar cargar con la codificación detectada
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
            except:
                # Fallback a otras codificaciones comunes
                for enc in ['latin1', 'ISO-8859-1', 'utf-16', 'windows-1252']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=enc)
                        break
                    except:
                        continue
            
            # Verificar columnas mínimas requeridas
            columnas_requeridas = {'Nombre', 'Sector', 'Estrato', 'Teléfono', 
                                  'Email', 'Dirección', 'Latitud', 'Longitud'}
            
            if not columnas_requeridas.issubset(df.columns):
                st.error(f"El archivo CSV debe contener estas columnas: {columnas_requeridas}")
                return None
            
            return df
        else:
            st.error("Por favor sube un archivo CSV válido")
            return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# ------------------ IA: Traductor de descripción a parámetros ------------------
def interpretar_prompt_con_ia(descripcion, api_key):
    """Transforma descripción en filtros usando IA"""
    if not api_key:
        st.error("API Key no proporcionada")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        prompt = f"""
        Eres un experto en análisis de datos comerciales. A partir de esta descripción:
        "{descripcion}"
        
        Devuelve un JSON con parámetros para filtrar un dataset de empresas. El formato debe ser:
        {{
            "sectores": ["sector1", "sector2"],
            "estratos": ["estrato1", "estrato2"], 
            "palabras_clave": ["palabra1", "palabra2"]
        }}
        
        Los sectores deben coincidir exactamente con los que existen en los datos.
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        
        texto = response.choices[0].message.content.strip()
        # Limpieza del JSON
        texto_limpio = re.sub(r'```json|```', '', texto).strip()
        return json.loads(texto_limpio)
        
    except Exception as e:
        st.error(f"Error con IA: {str(e)}")
        return None

# ------------------ Filtrar DataFrame ------------------
def filtrar_empresas(df, parametros):
    """Filtra el DataFrame según los parámetros generados"""
    try:
        # Filtro por sectores
        if 'sectores' in parametros and parametros['sectores']:
            df = df[df['Sector'].isin(parametros['sectores'])]
        
        # Filtro por estratos
        if 'estratos' in parametros and parametros['estratos']:
            df = df[df['Estrato'].isin(parametros['estratos'])]
        
        # Filtro por palabras clave (en nombre o dirección)
        if 'palabras_clave' in parametros and parametros['palabras_clave']:
            keywords = '|'.join(parametros['palabras_clave'])
            mask = df['Nombre'].str.contains(keywords, case=False, na=False) | \
                   df['Dirección'].str.contains(keywords, case=False, na=False)
            df = df[mask]
            
        return df
    except Exception as e:
        st.error(f"Error filtrando datos: {str(e)}")
        return pd.DataFrame()

# ------------------ Mapa Interactivo ------------------
def mostrar_mapa(df):
    """Muestra las empresas en un mapa interactivo"""
    if df.empty:
        st.warning("No hay datos para mostrar en el mapa")
        return
    
    # Crear mapa centrado en los datos
    mapa = folium.Map(
        location=[df['Latitud'].mean(), df['Longitud'].mean()],
        zoom_start=12,
        tiles='cartodbpositron'
    )
    
    # Agrupar marcadores para mejor visualización
    marker_cluster = folium.plugins.MarkerCluster().add_to(mapa)
    
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"<b>{row['Nombre']}</b><br>{row['Dirección']}<br>Tel: {row['Teléfono']}",
            icon=folium.Icon(color='blue', icon='building', prefix='fa')
        ).add_to(marker_cluster)
    
    st_folium(mapa, width=1200, height=600)

# ------------------ Interfaz Principal ------------------
def main():
    st.title("🧠 Prospectador Inteligente B2B")
    st.markdown("Carga tu base de datos de empresas y encuentra prospectos ideales con IA")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        deepseek_key = st.text_input("API Key de DeepSeek", type="password")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        
        # Mostrar instrucciones para el CSV
        with st.expander("ℹ️ Instrucciones para el CSV"):
            st.markdown("""
            El archivo CSV debe contener estas columnas:
            - **Nombre**: Nombre de la empresa
            - **Sector**: Sector/industria de la empresa
            - **Estrato**: Rango de empleados (ej. '51-100')
            - **Teléfono**: Número de contacto
            - **Email**: Correo electrónico
            - **Dirección**: Dirección completa
            - **Latitud**: Coordenada geográfica
            - **Longitud**: Coordenada geográfica
            
            Codificación recomendada: UTF-8
            """)
    
    # Cargar datos
    df = None
    if uploaded_file:
        with st.spinner("Cargando y detectando codificación del archivo..."):
            df = cargar_base_datos(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Archivo cargado correctamente con {len(df)} registros")
            
            # Mostrar vista previa
            with st.expander("🔍 Vista previa de los datos"):
                st.dataframe(df.head(3))
    
    # Entrada de descripción
    descripcion = st.text_area(
        "Describe tu cliente ideal:", 
        "Empresas de tecnología en CDMX con más de 50 empleados"
    )
    
    if st.button("🔍 Buscar Prospectos") and df is not None:
        if not deepseek_key:
            st.warning("Ingresa tu API Key de DeepSeek")
            return
            
        with st.spinner("Analizando descripción con IA..."):
            parametros = interpretar_prompt_con_ia(descripcion, deepseek_key)
        
        if parametros:
            st.success("✅ Parámetros generados:")
            st.json(parametros)
            
            with st.spinner("Filtrando empresas..."):
                df_filtrado = filtrar_empresas(df, parametros)
            
            if not df_filtrado.empty:
                st.success(f"✅ {len(df_filtrado)} empresas encontradas")
                
                # Mostrar resultados
                st.markdown("### 🗺 Mapa de Prospectos")
                mostrar_mapa(df_filtrado)
                
                st.markdown("### 📋 Lista de Empresas")
                st.dataframe(df_filtrado)
                
                # Botón de descarga
                csv = df_filtrado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar Resultados",
                    data=csv,
                    file_name="prospectos_filtrados.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No se encontraron empresas con esos criterios")
        else:
            st.error("No se pudieron generar parámetros de búsqueda")

if __name__ == "__main__":
    main()
