import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from openai import OpenAI
import json
import re
import chardet
from io import StringIO

# Configuración de la app
st.set_page_config(
    page_title="Prospectador Universal B2B",
    layout="wide",
    page_icon="🔍"
)

# ------------------ Carga Inteligente de CSV ------------------
def cargar_adaptar_csv(uploaded_file):
    """Carga y adapta automáticamente cualquier estructura de CSV"""
    try:
        # Detección de codificación
        rawdata = uploaded_file.read(100000)
        uploaded_file.seek(0)
        encoding = chardet.detect(rawdata)['encoding']
        
        # Leer CSV
        df = pd.read_csv(uploaded_file, encoding=encoding)
        
        # Limpieza básica de columnas
        df.columns = df.columns.str.strip().str.lower()
        
        return df
    except Exception as e:
        st.error(f"Error al procesar archivo: {str(e)}")
        return None

# ------------------ Análisis Automático de Columnas ------------------
def analizar_columnas(df, api_key):
    """Usa IA para identificar columnas relevantes"""
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        
        # Muestra de datos para el análisis
        muestra = df.head(3).to_dict(orient='records')
        
        prompt = f"""
        Analiza esta estructura de datos y devuelve un JSON que identifique:
        - nombre_empresa (columna con nombres de empresas)
        - sector (columna con sector/industria)
        - empleados (columna con número/rango de empleados)
        - telefono (columna con teléfonos)
        - email (columna con emails)
        - direccion (columna con direcciones)
        - latitud (columna con coordenadas de latitud)
        - longitud (columna con coordenadas de longitud)
        
        Ejemplo de datos: {muestra}
        
        Devuelve SOLO un JSON con los nombres exactos de las columnas identificadas.
        Si no existe una columna adecuada, usa null.
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        texto = response.choices[0].message.content.strip()
        texto_limpio = re.sub(r'```json|```', '', texto).strip()
        return json.loads(texto_limpio)
        
    except Exception as e:
        st.error(f"Error en análisis de columnas: {str(e)}")
        return None

# ------------------ Sistema de Filtrado Universal ------------------
def filtrar_universal(df, config, parametros):
    """Filtra los datos según la configuración de columnas y parámetros"""
    try:
        # Mapeo de columnas
        col_nombre = config.get('nombre_empresa')
        col_sector = config.get('sector')
        col_empleados = config.get('empleados')
        col_direccion = config.get('direccion')
        
        # Filtro por sector
        if col_sector and 'sectores' in parametros and parametros['sectores']:
            df = df[df[col_sector].str.lower().isin([s.lower() for s in parametros['sectores']])]
        
        # Filtro por empleados
        if col_empleados and 'estratos' in parametros and parametros['estratos']:
            df = df[df[col_empleados].str.lower().isin([e.lower() for e in parametros['estratos']])]
        
        # Filtro por palabras clave
        if 'palabras_clave' in parametros and parametros['palabras_clave']:
            keywords = '|'.join(parametros['palabras_clave'])
            mask = pd.Series(False, index=df.index)
            
            if col_nombre:
                mask = mask | df[col_nombre].str.contains(keywords, case=False, na=False)
            if col_direccion:
                mask = mask | df[col_direccion].str.contains(keywords, case=False, na=False)
            if col_sector:
                mask = mask | df[col_sector].str.contains(keywords, case=False, na=False)
                
            df = df[mask]
            
        return df
    except Exception as e:
        st.error(f"Error en filtrado: {str(e)}")
        return pd.DataFrame()

# ------------------ Generación de Mapa Inteligente ------------------
def generar_mapa(df, config):
    """Genera mapa interactivo según las columnas disponibles"""
    try:
        col_lat = config.get('latitud')
        col_lon = config.get('longitud')
        col_nombre = config.get('nombre_empresa')
        col_direccion = config.get('direccion')
        col_telefono = config.get('telefono')
        
        # Verificar si tenemos coordenadas
        if not col_lat or not col_lon or col_lat not in df.columns or col_lon not in df.columns:
            st.warning("No se encontraron coordenadas para generar el mapa")
            return
        
        # Convertir a numérico
        df[col_lat] = pd.to_numeric(df[col_lat], errors='coerce')
        df[col_lon] = pd.to_numeric(df[col_lon], errors='coerce')
        df = df.dropna(subset=[col_lat, col_lon])
        
        if df.empty:
            st.warning("No hay coordenadas válidas para mostrar")
            return
        
        # Crear mapa
        mapa = folium.Map(
            location=[df[col_lat].mean(), df[col_lon].mean()],
            zoom_start=12,
            tiles='cartodbpositron'
        )
        
        # Agregar marcadores
        for _, row in df.iterrows():
            popup_content = []
            if col_nombre:
                popup_content.append(f"<b>{row[col_nombre]}</b>")
            if col_direccion:
                popup_content.append(f"{row[col_direccion]}")
            if col_telefono:
                popup_content.append(f"Tel: {row[col_telefono]}")
                
            folium.Marker(
                location=[row[col_lat], row[col_lon]],
                popup="<br>".join(popup_content),
                icon=folium.Icon(color='blue', icon='building', prefix='fa')
            ).add_to(mapa)
        
        st_folium(mapa, width=1200, height=600)
    except Exception as e:
        st.error(f"Error generando mapa: {str(e)}")

# ------------------ Interfaz Principal ------------------
def main():
    st.title("🌐 Prospectador Universal B2B")
    st.markdown("Carga cualquier CSV de empresas y encuentra prospectos ideales con IA")
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 Configuración")
        api_key = st.text_input("API Key de DeepSeek", type="password")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        
        if uploaded_file:
            st.success("Archivo cargado correctamente")
    
    # Carga y análisis inicial
    df = None
    config_columnas = None
    
    if uploaded_file and api_key:
        with st.spinner("Analizando estructura del archivo..."):
            df = cargar_adaptar_csv(uploaded_file)
            
            if df is not None:
                config_columnas = analizar_columnas(df, api_key)
                
                if config_columnas:
                    st.success("Estructura del archivo identificada")
                    with st.expander("🔍 Configuración detectada"):
                        st.json(config_columnas)
                else:
                    st.error("No se pudo analizar la estructura del archivo")
    
    # Entrada de búsqueda
    if df is not None and config_columnas:
        descripcion = st.text_area(
            "Describe tu cliente ideal:",
            "Empresas de tecnología con más de 50 empleados"
        )
        
        if st.button("🔍 Buscar Prospectos"):
            with st.spinner("Generando parámetros de búsqueda..."):
                parametros = interpretar_prompt_con_ia(descripcion, api_key)
            
            if parametros:
                st.success("Parámetros generados:")
                st.json(parametros)
                
                with st.spinner("Filtrando datos..."):
                    df_filtrado = filtrar_universal(df, config_columnas, parametros)
                
                if not df_filtrado.empty:
                    st.success(f"✅ {len(df_filtrado)} empresas encontradas")
                    
                    # Mostrar resultados
                    st.markdown("### 🗺 Mapa de Prospectos")
                    generar_mapa(df_filtrado, config_columnas)
                    
                    st.markdown("### 📋 Resultados")
                    
                    # Crear vista optimizada
                    columnas_mostrar = []
                    if config_columnas.get('nombre_empresa'):
                        columnas_mostrar.append(config_columnas['nombre_empresa'])
                    if config_columnas.get('sector'):
                        columnas_mostrar.append(config_columnas['sector'])
                    if config_columnas.get('empleados'):
                        columnas_mostrar.append(config_columnas['empleados'])
                    if config_columnas.get('telefono'):
                        columnas_mostrar.append(config_columnas['telefono'])
                    if config_columnas.get('email'):
                        columnas_mostrar.append(config_columnas['email'])
                    if config_columnas.get('direccion'):
                        columnas_mostrar.append(config_columnas['direccion'])
                    
                    st.dataframe(df_filtrado[columnas_mostrar])
                    
                    # Exportación adaptada
                    st.markdown("### 📤 Exportar Resultados")
                    
                    # Opción 1: Exportar tal cual
                    csv_original = df_filtrado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar datos originales",
                        data=csv_original,
                        file_name="prospectos_original.csv",
                        mime="text/csv"
                    )
                    
                    # Opción 2: Exportar estandarizado
                    if st.button("Generar versión estandarizada"):
                        df_export = pd.DataFrame()
                        
                        if config_columnas.get('nombre_empresa'):
                            df_export['Empresa'] = df_filtrado[config_columnas['nombre_empresa']]
                        if config_columnas.get('sector'):
                            df_export['Sector'] = df_filtrado[config_columnas['sector']]
                        if config_columnas.get('empleados'):
                            df_export['Empleados'] = df_filtrado[config_columnas['empleados']]
                        if config_columnas.get('telefono'):
                            df_export['Teléfono'] = df_filtrado[config_columnas['telefono']]
                        if config_columnas.get('email'):
                            df_export['Email'] = df_filtrado[config_columnas['email']]
                        if config_columnas.get('direccion'):
                            df_export['Dirección'] = df_filtrado[config_columnas['direccion']]
                        if config_columnas.get('latitud') and config_columnas.get('longitud'):
                            df_export['Latitud'] = df_filtrado[config_columnas['latitud']]
                            df_export['Longitud'] = df_filtrado[config_columnas['longitud']]
                        
                        csv_estandar = df_export.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar datos estandarizados",
                            data=csv_estandar,
                            file_name="prospectos_estandarizados.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("No se encontraron empresas con esos criterios")

if __name__ == "__main__":
    main()
