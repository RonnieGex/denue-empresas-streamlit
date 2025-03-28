import streamlit as st
import pandas as pd
import folium
import requests
import json
import re
import time
from openai import OpenAI
from typing import Dict, Optional, List
from streamlit_folium import st_folium
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
import warnings

# ------------------ Configuración ------------------
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
st.set_page_config(
    page_title="Katalis Movistar - Prospectador B2B",
    layout="wide",
    page_icon="📡"
)

# ------------------ Constantes ------------------
DENUE_BASE_URL = "https://www.inegi.org.mx"
DENUE_API_PATH = "/app/api/denue/v1/consulta/BuscarAreaActEstr/{entidad}/0/0/0/0/{sector}/0/0/0/0/1/1000/0/{estrato}/{token}"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEFAULT_DESCRIPTION = "Empresas de tecnología en CDMX y EdoMex con +50 empleados"
TIMEOUT = 45  # Aumentado a 45 segundos
MAX_RETRIES = 5  # Más intentos para conexiones inestables
RETRY_DELAY = 15  # Mayor tiempo entre reintentos
BACKOFF_FACTOR = 2  # Factor de retroceso exponencial

# ------------------ Configuración avanzada de requests ------------------
def create_denue_session():
    """Crea una sesión requests con política de reintentos robusta para DENUE"""
    session = requests.Session()
    
    # Política de reintentos personalizada
    retry_strategy = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True
    )
    
    # Configuración del adaptador
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10,
        pool_block=True
    )
    
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Configuración de timeout a nivel de socket
    socket.setdefaulttimeout(TIMEOUT)
    
    return session

# ------------------ Función IA Optimizada ------------------
def interpretar_prompt_con_ia(descripcion: str, api_key: str) -> Optional[Dict]:
    """Transforma una descripción textual en parámetros estructurados para DENUE usando IA."""
    if not api_key or len(api_key) < 20:
        st.error("❌ API Key inválida o demasiado corta")
        return None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_API_URL,
            timeout=TIMEOUT
        )

        prompt = f"""
        ACTÚA COMO UN EXPERTO EN CLASIFICACIÓN DENUE. 
        Analiza esta descripción y devuelve EXCLUSIVAMENTE un JSON VÁLIDO SIN texto adicional:

        "{descripcion}"

        FORMATO REQUERIDO (usa códigos CLAE reales):
        {{
            "sectores_clae": ["code1", "code2"],
            "estrato": ["value1", "value2"],
            "ubicaciones": ["code1", "code2"],
            "palabras_clave": ["keyword1", "keyword2"]
        }}
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            timeout=TIMEOUT,
            response_format={"type": "json_object"}
        )
        
        if not response.choices:
            st.error("No se recibió respuesta de la API")
            return None
            
        raw_response = response.choices[0].message.content.strip()
        json_str = re.sub(r'^```json|```$', '', raw_response, flags=re.IGNORECASE).strip()
        return json.loads(json_str)
            
    except Exception as e:
        st.error(f"❌ Error en IA: {str(e)}")
        return None

# ------------------ Consulta DENUE con Protección Completa ------------------
def consultar_denue(entidad: str, sector: str, estrato: str, token: str) -> pd.DataFrame:
    """Consulta la API DENUE con manejo ultra-robusto de errores."""
    session = create_denue_session()
    url = f"{DENUE_BASE_URL}{DENUE_API_PATH.format(entidad=entidad, sector=sector, estrato=estrato, token=token)}"
    
    try:
        # Intento de conexión con manejo de errores a bajo nivel
        try:
            response = session.get(
                url,
                timeout=TIMEOUT,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                    'Accept': 'application/json',
                    'Connection': 'keep-alive',
                    'Accept-Encoding': 'gzip, deflate'
                },
                verify=True  # Verificación SSL estricta
            )
        except requests.exceptions.SSLError:
            # Fallback a verificación SSL menos estricta
            response = session.get(
                url,
                timeout=TIMEOUT,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                    'Accept': 'application/json',
                    'Connection': 'keep-alive'
                },
                verify=False
            )
        
        # Verificación adicional del estado
        if response.status_code == 200:
            try:
                data = response.json()
                if not isinstance(data, list):
                    st.warning("La API devolvió un formato inesperado")
                    return pd.DataFrame()
                
                empresas = []
                for e in data:
                    try:
                        empresas.append({
                            "Nombre": e[2] if len(e) > 2 else "N/A",
                            "Sector": e[4] if len(e) > 4 else "N/A",
                            "Estrato": e[5] if len(e) > 5 else "N/A",
                            "Teléfono": e[14] if len(e) > 14 else "N/A",
                            "Email": e[15] if len(e) > 15 else "N/A",
                            "Dirección": f"{e[8] if len(e) > 8 else ''} {e[9] if len(e) > 9 else ''}, {e[11] if len(e) > 11 else ''}, {e[13] if len(e) > 13 else ''}",
                            "Latitud": float(e[19]) if len(e) > 19 and e[19] else 0.0,
                            "Longitud": float(e[18]) if len(e) > 18 and e[18] else 0.0
                        })
                    except (IndexError, ValueError, TypeError) as e:
                        continue
                
                return pd.DataFrame(empresas).replace("", "N/A")
            
            except ValueError as e:
                st.error(f"Error decodificando JSON: {str(e)}")
                return pd.DataFrame()
        
        elif response.status_code == 404:
            st.warning("El servidor no encontró el recurso solicitado")
            return pd.DataFrame()
        elif response.status_code == 403:
            st.error("Acceso denegado - Token inválido o sin permisos")
            return pd.DataFrame()
        else:
            st.warning(f"El servidor respondió con código: {response.status_code}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error de conexión persistente con DENUE: {str(e)}")
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

# ------------------ Visualización ------------------
def mapa_empresas(df: pd.DataFrame) -> None:
    """Muestra un mapa interactivo con las empresas encontradas."""
    if df.empty:
        st.warning("No hay datos para mostrar en el mapa")
        return
    
    try:
        df = df[(df['Latitud'] != 0.0) & (df['Longitud'] != 0.0)].copy()
        if df.empty:
            st.warning("No hay ubicaciones válidas para mostrar")
            return
        
        avg_lat = df['Latitud'].mean()
        avg_lon = df['Longitud'].mean()
        
        mapa = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=11,
            tiles='cartodbpositron',
            control_scale=True
        )
        
        marker_cluster = folium.plugins.MarkerCluster().add_to(mapa)
        
        for _, row in df.iterrows():
            folium.Marker(
                location=[row['Latitud'], row['Longitud']],
                popup=folium.Popup(
                    f"<b>{row['Nombre']}</b><br>"
                    f"Sector: {row['Sector']}<br>"
                    f"Tel: {row['Teléfono']}<br>"
                    f"Email: {row['Email']}<br>"
                    f"Dirección: {row['Dirección']}",
                    max_width=300
                ),
                icon=folium.Icon(color='blue', icon='building', prefix='fa')
            ).add_to(marker_cluster)
        
        st_folium(mapa, width=1200, height=600, returned_objects=[])
    except Exception as e:
        st.error(f"❌ Error generando mapa: {str(e)}")

# ------------------ Interfaz Principal ------------------
def main():
    st.title("🧠 Prospectador Inteligente B2B")
    st.markdown("**Encuentra clientes ideales usando IA + datos DENUE**")
    
    with st.sidebar:
        st.header("🔑 Credenciales API")
        denue_token = st.text_input("Token DENUE", type="password")
        deepseek_key = st.text_input("API Key DeepSeek", type="password")
        
        st.markdown("---")
        st.header("⚙️ Configuración Avanzada")
        st.caption("Ajustes para conexiones inestables")
    
    descripcion = st.text_area(
        "**Describe tu cliente ideal:**",
        DEFAULT_DESCRIPTION,
        height=150
    )
    
    if st.button("🔍 Buscar Prospectos", type="primary"):
        if not denue_token or not deepseek_key:
            st.warning("⚠️ Ingresa ambas credenciales API")
            return

        with st.spinner("🔍 Analizando con IA..."):
            parametros = interpretar_prompt_con_ia(descripcion, deepseek_key)

        if not parametros:
            st.error("No se pudieron generar parámetros válidos")
            return
            
        st.success("✅ Parámetros generados")
        
        with st.expander("📋 Ver parámetros", expanded=False):
            st.json(parametros)
        
        # Proceso de búsqueda
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_df = pd.DataFrame()
        error_count = 0
        
        ubicaciones = parametros.get('ubicaciones', [])
        sectores = parametros.get('sectores_clae', [])
        estratos = parametros.get('estrato', [])
        
        total_combinaciones = len(ubicaciones) * len(sectores) * len(estratos)
        
        try:
            for i, estado in enumerate(ubicaciones):
                for j, sector in enumerate(sectores):
                    for k, estrato in enumerate(estratos):
                        combinacion_actual = i * len(sectores) * len(estratos) + j * len(estratos) + k + 1
                        status_text.text(f"🔍 Buscando: Estado {estado}, Sector {sector}, Estrato {estrato} ({combinacion_actual}/{total_combinaciones})")
                        
                        parcial = consultar_denue(estado, sector, estrato, denue_token)
                        if not parcial.empty:
                            total_df = pd.concat([total_df, parcial], ignore_index=True)
                        else:
                            error_count += 1
                            time.sleep(1)  # Pequeña pausa entre consultas fallidas
                            
                        progress_bar.progress(combinacion_actual / total_combinaciones)
            
            if not total_df.empty:
                st.success(f"✅ Encontradas {len(total_df)} empresas (Errores: {error_count})")
                
                # Mostrar resultados
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### 🗺 Mapa de Prospectos")
                    mapa_empresas(total_df)
                
                with col2:
                    st.markdown("### 📊 Resumen")
                    st.metric("Total Empresas", len(total_df))
                    st.metric("Sectores Únicos", total_df['Sector'].nunique())
                    st.metric("Ubicaciones Únicas", total_df['Dirección'].nunique())
                
                # Datos detallados
                st.markdown("### 📋 Detalle Completo")
                st.dataframe(
                    total_df,
                    column_config={
                        "Teléfono": st.column_config.TextColumn(width="medium"),
                        "Email": st.column_config.TextColumn(width="medium"),
                        "Dirección": st.column_config.TextColumn(width="large")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Exportar datos
                csv = total_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📤 Descargar CSV",
                    data=csv,
                    file_name="prospectos_movistar.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.warning(f"No se encontraron empresas con estos criterios (Errores: {error_count})")
                
        except Exception as e:
            st.error(f"Error en la búsqueda: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()
