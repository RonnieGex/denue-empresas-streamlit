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

# ------------------ Configuración ------------------
st.set_page_config(
    page_title="Katalis Movistar - Prospectador B2B",
    layout="wide",
    page_icon="📡"
)

# ------------------ Constantes ------------------
DENUE_API_URL = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr/{entidad}/0/0/0/0/{sector}/0/0/0/0/1/1000/0/{estrato}/{token}"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEFAULT_DESCRIPTION = "Empresas de tecnología en CDMX y EdoMex con +50 empleados"
TIMEOUT = 40  # Increased timeout
MAX_RETRIES = 5  # Maximum retry attempts
RETRY_DELAY = 10  # Seconds between retries
BACKOFF_FACTOR = 1.5  # Exponential backoff factor

# ------------------ Session Configuration ------------------
def create_retry_session():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ------------------ IA Function ------------------
def interpretar_prompt_con_ia(descripcion: str, api_key: str) -> Optional[Dict]:
    """Transform description into DENUE parameters using AI"""
    if not api_key or len(api_key) < 20:
        st.error("❌ Invalid or too short API Key")
        return None

    try:
        # Initialize OpenAI client with compatibility for different versions
        client_params = {
            'api_key': api_key,
            'base_url': DEEPSEEK_API_URL,
            'timeout': TIMEOUT
        }
        
        # Remove proxies parameter if it exists to prevent TypeError
        client = OpenAI(**client_params)

        prompt = f"""
        ACT AS A DENUE CLASSIFICATION EXPERT. 
        Analyze this description and return ONLY VALID JSON WITHOUT additional text:

        "{descripcion}"

        REQUIRED FORMAT (use real CLAE codes):
        {{
            "sectores_clae": ["code1", "code2"],
            "estrato": ["value1", "value2"],
            "ubicaciones": ["code1", "code2"],
            "palabras_clave": ["keyword1", "keyword2"]
        }}
        """
        
        # For OpenAI client version compatibility
        create_params = {
            'model': "deepseek-chat",
            'messages': [{"role": "user", "content": prompt}],
            'temperature': 0.1,
            'max_tokens': 500,
            'timeout': TIMEOUT
        }
        
        # Try different response formats for different versions
        try:
            create_params['response_format'] = {"type": "json_object"}
        except:
            pass

        response = client.chat.completions.create(**create_params)
        
        if not response.choices:
            st.error("No response received from API")
            return None
            
        raw_response = response.choices[0].message.content.strip()
        
        # Robust JSON extraction
        try:
            json_match = re.search(r'```json\s*({.*?})\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
            
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            
            parsed = json.loads(json_str)
            
            # Validate structure
            required_keys = ["sectores_clae", "estrato", "ubicaciones"]
            if not all(key in parsed for key in required_keys):
                raise ValueError("Invalid JSON structure")
                
            return parsed
            
        except (json.JSONDecodeError, AttributeError, ValueError) as je:
            st.error(f"Error processing response. Raw text:\n{raw_response[:300]}...")
            return None
            
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        return None

# ------------------ DENUE Query ------------------
def consultar_denue(entidad: str, sector: str, estrato: str, token: str) -> pd.DataFrame:
    """Query DENUE API with robust error handling"""
    session = create_retry_session()
    
    try:
        response = session.get(
            DENUE_API_URL.format(entidad=entidad, sector=sector, estrato=estrato, token=token),
            timeout=TIMEOUT,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
        )
        
        # Additional status verification
        if response.status_code == 200:
            try:
                data = response.json()
                if not isinstance(data, list):
                    st.warning("API returned unexpected format")
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
                st.error(f"JSON decoding error: {str(e)}")
                return pd.DataFrame()
        
        else:
            st.warning(f"Server responded with code: {response.status_code}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Persistent connection error: {str(e)}")
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return pd.DataFrame()

# ------------------ Visualization ------------------
def mapa_empresas(df: pd.DataFrame) -> None:
    """Display interactive map with companies"""
    if df.empty:
        st.warning("No data to display on map")
        return
    
    try:
        df = df[(df['Latitud'] != 0.0) & (df['Longitud'] != 0.0)].copy()
        if df.empty:
            st.warning("No valid locations to display")
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
        st.error(f"❌ Map generation error: {str(e)}")

# ------------------ Main Interface ------------------
def main():
    st.title("🧠 Intelligent B2B Prospector")
    st.markdown("**Find ideal clients using AI + DENUE data**")
    
    with st.sidebar:
        st.header("🔑 API Credentials")
        denue_token = st.text_input("DENUE Token", type="password")
        deepseek_key = st.text_input("DeepSeek API Key", type="password")
        
        st.markdown("---")
        st.header("⚙️ Advanced Settings")
        st.caption("Settings for unstable connections")
    
    descripcion = st.text_area(
        "**Describe your ideal client:**",
        DEFAULT_DESCRIPTION,
        height=150
    )
    
    if st.button("🔍 Search Prospects", type="primary"):
        if not denue_token or not deepseek_key:
            st.warning("⚠️ Enter both API credentials")
            return

        with st.spinner("🔍 Analyzing with AI..."):
            parametros = interpretar_prompt_con_ia(descripcion, deepseek_key)

        if not parametros:
            st.error("Could not generate valid parameters")
            return
            
        st.success("✅ Parameters generated")
        
        with st.expander("📋 View parameters", expanded=False):
            st.json(parametros)
        
        # Search process
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
                        status_text.text(f"🔍 Searching: State {estado}, Sector {sector}, Stratum {estrato} ({combinacion_actual}/{total_combinaciones})")
                        
                        parcial = consultar_denue(estado, sector, estrato, denue_token)
                        if not parcial.empty:
                            total_df = pd.concat([total_df, parcial], ignore_index=True)
                        else:
                            error_count += 1
                            
                        progress_bar.progress(combinacion_actual / total_combinaciones)
            
            if not total_df.empty:
                st.success(f"✅ Found {len(total_df)} companies (Errors: {error_count})")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### 🗺 Prospects Map")
                    mapa_empresas(total_df)
                
                with col2:
                    st.markdown("### 📊 Summary")
                    st.metric("Total Companies", len(total_df))
                    st.metric("Unique Sectors", total_df['Sector'].nunique())
                    st.metric("Unique Locations", total_df['Dirección'].nunique())
                
                # Detailed data
                st.markdown("### 📋 Complete Details")
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
                
                # Export data
                csv = total_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📤 Download CSV",
                    data=csv,
                    file_name="movistar_prospects.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.warning("No companies found with these criteria")
                
        except Exception as e:
            st.error(f"Search error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()
