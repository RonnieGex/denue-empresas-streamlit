import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import openai
import anthropic
import plotly.express as px

# Función para obtener recomendaciones desde distintos proveedores de IA
def get_recommendations_from_ai(api_key, provider, prompt):
    if provider == "OpenAI (GPT-4)":
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    elif provider == "Anthropic (Claude)":
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content
    elif provider == "DeepSeek":
        openai.api_key = api_key  # DeepSeek utiliza la misma interfaz que OpenAI
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de mercado."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    else:
        return "Proveedor no válido"
