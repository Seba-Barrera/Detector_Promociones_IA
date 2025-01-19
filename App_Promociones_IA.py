#########################################################################
# App de BenchMark de paginas con WebScrap + IA
#########################################################################


# https://platform.openai.com/account/api-keys
# https://openai.com/pricing


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Obtener versiones de paquetes instalados
# !pip list > requirements.txt

import streamlit as st

# librerias para data
import pandas as pd
import numpy as np
from IPython.display import display


# librerias para graficos
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# librerias de IA
from openai import OpenAI
from pydantic import BaseModel

# librerias para web scraping
from bs4 import BeautifulSoup
import requests



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#=======================================================================
# [B.1] Funcion de extraer texto de una pagina web
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def extraer_texto_url(url):

  try:
    # Obtener el contenido del sitio web
    respuesta = requests.get(url)
    respuesta.raise_for_status()
    soup = BeautifulSoup(respuesta.text, 'html.parser')
    
    # Eliminar elementos irrelevantes como scripts, estilos y botones
    for tag in soup(['script', 'style', 'button', 'nav', 'footer', 'form', 'aside']):
      tag.decompose()
    
    # Extraer encabezados y párrafos en orden
    texto_relevante = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
      # Obtener texto limpio y agregar salto de línea si es un encabezado
      contenido = tag.get_text(strip=True)
      if contenido:
        if tag.name.startswith('h'):
          texto_relevante.append(f"\n{contenido}\n")
        else:
          texto_relevante.append(contenido)
    
    # Combinar todo el texto en un solo string
    return '\n'.join(texto_relevante).strip()
  
  except requests.exceptions.RequestException as e:
    return ''
  

#=======================================================================
# [B.2] Funcion de extraer texto formateado usando IA
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def extraer_promociones_ia(
  url_web,
  texto_web,
  api_key_openAI
  ):
  
  # crear cliente de openAI
  cliente_OpenAI = OpenAI(api_key=api_key_openAI)


  # crear clase de formato de salida segun aspectos ingresados
  class Promociones(BaseModel):
    nombre_empresa: str
    color_empresa: str
    promociones: list[list[str]]


  # definir prompt del sistema 
  prompt_s = f'''
  Eres un experto en leer texto de contenido de sitios web de paginas de empresas e 
  identificar todas las promociones u ofertas (puede serte util por ejemplo buscar 
  palabras como "descuento" o "dcto" o el signo "%" para identificar texto 
  asociado a promociones). Dado del link de un sitio web de la empresa 
  se te pide identificar el nombre de la empresa y el color que mas la representa 
  segun su logo en formato RGB (ejemplo: "rgb(255, 165, 0)").
  Adicional a lo anterior, para cada una de las promociones que se detecten en el texto 
  debes generar una lista de 8 elementos con los siguientes aspectos:
  - Nombre de la promocion
  - Descripcion de la promocion
  - Rubro de la promocion (si es financiera, en comida, eventos, productos, etc)
  - Comercio de la promocion (si es una promocion en otra empresa o cadena, mencionarla)
  - Tipo de promocion (debe tener valor "porcentaje" si es un descuento porcentual 
  o "dinero" si es un descuento monetario)
  - Valor de la promocion (el valor del monto de la promocion o del porcentaje de descuento, 
  si no aplica ingresar un -1)
  - Dias de vigencia de la promocion (en caso de aplicar ingresar por ejemplo: 
  "lunes, miercoles, jueves", si la promocion es todos los dias, escribir expresamente todos
  los dias separados por coma como: "lunes,martes,miercoles,jueves,viernes,sabado,domingo",
  si como tambien por ejemplo si dice de lunes a miercoles, escribir: "lunes,martes,miercoles")
  - Restriccion de la promocion (si tienen limites de tiempo 
  u otros alcances de monto maximo)
'''
  '''

  # definir prompt del usuario  
  prompt_u = f'''
  El texto del contenido del sitio {url_web} es el siguiente: {texto_web}
  '''

  respuesta_ia = cliente_OpenAI.beta.chat.completions.parse(
    model='gpt-4o-2024-08-06',
    messages=[
      {'role': 'system', 'content': prompt_s},
      {'role': 'user', 'content': prompt_u},
      ],
    response_format=Promociones
    )

  respuesta_ia2 = respuesta_ia.choices[0].message.parsed


  columnas_promo = [
    'Nombre Promo',
    'Descripcion',
    'Rubro',
    'Comercio',
    'Tipo',
    'Valor',
    'Dias de vigencia',
    'Restricciones'
  ]
  
  
  largo_listas = [
    len(respuesta_ia2.promociones[x]) for x in range(len(respuesta_ia2.promociones))
    ]


  if min(largo_listas)==8 and  max(largo_listas)==8:

    df_promociones = pd.DataFrame(
      respuesta_ia2.promociones, 
      columns=columnas_promo
      )

    df_promociones['Sitio Web']=url_web
    df_promociones['Empresa']=respuesta_ia2.nombre_empresa
    df_promociones['Color Empresa']=respuesta_ia2.color_empresa

    df_promociones = df_promociones[
      ['Sitio Web','Empresa','Color Empresa']+columnas_promo
      ]

  else:
        
    df_promociones = pd.DataFrame(
      columns=columnas_promo
      )

  return df_promociones




#=======================================================================
# [B.3] Funcion de armar df consolidado
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def extraccion_promos_ia_web(
  lista_urls,
  api_key_openAI
  ):
  
  
  df_consolidado = pd.DataFrame([])
  for link in lista_urls:
    
    print(f'procesando link: {link}')
        
    texto_link = extraer_texto_url(
      url=link
      )
    
    if texto_link!='':
      df = extraer_promociones_ia(
        url_web = link,
        texto_web= texto_link,
        api_key_openAI= api_key_openAI
        )
      
      if len(df)>0:    
        df_consolidado = pd.concat([df_consolidado,df])
      
  
  #.................................
  # estandarizar rubro de promocion
  print('Procesando Rubros')
  
  class Rubro(BaseModel):
    rubro: list[str]
    
 
  # definir prompt del sistema 
  prompt_s = f'''
  Se te facilitara una lista de rubros y debes retornar una lista con la misma 
  cantidad de elementos, pero con los rubros estandarizados u homologados, es decir,
  si un elemento dice "comida" y otro "Alimentos", debes homologarlos a una categoria
  ("Comida" por ejemplo)
  '''

  # definir prompt del usuario  
  lista_rubros = list(df_consolidado['Rubro'])
  prompt_u = f'''
  la lista de rubros es la siguiente: {lista_rubros}
  '''
  
  
  
  cliente_OpenAI = OpenAI(api_key=api_key_openAI)
  respuesta_ia = cliente_OpenAI.beta.chat.completions.parse(
    model='gpt-4o-2024-08-06',
    messages=[
      {'role': 'system', 'content': prompt_s},
      {'role': 'user', 'content': prompt_u},
      ],
    response_format=Rubro
    )

  respuesta_ia2 = respuesta_ia.choices[0].message.parsed
  
  df_consolidado['Rubro'] = respuesta_ia2.rubro
  df_consolidado['Dias de vigencia']=df_consolidado['Dias de vigencia'].apply(
    lambda x: x.\
      replace('todos los días','lunes,martes,miércoles,jueves,viernes,sábado,domingo').\
      replace('todos los dias','lunes,martes,miércoles,jueves,viernes,sábado,domingo').\
      replace('á','a').\
      replace('é','e').\
      replace('í','i').\
      replace('ó','o').\
      replace('ú','u')
  )
  
  
  
  df_consolidado = df_consolidado.reset_index()
        
  return df_consolidado



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.set_page_config(layout='wide')

# titulo inicial 
st.markdown('## :globe_with_meridians: Extracción de promociones con IA :globe_with_meridians:')

# autoria 
st.sidebar.markdown('**Autor :point_right: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')

# ingresar OpenAI api key
usuario_api_key = st.sidebar.text_input(
  label='Tu OpenAI API key :key:',
  placeholder='Pega aca tu openAI API key',
  type='password'
  )


# ingresar listado de links separados por coma
texto_links = st.sidebar.text_area(
  'Ingresa aca los links de paginas desde donde quieras\nextraer promociones separados por ","',
  )

# colocar separador
st.sidebar.divider()

# colocar boton de procesar 
boton_procesar = st.sidebar.button(
  'Analizar Links', 
  icon='😃', 
  use_container_width=True,
  type='primary'
  )


#_____________________________________________________________________________
# comenzar a desplegar app una vez ingresado el archivo

if boton_procesar and len(texto_links)>0 and len(usuario_api_key)>0:
      
  # Crear df de respuestas y almacenarlo en session_state
  if 'df_promociones_links' not in st.session_state:   
          
    st.session_state.df_promociones_links = extraccion_promos_ia_web(
      lista_urls = [x.strip() for x in texto_links.split(',')],
      api_key_openAI = usuario_api_key
      )
  
  
  # Si el DataFrame ya está en session_state, continúa
if 'df_promociones_links' in st.session_state:  
  df_promociones_links = st.session_state.df_promociones_links
  
   
  
  # Crear tres tabs
  tab1, tab2, tab3 = st.tabs([
    ':date: Tabla detalle promociones', 
    ':bar_chart: Promociones segun rubro', 
    ':calendar: Promociones por dia'
    ])
  
  
  #...........................................................................
  # Detalle de las promociones
  
  with tab1:  
  
    st.data_editor(
      df_promociones_links[[
        'Sitio Web', 
        'Empresa', 
        'Nombre Promo',
        'Descripcion', 
        'Rubro', 
        'Comercio', 
        'Tipo', 
        'Valor', 
        'Dias de vigencia',
        'Restricciones'
        ]], 
      use_container_width=True, 
      disabled=True,
      hide_index=True
      )


  #...........................................................................
  # Grafico de promociones segun rubro
  
  with tab2:
    
    promo_rubros = df_promociones_links.groupby(
      ['Empresa','Rubro']
      ).agg(
        Cantidad = pd.NamedAgg(column = 'Empresa', aggfunc = len),
        Convenios = pd.NamedAgg(column = 'Comercio', aggfunc = lambda x: list(x))
      ).reset_index() 

    fig = px.bar(
      promo_rubros,
      x='Empresa',
      y='Cantidad',
      color = 'Rubro',
      hover_data=['Convenios'],  # Campo adicional en el hover
      title='Promociones segun rubro',
      )
     
    st.plotly_chart(fig)


  #...........................................................................
  # Tabla de promociones segun dia
  
  with tab3:
  
    # Crear un cuadro de texto para capturar la búsqueda
    col2a, col2b = st.columns([1,1])
    filtro3a = col2a.multiselect(
      'Rubro:',
      list(set(df_promociones_links['Rubro'])),
      list(set(df_promociones_links['Rubro'])),
      key='k_filtro3a'
      )
    filtro3b = col2b.multiselect(
      'Empresa:',
      list(set(df_promociones_links['Empresa'])),
      list(set(df_promociones_links['Empresa'])),
      key='k_filtro3b'
      )
    
    # Trabajar df
    promo_dias = df_promociones_links[
      ['Empresa','Nombre Promo','Comercio','Rubro','Dias de vigencia']
      ].assign(
      Dias=df_promociones_links['Dias de vigencia'].str.split(',')
      ).explode('Dias').reset_index(drop=True)
    
    promo_dias['valor'] = 1

    promo_dias2 = promo_dias.pivot_table(
      index = ['Rubro','Empresa','Nombre Promo','Comercio'],
      columns = 'Dias',
      values = 'valor',
      fill_value = 0
    ).reset_index()
    
    promo_dias2 = promo_dias2[
      ['Rubro','Empresa','Nombre Promo','Comercio',
      'lunes','martes','miercoles','jueves','viernes','sabado','domingo']
    ].sort_values(by=['Rubro','Empresa'])
        

    # Aplicar filtros
    if filtro3a:
      promo_dias3 = promo_dias2[
        promo_dias2['Rubro'].isin(filtro3a)
        ]      
    else:
      promo_dias3 = promo_dias2.copy()
      
    if filtro3b:
      promo_dias3 = promo_dias3[
        promo_dias3['Empresa'].isin(filtro3b)
        ]      
    else:
      promo_dias3 = promo_dias3.copy()


    # estetizar 
    promo_dias4 = promo_dias3.style.background_gradient(
      cmap='RdYlGn',
      axis=None,
      subset=['lunes','martes','miercoles','jueves','viernes','sabado','domingo']
      ).hide(axis='index')

    st.markdown(
      promo_dias4.render(), 
      unsafe_allow_html=True
      )
  


# !streamlit run BenchMark_WebScrap_IA_V3.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/45_App_BenchMark WebScrap + IA (12-01-25)/App/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit
