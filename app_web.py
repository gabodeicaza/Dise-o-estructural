"""Versión web (Streamlit) del Análisis Modal Espectral.  Ejecutar:  streamlit run app_web.py"""
import hashlib
import io
import json
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import ame_ntc as ntc
import ame_dimensionado as dim
import ame_pdf
import ame_sismico as ame
from ame_revision import COMBINACIONES
from ame_secciones import AYUDA_SECCIONES

NAVY, AZUL_OSC, AZUL, AZUL_MED, AZUL_CLARO, PALE, BG = (
    '#0B2545', '#13315C', '#1D4E89', '#3E7CB1', '#81A4CD', '#DCE9F7', '#F1F6FC')

COLS = ['h', 'ejes', 'seccion', 'E_col', 'viga', 'E_viga', 'artic', 'rod', 'art_sup', 'sin_losa', 'volado_izq', 'volado_der',
        'cargas', 'W']
COLS_COLUMNA = ['ejes', 'seccion', 'E_col', 'artic', 'rod', 'art_sup']       # se capturan en la tabla de columnas
ETIQ = {'h': 'h entrepiso (m)', 'ejes': 'Columnas en ejes', 'seccion': 'Sección de columnas (cm)',
        'E_col': 'E de las columnas', 'viga': 'Viga (cm)', 'E_viga': 'E de la viga', 'sin_losa': 'Tramos sin losa', 'artic': 'Articuladas abajo (ejes)', 'rod': 'Rodillos en la base (ejes)',
        'art_sup': 'Articuladas arriba (ejes)', 'cargas': 'Carga por crujía (t/m)',
        'W': 'o Peso W (t)'}
AYUDA = {'h': 'Altura del entrepiso en metros.',
         'ejes': 'Ejes que tienen columna en este nivel: "todos" (o vacío), "1 3" o "1-3".',
         'seccion': AYUDA_SECCIONES + '. Una sola para todo el nivel o una por columna: "25x50 Ø40 25x50". '
                    'Para girar una sección 90° (otra orientación en planta) agrega @90: "20x40@90".',
         'viga': 'Viga de este piso (solo con vigas flexibles). b x h con h vertical: 30x60. También T:50x10x60x15 o H:...',
         'artic': 'Solo para niveles superiores: ejes cuya columna está articulada en su extremo inferior. Los apoyos de la '
                  'base se eligen en "Apoyos en la base", arriba de esta tabla.',
         'E_col': 'E de las columnas de este nivel, en la unidad de E (o "acero"). Un valor para todas o uno por columna, '
                  'en el orden de los ejes: "221.36 221.36 221.36 2100". Vacío = el E general.',
         'E_viga': 'E de las vigas de este piso (solo con vigas flexibles), en la unidad de E o "acero". Vacío = el E general.',
         'sin_losa': 'Crujías (1, 2, 3… de izquierda a derecha) sin viga ni losa en este piso, por ejemplo "3". Su carga se '
                     'ignora. Si un eje queda sin viga en este piso, su columna lo cruza como una sola columna de varios entrepisos.',
         'rod': 'Ejes con apoyo de rodillos en la base (solo nivel 1): no transmite cortante, la columna no aporta '
                'rigidez lateral. Ej.: "4".',
         'art_sup': 'Ejes con la columna articulada en su parte superior (3EI/h³ si la base es empotrada).',
         'cargas': 'Un valor para todos los tramos o uno por tramo de izquierda a derecha: "4.7 3.2". Con volados, '
                   'incluye los volados (izq., crujías, der.), con 0 donde no hay losa: "2 2 4 0". Vacío si das el periodo medido.',
         'W': 'Opcional: peso total del nivel; sustituye a la carga por crujía.'}
ETIQ.update(volado_izq='Volado izq. (m)', volado_der='Volado der. (m)')
AYUDA.update(volado_izq='Volado izquierdo de este nivel (m). Vacío = el volado general de arriba. 0 = sin volado.',
             volado_der='Volado derecho de este nivel (m). Vacío = el volado general de arriba. 0 = sin volado.')
SUBGRUPOS = {'B': 'B · riesgo I–II (ASCE 7)', 'A2': 'A2 · riesgo III', 'A1': 'A1 · riesgo IV'}
VIGAS = ['Rígidas (marco de cortante)', 'Flexibles (marco plano)']
CLASES = ["Clase 1 (14000·√f'c)", "Clase 2 (8000·√f'c)"]
MODOS = ['Por secciones y cargas', 'Matrices K y M (ya calculadas)']
UNIDADES = ['kg, cm', 't, cm']
ORIENT_EI = ['Eje fuerte en la dirección del sismo (Ix)', 'Eje débil (Iy)']
CAPTURAS = ['Por nivel (una fila por nivel)', 'Por columna (una fila por columna)']
COLS_CONT = ['eje', 'desde', 'hasta', 'seccion', 'I', 'A', 'E', 'apoyo', 'art_sup']
CRITERIOS = ['Apuntes: el mayor entre el plato y T = 0', 'Máximo del espectro de diseño (como el código de MATLAB)']
MODOS_OPC = ['Todos los modos', 'Criterio de los apuntes (95 % de masa o T ≥ 0.4 s)', 'Los primeros N']
RDIST = ['Del plato (k2 = 0, como en tus hojas)', 'Del periodo fundamental (estricto)']
NORM_OPC = ['Primer nivel = 1 (apuntes)', 'Último nivel (azotea) = 1', 'Valor máximo = 1', 'Masa unitaria (φᵀMφ = 1)']
CLAVES_MANUAL = ['K_txt', 'M_txt', 'alturas_txt', 'pesos_txt']
SOLO_SECCIONES = ('crujias', 'volado_izq', 'volado_der', 'E', 'fc', 'clase', 'vigas', 'direccion', 'unidad_E', 'unidad_fc', 'unidad_seccion')  # no aplican al modo de matrices
ESPECTRO = ['a0', 'c', 'Ta', 'Tb', 'k', 'Ts']
SIMBOLO_APOYO = {'Empotrado': '▬  Empotrado', 'Articulado': '△  Articulado', 'Rodillo': '◬  Rodillo (sin cortante)'}
MAX_EJES = 20
DEFECTOS = dict(captura=CAPTURAS[0], crujias='', volado_izq='', volado_der='', subgrupo='B', intensidad='Base de diseño', direccion='X', estado='Automático', Ks='',
                Q='4', k1='1.0', combinacion='SRSS', zeta='', material='Concreto', E='', fc='', clase=CLASES[0],
                vigas=VIGAS[0], factor_Fu='1.1', mult_V='1', lim_dl='0.004', lim_sv='0.03', lim_oi='0.005',
                irreg=[], fuerte_torsion=False, fuerte_elev=False, zona='', regularidad='Regular', b_planta='',
                R1_otros=True, proyecto='', autor='', modo=MODOS[0], K_txt='', M_txt='', unidades=UNIDADES[0],
                alturas_txt='', pesos_txt='', criterio_estatico=CRITERIOS[0], unidad_espectro='cm/s²',
                unidad_E='kg/cm²', unidad_fc='kg/cm²', unidad_seccion='cm', modos=MODOS_OPC[0], modos_n='',
                R_dist=RDIST[0], Sa_modos='', e_s='', norm_modal=NORM_OPC[0], T_medido='', **{k: '' for k in ESPECTRO})
CAMPOS = list(DEFECTOS)
EJEMPLOS = ame.EJEMPLOS_MODELO

st.set_page_config(page_title='Análisis Modal Espectral', page_icon='🏢', layout='wide')

st.markdown(f"""
<style>
.cabecera {{background:{NAVY};padding:20px 28px;border-radius:12px;margin-bottom:18px}}
.cabecera h1 {{color:white;margin:0;font-size:2rem}}
.cabecera p {{color:{AZUL_CLARO};margin:4px 0 0 0}}
.kpi {{background:white;border:1px solid {AZUL_CLARO};border-top:4px solid {AZUL};border-radius:8px;padding:12px 16px}}
.kpi .t {{color:{AZUL_MED};font-size:.75rem;font-weight:700;letter-spacing:.04em}}
.kpi .n {{color:{AZUL};font-size:1.9rem;font-weight:700;line-height:1.2}}
.kpi .s {{color:{AZUL_MED};font-size:.8rem}}
.objetivo {{background:{PALE};border-left:5px solid {AZUL};border-radius:6px;padding:10px 14px;margin:6px 0 12px 0;color:{NAVY}}}
h3 {{color:{AZUL_OSC}}}
.stTabs [data-baseweb="tab-list"] button p {{font-size:1.02rem;font-weight:600}}
</style>
<div class="cabecera"><h1>Análisis Modal Espectral</h1>
<p>Marcos planos · NTC-S 2023 · fuerzas sísmicas por nivel, cortante basal, factor de escala y revisiones</p></div>
""", unsafe_allow_html=True)


def _limpio(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return bool(v) if isinstance(v, (bool, np.bool_)) else str(v)


def filas_de(df):
    """Filas de una tabla editada como lista de dicts (celdas vacías = '')."""
    return [{k: _limpio(v) for k, v in fila.items()} for _, fila in df.dropna(how='all').iterrows()]


def df_cols_de(filas):
    df = pd.DataFrame([{k: (bool(f.get(k)) if k == 'art_sup' else str(f.get(k, ''))) for k in COLS_CONT} for f in filas],
                      columns=COLS_CONT)
    return df.astype({k: (bool if k == 'art_sup' else object) for k in COLS_CONT})


def df_de(niveles):
    return pd.DataFrame([{k: str(r.get(k, '')) for k in COLS} for r in niveles], dtype=object)


def cargar(d):
    """Pone un proyecto (ejemplo o archivo) en el estado de la página."""
    d = ame.convertir_legacy(d)
    matrices = d.get('modo') == 'matrices'
    if not matrices and (not isinstance(d.get('niveles'), list) or 'crujias' not in d):
        raise ValueError('Formato de proyecto no reconocido.')
    d = dict(d)
    d['modo'] = MODOS[1] if matrices else MODOS[0]
    d['K_txt'], d['M_txt'] = d.get('K', ''), d.get('M', '')
    d['alturas_txt'], d['pesos_txt'] = d.get('alturas', ''), d.get('pesos', '')
    if d.get('unidades') not in UNIDADES:
        d['unidades'] = UNIDADES[0]
    if 'subgrupo' not in d:  # proyectos anteriores: "tipo" = Grupo B / Grupo A - sismo base / infrecuente
        grupo, sismo = ame.grupo_sismo(d.get('tipo', 'Grupo B'))
        d['subgrupo'] = 'B' if grupo == 'B' else 'A2'
        d.setdefault('intensidad', 'Infrecuente' if sismo == 'I' else 'Base de diseño')
    d['vigas'] = VIGAS[1] if str(d.get('vigas', '')).strip().lower().startswith('flex') else VIGAS[0]   # 'flexibles'/'rigidas'
    d['subgrupo'] = ntc.normalizar_subgrupo(d.get('subgrupo'))          # B1 y B2 de proyectos anteriores = B
    if d['subgrupo'] not in SUBGRUPOS:
        d['subgrupo'] = 'A2' if d['subgrupo'].startswith('A') else 'B'
    for k in CAMPOS:
        if matrices and k in SOLO_SECCIONES and k not in d:
            continue  # un ejemplo con K y M no pisa los datos de secciones que ya capturaste
        st.session_state[k] = d.get(k, DEFECTOS[k])
    st.session_state['estado'] = d.get('estado') if d.get('estado') in ame.ESTADOS else 'Automático'
    st.session_state['intensidad'] = ntc.normalizar_intensidad(st.session_state['intensidad'])
    fuente = ntc.espectro_fuente(st.session_state['intensidad'])  # frecuente y ocasional usan el de base de diseño
    esp = {k: dict(v) for k, v in (d.get('espectros') or {}).items()}
    if not esp:  # campos sueltos del formato anterior: pertenecen a la intensidad elegida
        esp = {fuente: {k: str(d.get(k, '')) for k in ESPECTRO}}
    st.session_state['_esp'] = esp
    st.session_state['_int_prev'] = st.session_state['intensidad']
    for k in ESPECTRO:
        st.session_state[k] = (esp.get(fuente) or esp.get(st.session_state['intensidad']) or {}).get(k, '')
    if isinstance(d.get('niveles'), list):
        st.session_state['df'] = df_de(d['niveles'])
        if isinstance(d.get('columnas'), list) and d['columnas']:
            filas_c = d['columnas']
        else:                                    # la tabla de columnas siempre queda lista, a partir de la captura por nivel
            try:
                filas_c = ame.columnas_desde_niveles(d)
            except (ValueError, KeyError, TypeError):
                filas_c = []
        st.session_state['df_cols'] = df_cols_de(filas_c)
        st.session_state['ver_cols'] = st.session_state.get('ver_cols', 0) + 1
        st.session_state['captura'] = CAPTURAS[1] if d.get('columnas') else CAPTURAS[0]
        st.session_state['_captura_prev'] = st.session_state['captura']
        st.session_state['_ult_niveles'] = [dict(r) for r in d['niveles']]
        st.session_state['_ult_cols'] = [dict(f) for f in filas_c]
        for i in range(1, MAX_EJES + 1):
            st.session_state[f'apoyo_{i}'] = 'Empotrado'
        try:                                     # los apoyos de la base del proyecto pasan a los selectores
            nej_ = len(ame.parse_vector(d.get('crujias', ''), 'Crujías')) + 1
            n1_ = d['niveles'][0] if d['niveles'] else {}
            for i, est in enumerate(ame.estados_apoyo(n1_.get('artic', ''), n1_.get('rod', ''), nej_), 1):
                st.session_state[f'apoyo_{i}'] = est
        except (ValueError, KeyError, TypeError):
            pass
    st.session_state['ver'] = st.session_state.get('ver', 0) + 1


def cambiar_intensidad():
    """Guarda el espectro de la intensidad anterior y muestra el de la nueva."""
    esp = st.session_state['_esp']
    esp[ntc.espectro_fuente(st.session_state['_int_prev'])] = {k: st.session_state[k] for k in ESPECTRO}
    nueva = st.session_state['intensidad']
    for k in ESPECTRO:
        st.session_state[k] = esp.get(ntc.espectro_fuente(nueva), {}).get(k, '')
    st.session_state['_int_prev'] = nueva


AUTOGUARDADO = Path(__file__).parent / '.autoguardado'      # copia de tu captura; sobrevive a reconexiones de la página


def ruta_auto(pid):
    return AUTOGUARDADO / f'{pid}.json'


if 'df' not in st.session_state:
    pid_ = str(st.query_params.get('p', ''))
    if not re.fullmatch(r'[0-9a-f]{12}', pid_):
        pid_ = uuid.uuid4().hex[:12]
    st.session_state['_pid'] = pid_
    st.query_params['p'] = pid_                  # la dirección de la página identifica tu proyecto
    recuperado_ = False
    try:
        if ruta_auto(pid_).exists():
            cargar(json.loads(ruta_auto(pid_).read_text(encoding='utf-8')))
            recuperado_ = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        recuperado_ = False
    if not recuperado_:
        cargar(next(iter(EJEMPLOS.values())))
    st.session_state['_recuperado'] = recuperado_
manual = st.session_state.get('modo') == MODOS[1]

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown('### Proyecto')
    st.text_input('Nombre del proyecto', key='proyecto')
    st.text_input('Elaboró', key='autor')
    st.selectbox('Ejemplo', list(EJEMPLOS), key='ejemplo')
    st.button('Cargar ejemplo', on_click=lambda: cargar(EJEMPLOS[st.session_state['ejemplo']]), width='stretch')
    arch = st.file_uploader('o abrir proyecto (.json)', type='json')
    if arch is not None and st.session_state.get('_arch') != arch.file_id:
        st.session_state['_arch'] = arch.file_id
        try:
            cargar(json.load(arch))
            st.rerun()
        except (ValueError, KeyError, TypeError, AttributeError):
            st.error('No se pudo leer ese archivo de proyecto.')
    if st.session_state.get('_recuperado'):
        st.success('Recuperé tu captura anterior: la página se había reiniciado.')
    st.caption('Tu captura se guarda sola mientras trabajas: si la página se reinicia, vuelve a abrir la misma dirección '
               '(termina en ?p=…) y se recupera. Para un respaldo seguro usa «Guardar proyecto (.json)» en Resultados.')
    st.divider()
    st.caption('Flujo: 1 · Estructura → 2 · Sismo y norma → 3 · Resultados. Las herramientas (calculadora de EI y '
               'dimensionado) están en la pestaña 4.')

T_est, T_sismo, T_res, T_herr = st.tabs(['1 · Estructura', '2 · Sismo y norma', '3 · Resultados', '4 · Herramientas'])

# ------------------------------------------------------------------ 1 · estructura
_capt_ahora = st.session_state['captura']
_capt_antes = st.session_state.get('_captura_prev', _capt_ahora)
if _capt_ahora != _capt_antes:       # al cambiar de modo, la otra tabla se llena con lo que ya capturaste
    try:
        if _capt_ahora == CAPTURAS[1]:
            st.session_state['df_cols'] = df_cols_de(ame.columnas_desde_niveles(
                dict(crujias=st.session_state['crujias'], niveles=st.session_state.get('_ult_niveles', []))))
            st.session_state['ver_cols'] += 1
        else:
            nuevo_d = ame.niveles_desde_columnas(dict(crujias=st.session_state['crujias'],
                                                      niveles=st.session_state.get('_ult_niveles', []),
                                                      columnas=st.session_state.get('_ult_cols', [])))
            st.session_state['df'] = df_de(nuevo_d['niveles'])
            st.session_state['ver'] += 1
            n1_ = nuevo_d['niveles'][0]
            try:
                nej_ = len(ame.parse_vector(st.session_state['crujias'], 'Crujías')) + 1
                for i, est in enumerate(ame.estados_apoyo(n1_.get('artic', ''), n1_.get('rod', ''), nej_), 1):
                    st.session_state[f'apoyo_{i}'] = est
            except ValueError:
                pass
        st.session_state.pop('_aviso_sync', None)
    except (ValueError, KeyError, TypeError) as e_sync:
        st.session_state['_aviso_sync'] = f'No se pudo pasar lo capturado al otro modo: {e_sync}'
st.session_state['_captura_prev'] = _capt_ahora

with T_est:
    st.caption('Paso 1 · Define la estructura: material, geometría, apoyos y cargas. '
               'Luego completa «Sismo y norma» y mira los «Resultados».')
    st.radio('¿Cómo defines la estructura?', MODOS, key='modo', horizontal=True,
             help='Por secciones y cargas el programa arma K y M. Si el ejercicio ya te da K y M, usa las matrices.')
    manual = st.session_state['modo'] == MODOS[1]
    st.text_input('Periodo fundamental medido T₁ (s), opcional', key='T_medido',
                  help='Si el problema da el periodo medido y pide el peso total: deja vacías las cargas, escribe el periodo '
                       'y el programa escala las masas para que T₁ sea ese valor. Si das cargas, se usan como distribución '
                       'relativa entre niveles.')

    with st.expander('Material y modelo', expanded=not manual):
        c1, c2, c3 = st.columns(3)
        c1.selectbox('Material', list(ame.MATERIALES), key='material')
        c2.selectbox('Vigas', VIGAS, key='vigas', disabled=manual,
                     help='Rígidas: modelo de cortante. Flexibles: marco plano con rigidez de columnas y vigas.')
        c3.radio('Dirección del sismo', ['X', 'Y'], key='direccion', horizontal=True, disabled=manual,
                 help='En una sección b x h, h está en la dirección X y b en la Y. Con Y se usa la inercia respecto al otro eje.')
        concreto = st.session_state['material'] == 'Concreto'
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.text_input('E del material', key='E', disabled=manual,
                      help="En la unidad de la derecha. Si lo dejas vacío: concreto 14000·√f'c (u 8000·√f'c), acero 2,040,000 kg/cm².")
        c2.selectbox('Unidad de E', list(ame.FACTOR_ESFUERZO), key='unidad_E', disabled=manual)
        c3.text_input("f'c del concreto", key='fc', disabled=(not concreto) or manual)
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Unidad de f'c", list(ame.FACTOR_ESFUERZO), key='unidad_fc', disabled=manual)
        c2.selectbox('Clase del concreto', CLASES, key='clase', disabled=(not concreto) or manual)
        c3.radio('Unidades de las secciones', ['cm', 'm'], key='unidad_seccion', horizontal=True, disabled=manual)

    with st.expander('Matrices K y M (ya calculadas)', expanded=manual):
        c1, c2 = st.columns(2)
        c1.text_area('Matriz K (pega desde Excel o escribe)', key='K_txt', height=170,
                     help='Filas en renglones; columnas separadas por espacios, tabulaciones o comas. También acepta [a b; c d].')
        c2.text_area('Matriz M (completa o solo la diagonal)', key='M_txt', height=170,
                     help='Puedes pegar la matriz completa o solo las masas: "3 3 1.5 1.5".')
        c1, c2, c3 = st.columns([1, 2, 2])
        c1.radio('Unidades de K y M', UNIDADES, key='unidades',
                 help='t, cm: K en t/cm y M en t·s²/cm. kg, cm: K en kg/cm y M en kg·s²/cm.')
        c2.text_input('Altura de cada entrepiso (m), de abajo hacia arriba', key='alturas_txt',
                      help='Por ejemplo "5 5 5 5". Se usa para distorsiones, momentos de volteo y el método estático.')
        c3.text_input('Pesos por nivel (t), opcional', key='pesos_txt',
                      help='Si lo dejas vacío se calcula W = m·g con la diagonal de M.')
        st.caption('Con matrices no hace falta dar secciones ni cargas: los pesos salen de las masas. '
                   'El orden de los niveles es de abajo hacia arriba, como en K y M.')

    with st.expander('Estructura por secciones y cargas (niveles de abajo hacia arriba)', expanded=not manual):
        st.radio('Captura de las columnas', CAPTURAS, key='captura', horizontal=True,
                 help='Por nivel: una fila por nivel con los ejes, secciones y E de sus columnas. Por columna: una fila por columna '
                      '(eje, niveles que abarca, sección, E, apoyo); es lo más directo cuando las columnas cambian de nivel a nivel.')
        por_columna = st.session_state['captura'] == CAPTURAS[1]
        if st.session_state.get('_aviso_sync'):
            st.warning(st.session_state['_aviso_sync'])
        if not por_columna:
            st.caption('Si cada columna tiene su propio E o I, o alguna cruza varios niveles, usa «Por columna»: '
                       'lo que ya capturaste se pasa solo a esa tabla.')
        st.text_input('Crujías (m), de izquierda a derecha', key='crujias',
                      help='Anchos de cada crujía separados por espacio: "4 5" define 2 crujías y 3 ejes (1, 2 y 3).')
        c1, c2, _ = st.columns([1, 1, 2])
        c1.text_input('Volado izquierdo (m) · todos los niveles', key='volado_izq',
                      help='Voladizo más allá del primer eje. Solo aporta peso; no cambia la rigidez lateral. '
                           'Cada nivel puede poner el suyo en la tabla.')
        c2.text_input('Volado derecho (m) · todos los niveles', key='volado_der',
                      help='Voladizo más allá del último eje. Cada nivel puede poner el suyo en la tabla.')
        try:
            nej = len(ame.parse_vector(st.session_state['crujias'], 'Crujías')) + 1
        except ValueError:
            nej = 0
        for _i in range(1, MAX_EJES + 1):
            st.session_state.setdefault(f'apoyo_{_i}', 'Empotrado')
        if por_columna:
            st.info('Los apoyos de la base se eligen en la columna "Apoyo" de la tabla de columnas.')
        else:
            st.markdown('**Apoyos en la base** · uno por eje, de izquierda a derecha')
            if 2 <= nej <= MAX_EJES:
                por_fila = min(nej, 6)
                for ini in range(1, nej + 1, por_fila):
                    cols_ap = st.columns(por_fila)
                    for i in range(ini, min(ini + por_fila, nej + 1)):
                        cols_ap[i - ini].selectbox(f'Eje {i}', list(ame.APOYOS), key=f'apoyo_{i}',
                                                   format_func=SIMBOLO_APOYO.get)
                st.caption('Empotrado: 12EI/h³ · Articulado: 3EI/h³ · Rodillo: no transmite cortante, la columna solo carga peso. '
                           'La vista previa dibuja cada apoyo para que confirmes.')
            else:
                st.caption('Escribe las crujías para elegir los apoyos de cada eje.')
        flexibles = st.session_state['vigas'].startswith('Flex')
        conf = {k: st.column_config.TextColumn(ETIQ[k], help=AYUDA[k]) for k in COLS}
        conf['rod'] = None  # los rodillos se eligen en "Apoyos en la base"
        if por_columna:
            for k_ in COLS_COLUMNA:
                conf[k_] = None  # esas columnas se capturan en la tabla de columnas
        if not flexibles:
            conf['viga'] = None  # oculta las columnas de viga
            conf['E_viga'] = None
        if por_columna:
            st.markdown('**Niveles** · altura, vigas, tramos sin losa, volados y cargas')
        edit = st.data_editor(st.session_state['df'], key=f'niv_{st.session_state["ver"]}', num_rows='dynamic',
                              width='stretch', column_config=conf)
        st.caption('Una fila por nivel (la primera es la base; agrega filas con el + de la tabla). '
                   + ('' if por_columna else 'Las columnas se ubican por eje: si un nivel no tiene columna en algún eje, no lo pongas en '
                      '"Columnas en ejes". ') + 'Pasa el cursor sobre los encabezados para ver ejemplos de sintaxis.')
        edit_cols = None
        if por_columna:
            st.markdown('**Columnas** · una fila por columna, con los niveles que abarca')
            conf_c = {'eje': st.column_config.TextColumn('Eje', help='Número de eje (1, 2, 3… de izquierda a derecha).'),
                      'desde': st.column_config.TextColumn('Desde nivel', help='Nivel donde empieza (vacío = 1).'),
                      'hasta': st.column_config.TextColumn('Hasta nivel', help='Nivel donde termina (vacío = el último). '
                                                           'Una columna que cruza un piso sin viga es una sola fila.'),
                      'seccion': st.column_config.TextColumn('Sección (cm)', help=AYUDA_SECCIONES + ' Déjala vacía si das la '
                                                             'inercia I directamente.'),
                      'I': st.column_config.TextColumn('I (cm⁴)', help='Inercia de esta columna en la dirección del sismo. '
                                                       'Si la llenas, se usa en lugar de la sección.'),
                      'A': st.column_config.TextColumn('A (cm²)', help='Área, opcional (solo con vigas flexibles). '
                                                       'Sin área la columna es axialmente rígida.'),
                      'E': st.column_config.TextColumn('E', help='E de esta columna, en la unidad de E, o "acero". '
                                                       'Vacío = el E general.'),
                      'apoyo': st.column_config.SelectboxColumn('Apoyo en su base', options=list(ame.APOYOS), required=False,
                                                                help='Cómo termina la columna abajo (base de la estructura si empieza '
                                                                     'en el nivel 1). Rodillo solo en la base.'),
                      'art_sup': st.column_config.CheckboxColumn('Articulada arriba')}
            edit_cols = st.data_editor(st.session_state['df_cols'], key=f'cols_{st.session_state["ver_cols"]}',
                                       num_rows='dynamic', width='stretch', column_config=conf_c)
            if st.button('Rellenar la tabla de columnas desde la captura por nivel'):
                try:
                    filas_c = ame.columnas_desde_niveles(dict(crujias=st.session_state['crujias'], niveles=filas_de(edit)))
                    st.session_state['df_cols'] = df_cols_de(filas_c)
                    st.session_state['ver_cols'] += 1
                    st.rerun()
                except ValueError as e_cols:
                    st.warning(str(e_cols))


# ------------------------------------------------------------------ 2 · sismo y norma
with T_sismo:
    st.caption('Paso 2 · Objetivo de diseño, espectro del sitio y datos de la norma.')
    st.markdown('### Objetivo de diseño (NTC-S 2023)')
    st.selectbox('Grupo / subgrupo', list(SUBGRUPOS), key='subgrupo', format_func=SUBGRUPOS.get,
                 help='Grupo B, o Grupo A con sus subgrupos A2 y A1 (A1 es el más exigente). El riesgo equivale a la categoría de ASCE 7.')
    st.selectbox('Intensidad sísmica', ntc.INTENSIDADES, key='intensidad', on_change=cambiar_intensidad,
                 format_func=lambda i: ntc.ETIQUETA_INTENSIDAD.get(i, i),
                 help='Normalmente se usa el sismo base de diseño. MC = máximo considerado = infrecuente. Frecuente y '
                      'ocasional usan el espectro de base de diseño × Ks (× 2.5 Ks en ocasional).')
    grupo_sel = ntc.SUBGRUPOS[st.session_state['subgrupo']]
    objetivo = ntc.objetivo_diseno(grupo_sel, st.session_state['intensidad'], st.session_state['subgrupo'])
    st.markdown(f'<div class="objetivo"><b>{objetivo["desempeno"]}</b><br>{objetivo["espectro"]}<br>'
                f'<small>{objetivo["revisa"]}</small></div>', unsafe_allow_html=True)
    if objetivo['nota']:
        st.caption(objetivo['nota'])
    st.selectbox('Estado de cálculo', ['Automático', *ame.ESTADOS], key='estado',
                 help='Automático: el que corresponde al objetivo de diseño. Puedes forzar otro estado.')
    st.text_input('Ks (limitación de daños)', key='Ks',
                  help='Sa(BD)·Ks. Vacío = automático con la ecuación 3.1.1 de la norma (según Ts). Escribe un valor si el '
                       'ejercicio te da otro (en tus hojas, 0.25).')
    try:
        _ts_txt = (st.session_state['Ts'] if ntc.espectro_fuente(st.session_state['intensidad']) == 'Base de diseño'
                   else st.session_state['_esp'].get('Base de diseño', {}).get('Ts', ''))
        _ts_ks = ame.parse_num(_ts_txt, 'Ts')
        st.caption(f'Ks automático = {ntc.ks_norma(_ts_ks):.4f} (zona {ntc.zona_ks(_ts_ks)}, Ts = {_ts_ks:g} s): '
                   '1/6 si Ts ≤ 0.5 · 1/[6 − 4(Ts − 0.5)] entre 0.5 y 1.0 · 1/4 si Ts ≥ 1.0. '
                   'Si escribes un valor arriba, se usa ese.')
    except (ValueError, KeyError):
        pass
    st.text_input('Q (varios valores separados por espacio)', key='Q',
                  help='Se usa el primero para las fuerzas. En ocupación inmediata se usa Q = 1.')
    st.caption('Q: 4 ductilidad alta (daño severo) · 3 media (daño importante) · 2 media (daño medio) · '
               '1.5 baja (poco daño) · 1 elástica.')
    st.text_input('k1', key='k1')
    st.caption('k1: 0.8 con menos de 3 crujías resistentes · 1.0 con 3 o más · 1.25 en sistemas duales.')
    st.divider()
    st.markdown(f'### Espectro SASID · {ntc.espectro_fuente(st.session_state["intensidad"])}')
    c1, c2 = st.columns(2)
    for i, (et, k) in enumerate([('a0 (cm/s²)', 'a0'), ('c (cm/s²)', 'c'), ('Ta (s)', 'Ta'),
                                 ('Tb (s)', 'Tb'), ('k', 'k'), ('Ts (s)', 'Ts')]):
        (c1 if i % 2 == 0 else c2).text_input(et, key=k)
    st.caption('Captura el espectro de base de diseño y el infrecuente (MC); se conservan al cambiar de intensidad. '
               'En frecuente y ocasional se usa el de base de diseño.')

    st.divider()
    st.markdown('### Análisis')
    st.selectbox('Combinación modal', list(COMBINACIONES), key='combinacion',
                 help='SRSS: raíz de la suma de cuadrados. CQC: considera la correlación entre modos cercanos. '
                      'Suma absoluta: cota superior.')
    c1, c2 = st.columns(2)
    c1.text_input('Factor Fu (FC)', key='factor_Fu')
    c2.text_input('V estático ×', key='mult_V')

    st.divider()
    with st.expander('Irregularidades (corrección de γmáx)'):
        st.multiselect('Condiciones que se cumplen', list(ntc.IRREGULARIDADES), key='irreg',
                       format_func=lambda c: f'{c} · {ntc.IRREGULARIDADES[c]}',
                       help='1 condición: γc = 0.8 · 2: 0.7 · 3 o más: 0.6. Solo aplica a ocupación inmediata y seguridad de vida.')
        st.checkbox('Fuertemente irregular por torsión (5.2.2)', key='fuerte_torsion')
        st.checkbox('Fuertemente irregular en elevación (5.3.3)', key='fuerte_elev')
        gc, gc_txt = ntc.gamma_c(st.session_state['irreg'], st.session_state['fuerte_torsion'],
                                 st.session_state['fuerte_elev'])
        st.caption(gc_txt)

    with st.expander('Otros datos normativos'):
        st.text_input('Amortiguamiento ζ', key='zeta', help='Vacío: 0.05 concreto, 0.03 acero. Se usa en CQC; '
                      'el espectro de SASID debe pedirse con el mismo ζ.')
        st.checkbox('R = 1 en materiales distintos del concreto', key='R1_otros',
                    help='Los apuntes indican R = 1 para otros materiales.')
        st.selectbox('Coeficiente del método estático', CRITERIOS, key='criterio_estatico',
                     help='Los apuntes toman el mayor entre c/(Q′R′) en el plato y a0/(Q′R′) en T = 0. '
                          'El código de MATLAB toma el máximo del espectro de diseño completo.')
        st.selectbox('Zona geotécnica', ['', 'I', 'II', 'III'], key='zona',
                     format_func=lambda z: z or '—', help='Para verificar si el método estático es aplicable.')
        st.radio('Regularidad', ['Regular', 'Irregular'], key='regularidad', horizontal=True)
        st.text_input('Dimensión de la planta perpendicular al análisis, b (m)', key='b_planta',
                      help='Para calcular la excentricidad accidental y los momentos torsionantes.')

    with st.expander('Unidades y opciones avanzadas'):
        st.radio('Unidades del espectro (a0 y c)', ['cm/s²', 'g'], key='unidad_espectro', horizontal=True,
                 help='Tus hojas dan el espectro en g y en cm/s². Si lo capturas en g se convierte (× 981).')
        st.selectbox('Modos que se combinan', MODOS_OPC, key='modos',
                     help='Para ejercicios que piden usar solo algunos modos. Los apuntes piden el 95 % de la masa '
                          'efectiva o todos los modos con T ≥ 0.4 s.')
        st.text_input('N (si eliges "Los primeros N")', key='modos_n')
        st.selectbox('R′ para distorsiones y cimentación', RDIST, key='R_dist',
                     help='En tus hojas QR′ = 4(2) = 8: R′ del plato con k2 = 0. La norma indica el periodo fundamental.')
        st.text_input('Ordenadas Sa/(Q′R′) por modo (opcional)', key='Sa_modos',
                      help='Si el ejercicio ya da las aceleraciones espectrales de cada modo (por ejemplo "0.0894 0.0879 '
                           '0.0888" en g), se usan en lugar de las del espectro. Van en las unidades del espectro.')
        st.selectbox('Normalización de las formas modales', NORM_OPC, key='norm_modal',
                     help='Solo cambia cómo se presentan los modos (y γ); las fuerzas y los periodos no cambian.')
        st.text_input('Excentricidad estática e_s (m, opcional)', key='e_s',
                      help='Con la dimensión b de la planta calcula e = e_s ± e_a para los momentos torsionantes.')

    st.divider()
    st.markdown('### Límites de distorsión')
    st.caption('Los γmáx dependen del sistema (tablas 4.3.x de la norma): confírmalos.')
    c1, c2, c3 = st.columns(3)
    c1.text_input('Daños', key='lim_dl', help='0.002 ó 0.004')
    c2.text_input('Vida', key='lim_sv')
    c3.text_input('Ocup.', key='lim_oi', help='Vacío = sin límite')


# ------------------------------------------------------------------ datos y cálculo
datos = {k: st.session_state[k] for k in CAMPOS if k not in ESPECTRO and k not in CLAVES_MANUAL}
datos['modo'] = 'matrices' if manual else 'secciones'
datos.update(K=st.session_state['K_txt'], M=st.session_state['M_txt'], alturas=st.session_state['alturas_txt'],
             pesos=st.session_state['pesos_txt'])
datos['vigas'] = 'flexibles' if flexibles else 'rigidas'
esp = {k: dict(v) for k, v in st.session_state['_esp'].items()}
esp[ntc.espectro_fuente(st.session_state['intensidad'])] = {k: st.session_state[k] for k in ESPECTRO}
datos['espectros'] = esp
datos['niveles'] = [{k: ('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for k, v in fila.items()}
                    for _, fila in edit.dropna(how='all').iterrows()]
if por_columna and not manual:
    datos['columnas'] = filas_de(edit_cols)
elif datos['niveles'] and 2 <= nej <= MAX_EJES and not manual:       # los selectores mandan sobre los apoyos del nivel 1
    try:
        ejes1 = ame.parse_ejes(datos['niveles'][0].get('ejes', ''), nej, 'nivel 1', range(1, nej + 1))
    except ValueError:
        ejes1 = list(range(1, nej + 1))
    art_, rod_ = ame.apoyos_a_texto([st.session_state.get(f'apoyo_{i}', 'Empotrado') for i in range(1, nej + 1)])
    datos['niveles'][0]['artic'] = ' '.join(str(e) for e in art_ if e in ejes1)
    datos['niveles'][0]['rod'] = ' '.join(str(e) for e in rod_ if e in ejes1)
st.session_state['_ult_niveles'] = [dict(r) for r in datos['niveles']]          # con los apoyos de los selectores aplicados
if edit_cols is not None:
    st.session_state['_ult_cols'] = filas_de(edit_cols)
firma = hashlib.md5(json.dumps(datos, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
if st.session_state.get('_firma_auto') != firma and st.session_state.get('_pid'):
    try:                                          # autoguardado de lo que llevas capturado (aunque tenga errores)
        AUTOGUARDADO.mkdir(exist_ok=True)
        ruta_auto(st.session_state['_pid']).write_text(json.dumps(datos, ensure_ascii=False), encoding='utf-8')
        st.session_state['_firma_auto'] = firma
    except OSError:
        pass

modelo = res = error_modelo = None
try:
    modelo = ame.preparar(datos)
except ValueError as e:
    error_modelo = f'Revisa los datos: {e}'
if modelo is not None:
    try:
        res = ame.calcular(modelo)
    except (np.linalg.LinAlgError, ValueError) as e:
        error_modelo = f'No se pudo calcular: {e if isinstance(e, ValueError) else "las matrices son singulares. Revisa secciones y alturas."}'


def kpi(col, titulo, num, sub):
    col.markdown(f'<div class="kpi"><div class="t">{titulo}</div><div class="n">{num}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def mostrar(df, dec=4):
    st.dataframe(df, hide_index=True, width='stretch',
                 column_config={c: st.column_config.NumberColumn(format=f'%.{dec}f') for c in df.columns[1:]
                                if pd.api.types.is_numeric_dtype(df[c])})


with T_est:
    if error_modelo:
        st.error(error_modelo)
        if por_columna and not manual:
            borrador = dict(crujias=st.session_state['crujias'], niveles=datos['niveles'], columnas=datos.get('columnas'))
            _, _, _, faltan = ame.cobertura_columnas(borrador)
            with st.expander('Dibujo de las columnas capturadas', expanded=True):
                st.pyplot(ame.fig_captura_columnas(borrador))
                if faltan:
                    st.caption('Los niveles en rojo no tienen columnas en la tabla: agrega filas con ese nivel en «Desde / Hasta», '
                               'o extiende las columnas existentes.')
            if faltan and st.button('Extender las columnas hasta los niveles sin columnas'):
                nuevas, hechos = ame.completar_columnas(datos['columnas'], len(datos['niveles']))
                if hechos:
                    st.session_state['df_cols'] = df_cols_de(nuevas)
                    st.session_state['ver_cols'] += 1
                    st.rerun()
                else:
                    st.warning('No hay columnas que terminen justo debajo de esos niveles; agrega las filas a mano.')
    elif manual:
        st.info('Con matrices no hay geometría que dibujar: revisa K, M, las alturas y los pesos derivados en la pestaña '
                '«Resultados».')
    else:
        with st.expander('Vista previa de la estructura', expanded=True):
            st.pyplot(ame.fig_estructura(modelo))
            st.caption('Revisa que ejes, secciones, articulaciones, vigas y cargas sean los que esperas antes de leer los resultados.')
        st.markdown('**Columnas del modelo** · valores efectivos de cada columna (las que cruzan niveles aparecen una sola vez)')
        tabla_col = pd.DataFrame([{
            'Eje': f['eje'], 'Niveles': f'{f["desde"]}' + (f' a {f["hasta"]}' if f['hasta'] != f['desde'] else ''),
            'Sección': f['seccion'], 'E (kg/cm²)': f['E'], 'I (cm⁴)': f['I'], 'A (cm²)': f['A'] if f['A'] else np.nan,
            'EI (t·m²)': f['EI_tm2'], 'Longitud (m)': f['L'], 'Rigidez lateral k (kg/cm)': f['k'], 'Apoyo': f['apoyo']}
            for f in ame.tabla_columnas(modelo)])
        st.dataframe(tabla_col, hide_index=True, width='stretch',
                     column_config={c: st.column_config.NumberColumn(format='%.6g')
                                    for c in ('E (kg/cm²)', 'I (cm⁴)', 'A (cm²)', 'EI (t·m²)', 'Longitud (m)',
                                              'Rigidez lateral k (kg/cm)')})

# ------------------------------------------------------------------ 3 · resultados
with T_res:
    if res is None:
        st.error(error_modelo)
        st.info('Corrige los datos en las pestañas «Estructura» y «Sismo y norma».')
    else:
        p, n = res['p'], res['p']['n']
        fe = res['factor_escala']
        nm = res['normativa']


        st.markdown(f'<div class="objetivo"><b>Grupo {p["subgrupo"]} · sismo {p["intensidad"].lower()}</b> → '
                    f'{p["objetivo"]["desempeno"]} · estado de cálculo: <b>{p["estado"]}</b><br>'
                    f'<small>{p["objetivo"]["espectro"]} · {p["objetivo"]["revisa"]}</small></div>', unsafe_allow_html=True)
        aj = p.get('ajuste_periodo')
        if aj:
            st.success(f'**Peso total del marco: W = {aj["W_total"]:.3f} t** (masa total {aj["m_total"]:.4f} kg·s²/cm), obtenido '
                       f'del periodo medido T₁ = {aj["T_medido"]:g} s. Con los pesos dados el periodo calculado era '
                       f'{aj["T_calculado"]:.4f} s.')
        alertas = [t for s, t in res['avisos'] if s == 'alerta']
        avisos = [t for s, t in res['avisos'] if s == 'aviso']
        infos = [t for s, t in res['avisos'] if s == 'info']
        if alertas:
            st.error('**Alertas**\n\n' + '\n'.join(f'- {t}' for t in alertas))
        if avisos:
            st.warning('**Avisos de coherencia**\n\n' + '\n'.join(f'- {t}' for t in avisos))
        if infos:
            with st.expander(f'Notas ({len(infos)})'):
                for t in infos:
                    st.write('• ' + t)
        if not res['avisos']:
            st.success('Sin avisos de coherencia.')

        k = st.columns(4)
        kpi(k[0], 'PERIODO FUNDAMENTAL', f'{res["T"][0]:.4f} s', f'{n} niveles · γ₁ = {res["gam"][0]:.3f}')
        kpi(k[1], 'CORTANTE BASAL DINÁMICO', f'{res["V_din"]:.3f} t', f'Σ Mvu = {res["tabla_din"]["Mvu"].sum():.2f} t·m')
        kpi(k[2], 'CORTANTE BASAL ESTÁTICO', f'{res["V_est"]:.3f} t', f'C = {res["Fmax_est"]:.4f} g · gobierna {res["estatico"]["gobierna"].split(" (")[0]}')
        kpi(k[3], 'FACTOR DE ESCALA', f'{fe:.4f}', 'Escalar fuerzas dinámicas' if fe > 1 else 'El dinámico ya cumple (≤ 1)')
        st.write('')

        # descargas (Excel y PDF se generan al pedirlos: son lentos)
        d1, d2, d3, _ = st.columns([1, 1, 1, 1])
        if d1.button('Preparar Excel', width='stretch'):
            buf = io.BytesIO()
            ame.exportar_excel(res, buf)
            st.session_state['excel'] = (firma, buf.getvalue())
        if d2.button('Preparar memoria PDF', width='stretch', type='primary'):
            with st.spinner('Generando memoria de cálculo…'):
                buf = io.BytesIO()
                ame_pdf.memoria_pdf(res, buf, proyecto=datos['proyecto'], autor=datos['autor'])
                st.session_state['pdf'] = (firma, buf.getvalue())
        d3.download_button('Guardar proyecto (.json)', json.dumps(datos, indent=2, ensure_ascii=False), 'proyecto_AME.json',
                           mime='application/json', width='stretch')
        b1, b2, _ = st.columns([1, 1, 2])
        for clave, col, etiqueta, nombre, mime in (
                ('excel', b1, 'Descargar Excel', 'resultados_AME.xlsx',
                 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
                ('pdf', b2, 'Descargar memoria PDF', 'memoria_AME.pdf', 'application/pdf')):
            listo = st.session_state.get(clave)
            if listo and listo[0] == firma:
                col.download_button(etiqueta, listo[1], nombre, mime=mime, width='stretch')
            elif listo:
                col.caption(f'El archivo ({clave.upper()}) quedó desactualizado: vuelve a prepararlo.')


        NOMBRES = ['Resumen', 'Fuerzas por nivel', 'Análisis estático', 'Análisis elástico', 'Revisiones', 'Normativa', 'Modos',
                   'Detalle completo', 'Formas modales', 'Espectro']
        tab = dict(zip(NOMBRES, st.tabs(NOMBRES)))
        with tab['Resumen']:
            est_txt = ('estático ' + ('aplicable' if nm['estatico']['aplica'] else 'no aplicable (referencia)')) if 'tabla_est' in res else ''
            st.markdown(f'**Análisis de diseño: {p["estado"]}** · {res["n_modos"]} de {n} modos · {p["combinacion"]}'
                        f' · {est_txt} · elástico sin reducir')
            enc, filas = ame.tabla_comparacion_metodos(res)
            mostrar(pd.DataFrame(filas, columns=enc), 3)
            if fe > 1:
                st.info(f'V estático / V dinámico = {fe:.4f} > 1: para diseñar se usan las fuerzas dinámicas escaladas por ese factor.')
            st.pyplot(ame.fig_comparacion(res))
            st.markdown('**Fu y Vu por nivel de cada análisis**')
            enc, filas = ame.tabla_niveles_metodos(res)
            mostrar(pd.DataFrame(filas, columns=enc), 3)
        with tab['Fuerzas por nivel']:
            st.pyplot(ame.fig_diagramas(res))
            t = res['tabla_din']
            st.markdown('**Análisis dinámico (modal espectral)**')
            mostrar(pd.DataFrame({'Nivel': t['nivel'], 'h (m)': t['h'], 'F (t)': t['F'], 'Fu (t)': t['Fu'],
                                  'Vu (t)': t['Vu'], 'Mvu (t·m)': t['Mvu'], 'M volteo entrepiso (t·m)': t['Mv']}))
            if res['factor_aplicado'] > 1:
                te = res['tabla_din_esc']
                st.markdown(f'**Fuerzas dinámicas escaladas (× {res["factor_aplicado"]:.4f})**')
                mostrar(pd.DataFrame({'Nivel': te['nivel'], 'h (m)': te['h'], 'F (t)': te['F'], 'Fu (t)': te['Fu'],
                                      'Vu (t)': te['Vu'], 'Mvu (t·m)': te['Mvu'], 'M volteo entrepiso (t·m)': te['Mv']}))
                st.caption('El factor de escala (V estático / V dinámico) es mayor que 1: estas son las fuerzas para el diseño.')
            else:
                st.caption('El factor de escala es menor o igual que 1: las fuerzas dinámicas ya cumplen y no se escalan.')
            st.caption('M volteo entrepiso = momento de volteo en la base de ese entrepiso (Σ Fu·(z − z de la base)); el del '
                       'primer entrepiso es el momento de volteo total en la cimentación.')
        with tab['Análisis estático']:
            e, se, ap = res['tabla_est'], res['estatico'], nm['estatico']
            if ap['aplica']:
                st.success('El método estático es aplicable' + (f' (H = {ap["H"]:.1f} m ≤ {ap["limite"]:g} m)' if ap['limite'] else '') + '.')
            else:
                st.warning('Según los apuntes el método estático **no aplica** (grupo A o altura excedida): '
                           'el factor de escala es solo de referencia.')
            st.markdown('**Coeficiente sísmico** · C = máx { c/(Q′R′) en el plato ; a₀/(Q′R′) en T = 0 }')
            enc, filas = ame.tabla_derivacion_estatica(res)
            mostrar(pd.DataFrame(filas, columns=enc))
            c = st.columns(4)
            c[0].metric('C (g)', f'{se["C"]:.5f}')
            c[1].metric('Gobierna', se['gobierna'].split(' (')[0].replace('el ', '', 1))
            c[2].metric('ΣW (t)', f'{se["W_T"]:.2f}')
            c[3].metric('V estático (t)', f'{res["V_est"]:.3f}')
            st.pyplot(ame.fig_diagramas(res, 'estatico'))
            st.markdown('**Fuerzas por nivel**  ·  Fᵢ = C·(Wᵢhᵢ / ΣWᵢhᵢ)·ΣW')
            mostrar(pd.DataFrame({'Nivel': e['nivel'], 'h (m)': e['h'], 'Wi (t)': e['Wi'], 'Wihi (t·m)': e['hWi'],
                                  'Fi (t)': e['Fi'], 'Fu (t)': e['Fu'], 'Vu (t)': e['Vu'], 'Mvu (t·m)': e['Mvu'],
                                  'M volteo entrepiso (t·m)': e['Mv']}))
            st.caption(f'Comprobación: ΣFᵢ = {e["Fi"].sum():.4f} t = C·ΣW = {se["C"]:.5f} × {se["W_T"]:.2f} · '
                       f'ΣMvu = {e["Mvu"].sum():.2f} t·m')
            st.markdown(f'**Distorsiones por el método estático** ({p["estado"]})')
            enc, filas = ame.tabla_distorsiones_estatico(res)
            mostrar(pd.DataFrame(filas, columns=enc), 5)
            st.caption('Desplazamientos de K⁻¹·Fᵢ × ' + ('1 (limitación de daños: sin FC ni Q·R′)' if res['clave'] == 'DL'
                       else f'FC·Q·R′ = {se["amp"]:.4f} (R′ del plato)') + ', comparados con el análisis dinámico.')
            st.markdown('**Factor de escala**')
            c = st.columns(4)
            c[0].metric('V estático (t)', f'{res["V_est"] * p["mult_V"]:.3f}')
            c[1].metric('V dinámico (t)', f'{res["V_din"]:.3f}')
            c[2].metric('Factor V est. / V din.', f'{fe:.4f}')
            c[3].metric('Se aplica', f'× {res["factor_aplicado"]:.4f}' if res['factor_aplicado'] > 1 else 'No hace falta')
        with tab['Análisis elástico']:
            el = res['elastico']
            st.info('Espectro **sin reducir** (Q′ = 1, R′ = 1), el "espectro transparente" de los apuntes: la demanda que tendría la '
                    'estructura si no se redujera por ductilidad ni sobrerresistencia. Mismo modelo, modos y combinación.')
            c = st.columns(4)
            c[0].metric('V elástico (t)', f'{el["V_el"]:.3f}')
            c[1].metric('V de diseño (t)', f'{res["V_din"]:.3f}')
            c[2].metric('V el. / V diseño', f'{el["V_el"] / res["V_din"]:.3f}')
            c[3].metric('Σ Mvu elástico (t·m)', f'{el["tabla"]["Mvu"].sum():.2f}')
            st.pyplot(ame.fig_elastico(res))
            st.markdown('**Ordenadas espectrales por modo**')
            mostrar(pd.DataFrame({'Modo': np.arange(1, n + 1), 'T (s)': res['T'], 'Sa elástico (cm/s²)': el['Sa'],
                                  'Sa de diseño (cm/s²)': res['Sa_modal'][:, 0], 'Relación': el['Sa'] / res['Sa_modal'][:, 0]}))
            st.pyplot(ame.fig_diagramas(res, 'elastico'))
            te = el['tabla']
            st.markdown('**Fuerzas elásticas por nivel (dinámico)**')
            mostrar(pd.DataFrame({'Nivel': te['nivel'], 'h (m)': te['h'], 'F (t)': te['F'], 'Fu (t)': te['Fu'],
                                  'Vu (t)': te['Vu'], 'Mvu (t·m)': te['Mvu'], 'M volteo entrepiso (t·m)': te['Mv']}))
            if 'est' in el:
                ee = el['est']['tabla']
                st.markdown(f'**Fuerzas elásticas por el método estático** · C = máx(c, a₀) = {el["est"]["C"]:.4f} g')
                mostrar(pd.DataFrame({'Nivel': ee['nivel'], 'h (m)': ee['h'], 'Wi (t)': ee['Wi'], 'Fi (t)': ee['Fi'],
                                      'Fu (t)': ee['Fu'], 'Vu (t)': ee['Vu'], 'Mvu (t·m)': ee['Mvu'],
                                      'M volteo entrepiso (t·m)': ee['Mv']}))
            st.markdown('**Fuerza y distorsión elásticas contra las de diseño**')
            enc, filas = ame.tabla_elastico_comparacion(res)
            mostrar(pd.DataFrame(filas, columns=enc), 4)
            st.caption('Las distorsiones de diseño incluyen la amplificación FC·Q·R′; las elásticas no se amplifican.')
        with tab['Revisiones']:
            rev = res['revision']
            st.markdown('**Distorsiones de entrepiso por estado límite** (★ = estado de diseño)')
            lims = [[(e['nombre'] + (' ★' if k_ == res['clave'] else '')),
                     e['limite_base'] if e['limite_base'] is not None else np.nan, e['gamma_c'],
                     e['limite'] if e['limite'] is not None else np.nan] for k_, e in rev['estados'].items()]
            mostrar(pd.DataFrame(lims, columns=['Estado límite', 'γmáx', 'γc', 'Límite aplicado']), 4)
            enc, filas = ame.tabla_distorsiones(res)
            mostrar(pd.DataFrame(filas, columns=enc), 5)
            st.caption('Seguridad de vida y ocupación inmediata: desplazamientos × FC·Q·R′ con R′ '
                       + ('del plato (k2 = 0, como en tus hojas)' if p.get('R_dist', 'plato') == 'plato' else 'del periodo fundamental')
                       + '. Limitación de daños: Sa·Ks, sin factor de carga.')
            st.pyplot(ame.fig_distorsiones(res))
            pdt = ame.tabla_pdelta(res)
            if pdt:
                st.markdown('**Efectos P-Δ (θ = P·δ / (V·h))**')
                mostrar(pd.DataFrame(pdt[1], columns=pdt[0]))
                st.caption(f'Irregularidad de masa: {rev["irreg_masa"] or "ninguna"} · piso blando: '
                           f'{rev["irreg_rigidez"] or "ninguno"}. θ ≤ 0.10 se puede ignorar; θ > 0.25 posible inestabilidad.')
        with tab['Normativa']:
            st.markdown('**Factores y reglas aplicadas**')
            st.dataframe(pd.DataFrame(nm['factores'], columns=['Concepto', 'Valor', 'Regla o fuente']), hide_index=True,
                         width='stretch')
            v = nm.get('vmin')
            if v:
                st.markdown('**Cortante basal mínimo** · V ≥ FC · a_min · W')
                c = st.columns(4)
                c[0].metric('a_min (g)', f'{v["a_min_g"]:.5f}')
                c[1].metric('V mínimo (t)', f'{v["V_min"]:.3f}')
                c[2].metric('V dinámico (t)', f'{v["V_din"]:.3f}')
                c[3].metric('Resultado', 'Cumple' if v['cumple'] else f'× {v["factor"]:.3f}')
            cim = nm.get('cimentacion')
            if cim:
                st.markdown('**Cimentación** · elementos mecánicos × 0.65·R′')
                c = st.columns(3)
                c[0].metric('0.65·R′', f'{cim["factor"]:.4f}')
                c[1].metric('V cimentación (t)', f'{cim["V"]:.3f}')
                c[2].metric('M volteo (t·m)', f'{cim["M"]:.3f}')
            tor = nm.get('torsion')
            if tor:
                st.markdown(f'**Torsión accidental** · b = {tor["b"]:g} m')
                mostrar(pd.DataFrame({'Nivel': np.arange(n, 0, -1), 'eₐ (m)': tor['e_a'][::-1], 'Fu (t)': tor['Fu'][::-1],
                                      'Mt (t·m)': tor['Mt'][::-1]}))
                if 'e_s' in tor:
                    st.markdown(f'**Excentricidad de diseño** · e_s = {tor["e_s"]:g} m: e = e_s ± e_a (dinámico, factor 1.0)'
                                + (' y 1.5·e_s ± e_a (estático)' if 'e1_est' in tor else ''))
                    df = {'Nivel': np.arange(n, 0, -1), 'e₁ dinámico (m)': tor['e1_din'][::-1], 'e₂ dinámico (m)': tor['e2_din'][::-1],
                          'Mt₁ dinámico (t·m)': tor['Mt1_din'][::-1], 'Mt₂ dinámico (t·m)': tor['Mt2_din'][::-1]}
                    if 'e1_est' in tor:
                        df.update({'e₁ estático (m)': tor['e1_est'][::-1], 'e₂ estático (m)': tor['e2_est'][::-1],
                                   'Mt₁ estático (t·m)': tor['Mt1_est'][::-1], 'Mt₂ estático (t·m)': tor['Mt2_est'][::-1]})
                    mostrar(pd.DataFrame(df))
                st.caption('El efecto bidireccional 100% + 30% no se calcula (necesita el análisis en las dos direcciones).')
            else:
                st.caption('Captura "b" en Otros datos normativos para ver la torsión accidental.')
            es = nm['estatico']
            st.markdown('**Método estático**')
            st.write(f'H = {es["H"]:.1f} m' + (f' · límite {es["limite"]:g} m (zona {es["zona"]}, '
                                               f'{"regular" if es["regular"] else "irregular"})' if es['limite'] else '')
                     + ' → ' + ('aplicable' if es['aplica'] else '**no aplicable** (grupo A o altura excedida): el factor de escala es solo de referencia'))
            mo = nm['modos']
            st.markdown('**Participación modal**')
            mostrar(pd.DataFrame({'Modo': np.arange(1, n + 1), 'Masa efectiva': mo['m_ef'], 'Acumulada': mo['acum']}))
            st.caption(f'{mo["n95"]} modo(s) alcanzan 95%; modos con T ≥ 0.4 s: {mo["n_T04"]}. Los apuntes piden más de 3 modos, '
                       'el 95% o todos los modos con T ≥ 0.4 s.')
        with tab['Modos']:
            qr = {f[0]: f[-1] for f in res['QR'][::len(res['Qs'])]}
            tabla_modos = {'Modo': np.arange(1, n + 1), 'T (s)': res['T'], 'f (Hz)': res['f'], 'ω (rad/s)': res['w'],
                           'γ': res['gam'], f'{ame.rotulo_sa(res)} (cm/s²)': res['Sa_modal'][:, 0]}
            if res['clave'] != 'DL':
                tabla_modos["Q·R'"] = [qr[j + 1] for j in range(n)]
            else:
                st.caption('Limitación de daño: Sa(BD)·Ks, sin dividir entre Q′R′ ni factor de carga.')
            mostrar(pd.DataFrame(tabla_modos))
        with tab['Detalle completo']:
            st.code(ame.reporte_texto(res), language=None)
        with tab['Formas modales']:
            st.pyplot(ame.fig_modal(res))
        with tab['Espectro']:
            st.pyplot(ame.fig_espectro(res))


# ------------------------------------------------------------------ 4 · herramientas
with T_herr:
    sub_calc, sub_dim = st.tabs(['Calculadora de EI', 'Dimensionar · resolver al revés'])
    with sub_calc:
        st.caption('Cada fila es un miembro con su propia sección y su propio E. Agrega las filas que necesites con el + de la '
                   'tabla. Obtienes I, EI y, si das la altura, la rigidez lateral de la columna. Copia el texto de la sección '
                   'a la tabla de estructura.')
        if 'calc_df' not in st.session_state:
            st.session_state['calc_df'] = pd.DataFrame(
                [dict(seccion='I:3688/A:26.8/I2:98', E='2.1e6', unidad_E='kg/cm²', orientacion=ORIENT_EI[0], h=''),
                 dict(seccion='50x50', E='221.35944', unidad_E='t/cm²', orientacion=ORIENT_EI[0], h='3.5')], dtype=object)
        unidad_calc = st.radio('Unidades de las secciones', ['cm', 'm'], key='calc_us', horizontal=True)
        edit_calc = st.data_editor(
            st.session_state['calc_df'], key='calc_tab', num_rows='dynamic', width='stretch',
            column_config={
                'seccion': st.column_config.TextColumn('Sección', help='Misma sintaxis de la tabla de estructura: 50x50, Ø60, '
                                                       'I:3688/A:26.8/I2:98 (inercia en cm⁴, área en cm²), H:…, T:…. '
                                                       'Agrega @90 para girarla.'),
                'E': st.column_config.TextColumn('E', help='El E de ESTA sección, en la unidad de la columna de al lado.'),
                'unidad_E': st.column_config.SelectboxColumn('Unidad de E', options=list(ame.FACTOR_ESFUERZO), required=False),
                'orientacion': st.column_config.SelectboxColumn('Orientación respecto al sismo', options=ORIENT_EI,
                                                                required=False),
                'h': st.column_config.TextColumn('Altura h (m)', help='Opcional: para obtener 12EI/h³ y 3EI/h³.')})
        filas_calc, errores_calc = ame.propiedades_tabla(filas_de(edit_calc), unidad_calc)
        for i_err, msg_err in errores_calc:
            st.warning(f'Fila {i_err}: {msg_err}')
        if filas_calc:
            hay_h = any('k12' in f for f in filas_calc)
            tabla_calc = pd.DataFrame([{
                'Fila': f['fila'], 'Sección': f['seccion'], 'E (kg/cm²)': f['E'], 'I (cm⁴)': f['I'],
                'I otra dirección (cm⁴)': f['I_otra'], 'A (cm²)': f['A'] if f['A'] else np.nan,
                'r (cm)': f['r'] if f['r'] else np.nan, 'EI (kg·cm²)': f['EI'], 'EI (t·cm²)': f['EI_tcm2'],
                'EI (t·m²)': f['EI_tm2'],
                **({'12EI/h³ (t/cm)': f.get('k12_tcm', np.nan), '3EI/h³ (t/cm)': f.get('k3_tcm', np.nan),
                    '12EI/h³ (kg/cm)': f.get('k12', np.nan), '3EI/h³ (kg/cm)': f.get('k3', np.nan)} if hay_h else {})}
                for f in filas_calc])
            st.dataframe(tabla_calc, hide_index=True, width='stretch',
                         column_config={c: st.column_config.NumberColumn(format='%.6g') for c in tabla_calc.columns[2:]})
            st.caption('Rigidez lateral: 12EI/h³ si la columna está empotrada arriba y abajo (vigas rígidas), 3EI/h³ si está '
                       'articulada en un extremo.')
        elif not errores_calc:
            st.info('Escribe al menos una sección y su E.')
    with sub_dim:
        if res is None:
            st.info('Primero completa una estructura válida en las pestañas «Estructura» y «Sismo y norma».')
        else:
            st.markdown('**Resolver al revés** · en lugar de dar la sección y revisar, das la meta y el programa busca el tamaño de '
                        'las columnas que hace falta.')
            if manual:
                st.info('El dimensionado necesita la estructura por secciones; con matrices K y M no hay secciones que ajustar.')
            else:
                niv_ops = list(range(1, n + 1))
                c1, c2, c3 = st.columns([2, 1, 1])
                sel = c1.multiselect('Niveles a dimensionar (sus columnas se reemplazan)', niv_ops, default=niv_ops, key=f'dim_niv_{n}')
                forma = c2.selectbox('Forma de la sección', list(dim.FORMAS), key='dim_forma')
                rel_txt = c3.text_input('Relación h/b', value='1.5', key='dim_rel', disabled=forma != 'Rectangular',
                                        help='Solo para rectangulares: h (en la dirección del sismo) = relación × b.')
                meta = st.radio('Meta', ['Cumplir la distorsión', 'Alcanzar un periodo'], key='dim_meta', horizontal=True)
                c1, c2, c3, c4 = st.columns(4)
                paso_txt = c1.text_input('Paso (cm)', value='5', key='dim_paso')
                lmax_txt = c2.text_input('Lado máximo (cm)', value='150', key='dim_lmax')
                if meta == 'Cumplir la distorsión':
                    lini_txt = c3.text_input('Lado inicial (cm)', value='20', key='dim_lini')
                    lim_txt = c4.text_input('Límite γ (opcional)', key='dim_lim',
                                            help='Sustituye al límite del estado de diseño. Vacío = el de "Límites de distorsión".')
                    estados = st.radio('Estados que deben cumplir', ['Solo el estado de diseño', 'Todos los estados con límite'],
                                       key='dim_estados', horizontal=True)
                else:
                    lini_txt = c3.text_input('Lado mínimo (cm)', value='10', key='dim_lini2')
                    lim_txt = ''
                    T_txt = c4.text_input('Periodo objetivo T₁ (s)', key='dim_T')
                if st.button('Dimensionar', type='primary', key='dim_ir'):
                    try:
                        num = lambda t, nombre: ame.parse_num(t, nombre)
                        rel = num(rel_txt, 'Relación h/b') if forma == 'Rectangular' else 1.0
                        if meta == 'Cumplir la distorsión':
                            r_dim = dim.dimensionar_distorsion(
                                datos, sel, forma=forma, rel=rel, paso=num(paso_txt, 'Paso'), lado_ini=num(lini_txt, 'Lado inicial'),
                                lado_max=num(lmax_txt, 'Lado máximo'),
                                estados='diseno' if estados.startswith('Solo') else 'todos',
                                limite=num(lim_txt, 'Límite γ') if lim_txt.strip() else None)
                        else:
                            r_dim = dim.dimensionar_periodo(datos, sel, num(T_txt, 'Periodo objetivo'), forma=forma, rel=rel,
                                                            paso=num(paso_txt, 'Paso'), lado_min=num(lini_txt, 'Lado mínimo'),
                                                            lado_max=num(lmax_txt, 'Lado máximo'))
                        r_dim['firma'] = firma
                        st.session_state['dim_res'] = r_dim
                    except ValueError as e_dim:
                        st.session_state.pop('dim_res', None)
                        st.error(str(e_dim))
                r_dim = st.session_state.get('dim_res')
                if r_dim and r_dim.get('firma') != firma:
                    st.caption('Cambiaste los datos de la estructura: vuelve a dimensionar.')
                elif r_dim:
                    pr = r_dim['res']['p']
                    fila_i = lambda txt: float(ame.inercia(ame.parse_secciones(txt, 'dim')[0], pr['direccion']))
                    if r_dim['meta'] == 'distorsion':
                        (st.success if r_dim['cumple'] else st.warning)(
                            ('Todos los entrepisos cumplen' if r_dim['cumple'] else 'No se logró cumplir con el lado máximo')
                            + ' · ' + ' · '.join(f'{r_dim["res"]["revision"]["estados"][c]["nombre"]}: γ ≤ {lim:g}'
                                                 for c, lim in r_dim['usados'])
                            + f' · {r_dim["iteraciones"]} pasos')
                        tabla_dim = pd.DataFrame([{
                            'Nivel': i, 'Sección (cm)': r_dim['textos'][i], 'I (cm⁴)': fila_i(r_dim['textos'][i]),
                            'γ': r_dim['dist'][i - 1], 'γ / límite': float(r_dim['razones'][i - 1])} for i in r_dim['niveles']])
                        mostrar(tabla_dim, 5)
                        st.caption(f'Resultado: T₁ = {r_dim["T"][0]:.4f} s · el menor lado, en pasos, con el que cada entrepiso cumple.')
                    else:
                        st.success(f'Con una sección {r_dim["texto"]} cm en los niveles elegidos T₁ = {r_dim["T"]:.4f} s '
                                   f'(≤ {r_dim["T_obj"]:g} s).')
                        c1, c2, c3 = st.columns(3)
                        c1.metric('Lado exacto (cm)', f'{r_dim["lado_exacto"]:.2f}')
                        c2.metric('Lado redondeado (cm)', f'{r_dim["lado"]:g}')
                        c3.metric('I exacta (cm⁴)', f'{r_dim["I_exacta"]:,.0f}')
                        st.caption(f'T₁ con el lado exacto = {r_dim["T_exacto"]:.5f} s (el objetivo). El redondeo va hacia arriba, '
                                   'así el periodo queda igual o menor.')
                    if not por_columna:
                        if st.button('Aplicar estas secciones a la tabla de niveles', key='dim_aplicar'):
                            df_nuevo = edit.dropna(how='all').reset_index(drop=True)
                            textos = r_dim['textos'] if r_dim['meta'] == 'distorsion' else {i: r_dim['texto'] for i in r_dim['niveles']}
                            for i, txt in textos.items():
                                df_nuevo.loc[i - 1, 'seccion'] = txt
                            st.session_state['df'] = df_nuevo
                            st.session_state['ver'] += 1
                            st.session_state.pop('dim_res', None)
                            st.rerun()
                    else:
                        st.caption('Con la captura por columna, escribe estas secciones en la tabla de columnas.')
