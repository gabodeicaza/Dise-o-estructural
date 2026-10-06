"""Versión web (Streamlit) del Análisis Modal Espectral.  Ejecutar:  streamlit run app_web.py"""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

import ame_pdf
import ame_sismico as ame
from ame_revision import COMBINACIONES
from ame_secciones import AYUDA_SECCIONES

NAVY, AZUL_OSC, AZUL, AZUL_MED, AZUL_CLARO, PALE, BG = (
    '#0B2545', '#13315C', '#1D4E89', '#3E7CB1', '#81A4CD', '#DCE9F7', '#F1F6FC')

COLS = ['h', 'ejes', 'seccion', 'viga', 'artic', 'cargas', 'W']
ETIQ = {'h': 'h entrepiso (m)', 'ejes': 'Columnas en ejes', 'seccion': 'Sección de columnas (cm)',
        'viga': 'Viga (cm)', 'artic': 'Articuladas en la base (ejes)', 'cargas': 'Carga por crujía (t/m)',
        'W': 'o Peso W (t)'}
AYUDA = {'h': 'Altura del entrepiso en metros.',
         'ejes': 'Ejes que tienen columna en este nivel: "todos" (o vacío), "1 3" o "1-3".',
         'seccion': AYUDA_SECCIONES + '. Una sola para todo el nivel o una por columna: "25x50 Ø40 25x50".',
         'viga': 'Viga de este piso (solo con vigas flexibles). b x h con h vertical: 30x60. También T:50x10x60x15 o H:...',
         'artic': 'Ejes cuya columna está articulada en su base (normalmente solo en el nivel 1): "2".',
         'cargas': 'Un valor para todas las crujías o uno por crujía: "4.7 3.2".',
         'W': 'Opcional: peso total del nivel; sustituye a la carga por crujía.'}
TIPOS = ['Grupo B', 'Grupo A - sismo base', 'Grupo A - sismo infrecuente']
VIGAS = ['Rígidas (marco de cortante)', 'Flexibles (marco plano)']
CLASES = ["Clase 1 (14000·√f'c)", "Clase 2 (8000·√f'c)"]
DEFECTOS = dict(crujias='', tipo=TIPOS[0], direccion='X', estado=ame.ESTADOS[0], Ks='0.25', Q='4', k1='1.0',
                combinacion='SRSS', zeta='0.05', material='Concreto', E='', fc='', clase=CLASES[0], vigas=VIGAS[0],
                factor_Fu='1.1', mult_V='1', lim_dl='0.004', lim_sv='0.03', lim_oi='',
                a0='', c='', Ta='', Tb='', k='', Ts='', proyecto='', autor='')
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
h3 {{color:{AZUL_OSC}}}
</style>
<div class="cabecera"><h1>Análisis Modal Espectral</h1>
<p>Marcos planos · fuerzas sísmicas por nivel, cortante basal, factor de escala y revisiones</p></div>
""", unsafe_allow_html=True)


def df_de(niveles):
    return pd.DataFrame([{k: str(r.get(k, '')) for k in COLS} for r in niveles], dtype=object)


def cargar(d):
    """Pone un proyecto (ejemplo o archivo) en el estado de la página."""
    d = ame.convertir_legacy(d)
    if not isinstance(d.get('niveles'), list) or 'crujias' not in d:
        raise ValueError('Formato de proyecto no reconocido.')
    for k in CAMPOS:
        st.session_state[k] = d.get(k, DEFECTOS[k])
    st.session_state['df'] = df_de(d['niveles'])
    st.session_state['ver'] = st.session_state.get('ver', 0) + 1


if 'df' not in st.session_state:
    cargar(next(iter(EJEMPLOS.values())))

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

    st.markdown('### Edificación y sismo')
    st.selectbox('Tipo', TIPOS, key='tipo')
    st.radio('Dirección del sismo', ['X', 'Y'], key='direccion', horizontal=True,
             help='En una sección b x h, h está en la dirección X y b en la Y. Con Y se usa la inercia respecto al otro eje.')
    st.selectbox('Estado límite de diseño', ame.ESTADOS, key='estado',
                 help='Seguridad de vida: Sa/(Q′R′). Ocupación inmediata: Q′ = 1. Limitación de daños: Sa·Ks.')
    st.text_input('Ks (solo limitación de daños)', key='Ks', disabled=st.session_state['estado'] != 'Limitación de daños',
                  help='Factor del espectro para limitación de daños (0.25 en tus hojas).')
    st.text_input('Q (varios valores separados por espacio)', key='Q',
                  help='Se usa el primero para las fuerzas. En ocupación inmediata se usa Q′ = 1.')
    st.text_input('k1', key='k1')

    st.markdown('### Análisis')
    st.selectbox('Combinación modal', list(COMBINACIONES), key='combinacion',
                 help='SRSS: raíz de la suma de cuadrados. CQC: considera la correlación entre modos cercanos. '
                      'Suma absoluta: cota superior.')
    st.text_input('Amortiguamiento ζ (solo CQC)', key='zeta', disabled=st.session_state['combinacion'] != 'CQC')
    st.selectbox('Vigas', VIGAS, key='vigas',
                 help='Rígidas: modelo de cortante. Flexibles: marco plano con rigidez de columnas y vigas.')

    st.markdown('### Material')
    st.selectbox('Material', list(ame.MATERIALES), key='material')
    c1, c2 = st.columns(2)
    c1.text_input('E (kg/cm²)', key='E',
                  help="Si lo dejas vacío: concreto 14000·√f'c (u 8000·√f'c), acero 2,040,000.")
    concreto = st.session_state['material'] == 'Concreto'
    c2.text_input("f'c (kg/cm²)", key='fc', disabled=not concreto)
    st.selectbox('Clase del concreto', CLASES, key='clase', disabled=not concreto)
    c1, c2 = st.columns(2)
    c1.text_input('Factor Fu (FC)', key='factor_Fu')
    c2.text_input('V estático ×', key='mult_V')

    st.markdown('### Límites de distorsión')
    st.caption('Valores de ejemplo: confírmalos con tu reglamento.')
    c1, c2, c3 = st.columns(3)
    c1.text_input('Daños', key='lim_dl')
    c2.text_input('Vida', key='lim_sv')
    c3.text_input('Ocup.', key='lim_oi', help='Vacío = sin límite')

    st.markdown('### Espectro del sitio (SASID)')
    c1, c2 = st.columns(2)
    for i, (et, k) in enumerate([('a0 (cm/s²)', 'a0'), ('c (cm/s²)', 'c'), ('Ta (s)', 'Ta'),
                                 ('Tb (s)', 'Tb'), ('k', 'k'), ('Ts (s)', 'Ts')]):
        (c1 if i % 2 == 0 else c2).text_input(et, key=k)

# ------------------------------------------------------------------ estructura
st.markdown('### Estructura (niveles de abajo hacia arriba)')
st.text_input('Crujías (m), de izquierda a derecha', key='crujias',
              help='Anchos de cada crujía separados por espacio: "4 5" define 2 crujías y 3 ejes (1, 2 y 3).')
flexibles = st.session_state['vigas'].startswith('Flex')
conf = {k: st.column_config.TextColumn(ETIQ[k], help=AYUDA[k]) for k in COLS}
if not flexibles:
    conf['viga'] = None  # oculta la columna
edit = st.data_editor(st.session_state['df'], key=f'niv_{st.session_state["ver"]}', num_rows='dynamic',
                      width='stretch', column_config=conf)
st.caption('Una fila por nivel (la primera es la base; agrega filas con el + de la tabla). Las columnas se ubican '
           'por eje: si un nivel no tiene columna en algún eje, no lo pongas en "Columnas en ejes". '
           'Pasa el cursor sobre los encabezados para ver ejemplos de sintaxis.')

datos = {k: st.session_state[k] for k in CAMPOS}
datos['vigas'] = 'flexibles' if flexibles else 'rigidas'
datos['niveles'] = [{k: ('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for k, v in fila.items()}
                    for _, fila in edit.dropna(how='all').iterrows()]
firma = hashlib.md5(json.dumps(datos, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

try:
    modelo = ame.preparar_modelo(datos)
except ValueError as e:
    st.error(f'Revisa los datos: {e}')
    st.stop()

with st.expander('Vista previa de la estructura', expanded=True):
    st.pyplot(ame.fig_estructura(modelo))
    st.caption('Revisa que ejes, secciones, articulaciones, vigas y cargas sean los que esperas antes de leer los resultados.')

try:
    res = ame.calcular(modelo)
except (np.linalg.LinAlgError, ValueError) as e:
    st.error(f'No se pudo calcular: {e if isinstance(e, ValueError) else "las matrices son singulares. Revisa secciones y alturas."}')
    st.stop()

# ------------------------------------------------------------------ resultados
p, n = res['p'], res['p']['n']
fe = res['factor_escala']


def kpi(col, titulo, num, sub):
    col.markdown(f'<div class="kpi"><div class="t">{titulo}</div><div class="n">{num}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


st.markdown('### Resultados')
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
kpi(k[2], 'CORTANTE BASAL ESTÁTICO', f'{res["V_est"]:.3f} t', f'Sa/(Q′R′) máx = {res["Fmax_est"]:.4f} g')
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


def mostrar(df, dec=4):
    st.dataframe(df, hide_index=True, width='stretch',
                 column_config={c: st.column_config.NumberColumn(format=f'%.{dec}f') for c in df.columns[1:]
                                if pd.api.types.is_numeric_dtype(df[c])})


tabs = st.tabs(['Fuerzas por nivel', 'Revisiones', 'Modos', 'Detalle completo', 'Formas modales', 'Espectro'])
with tabs[0]:
    t = res['tabla_din']
    st.markdown('**Análisis dinámico (modal espectral)**')
    mostrar(pd.DataFrame({'Nivel': t['nivel'], 'h (m)': t['h'], 'F (t)': t['F'], 'Fu (t)': t['Fu'],
                          'Vu (t)': t['Vu'], 'Mvu (t·m)': t['Mvu']}))
    e = res['tabla_est']
    st.markdown('**Análisis estático (para el factor de escala)**')
    mostrar(pd.DataFrame({'Nivel': e['nivel'], 'h (m)': e['h'], 'Wi (t)': e['Wi'], 'Wihi (t·m)': e['hWi'],
                          'Fi (t)': e['Fi'], 'Fu (t)': e['Fu'], 'Vu (t)': e['Vu'], 'Mvu (t·m)': e['Mvu']}))
    st.pyplot(ame.fig_fuerzas(res))
with tabs[1]:
    rev = res['revision']
    st.markdown('**Distorsiones de entrepiso por estado límite**')
    enc, filas = ame.tabla_distorsiones(res)
    mostrar(pd.DataFrame(filas, columns=enc), 5)
    st.caption('Seguridad de vida y ocupación inmediata dan la misma distorsión (desplazamientos iguales): '
               'solo cambia el límite que debe cumplirse.')
    st.pyplot(ame.fig_distorsiones(res))
    pdt = ame.tabla_pdelta(res)
    if pdt:
        st.markdown('**Efectos P-Δ (θ = P·δ / (V·h))**')
        mostrar(pd.DataFrame(pdt[1], columns=pdt[0]))
        st.caption(f'Irregularidad de masa: {rev["irreg_masa"] or "ninguna"} · piso blando: '
                   f'{rev["irreg_rigidez"] or "ninguno"}. θ ≤ 0.10 se puede ignorar; θ > 0.25 posible inestabilidad.')
with tabs[2]:
    qr = {f[0]: f[-1] for f in res['QR'][::len(res['Qs'])]}
    mostrar(pd.DataFrame({'Modo': np.arange(1, n + 1), 'T (s)': res['T'], 'f (Hz)': res['f'], 'ω (rad/s)': res['w'],
                          'γ': res['gam'], "Sa/(Q'R') (cm/s²)": res['Sa_modal'][:, 0],
                          "Q·R'": [qr[j + 1] for j in range(n)]}))
with tabs[3]:
    st.code(ame.reporte_texto(res), language=None)
with tabs[4]:
    st.pyplot(ame.fig_modal(res))
with tabs[5]:
    st.pyplot(ame.fig_espectro(res))
