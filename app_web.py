"""Versión web (Streamlit) del Análisis Modal Espectral.  Ejecutar:  streamlit run app_web.py"""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

import ame_ntc as ntc
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
         'artic': 'Ejes cuya columna está articulada en su base (normalmente solo en el nivel 1): "2". Vacío = todas empotradas.',
         'cargas': 'Un valor para todas las crujías o uno por crujía: "4.7 3.2".',
         'W': 'Opcional: peso total del nivel; sustituye a la carga por crujía.'}
SUBGRUPOS = {'B2': 'B2 · riesgo I (ASCE 7)', 'B1': 'B1 · riesgo II', 'A2': 'A2 · riesgo III', 'A1': 'A1 · riesgo IV'}
VIGAS = ['Rígidas (marco de cortante)', 'Flexibles (marco plano)']
CLASES = ["Clase 1 (14000·√f'c)", "Clase 2 (8000·√f'c)"]
MODOS = ['Por secciones y cargas', 'Matrices K y M (ya calculadas)']
UNIDADES = ['kg, cm', 't, cm']
CLAVES_MANUAL = ['K_txt', 'M_txt', 'alturas_txt', 'pesos_txt']
SOLO_SECCIONES = ('crujias', 'E', 'fc', 'clase', 'vigas', 'direccion')  # no aplican al modo de matrices
ESPECTRO = ['a0', 'c', 'Ta', 'Tb', 'k', 'Ts']
DEFECTOS = dict(crujias='', subgrupo='B1', intensidad='Base de diseño', direccion='X', estado='Automático', Ks='0.25',
                Q='4', k1='1.0', combinacion='SRSS', zeta='', material='Concreto', E='', fc='', clase=CLASES[0],
                vigas=VIGAS[0], factor_Fu='1.1', mult_V='1', lim_dl='0.004', lim_sv='0.03', lim_oi='0.005',
                irreg=[], fuerte_torsion=False, fuerte_elev=False, zona='', regularidad='Regular', b_planta='',
                R1_otros=True, proyecto='', autor='', modo=MODOS[0], K_txt='', M_txt='', unidades=UNIDADES[0],
                alturas_txt='', pesos_txt='', **{k: '' for k in ESPECTRO})
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
</style>
<div class="cabecera"><h1>Análisis Modal Espectral</h1>
<p>Marcos planos · NTC-S 2023 · fuerzas sísmicas por nivel, cortante basal, factor de escala y revisiones</p></div>
""", unsafe_allow_html=True)


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
        d['subgrupo'] = grupo + '1'
        d.setdefault('intensidad', 'Infrecuente' if sismo == 'I' else 'Base de diseño')
    for k in CAMPOS:
        if matrices and k in SOLO_SECCIONES and k not in d:
            continue  # un ejemplo con K y M no pisa los datos de secciones que ya capturaste
        st.session_state[k] = d.get(k, DEFECTOS[k])
    st.session_state['estado'] = d.get('estado') if d.get('estado') in ame.ESTADOS else 'Automático'
    esp = {k: dict(v) for k, v in (d.get('espectros') or {}).items()}
    if not esp:  # campos sueltos del formato anterior: pertenecen a la intensidad elegida
        esp = {st.session_state['intensidad']: {k: str(d.get(k, '')) for k in ESPECTRO}}
    st.session_state['_esp'] = esp
    st.session_state['_int_prev'] = st.session_state['intensidad']
    for k in ESPECTRO:
        st.session_state[k] = esp.get(st.session_state['intensidad'], {}).get(k, '')
    if isinstance(d.get('niveles'), list):
        st.session_state['df'] = df_de(d['niveles'])
    st.session_state['ver'] = st.session_state.get('ver', 0) + 1


def cambiar_intensidad():
    """Guarda el espectro de la intensidad anterior y muestra el de la nueva."""
    esp = st.session_state['_esp']
    esp[st.session_state['_int_prev']] = {k: st.session_state[k] for k in ESPECTRO}
    nueva = st.session_state['intensidad']
    for k in ESPECTRO:
        st.session_state[k] = esp.get(nueva, {}).get(k, '')
    st.session_state['_int_prev'] = nueva


if 'df' not in st.session_state:
    cargar(next(iter(EJEMPLOS.values())))
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

    st.markdown('### Objetivo de diseño (NTC-S 2023)')
    st.selectbox('Grupo / subgrupo', list(SUBGRUPOS), key='subgrupo', format_func=SUBGRUPOS.get,
                 help='Grupo B: B1 y B2. Grupo A: A1 y A2. El riesgo equivale a la categoría de ASCE 7.')
    st.selectbox('Intensidad sísmica', ntc.INTENSIDADES, key='intensidad', on_change=cambiar_intensidad,
                 help='Normalmente se usa el sismo base de diseño. El espectro SASID se guarda por intensidad.')
    grupo_sel = ntc.SUBGRUPOS[st.session_state['subgrupo']]
    objetivo = ntc.objetivo_diseno(grupo_sel, st.session_state['intensidad'])
    st.markdown(f'<div class="objetivo"><b>{objetivo["desempeno"]}</b><br>{objetivo["espectro"]}<br>'
                f'<small>{objetivo["revisa"]}</small></div>', unsafe_allow_html=True)
    st.selectbox('Estado de cálculo', ['Automático', *ame.ESTADOS], key='estado',
                 help='Automático: el que corresponde al objetivo de diseño. Puedes forzar otro estado.')
    st.text_input('Ks (limitación de daños)', key='Ks', help='Sa(BD)·Ks. En tus hojas Ks = 0.25.')
    st.text_input('Q (varios valores separados por espacio)', key='Q',
                  help='Se usa el primero para las fuerzas. En ocupación inmediata se usa Q = 1.')
    st.caption('Q: 4 ductilidad alta (daño severo) · 3 media (daño importante) · 2 media (daño medio) · '
               '1.5 baja (poco daño) · 1 elástica.')
    st.text_input('k1', key='k1')
    st.caption('k1: 0.8 con menos de 3 crujías resistentes · 1.0 con 3 o más · 1.25 en sistemas duales.')
    st.radio('Dirección del sismo', ['X', 'Y'], key='direccion', horizontal=True,
             help='En una sección b x h, h está en la dirección X y b en la Y. Con Y se usa la inercia respecto al otro eje.',
             disabled=manual)

    st.markdown(f'### Espectro SASID · {st.session_state["intensidad"]}')
    c1, c2 = st.columns(2)
    for i, (et, k) in enumerate([('a0 (cm/s²)', 'a0'), ('c (cm/s²)', 'c'), ('Ta (s)', 'Ta'),
                                 ('Tb (s)', 'Tb'), ('k', 'k'), ('Ts (s)', 'Ts')]):
        (c1 if i % 2 == 0 else c2).text_input(et, key=k)
    st.caption('Captura el espectro de la intensidad elegida; el de cada intensidad se conserva al cambiar.')

    st.markdown('### Análisis')
    st.selectbox('Combinación modal', list(COMBINACIONES), key='combinacion',
                 help='SRSS: raíz de la suma de cuadrados. CQC: considera la correlación entre modos cercanos. '
                      'Suma absoluta: cota superior.')
    st.selectbox('Vigas', VIGAS, key='vigas', disabled=manual,
                 help='Rígidas: modelo de cortante. Flexibles: marco plano con rigidez de columnas y vigas.')

    st.markdown('### Material')
    st.selectbox('Material', list(ame.MATERIALES), key='material')
    c1, c2 = st.columns(2)
    c1.text_input('E (kg/cm²)', key='E', disabled=manual,
                  help="Si lo dejas vacío: concreto 14000·√f'c (u 8000·√f'c), acero 2,040,000.")
    concreto = st.session_state['material'] == 'Concreto'
    c2.text_input("f'c (kg/cm²)", key='fc', disabled=(not concreto) or manual)
    st.selectbox('Clase del concreto', CLASES, key='clase', disabled=(not concreto) or manual)
    c1, c2 = st.columns(2)
    c1.text_input('Factor Fu (FC)', key='factor_Fu')
    c2.text_input('V estático ×', key='mult_V')

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
        st.selectbox('Zona geotécnica', ['', 'I', 'II', 'III'], key='zona',
                     format_func=lambda z: z or '—', help='Para verificar si el método estático es aplicable.')
        st.radio('Regularidad', ['Regular', 'Irregular'], key='regularidad', horizontal=True)
        st.text_input('Dimensión de la planta perpendicular al análisis, b (m)', key='b_planta',
                      help='Para calcular la excentricidad accidental y los momentos torsionantes.')

    st.markdown('### Límites de distorsión')
    st.caption('Los γmáx dependen del sistema (tablas 4.3.x de la norma): confírmalos.')
    c1, c2, c3 = st.columns(3)
    c1.text_input('Daños', key='lim_dl', help='0.002 ó 0.004')
    c2.text_input('Vida', key='lim_sv')
    c3.text_input('Ocup.', key='lim_oi', help='Vacío = sin límite')

# ------------------------------------------------------------------ estructura
st.radio('¿Cómo defines la estructura?', MODOS, key='modo', horizontal=True,
         help='Por secciones y cargas el programa arma K y M. Si el ejercicio ya te da K y M, usa las matrices.')
manual = st.session_state['modo'] == MODOS[1]

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

datos = {k: st.session_state[k] for k in CAMPOS if k not in ESPECTRO and k not in CLAVES_MANUAL}
datos['modo'] = 'matrices' if manual else 'secciones'
datos.update(K=st.session_state['K_txt'], M=st.session_state['M_txt'], alturas=st.session_state['alturas_txt'],
             pesos=st.session_state['pesos_txt'])
datos['vigas'] = 'flexibles' if flexibles else 'rigidas'
esp = {k: dict(v) for k, v in st.session_state['_esp'].items()}
esp[st.session_state['intensidad']] = {k: st.session_state[k] for k in ESPECTRO}
datos['espectros'] = esp
datos['niveles'] = [{k: ('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for k, v in fila.items()}
                    for _, fila in edit.dropna(how='all').iterrows()]
firma = hashlib.md5(json.dumps(datos, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

try:
    modelo = ame.preparar(datos)
except ValueError as e:
    st.error(f'Revisa los datos: {e}')
    st.stop()

if manual:
    st.info('Con matrices no hay geometría que dibujar: revisa K, M, las alturas y los pesos derivados en las pestañas '
            'de resultados.')
else:
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
nm = res['normativa']


def kpi(col, titulo, num, sub):
    col.markdown(f'<div class="kpi"><div class="t">{titulo}</div><div class="n">{num}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


st.markdown('### Resultados')
st.markdown(f'<div class="objetivo"><b>Grupo {p["subgrupo"]} · sismo {p["intensidad"].lower()}</b> → '
            f'{p["objetivo"]["desempeno"]} · estado de cálculo: <b>{p["estado"]}</b><br>'
            f'<small>{p["objetivo"]["espectro"]} · {p["objetivo"]["revisa"]}</small></div>', unsafe_allow_html=True)
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


tabs = st.tabs(['Fuerzas por nivel', 'Revisiones', 'Normativa', 'Modos', 'Detalle completo', 'Formas modales', 'Espectro'])
with tabs[0]:
    st.pyplot(ame.fig_diagramas(res))
    t = res['tabla_din']
    st.markdown('**Análisis dinámico (modal espectral)**')
    mostrar(pd.DataFrame({'Nivel': t['nivel'], 'h (m)': t['h'], 'F (t)': t['F'], 'Fu (t)': t['Fu'],
                          'Vu (t)': t['Vu'], 'Mvu (t·m)': t['Mvu']}))
    e = res['tabla_est']
    st.markdown('**Análisis estático (para el factor de escala)**')
    mostrar(pd.DataFrame({'Nivel': e['nivel'], 'h (m)': e['h'], 'Wi (t)': e['Wi'], 'Wihi (t·m)': e['hWi'],
                          'Fi (t)': e['Fi'], 'Fu (t)': e['Fu'], 'Vu (t)': e['Vu'], 'Mvu (t·m)': e['Mvu']}))
with tabs[1]:
    rev = res['revision']
    st.markdown('**Distorsiones de entrepiso por estado límite** (★ = estado de diseño)')
    lims = [[(e['nombre'] + (' ★' if k_ == res['clave'] else '')),
             e['limite_base'] if e['limite_base'] is not None else np.nan, e['gamma_c'],
             e['limite'] if e['limite'] is not None else np.nan] for k_, e in rev['estados'].items()]
    mostrar(pd.DataFrame(lims, columns=['Estado límite', 'γmáx', 'γc', 'Límite aplicado']), 4)
    enc, filas = ame.tabla_distorsiones(res)
    mostrar(pd.DataFrame(filas, columns=enc), 5)
    st.caption('Seguridad de vida y ocupación inmediata: desplazamientos × FC·Q·R′ con R′ del periodo fundamental. '
               'Limitación de daños: Sa·Ks, sin factor de carga.')
    st.pyplot(ame.fig_distorsiones(res))
    pdt = ame.tabla_pdelta(res)
    if pdt:
        st.markdown('**Efectos P-Δ (θ = P·δ / (V·h))**')
        mostrar(pd.DataFrame(pdt[1], columns=pdt[0]))
        st.caption(f'Irregularidad de masa: {rev["irreg_masa"] or "ninguna"} · piso blando: '
                   f'{rev["irreg_rigidez"] or "ninguno"}. θ ≤ 0.10 se puede ignorar; θ > 0.25 posible inestabilidad.')
with tabs[2]:
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
        st.caption('Sin excentricidad estática eₛ ni efecto bidireccional 100% + 30%.')
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
with tabs[3]:
    qr = {f[0]: f[-1] for f in res['QR'][::len(res['Qs'])]}
    mostrar(pd.DataFrame({'Modo': np.arange(1, n + 1), 'T (s)': res['T'], 'f (Hz)': res['f'], 'ω (rad/s)': res['w'],
                          'γ': res['gam'], "Sa/(Q'R') (cm/s²)": res['Sa_modal'][:, 0],
                          "Q·R'": [qr[j + 1] for j in range(n)]}))
with tabs[4]:
    st.code(ame.reporte_texto(res), language=None)
with tabs[5]:
    st.pyplot(ame.fig_modal(res))
with tabs[6]:
    st.pyplot(ame.fig_espectro(res))
