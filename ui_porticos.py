"""Pestaña «6 · Pórtico y armado»: envolventes de M y V por viga (sismo de ida, regreso y gravedad) y acero requerido."""
import io

import numpy as np
import pandas as pd
import streamlit as st

import ame_parcial3 as p3
import ame_porticos as pt

COLS = ['Nivel', 'h (m)', 'Col b (cm)', 'Col h (cm)', 'Viga b (cm)', 'Viga h (cm)', 'CM (t/m)', 'CV máx (t/m)', 'CV acc (t/m)',
        'F sismo (t)']
FUENTES_F = {'Dinámico (SRSS)': 'tabla_din', 'Dinámico escalado al estático': 'tabla_din_esc', 'Estático': 'tabla_est'}


def _defecto(n=3):
    return pd.DataFrame({'Nivel': range(1, n + 1), 'h (m)': [3.0] * n, 'Col b (cm)': [40.0] * n, 'Col h (cm)': [40.0] * n,
                         'Viga b (cm)': [25.0] * n, 'Viga h (cm)': [50.0] * n, 'CM (t/m)': [2.0] * n,
                         'CV máx (t/m)': [0.5] * n, 'CV acc (t/m)': [0.3] * n,
                         'F sismo (t)': [3.0 * (i + 1) for i in range(n)]})


def _traer(res_sismo, crujias, fuente):
    t = res_sismo[FUENTES_F[fuente]]
    F = np.asarray(t['F'], float)[::-1]
    alt = np.asarray(res_sismo.get('alturas_acum'), float)
    h = np.diff(np.concatenate([[0.0], alt]))
    df = st.session_state.get('p6_df', _defecto(len(F)))
    if len(df) != len(F):
        df = _defecto(len(F))
    df = df.copy()
    df['h (m)'] = h
    df['F sismo (t)'] = F
    st.session_state['p6_df'] = df
    if crujias:
        st.session_state['p6_spans'] = crujias


def _fig(f):
    st.pyplot(f, width='stretch')
    import matplotlib.pyplot as plt
    plt.close(f)


def mostrar(res_sismo=None, crujias_txt=''):
    st.caption('Análisis de un pórtico plano de concreto con CM, CV y sismo. El sismo se resuelve de ida (+) y de regreso (−) '
               'y se combina con la gravedad; la envolvente da el momento y el cortante máximos de cada viga, con los que se '
               'calcula el acero y se propone el armado (ductilidad baja, apuntes pp. 31–35).')
    if 'p6_df' not in st.session_state:
        st.session_state['p6_df'] = _defecto()
    st.session_state.setdefault('p6_spans', '4 5')
    c1, c2, c3 = st.columns(3)
    c1.text_input('Crujías (m), de izquierda a derecha', key='p6_spans')
    fuente = c2.selectbox('Fuerzas sísmicas tomadas de', list(FUENTES_F), key='p6_fuente')
    ok_res = res_sismo is not None and FUENTES_F[fuente] in res_sismo
    c3.write('')
    c3.button('Traer alturas, crujías y fuerzas del análisis', disabled=not ok_res, key='p6_traer',
              on_click=lambda: _traer(res_sismo, crujias_txt, fuente),
              help='Copia las alturas y las fuerzas por nivel (sin factor de carga) del análisis sísmico y las crujías de la '
                   'pestaña Estructura. Las secciones y las cargas las capturas aquí.')
    c1, c2, c3, c4 = st.columns(4)
    fc = c1.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='p6_fc')
    fy = c2.number_input('fy (kg/cm²)', 2000.0, 6000.0, 4200.0, key='p6_fy')
    clase = c3.selectbox('E del concreto', ["14000·√f′c (clase 1)", "8000·√f′c (clase 2)"], key='p6_clase')
    E = (14000 if clase.startswith('14') else 8000) * np.sqrt(fc)
    c4.metric('E (kg/cm²)', f'{E:,.0f}')
    c1, c2, c3, c4 = st.columns(4)
    grupo = c1.radio('Grupo', ['B', 'A'], horizontal=True, key='p6_grupo')
    base = c2.radio('Base de las columnas', ['Empotrada', 'Articulada'], horizontal=True, key='p6_base')
    fi_vig = c3.number_input('Factor de I en vigas (agrietamiento)', 0.1, 1.0, 1.0, 0.05, key='p6_fiv',
                             help='1.0 = sección gruesa. Si tu profesor usa inercia agrietada, cámbialo.')
    fi_col = c4.number_input('Factor de I en columnas', 0.1, 1.0, 1.0, 0.05, key='p6_fic')
    c1, c2, c3 = st.columns(3)
    paño = c1.checkbox('Diseñar en el paño de la columna (como ETABS)', True, key='p6_paño')
    estab = c2.checkbox('Incluir estabilidad (0.9 CM ± sismo)', False, key='p6_estab')
    r_rec = c3.number_input('r (cm) a centroide del acero', 2.0, 15.0, 5.0, key='p6_r')
    df = st.data_editor(st.session_state['p6_df'], num_rows='dynamic', hide_index=True, width='stretch', key='p6_editor')
    df = df.dropna()
    if len(df) < 1:
        st.info('Captura al menos un nivel.')
        return
    try:
        spans = [float(x) for x in st.session_state['p6_spans'].replace(',', ' ').split()]
        if not spans or min(spans) <= 0:
            raise ValueError
    except ValueError:
        st.error('Escribe las crujías como números positivos separados por espacios, por ejemplo «4 5».')
        return
    n = len(df)
    try:
        res = pt.resolver(spans, list(df['h (m)']), list(zip(df['Col b (cm)'], df['Col h (cm)'])),
                          list(zip(df['Viga b (cm)'], df['Viga h (cm)'])), E,
                          dict(CM=list(df['CM (t/m)']), CV=list(df['CV máx (t/m)']), CVacc=list(df['CV acc (t/m)'])),
                          list(df['F sismo (t)']), base, fi_col, fi_vig)
    except np.linalg.LinAlgError:
        st.error('El pórtico es inestable o inconsistente: revisa las secciones y los apoyos.')
        return
    combos = pt.combinaciones(grupo, estab)
    env, mats = pt.envolvente(res, combos)
    ux, dist = pt.desplazamientos(res)
    F = list(df['F sismo (t)'])
    cargas = dict(CM=list(df['CM (t/m)']), CV=list(df['CV máx (t/m)']), CVacc=list(df['CV acc (t/m)']))
    with st.expander('Combinaciones de carga usadas'):
        st.dataframe(pd.DataFrame([[k] + list(v) for k, v in combos.items()],
                                  columns=['Combinación', 'CM', 'CV máx', 'CV acc', 'Sismo']), hide_index=True, width='stretch')
        st.caption('Sismo de ida (+) y de regreso (−): las fuerzas laterales del análisis se aplican en un sentido y en el '
                   'contrario; la envolvente toma, en cada punto, el mayor y el menor momento y cortante de todas las combinaciones.')
    tabs = st.tabs(['Modelo', 'Gravedad', 'Sismo de ida', 'Sismo de regreso', 'Envolvente', 'Deformadas', 'Diseño por viga',
                    'Resumen de acero'])
    tipo_sel = lambda clave: 'Momento M' if st.session_state.get(clave, 'Momento M') == 'Momento M' else 'Cortante V'

    def selector(clave):
        return 'M' if st.radio('Diagrama', ['Momento M', 'Cortante V'], horizontal=True, key=clave) == 'Momento M' else 'V'

    with tabs[0]:
        _fig(pt.figura_cargas(res, F, cargas))
        st.caption('Distorsiones del sismo de ida sin factor de carga: ' + ' · '.join(
            f'N{i + 1}: {d:.5f}' for i, d in enumerate(dist)) + '. Revisa con tu análisis sísmico los límites γ.')
    with tabs[1]:
        tp = selector('p6_t1')
        _fig(pt.figura_diagrama(res, mats['Gravedad'], tp, f'Gravedad {combos["Gravedad"][:2]}', None))
    with tabs[2]:
        tp = selector('p6_t2')
        _fig(pt.figura_diagrama(res, mats['Sismo ida (+)'], tp, 'Sismo de ida: 1.1(CM + CVacc) + 1.1 S'))
    with tabs[3]:
        tp = selector('p6_t3')
        _fig(pt.figura_diagrama(res, mats['Sismo regreso (−)'], tp, 'Sismo de regreso: 1.1(CM + CVacc) − 1.1 S'))
    with tabs[4]:
        tp = selector('p6_t4')
        mx = {k: dict(x=v['x'], M=v['Mmax'], V=v['Vmax']) for k, v in env.items()}
        mn = {k: dict(x=v['x'], M=v['Mmin'], V=v['Vmin']) for k, v in env.items()}
        _fig(pt.figura_diagrama(res, mx, tp, 'Envolvente (máximo + mínimo de todas las combinaciones)', extra=mn))
        st.caption('Amarillo/azul: valores positivos (en vigas, tensión abajo); rojo/naranja: negativos (tensión arriba).')
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            _fig(pt.figura_deformada(res, 'S', 1.0, 'Sismo de ida'))
        with c2:
            _fig(pt.figura_deformada(res, 'S', -1.0, 'Sismo de regreso'))
    col_h = list(df['Col h (cm)'])
    dis = pt.resumen_vigas(env, list(zip(df['Viga b (cm)'], df['Viga h (cm)'])), fc, fy, col_h, r_rec, paño)
    with tabs[6]:
        claves = list(dis)
        nombre = st.selectbox('Viga', [f'Piso {k[1]} · crujía {k[2]}' for k in claves], key='p6_viga')
        d = dis[claves[[f'Piso {k[1]} · crujía {k[2]}' for k in claves].index(nombre)]]
        (st.success if d['ok'] else st.error)(
            f'Viga {d["b"]:g}×{d["h"]:g} cm, L = {d["L"]:g} m: ' + ('el acero cabe y cumple.' if d['ok'] else 'revisa la sección (As > As,máx o no alcanza).'))
        _fig(pt.figura_viga(env, d, mats))
        filas = []
        for s in d['secciones']:
            a = s['armado']
            filas.append([s['posicion'].capitalize(), s['lecho'], s['x'], s['Mu'], s['As'] if s['As'] is not None else np.nan,
                          a['armado'] if a else '—', a['As'] if a else np.nan, a['MR'] if a else np.nan,
                          'sí' if (a and a['cabe_1_lecho']) else ('2 lechos' if a else '—')])
        st.dataframe(pd.DataFrame(filas, columns=['Sección', 'Lecho', 'x (m)', 'Mu (t·m)', 'As calc (cm²)', 'Armado', 'As real (cm²)',
                                                  'MR (t·m)', 'Cabe en 1 lecho']), hide_index=True, width='stretch',
                     column_config={c: st.column_config.NumberColumn(format='%.2f') for c in
                                    ['x (m)', 'Mu (t·m)', 'As calc (cm²)', 'As real (cm²)', 'MR (t·m)']})
        c = d['cortante']
        st.markdown(f'**Estribos:** Vu máx = {d["Vu"]:.2f} t → ' + (f'E#3 @ {c["s_usar"]} cm' if c.get('s_usar') else 'excede el límite'))
        with st.expander('Cálculo desarrollado de cada sección', expanded=True):
            for s in d['secciones']:
                st.markdown(f'**{s["posicion"].capitalize()} · lecho {s["lecho"]}**')
                for lin in s['lineas']:
                    st.markdown(f'- {lin}')
                if s['armado']:
                    a = s['armado']
                    st.markdown(f'- Propuesta: **{a["armado"]}** = {a["As"]:.2f} cm² ≥ {s["As"]:.2f}; MR = {a["MR"]:.2f} t·m '
                                f'≥ Mu = {s["Mu"]:.2f} t·m')
        with st.expander('Cortante desarrollado'):
            for lin in c['pasos']:
                st.markdown(f'- {lin}')
        st.caption('El armado superior en cada apoyo y el inferior corrido salen de las seis secciones. Verifica siempre MR '
                   'con el área real elegida. Al menos 2 varillas corridas por lecho (apuntes p. 32).')
    with tabs[7]:
        filas = []
        for k, d in dis.items():
            g = lambda pn, lc: next(s for s in d['secciones'] if s['posicion'] == pn and s['lecho'] == lc)
            fila = [k[1], k[2], f'{d["b"]:g}×{d["h"]:g}', d['L']]
            for pn, lc in (('izquierdo', 'superior'), ('centro', 'inferior'), ('derecho', 'superior')):
                s = g(pn, lc)
                fila += [s['Mu'], s['As'] if s['As'] is not None else np.nan, s['armado']['armado'] if s['armado'] else '—']
            fila.append(d['inferior_corrido']['armado']['armado'] if d['inferior_corrido']['armado'] else '—')
            fila.append(f'E#3@{d["cortante"]["s_usar"]}' if d['cortante'].get('s_usar') else '—')
            fila.append('✔' if d['ok'] else '✘')
            filas.append(fila)
        enc = ['Piso', 'Crujía', 'b×h', 'L (m)', 'Mu− izq', 'As izq', 'Sup. izq', 'Mu+ centro', 'As centro', 'Inf. centro',
               'Mu− der', 'As der', 'Sup. der', 'Inf. corrido', 'Estribos', 'OK']
        tabla = pd.DataFrame(filas, columns=enc)
        st.dataframe(tabla, hide_index=True, width='stretch',
                     column_config={c: st.column_config.NumberColumn(format='%.2f') for c in
                                    ['L (m)', 'Mu− izq', 'As izq', 'Mu+ centro', 'As centro', 'Mu− der', 'As der']})
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine='openpyxl') as w:
            tabla.to_excel(w, index=False, sheet_name='Armado de vigas')
        st.download_button('Descargar resumen de armado (.xlsx)', bio.getvalue(), 'armado_vigas.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
