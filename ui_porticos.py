"""Examen 3 · Pórtico: envolventes de M y V por viga (sismo de ida, regreso y gravedad), acero requerido,
armado con la varilla que elige el usuario y render de las secciones."""
import io

import numpy as np
import pandas as pd
import streamlit as st

import ame_parcial3 as p3
import ame_porticos as pt

COLS = ['Nivel', 'h (m)', 'Col b (cm)', 'Col h (cm)', 'Viga b (cm)', 'Viga h (cm)', 'CM (t/m)', 'CV máx (t/m)', 'CV acc (t/m)',
        'F sismo (t)']
COLS_TXT = ['Nivel', 'CM (t/m)', 'CV máx (t/m)', 'CV acc (t/m)', 'F sismo (t)']
FUENTES_F = {'Dinámico (SRSS)': 'tabla_din', 'Dinámico escalado al estático': 'tabla_din_esc', 'Estático': 'tabla_est'}
VARS = list(pt.DIAMETROS)
OPC_MODELO = ['Modelo del examen 2 (tramos sin losa, columnas continuas, secciones por nivel, volados)',
              'Pórtico regular (captura rápida)']


def precargar(p):
    """Llena la pestaña con los datos de un ejemplo (clave 'porticos' del ejemplo): cargas por nivel, fuerzas y opciones."""
    n = len(p['CM'])
    st.session_state['p6_ver'] = st.session_state.get('p6_ver', 0) + 1
    st.session_state['p6_dft'] = pd.DataFrame({'Nivel': range(1, n + 1), 'CM (t/m)': list(p['CM']), 'CV máx (t/m)': ['0'] * n,
                                               'CV acc (t/m)': ['0'] * n, 'F sismo (t)': list(p['F'])})
    st.session_state.update({'p6_fm': OPC_MODELO[0], 'p6_fc': p['fc'], 'p6_fy': p['fy'], 'p6_grupo': p['grupo'],
                             'p6_pp': p['peso_propio'], 'p6_grav': p['gravedad'], 'p6_paño': p['paño'], 'p6_est': p['estribo'], 'p6_rec': p['rec']})
VARS_BARRA = [v for v in VARS if v != '#10'] + ['#10']


def _g(x):
    return f'{x:g}'


def _numeros(df, cols):
    """Las tablas se capturan como texto (así se pueden escribir decimales como 3.5 o 3,5) y aquí se pasan a número.
    Devuelve el DataFrame numérico o None si alguna celda no es un número."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c].astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')
    return out if not out[cols].isna().any().any() else None


def _defecto(n=3):
    return pd.DataFrame({'Nivel': range(1, n + 1), 'h (m)': ['3'] * n, 'Col b (cm)': ['40'] * n, 'Col h (cm)': ['40'] * n,
                         'Viga b (cm)': ['25'] * n, 'Viga h (cm)': ['50'] * n, 'CM (t/m)': ['2'] * n,
                         'CV máx (t/m)': ['0.5'] * n, 'CV acc (t/m)': ['0.3'] * n,
                         'F sismo (t)': [_g(3.0 * (i + 1)) for i in range(n)]})


def _defecto_txt(n):
    return pd.DataFrame({'Nivel': range(1, n + 1), 'CM (t/m)': ['2'] * n, 'CV máx (t/m)': ['0.5'] * n, 'CV acc (t/m)': ['0.3'] * n,
                         'F sismo (t)': [_g(3.0 * (i + 1)) for i in range(n)]})


def _traer(res_sismo, crujias, fuente):
    t = res_sismo[FUENTES_F[fuente]]
    F = np.asarray(t['F'], float)[::-1]
    alt = np.asarray(res_sismo.get('alturas_acum'), float)
    h = np.diff(np.concatenate([[0.0], alt]))
    df = st.session_state.get('p6_df', _defecto(len(F)))
    if len(df) != len(F):
        df = _defecto(len(F))
    df = df.copy()
    df['h (m)'] = [_g(round(x, 4)) for x in h]
    df['F sismo (t)'] = [_g(round(x, 4)) for x in F]
    st.session_state['p6_ver'] = st.session_state.get('p6_ver', 0) + 1
    st.session_state['p6_df'] = df
    tx = st.session_state.get('p6_dft')
    if tx is None or len(tx) != len(F):
        tx = _defecto_txt(len(F))
    tx = tx.copy()
    tx['F sismo (t)'] = [_g(round(x, 4)) for x in F]
    st.session_state['p6_dft'] = tx
    if crujias:
        st.session_state['p6_spans'] = crujias


def _fig(f, dpi=170, nombre='diagrama'):
    """Muestra la figura como imagen de alta resolución: con el icono de pantalla completa (arriba a la derecha de la imagen)
    se amplía a todo el ancho de la pantalla y los valores se leen sin amontonarse."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    f.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(f)
    st.image(buf.getvalue(), width='stretch')
    st.download_button('Descargar imagen (PNG)', buf.getvalue(), f'{nombre}.png', 'image/png', key=f'dl_{nombre}_{st.session_state.get("_n_img", 0)}')
    st.session_state['_n_img'] = st.session_state.get('_n_img', 0) + 1


def _etq(clave, ne):
    return pt.etiqueta_viga(clave, ne)


def mostrar(res_sismo=None, crujias_txt='', modelo=None, datos=None):
    st.caption('Análisis de un pórtico plano de concreto con CM, CV y sismo. El sismo se resuelve de ida (+) y de regreso (−) '
               'y se combina con la gravedad; la envolvente da el momento y el cortante máximos de cada viga, con los que se '
               'calcula el acero. Tú eliges la varilla; la app propone el número de barras y revisa que quepan en el ancho '
               '(ductilidad baja, apuntes pp. 31–35).')
    usa_modelo = False
    hay_modelo = modelo is not None and datos is not None and modelo.get('vigas') == 'flexibles'
    if st.session_state.get('p6_fm') not in OPC_MODELO:         # proyectos guardados con los nombres anteriores
        anterior = str(st.session_state.pop('p6_fm', '') or '')
        if anterior.startswith('Modelo'):
            st.session_state['p6_fm'] = OPC_MODELO[0]
        elif anterior.startswith('Pórtico'):
            st.session_state['p6_fm'] = OPC_MODELO[1]
    fuente_modelo = st.radio('Geometría del pórtico', OPC_MODELO, key='p6_fm', index=0 if hay_modelo else 1,
                             help='La viga continua sin columnas está en su propia pestaña: «Examen 3 → Viga continua».')
    usa_modelo = fuente_modelo.startswith('Modelo')
    if usa_modelo and not hay_modelo:
        st.warning('Para usar el modelo del examen 2 (pestaña «1 · Estructura») elige **Flexibles (marco plano)** en el tipo de vigas y captura la viga '
                   'de cada piso. Mientras tanto usa el pórtico regular.')
        return
    fuente = st.selectbox('Fuerzas sísmicas tomadas de', list(FUENTES_F), key='p6_fuente')
    ok_res = res_sismo is not None and FUENTES_F[fuente] in res_sismo

    c1, c2, c3, c4 = st.columns(4)
    fc = c1.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='p6_fc')
    fy = c2.number_input('fy (kg/cm²)', 2000.0, 6000.0, 4200.0, key='p6_fy')
    clase = c3.selectbox('E del concreto', ["14000·√f′c (clase 1)", "8000·√f′c (clase 2)"], key='p6_clase')
    E = (14000 if clase.startswith('14') else 8000) * np.sqrt(fc)
    c4.metric('E (kg/cm²)', f'{E:,.0f}')
    c1, c2, c3, c4 = st.columns(4)
    grupo = c1.radio('Grupo', ['B', 'A'], horizontal=True, key='p6_grupo')
    estab = c2.checkbox('Incluir estabilidad (0.9 CM ± sismo)', False, key='p6_estab')
    rec = c3.number_input('Recubrimiento libre (cm)', 1.5, 8.0, 4.0, 0.5, key='p6_rec')
    est_var = c4.selectbox('Estribo', ['#3', '#4'], key='p6_est')
    c1, c2, c3 = st.columns(3)
    agrupar = c1.checkbox('Permitir agrupar barras de 2 en 2 en un lecho', True, key='p6_agr',
                          help='Si no caben sueltas en el ancho, se ponen de 2 en 2 pegadas antes de pasar a un segundo '
                               'lecho (como en el ejercicio de clase: 6#8 en b = 35 cm).')
    usa_pp = c2.checkbox('Agregar peso propio de la viga a CM', False, key='p6_pp',
                         help='Suma γ·b·h (t/m) de la viga de cada nivel a la carga muerta de todos sus tramos.')
    gam = c3.number_input('γ del concreto (t/m³)', 2.0, 2.6, 2.4, 0.05, key='p6_gam', disabled=not usa_pp)
    pp = gam if usa_pp else 0.0
    paño = st.checkbox('Diseñar en el paño de la columna (como ETABS); desactívalo para usar el momento en el eje del apoyo', True,
                       key='p6_paño', help='En el ejercicio de clase se diseña con el valor del diagrama en el eje del apoyo.')
    usa_grav = st.checkbox('Incluir la combinación de gravedad (1.3 CM + 1.5 CV; Grupo A: 1.5 / 1.7) en la envolvente', True,
                           key='p6_grav', help='Desactívala para diseñar solo con CM + CVacc ± sismo, como en el ejercicio de clase.')

    if usa_modelo:
        n = len(datos['niveles'])
        st.info('La geometría (columnas por eje, secciones por nivel, vigas, tramos sin losa, columnas que cruzan pisos y '
                'apoyos) es la de «Examen 2 → 1 · Estructura». Aquí solo captura las cargas por tramo (un valor para todos o uno por tramo: '
                'volado izq. + crujías + volado der., t/m) y las fuerzas sísmicas.')
        import ame_sismico as ame
        ej17 = next((v['porticos'] for k, v in ame.EJEMPLOS_MODELO.items() if k.startswith('Ejemplo 17')), None)
        if ej17 and str(datos.get('crujias', '')).split() == ['5', '6', '5', '5', '7'] and n == len(ej17['CM']):
            st.button('Llenar cargas, fuerzas y opciones con los datos del ejercicio de clase (Ejemplo 17)', key='p6_ej17',
                      on_click=lambda: precargar(ej17),
                      help='CM = 10, 10, 10.8, 11.6, 10 t/m; F = 5, 5, 10, 15, 20 t; Grupo A con peso propio; solo combinaciones con '
                           'sismo; momentos en el eje del apoyo; estribo #4 y recubrimiento de 3 cm.')
        tx = st.session_state.get('p6_dft')
        if tx is None or len(tx) != n:
            tx = _defecto_txt(n)
            if ok_res:
                tx['F sismo (t)'] = [_g(round(x, 4)) for x in np.asarray(res_sismo[FUENTES_F[fuente]]['F'], float)[::-1]]
            st.session_state['p6_dft'] = tx
        st.button('Traer fuerzas sísmicas del análisis', disabled=not ok_res, key='p6_traer2',
                  on_click=lambda: _traer(res_sismo, '', fuente))
        conf = {c: st.column_config.TextColumn(c) for c in COLS_TXT[1:4]}
        conf['F sismo (t)'] = st.column_config.TextColumn('F sismo (t)')
        conf['Nivel'] = st.column_config.NumberColumn('Nivel', format='%d', disabled=True)
        tx = st.data_editor(st.session_state['p6_dft'], hide_index=True, width='stretch', key=f'p6_editor_t_{st.session_state.get("p6_ver", 0)}', column_config=conf,
                            disabled=['Nivel'])
        try:
            cargas_txt = [dict(CM=r['CM (t/m)'], CV=r['CV máx (t/m)'], CVacc=r['CV acc (t/m)']) for _, r in tx.iterrows()]
            geo = pt.geo_modelo(modelo, datos, cargas_txt, pp)
            geo['E'] = E
            F = [float(str(x).strip().replace(',', '.')) for x in tx['F sismo (t)']]
            res = pt.resolver_geo(geo, F)
        except ValueError as e:
            st.error(str(e) if 'could not convert' not in str(e) else 'Escribe las fuerzas sísmicas como números (por ejemplo 3.5).')
            return
    else:
        if 'p6_df' not in st.session_state:
            st.session_state['p6_df'] = _defecto()
        st.session_state.setdefault('p6_spans', '4 5')
        c1, c2, c3 = st.columns(3)
        c1.text_input('Crujías (m), de izquierda a derecha', key='p6_spans')
        base = c2.radio('Base de las columnas', ['Empotrada', 'Articulada'], horizontal=True, key='p6_base')
        c3.write('')
        c3.button('Traer alturas, crujías y fuerzas del análisis', disabled=not ok_res, key='p6_traer',
                  on_click=lambda: _traer(res_sismo, crujias_txt, fuente))
        c1, c2 = st.columns(2)
        fi_vig = c1.number_input('Factor de I en vigas (agrietamiento)', 0.1, 1.0, 1.0, 0.05, key='p6_fiv')
        fi_col = c2.number_input('Factor de I en columnas', 0.1, 1.0, 1.0, 0.05, key='p6_fic')
        conf = {c: st.column_config.TextColumn(c) for c in COLS[1:]}
        conf['Nivel'] = st.column_config.NumberColumn('Nivel', step=1, format='%d', disabled=True)
        df = st.data_editor(st.session_state['p6_df'], num_rows='dynamic', hide_index=True, width='stretch', key=f'p6_editor_{st.session_state.get("p6_ver", 0)}',
                            column_config=conf).dropna()
        if len(df) < 1:
            st.info('Captura al menos un nivel.')
            return
        df = _numeros(df, COLS[1:])
        if df is None:
            st.error('Hay celdas vacías o que no son números. Escribe decimales con punto o coma (por ejemplo 3.5).')
            return
        try:
            spans = [float(x) for x in st.session_state['p6_spans'].replace(',', ' ').split()]
            if not spans or min(spans) <= 0:
                raise ValueError
        except ValueError:
            st.error('Escribe las crujías como números positivos separados por espacios, por ejemplo «4 5».')
            return
        F = list(df['F sismo (t)'])
        try:
            res = pt.resolver(spans, list(df['h (m)']), list(zip(df['Col b (cm)'], df['Col h (cm)'])),
                              list(zip(df['Viga b (cm)'], df['Viga h (cm)'])), E,
                              dict(CM=list(df['CM (t/m)']), CV=list(df['CV máx (t/m)']), CVacc=list(df['CV acc (t/m)'])),
                              F, base, fi_col, fi_vig, pp)
        except (np.linalg.LinAlgError, ValueError):
            st.error('El pórtico es inestable o inconsistente: revisa las secciones y los apoyos.')
            return
        geo = res['geo']

    ne = res['ne']
    todas = pt.combinaciones(grupo, estab)
    combos = pt.combinaciones(grupo, estab, gravedad=usa_grav)
    env, mats = pt.envolvente(res, combos)
    ux, dist = pt.desplazamientos(res)
    with st.expander('Combinaciones de carga usadas'):
        st.dataframe(pd.DataFrame([[k] + list(v) for k, v in combos.items()],
                                  columns=['Combinación', 'CM', 'CV máx', 'CV acc', 'Sismo']), hide_index=True, width='stretch')
        st.caption('La envolvente toma, en cada punto, el mayor y el menor momento y cortante de todas las combinaciones.')

    st.markdown('##### Tamaño y etiquetas de los renders')
    c1, c2, c3, c4 = st.columns(4)
    etiquetas = c1.selectbox('Etiquetas', ['extremos', 'todas', 'ninguna'], key='p6_etq',
                             help='«extremos»: valores en los apoyos y el máximo de cada viga; «todas»: más valores; «ninguna»: solo la forma.')
    piso_sel = c2.selectbox('Mostrar', ['Todo el pórtico'] + [f'Solo el piso {k}' for k in range(1, res['n'] + 1)], key='p6_piso')
    piso = None if piso_sel.startswith('Todo') else int(piso_sel.split()[-1])
    ancho = c3.slider('Ancho de la figura (pulgadas)', 8, 30, int(round(pt._tam(res)[0])), key='p6_ancho',
                      help='10 pulgadas se lee bien en la página. Más ancho da más espacio entre etiquetas pero la letra se ve más chica '
                           '(usa la pantalla completa de la imagen).')
    alto = c4.slider('Alto de la figura (pulgadas)', 4, 30, int(round(pt._tam(res)[1])), key='p6_alto')
    c1, c2, c3 = st.columns(3)
    elementos = {'Solo vigas': 'vigas', 'Vigas y columnas': 'todos', 'Solo columnas': 'columnas'}[
        c1.selectbox('Diagramas de', ['Solo vigas', 'Vigas y columnas', 'Solo columnas'], key='p6_elem',
                     help='Con «Solo vigas» los diagramas no se enciman en los nudos; es lo que se diseña en el examen 3.')]
    cols_etq = c2.checkbox('Mostrar valores en las columnas', False, key='p6_etq_col',
                           help='Apagado: solo se rotulan las vigas, que es lo que se diseña. Enciéndelo para ver también las columnas.')
    fuente = c3.slider('Tamaño de las etiquetas', 8, 16, 11, key='p6_fuente_etq')
    st.caption('Tip: pasa el cursor sobre la imagen y usa el icono de pantalla completa (arriba a la derecha) para ver todo el pórtico '
               'a todo el ancho; con «Solo el piso N» el diagrama sale aún más grande.')
    kw = dict(etiquetas=etiquetas, piso=piso, ancho=ancho, alto=alto, columnas=cols_etq, fuente=fuente, elementos=elementos)

    tabs = st.tabs(['Modelo', 'Gravedad', 'Sismo de ida', 'Sismo de regreso', 'Envolvente', 'Deformadas', 'Diseño por viga',
                    'Viga continua del nivel', 'Secciones (render)', 'Resumen de acero'])

    def selector(clave):
        return 'M' if st.radio('Diagrama', ['Momento M', 'Cortante V'], horizontal=True, key=clave) == 'Momento M' else 'V'

    with tabs[0]:
        _fig(pt.figura_cargas(res, F))
        st.caption('Distorsiones del sismo de ida sin factor de carga: ' + ' · '.join(
            f'N{i + 1}: {d:.5f}' for i, d in enumerate(dist)) + '. Revisa con tu análisis sísmico los límites γ.')
    with tabs[1]:
        _fig(pt.figura_diagrama(res, pt.combinar(res, todas['Gravedad']), selector('p6_t1'), f'Gravedad {todas["Gravedad"][:2]}', **kw))
    with tabs[2]:
        _fig(pt.figura_diagrama(res, mats['Sismo ida (+)'], selector('p6_t2'), 'Sismo de ida: 1.1(CM + CVacc) + 1.1 S', **kw))
    with tabs[3]:
        _fig(pt.figura_diagrama(res, mats['Sismo regreso (−)'], selector('p6_t3'), 'Sismo de regreso: 1.1(CM + CVacc) − 1.1 S', **kw))
    with tabs[4]:
        tp = selector('p6_t4')
        mx = {k: dict(x=v['x'], p=v['p'], M=v['Mmax'], V=v['Vmax']) for k, v in env.items()}
        mn = {k: dict(x=v['x'], p=v['p'], M=v['Mmin'], V=v['Vmin']) for k, v in env.items()}
        _fig(pt.figura_diagrama(res, mx, tp, 'Envolvente (máximo + mínimo de todas las combinaciones)', extra=mn, **kw))
        st.caption('Amarillo/azul: valores positivos (en vigas, tensión abajo); rojo/naranja: negativos (tensión arriba).')
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            _fig(pt.figura_deformada(res, 'S', 1.0, 'Sismo de ida'))
        with c2:
            _fig(pt.figura_deformada(res, 'S', -1.0, 'Sismo de regreso'))

    claves = [k for k in env if k[0] == 'b']
    if not claves:
        st.info('No hay vigas con carga en el modelo.')
        return
    nombres = [_etq(k, ne) for k in claves]
    # elección de varilla por viga (default: regla h/10)
    with st.expander('Varilla elegida por viga (regla h/10 como sugerencia)', expanded=True):
        st.caption('Regla del profesor: número de varilla ≈ h(cm)/10. No existen la #7 ni la #9: tú decides cuál usar. '
                   'Si no cambias nada se usa la sugerida.')
        filas = []
        for k, nm in zip(claves, nombres):
            h = env[k]['h']
            sug, txt = pt.varilla_sugerida(h)
            filas.append(dict(Viga=nm, b=env[k]['b'], h=h, sugerida=txt, superior=sug, inferior=sug))
        base_df = pd.DataFrame(filas)
        key_ed = 'p6_vars_' + str(hash(tuple(nombres) + tuple(base_df['h'])))
        ed = st.data_editor(base_df, hide_index=True, width='stretch', key=key_ed, disabled=['Viga', 'b', 'h', 'sugerida'],
                            column_config={'superior': st.column_config.SelectboxColumn('Varilla superior', options=VARS, required=True),
                                           'inferior': st.column_config.SelectboxColumn('Varilla inferior', options=VARS, required=True),
                                           'b': st.column_config.NumberColumn(format='%g'), 'h': st.column_config.NumberColumn(format='%g')})
    elec = {k: (r['superior'], r['inferior']) for k, (_, r) in zip(claves, ed.iterrows())}
    dis = pt.resumen_vigas(env, res, fc, fy, elec, est_var, rec, agrupar, paño)

    # advertencia viga más ancha que la columna
    for k in claves:
        _, kk, j = k
        ch = [c for c in geo['niveles'][kk - 1]['cols'] if c['eje'] in (j, j + 1)]
        if ch and env[k]['b'] > min(c['b'] for c in ch) + 1e-9:
            st.warning(f'{_etq(k, ne)}: la viga ({env[k]["b"]:g} cm) es más ancha que una de sus columnas '
                       f'({min(c["b"] for c in ch):g} cm).')

    with tabs[6]:
        nombre = st.selectbox('Viga', nombres, key='p6_viga')
        d = dis[claves[nombres.index(nombre)]]
        (st.success if d['ok'] else st.error)(
            f'Viga {d["b"]:g}×{d["h"]:g} cm, L = {d["L"]:g} m: ' + ('el acero cabe y cumple.' if d['ok'] else
                                                                  'revisa: As > As,máx, no alcanza el MR, no caben las barras o el cortante.'))
        _fig(pt.figura_viga(env, d, mats))
        filas = []
        for s in d['secciones']:
            a = s['armado']
            filas.append([s['posicion'].capitalize(), s['lecho'], s['x'], s['Mu'], s['As'] if s['As'] is not None else np.nan,
                          a['armado'] if a else '—', ' + '.join(map(str, a['capas'])) if a else '—', a['As'] if a else np.nan,
                          a['MR'] if a else np.nan, 'sí' if (a and a['cabe']) else 'no'])
        st.dataframe(pd.DataFrame(filas, columns=['Sección', 'Lecho', 'x (m)', 'Mu (t·m)', 'As calc (cm²)', 'Armado', 'Barras por lecho',
                                                  'As real (cm²)', 'MR (t·m)', 'Cabe']), hide_index=True, width='stretch',
                     column_config={c: st.column_config.NumberColumn(format='%.2f') for c in
                                    ['x (m)', 'Mu (t·m)', 'As calc (cm²)', 'As real (cm²)', 'MR (t·m)']})
        c = d['cortante']
        st.markdown(f'**Estribos:** Vu máx = {d["Vu"]:.2f} t → ' + (f'E{d["est_var"]} ({c["ramas"]} ramas) @ {c["s_usar"]} cm'
                                                                  if c.get('s_usar') else 'excede el límite'))
        with st.expander('Cálculo desarrollado de cada sección', expanded=True):
            for s in d['secciones']:
                st.markdown(f'**{s["posicion"].capitalize()} · lecho {s["lecho"]}**')
                for lin in s['lineas']:
                    st.markdown(f'- {lin}')
        with st.expander('Cortante desarrollado'):
            for lin in c['pasos']:
                st.markdown(f'- {lin}')
        st.caption('El armado superior en cada apoyo y el inferior corrido salen de las seis secciones. '
                   'Al menos 2 varillas corridas por lecho (apuntes p. 32).')
    with tabs[7]:
        niveles = sorted({k[1] for k in claves})
        c1, c2, c3 = st.columns(3)
        nivel = c1.selectbox('Nivel', niveles, key='p6_nivel')
        h_n = next(env[k]['h'] for k in claves if k[1] == nivel)
        sug, txt_sug = pt.varilla_sugerida(h_n)
        vs = c2.selectbox('Varilla superior', VARS, index=VARS.index(sug), key=f'p6_nvs_{nivel}_{sug}')
        vi = c3.selectbox('Varilla inferior', VARS, index=VARS.index(sug), key=f'p6_nvi_{nivel}_{sug}')
        st.caption(txt_sug + ' Los renders y tablas de esta pestaña usan estas varillas para toda la viga del nivel.')
        apoyo_unico = st.checkbox('Un solo momento por apoyo: el mayor de ambos lados (como en clase)', True, key='p6_apoyo',
                                  help='En un apoyo interior el diagrama tiene un valor a cada lado de la columna; en clase se dibujó y se '
                                       'diseñó con el mayor, con el mismo acero a ambos lados.')
        niv = pt.armado_nivel(env, res, nivel, fc, fy, vs, vi, est_var, rec, agrupar, paño, apoyo_unico)
        if niv is None:
            st.info('Ese nivel no tiene vigas con carga.')
        else:
            ok_v = niv['MR_max'] >= niv['Mu_max'] and niv['V_lim'] >= niv['Vu_max']
            (st.success if ok_v else st.error)('La sección cumple las revisiones de la viga completa.' if ok_v else
                                               'La sección no cumple una revisión de la viga completa: cámbiala.')
            st.markdown('**Procedimiento** (como en clase)')
            st.markdown('0. Revisiones: ' + niv['pasos'][0] + '  \n   ' + niv['pasos'][1])
            st.markdown('1. Diseño por flexión en cada punto a), b), c)… (caras de apoyo y centros): tabla de abajo.')
            st.markdown('2. ' + niv['pasos'][2])
            st.markdown('3. ' + niv['pasos'][3])
            st.markdown('4. Anclaje: la longitud de cada bastón ya incluye d más allá del punto donde deja de requerirse.')
            _fig(pt.figura_alzado(niv, res))
            filas = []
            for q in niv['puntos']:
                for lecho, nf in (('superior', q['n_sup']), ('inferior', q['n_inf'])):
                    s_ = next(s for s in niv['dis'][q['clave']]['secciones'] if s['posicion'] == q['posicion'] and s['lecho'] == lecho)
                    a = s_['armado']
                    filas.append([f'{q["letra"]})', 'sup.' if lecho == 'superior' else 'inf.', q['x'], s_['Mu'],
                                  s_['As'] if s_['As'] is not None else np.nan, a['armado'] if a else '—',
                                  a['As'] if a else np.nan, f'{nf}{vs if lecho == "superior" else vi}'])
            st.dataframe(pd.DataFrame(filas, columns=['Punto', 'Lecho', 'x (m)', 'Mu (t·m)', 'As req/usar (cm²)', 'As real propuesta',
                                                      'As real (cm²)', 'Barras finales (corrido + bastones)']),
                         hide_index=True, width='stretch',
                         column_config={c: st.column_config.NumberColumn(format='%.2f') for c in
                                        ['x (m)', 'Mu (t·m)', 'As req/usar (cm²)', 'As real (cm²)']})
            if niv['bastones']:
                st.markdown('**Bastones**')
                st.dataframe(pd.DataFrame([[q['lecho'], f'1{q["varilla"]}', q['req_ini'], q['req_fin'], q['x_ini'], q['x_fin'], q['L']]
                                           for q in niv['bastones']],
                                          columns=['Lecho', 'Barra', 'Requerido desde (m)', 'Requerido hasta (m)', 'Colocado desde (m)',
                                                   'Colocado hasta (m)', 'Longitud (m)']),
                             hide_index=True, width='stretch',
                             column_config={c: st.column_config.NumberColumn(format='%.2f') for c in
                                            ['Requerido desde (m)', 'Requerido hasta (m)', 'Colocado desde (m)', 'Colocado hasta (m)', 'Longitud (m)']})
            else:
                st.info('El corrido cubre todo el diagrama: no hacen falta bastones.')
            st.caption('La longitud de desarrollo del anclaje se toma como d (regla de clase). Verifica con tu profesor si además '
                       'pide la Ld de la NTC. Los puntos a), b)… son caras de apoyo y centros de cada crujía.')
    with tabs[8]:
        nombre2 = st.selectbox('Viga', nombres, key='p6_viga2')
        d2 = dis[claves[nombres.index(nombre2)]]
        _fig(pt.figura_secciones(d2))
        st.caption('Se revisa la separación libre entre barras (≥ máx(Ø, 2.5 cm)) con el ancho útil descontando recubrimiento y '
                   'estribo. Las barras en rojo no cumplen la separación; si no caben en 3 lechos aumenta b, h o la varilla.')
    with tabs[9]:
        filas = []
        for k, d in dis.items():
            g = lambda pn, lc: next(s for s in d['secciones'] if s['posicion'] == pn and s['lecho'] == lc)
            fila = [_etq(k, ne), f'{d["b"]:g}×{d["h"]:g}', d['L']]
            for pn, lc in (('izquierdo', 'superior'), ('centro', 'inferior'), ('derecho', 'superior')):
                s = g(pn, lc)
                fila += [s['Mu'], s['As'] if s['As'] is not None else np.nan, s['armado']['armado'] if s['armado'] else '—']
            ic = d['inferior_corrido']['armado']
            fila.append(ic['armado'] if ic else '—')
            fila.append(f'E{d["est_var"]}@{d["cortante"]["s_usar"]}' if d['cortante'].get('s_usar') else '—')
            fila.append('✔' if d['ok'] else '✘')
            filas.append(fila)
        enc = ['Viga', 'b×h', 'L (m)', 'Mu− izq', 'As izq', 'Sup. izq', 'Mu+ centro', 'As centro', 'Inf. centro',
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
