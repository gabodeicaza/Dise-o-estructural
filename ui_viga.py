"""Opción «Una sola viga» de la pestaña 6: viga continua sin pórtico, como el ejemplo de clase (volados, cargas puntuales y uniformes,
peso propio). Cargas factorizadas, diagramas de M y V, revisión de la sección, acero a flexión, armado (corridas y bastones) y
estribos por zona, con el procedimiento del pizarrón."""
import io

import numpy as np
import pandas as pd
import streamlit as st

import ame_parcial3 as p3
import ame_porticos as pt
import ame_viga as av


def _numeros(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c].astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')
    return out if not out[cols].isna().any().any() else None


def _desarrollo(f, titulo=None):
    if titulo:
        st.markdown(f'##### {titulo}')
    for tit, tx in f['tex']:
        st.markdown(f'**{tit}**')
        st.latex(tx)


def _imagen(fig, nombre):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    st.image(buf.getvalue(), width='stretch')
    st.download_button('Descargar imagen (PNG)', buf.getvalue(), f'{nombre}.png', 'image/png', key=f'dl_{nombre}')


def _defecto(n):
    """Cargas puntuales por omisión (CM, CVmáx): las de la viga de 13 m de clase (P1, P2 y P3) si hay 4 segmentos; si no, ceros."""
    if n == 4:
        return ['1', '0', '3', '0', '2'], ['1', '0', '4', '0', '3']
    return ['0'] * (n + 1), ['0'] * (n + 1)


def mostrar():
    st.caption('Una viga sola, sin columnas, como el ejemplo de clase: das los segmentos, dónde están los apoyos, las cargas uniformes '
               'y las puntuales. La app factoriza las cargas, resuelve momentos y cortantes, calcula el acero en cada apoyo y claro '
               'con el procedimiento del pizarrón, arma la viga (corridas y bastones) y propone los estribos. '
               'Viene cargado el ejemplo de clase (Grupo B, b = 40, h = 80); con tope de 30 cm sus estribos salen @30, como en la hoja.')
    c1, c2, c3, c4 = st.columns(4)
    b = c1.number_input('b (cm)', 10.0, 300.0, 40.0, key='v1_b')
    h = c2.number_input('h (cm)', 20.0, 300.0, 80.0, key='v1_h')
    fc = c3.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='v1_fc')
    fy = c4.number_input('fy (kg/cm²)', 2000.0, 6000.0, 4200.0, key='v1_fy')

    seg_txt = st.text_input('Segmentos (m), de izquierda a derecha', '2 4 4 3', key='v1_seg',
                            help='Cada número es la distancia entre dos nodos consecutivos. Los nodos son los extremos, los apoyos y '
                                 'los puntos donde hay una carga puntual. Ejemplo de clase: 2 4 4 3 (apoyos en x = 2 y x = 10, P2 en x = 6).')
    try:
        largos = [float(x) for x in seg_txt.replace(',', ' ').split()]
        if len(largos) < 1 or min(largos) <= 0:
            raise ValueError
    except ValueError:
        st.error('Escribe los segmentos como números positivos separados por espacios, por ejemplo «2 4 4 3».')
        return
    n = len(largos)
    xn = np.concatenate([[0.0], np.cumsum(largos)])
    pcm0, pcv0 = _defecto(n)

    ap_def = '2 10' if n == 4 else f'0 {xn[-1]:g}'
    ap_txt = st.text_input('Posición de los apoyos (x, m)', ap_def, key=f'v1_apoyos_{n}',
                           help='Escribe las x de los apoyos separadas por espacios; deben coincidir con algún nodo (la suma acumulada de los '
                                'segmentos). Los demás nodos quedan sin apoyo y solo sirven para colocar cargas puntuales. '
                                'Ejemplo de clase: 2 10. Viga de 3 claros de 5 m con P a la mitad de cada uno: segmentos «2.5 2.5 2.5 2.5 2.5 2.5» '
                                'y apoyos «0 5 10 15».')
    nodos_txt = ', '.join(f'{x:g}' for x in xn)
    try:
        xs_ap = [float(t) for t in ap_txt.replace(',', ' ').split()]
    except ValueError:
        st.error('Escribe las posiciones de los apoyos como números separados por espacios, por ejemplo «0 5 10 15».')
        return
    tipos = ['—'] * (n + 1)
    for xa in xs_ap:
        k = np.flatnonzero(np.abs(xn - xa) < 1e-6)
        if not len(k):
            st.error(f'Hay un apoyo en x = {xa:g} m, pero ahí no hay un nodo. Los nodos están en x = {nodos_txt} m: ajusta los segmentos o '
                     'la posición del apoyo.')
            return
        tipos[int(k[0])] = 'Articulado'
    c1, c2 = st.columns(2)
    emp_i = c1.checkbox('Empotrar el extremo izquierdo', False, key='v1_emp_i', disabled=tipos[0] == '—',
                        help='Solo si hay un apoyo en x = 0.')
    emp_d = c2.checkbox('Empotrar el extremo derecho', False, key='v1_emp_d', disabled=tipos[-1] == '—',
                        help='Solo si hay un apoyo en el último nodo.')
    if emp_i and tipos[0] != '—':
        tipos[0] = 'Empotrado'
    if emp_d and tipos[-1] != '—':
        tipos[-1] = 'Empotrado'

    st.markdown('**Nodos y cargas puntuales (t, sin factorizar)**')
    nod = pd.DataFrame({'Nodo': [f'{k + 1}  (x = {x:g} m)' for k, x in enumerate(xn)], 'Apoyo': [t if t != '—' else 'sin apoyo' for t in tipos],
                        'P CM (t)': pcm0, 'P CVmáx (t)': pcv0})
    nod = st.data_editor(nod, hide_index=True, width='stretch', key=f'v1_nodos_{n}', disabled=['Nodo', 'Apoyo'],
                         column_config={'Nodo': st.column_config.TextColumn('Nodo'),
                                        'Apoyo': st.column_config.TextColumn('Apoyo'),
                                        'P CM (t)': st.column_config.TextColumn('P CM (t)'),
                                        'P CVmáx (t)': st.column_config.TextColumn('P CVmáx (t)')})
    st.markdown('**Cargas uniformes por segmento (t/m, sin factorizar)**')
    seg = pd.DataFrame({'Segmento': [f'{i + 1}  ({xn[i]:g} → {xn[i + 1]:g} m)' for i in range(n)], 'CM (t/m)': ['1'] * n, 'CV máx (t/m)': ['0.5'] * n})
    seg = st.data_editor(seg, hide_index=True, width='stretch', key=f'v1_segs_{n}', disabled=['Segmento'],
                         column_config={'Segmento': st.column_config.TextColumn('Segmento'),
                                        'CM (t/m)': st.column_config.TextColumn('CM (t/m)'),
                                        'CV máx (t/m)': st.column_config.TextColumn('CV máx (t/m)')})
    nod = _numeros(nod.assign(**{'P CM (t)': nod['P CM (t)'].fillna('0'), 'P CVmáx (t)': nod['P CVmáx (t)'].fillna('0')}), ['P CM (t)', 'P CVmáx (t)'])
    seg = _numeros(seg, ['CM (t/m)', 'CV máx (t/m)'])
    if nod is None or seg is None or len(nod) != n + 1 or len(seg) != n:
        st.error('Hay celdas vacías o que no son números. Escribe decimales con punto o coma (por ejemplo 1.5).')
        return

    c1, c2, c3, c4 = st.columns(4)
    grupo = c1.radio('Grupo', ['B', 'A'], horizontal=True, key='v1_grupo', help='Grupo B: 1.3 CM + 1.5 CV. Grupo A: 1.5 CM + 1.7 CV.')
    alternar = c2.checkbox('Alternar la carga viva', False, key='v1_alt',
                           help='Apagado (como en clase): la CV actúa en toda la viga. Encendido: se coloca solo donde más desfavorece '
                                'cada sección (patrones por claro y por volado).')
    usa_pp = c3.checkbox('Agregar peso propio', True, key='v1_pp', help='γ·b·h (t/m) sumado a la carga muerta de todos los segmentos.')
    gam = c4.number_input('γ (t/m³)', 2.0, 2.6, 2.4, 0.05, key='v1_gam', disabled=not usa_pp)
    pp = gam * b * h / 1e4 if usa_pp else 0.0
    ancho_ap = st.number_input('Ancho del apoyo (cm): 0 = diseñar con el momento y el cortante en el eje del apoyo', 0.0, 300.0, 0.0, 5.0,
                               key='v1_ap', help='Mayor que 0: se diseña en la cara del apoyo (a la mitad del ancho a cada lado).')

    cm, cv = list(seg['CM (t/m)']), list(seg['CV máx (t/m)'])
    try:
        env = av.envolvente_viga(xn, tipos, cm, cv, list(nod['P CM (t)']), list(nod['P CVmáx (t)']), grupo, pp, alternar)
    except (ValueError, np.linalg.LinAlgError) as e:
        st.error(str(e) if isinstance(e, ValueError) else 'La viga es inestable: revisa los apoyos.')
        return
    puntos = av.puntos_diseno(env, ancho_ap)
    cortes = av.cortantes_diseno(env, ancho_ap)
    if not puntos:
        st.info('Con estos datos no hay momentos que diseñar: revisa los apoyos y las cargas.')
        return
    Mu_max = max(p['Mu'] for p in puntos)
    Vu_max = max(c['Vu'] for c in cortes)
    fcm, fcv = env['fcm'], env['fcv']

    st.markdown('#### 0 · Cargas factorizadas')
    if usa_pp:
        st.latex(rf"p.p. = \gamma\,b\,h = {gam:g}({b / 100:g})({h / 100:g}) = {pp:.3f}\ \text{{t/m}}")
    vistos = set()
    for i in range(n):
        clave = (cm[i], cv[i])
        if clave in vistos:
            continue
        vistos.add(clave)
        iguales = [j + 1 for j in range(n) if (cm[j], cv[j]) == clave]
        st.latex(rf"w_u = F_{{CM}}(CM + p.p.) + F_{{CV}}\,CV_{{m\acute{{a}}x}} = {fcm}({cm[i]:g} + {pp:.3f}) + {fcv}({cv[i]:g}) = "
                 rf"{env['wu'][i]:.3f}\ \text{{t/m}}\qquad (\text{{segmentos }}{', '.join(map(str, iguales))})")
    for k in range(n + 1):
        if nod['P CM (t)'][k] or nod['P CVmáx (t)'][k]:
            st.latex(rf"P_{{u,{k + 1}}} = {fcm}({nod['P CM (t)'][k]:g}) + {fcv}({nod['P CVmáx (t)'][k]:g}) = {env['Pu'][k]:.2f}\ \text{{t}}"
                     rf"\qquad (x = {xn[k]:g}\ \text{{m}})")

    st.markdown('#### 1 · Diagramas de momento y cortante')
    _imagen(av.figura_viga_sola(env, f'Mu y Vu · Grupo {grupo} ({fcm} CM + {fcv} CV)'), 'viga_sola')
    st.caption('Amarillo: momento positivo (tensión abajo); rojo: negativo (tensión arriba). Reacciones factorizadas (t): '
               + ' · '.join(f'apoyo {s + 1} (x = {xn[k]:g}): {env["R"][k]:.2f}' for s, k in enumerate(env['ap'])))

    st.markdown('#### 2 · Propuesta de sección y revisiones')
    c1, c2, c3, c4 = st.columns(4)
    sug, txt_sug = pt.varilla_sugerida(h)
    var = c1.selectbox('Varilla longitudinal', list(pt.DIAMETROS), index=list(pt.DIAMETROS).index(sug), key=f'v1_var_{sug}', help=txt_sug)
    est = c2.selectbox('Estribo', ['#3', '#4'], index=1, key='v1_est')
    ramas = c3.selectbox('Ramas del estribo', [2, 4], index=1 if b >= 40 else 0, key=f'v1_ramas_{b >= 40}',
                         help='Si b ≥ 40 cm se usan estribos de 4 ramas.')
    rec = c4.number_input('Recubrimiento libre (cm)', 1.5, 8.0, 3.0, 0.5, key='v1_rec')
    c1, c2, c3, c4 = st.columns(4)
    r_fija = c1.checkbox('Usar r fija al centroide (como en clase)', True, key='v1_rfija',
                         help='Activado: d = h − r con la r que escribas (clase: r = 5 cm, d = 75 cm para h = 80). '
                              'Desactivado: la d sale de la posición real de las barras.')
    r_val = c2.number_input('r (cm)', 2.0, 20.0, 5.0, 0.5, key='v1_r', disabled=not r_fija)
    agrupar = c3.checkbox('Barras agrupadas de 2 en 2', True, key='v1_agr')
    s_tope = c4.number_input('Tope de la separación (cm)', 10.0, 40.0, 25.0, 1.0, key='v1_stope',
                             help='El profesor prefiere 25 cm cuando rige el tope (mayor seguridad en obra). Los apuntes y la hoja de la viga '
                                  'de 13 m usan 30 cm: pon 30 para reproducirla.')
    st.caption(txt_sug)
    rf = r_val if r_fija else None
    est_db, a_est = pt.DIAMETROS[est], p3.VARILLAS[est]
    ab = p3.VARILLAS[var]
    viga = pt.ficha_viga(b, h, fc, fy, var, est, ramas, rec, rf, Mu_max=Mu_max, Vu_max=Vu_max)
    (st.success if viga['ok_flexion'] and viga['ok_cortante'] else st.error)(
        f'Mu máx = {Mu_max:.2f} t·m, Vu máx = {Vu_max:.2f} t: la sección cumple (subreforzada y sin exceder el cortante).'
        if viga['ok_flexion'] and viga['ok_cortante'] else
        f'Mu máx = {Mu_max:.2f} t·m, Vu máx = {Vu_max:.2f} t: la sección NO cumple una revisión global; cámbiala antes de armar.')
    _desarrollo(viga)

    st.markdown('#### 3 · Diseño por flexión')
    fichas = [pt.ficha_punto(p['Mu'], b, h, fc, fy, var, rec, est_db, rf, agrupar, lecho=p['lecho']) for p in puntos]
    filas = [[p['nombre'], p['lecho'], p['Mu'], f['As_req'], f['As_usar'], f['armado'], f['As_real'], f['MR'], f['sep_libre'],
              '✔' if f['ok'] else '✘'] for p, f in zip(puntos, fichas)]
    st.dataframe(pd.DataFrame(filas, columns=['Punto', 'Lecho', 'Mu (t·m)', 'As req (cm²)', 'As usar (cm²)', 'Armado', 'As real (cm²)',
                                              'MR (t·m)', 'Sep. libre (cm)', 'OK']).round(2), hide_index=True, width='stretch')
    for p, f in zip(puntos, fichas):
        with st.container(border=True):
            _desarrollo(f, f'{p["nombre"]}:  Mu = {p["Mu"]:.2f} t·m  ' + ('✔' if f['ok'] else '✘ revisa'))
    import matplotlib.pyplot as plt
    fig = pt.figura_puntos(fichas, b, h, rec, est_db, [p['nombre'].split(' (')[0] for p in puntos])
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    st.markdown('#### 4 · Armado longitudinal: corridas y bastones')
    d_c = viga['d']
    arm = av.armado_longitudinal(env, puntos, fichas, b, fc, fy, var, d_c, viga['lim']['As_min'], ab)
    for t in arm['lineas']:
        st.markdown(f'- {t}')
    st.caption('Regla de clase: mínimo 2 varillas corridas, o el As,mín, o el 50 % de la cantidad máxima de varillas en una sección; '
               'el acero positivo se corre a todo lo largo de la viga y se permiten bastones en las secciones críticas.')
    if arm['bastones']:
        st.dataframe(pd.DataFrame([[bs['punto'], bs['lecho'], f'+{bs["n"]}{var}', bs['req0'], bs['req1'], bs['x0'], bs['x1'], bs['x1'] - bs['x0']]
                                   for bs in arm['bastones']],
                                  columns=['Punto', 'Lecho', 'Bastón', 'Requerido desde (m)', 'Requerido hasta (m)',
                                           'Colocado desde (m)', 'Colocado hasta (m)', 'Longitud (m)']).round(2),
                     hide_index=True, width='stretch')
        st.caption('Cada bastón se prolonga una distancia d = '
                   f'{d_c:.1f} cm más allá de donde la envolvente de Mu ya no excede el MR de las corridas.')
    else:
        st.markdown('- No hacen falta bastones: las corridas cubren el momento en toda la viga.')

    st.markdown('#### 5 · Estribos por zona')
    Av = a_est * ramas
    for tit, tx in p3.limites_cortante_tex(b, d_c, fc, Av, ramas, a_est):
        st.markdown(f'*{tit}*')
        st.latex(tx)
    as_apoyo = {p['s']: f['As_real'] for p, f in zip(puntos, fichas) if p['tipo'] == 'apoyo' and f['As_real']}
    as_claro = {p['s']: f['As_real'] for p, f in zip(puntos, fichas) if p['tipo'] == 'claro' and f['As_real']}
    zonas = av.zonas_cortante(env)
    seps, resumen = [], []
    for i, z in enumerate(zonas, 1):
        tipo, s = z['ref']
        as_t = (as_apoyo if tipo == 'apoyo' else as_claro).get(s, viga['lim']['As_min'])
        cf = p3.cortante_ficha(z['Vu'] * 1000, b, d_c, fc, 4200.0, Av, as_t, s_tope)
        seps.append(cf['s_usar'] if cf['ok'] else None)
        with st.container(border=True):
            st.markdown(f'##### Zona {i}: x = {z["x0"]:g} → {z["x1"]:g} m,  Vu = {z["Vu"]:.2f} t  (As en tensión = {as_t:.2f} cm², '
                        f'{"del apoyo " + str(s + 1) if tipo == "apoyo" else "del claro " + str(s + 1)})')
            for tit, tx in cf['tex']:
                st.markdown(f'*{tit}*')
                st.latex(tx)
            if cf['ok']:
                st.success(f'E {est}, {ramas} ramas @ {cf["s_usar"]} cm')
            else:
                st.error('Vu excede el límite de la sección: agranda b o h.')
        resumen.append([f'{z["x0"]:g} → {z["x1"]:g}', z['Vu'], as_t, cf['VcR'] / 1000, cf['s_max'], f'E {est}, {ramas}r @ {cf["s_usar"]} cm' if cf['ok'] else 'NO'])
    st.markdown('**Resumen de estribos**')
    st.dataframe(pd.DataFrame(resumen, columns=['Zona (m)', 'Vu (t)', 'As tens. (cm²)', 'VcR (t)', 'smáx (cm)', 'Estribos']).round(2),
                 hide_index=True, width='stretch')
    st.caption('En los extremos de claro (a 2h de la cara del apoyo) con sismo, la separación debe cumplir además s ≤ mín{d/4; 8 db long.; '
               '24 db est.; 30 cm} (integridad estructural).')

    st.markdown('#### 6 · Alzado del armado')
    r_dibujo = r_val if r_fija else rec + est_db + pt.DIAMETROS[var] / 2
    _imagen(av.figura_alzado_viga(env, arm, zonas, seps, h, r_dibujo, var, est), 'alzado_viga')
