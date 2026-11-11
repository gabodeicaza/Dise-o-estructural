"""Examen 3 · Viga continua (sin pórtico), como el ejemplo de clase y el examen: volados, cargas puntuales y uniformes, peso propio,
segmentos inclinados y combinación accidental. Cargas factorizadas, diagramas de M y V, revisión de la sección, acero a flexión (con el
FR de NTC-2023 por deformación), armado (corridas y bastones), estribos por zona con la zona de confinamiento 2h, acero de piel, flechas
(ELS) y alzado, con el procedimiento del pizarrón."""
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
    _calculadora_losas()
    st.markdown('**Cargas uniformes por segmento (t/m de proyección horizontal, sin factorizar)**')
    seg = pd.DataFrame({'Segmento': [f'{i + 1}  ({xn[i]:g} → {xn[i + 1]:g} m)' for i in range(n)], 'CM (t/m)': ['1'] * n, 'CV máx (t/m)': ['0.5'] * n,
                        'Desnivel (m)': ['0'] * n})
    seg = st.data_editor(seg, hide_index=True, width='stretch', key=f'v1_segs_{n}', disabled=['Segmento'],
                         column_config={'Segmento': st.column_config.TextColumn('Segmento'),
                                        'CM (t/m)': st.column_config.TextColumn('CM (t/m)'),
                                        'CV máx (t/m)': st.column_config.TextColumn('CV máx (t/m)', help='En la combinación accidental escribe aquí la CV instantánea (CVa).'),
                                        'Desnivel (m)': st.column_config.TextColumn('Desnivel (m)', help='Diferencia de altura entre los extremos del segmento '
                                                                                    '(viga inclinada, como la diagonal del examen 3). 0 = horizontal.')})
    nod = _numeros(nod.assign(**{'P CM (t)': nod['P CM (t)'].fillna('0'), 'P CVmáx (t)': nod['P CVmáx (t)'].fillna('0')}), ['P CM (t)', 'P CVmáx (t)'])
    seg = _numeros(seg.assign(**{'Desnivel (m)': seg['Desnivel (m)'].fillna('0')}), ['CM (t/m)', 'CV máx (t/m)', 'Desnivel (m)'])
    if nod is None or seg is None or len(nod) != n + 1 or len(seg) != n:
        st.error('Hay celdas vacías o que no son números. Escribe decimales con punto o coma (por ejemplo 1.5).')
        return

    c1, c2, c3, c4 = st.columns(4)
    grupo = c1.radio('Combinación', list(av.FACTORES), horizontal=True, key='v1_grupo',
                     help='Grupo B: 1.3 CM + 1.5 CV. Grupo A: 1.5 CM + 1.7 CV. Accidental: 1.1 (CM + CVa), con la CV instantánea en la tabla.')
    alternar = c2.checkbox('Alternar la carga viva', False, key='v1_alt',
                           help='Apagado (como en clase): la CV actúa en toda la viga. Encendido: se coloca solo donde más desfavorece '
                                'cada sección (patrones por claro y por volado).')
    usa_pp = c3.checkbox('Agregar peso propio', True, key='v1_pp', help='γ·b·h (t/m) sumado a la carga muerta de todos los segmentos.')
    gam = c4.number_input('γ (t/m³)', 2.0, 2.6, 2.4, 0.05, key='v1_gam', disabled=not usa_pp)
    pp = gam * b * h / 1e4 if usa_pp else 0.0
    dz = np.array(seg['Desnivel (m)'], float)
    cos = np.array(largos) / np.hypot(largos, dz)
    inclinada = bool(np.any(np.abs(dz) > 1e-9))
    pp_seg = pp / cos
    ancho_ap = st.number_input('Ancho del apoyo (cm): 0 = diseñar con el momento y el cortante en el eje del apoyo', 0.0, 300.0, 0.0, 5.0,
                               key='v1_ap', help='Mayor que 0: se diseña en la cara del apoyo (a la mitad del ancho a cada lado).')

    cm, cv = list(seg['CM (t/m)']), list(seg['CV máx (t/m)'])
    try:
        env = av.envolvente_viga(xn, tipos, cm, cv, list(nod['P CM (t)']), list(nod['P CVmáx (t)']), grupo, pp_seg, alternar, cos=cos)
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
    if inclinada:
        st.info('Segmentos inclinados: ' + '; '.join(f'segmento {i + 1}: θ = {np.degrees(np.arccos(cos[i])):.1f}°, longitud real = '
                                                       f'{largos[i] / cos[i]:.2f} m, p.p. por metro horizontal = {pp_seg[i]:.3f} t/m'
                                                       for i in range(n) if abs(dz[i]) > 1e-9)
                + '. Las cargas se dan por metro horizontal y los apoyos son verticales, así que los momentos son los de la proyección '
                  'horizontal; el cortante perpendicular al eje es V·cos θ. (En un pórtico, la acción de marco cambia los momentos.)')
    vistos = set()
    for i in range(n):
        clave = (cm[i], cv[i], round(pp_seg[i], 6))
        if clave in vistos:
            continue
        vistos.add(clave)
        iguales = [j + 1 for j in range(n) if (cm[j], cv[j], round(pp_seg[j], 6)) == clave]
        st.latex(rf"w_u = F_{{CM}}(CM + p.p.) + F_{{CV}}\,CV = {fcm}({cm[i]:g} + {pp_seg[i]:.3f}) + {fcv}({cv[i]:g}) = "
                 rf"{env['wu'][i]:.3f}\ \text{{t/m}}\qquad (\text{{segmentos }}{', '.join(map(str, iguales))})")
    for k in range(n + 1):
        if nod['P CM (t)'][k] or nod['P CVmáx (t)'][k]:
            st.latex(rf"P_{{u,{k + 1}}} = {fcm}({nod['P CM (t)'][k]:g}) + {fcv}({nod['P CVmáx (t)'][k]:g}) = {env['Pu'][k]:.2f}\ \text{{t}}"
                     rf"\qquad (x = {xn[k]:g}\ \text{{m}})")

    st.markdown('#### 1 · Diagramas de momento y cortante')
    _imagen(av.figura_viga_sola(env, f'Mu y Vu · {"Grupo " + grupo if grupo != "Accidental" else "Accidental"} ({fcm} CM + {fcv} CV)'), 'viga_sola')
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
    c1, c2, c3, c4 = st.columns(4)
    usa_conf = c1.checkbox('Zona de confinamiento 2h', True, key='v1_conf',
                           help='Integridad estructural (examen 3): a 2h de cada apoyo, s ≤ mín{d/4; 8 db; 24 db est.; tope}.')
    piel_var = c2.selectbox('Varilla del acero de piel', ['#3', '#4', '#5'], index=['#3', '#4', '#5'].index(est) if est in ('#3', '#4') else 0,
                            key=f'v1_pielvar_{est}', help='En el examen se usó la misma varilla que el estribo.')
    piel_umbral = c3.number_input('Acero de piel si h > (cm)', 30.0, 200.0, 60.0, 5.0, key='v1_pielh',
                                  help='Criterio de la hoja del examen: h > 60 cm. Verifícalo contra la NTC que te pidan.')
    piel_s = c4.number_input('Separación máx. del acero de piel (cm)', 10.0, 40.0, 20.0, 1.0, key='v1_piels')
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
    frs = [p3.fr_deformacion(f['As_real'], b, f['d'], fc, fy) if f['As_real'] else None for f in fichas]
    filas = [[p['nombre'], p['lecho'], p['Mu'], f['As_req'], f['As_usar'], f['armado'], f['As_real'], f['MR'],
              fr['et'] if fr else np.nan, fr['FR'] if fr else np.nan, f['MR'] * fr['FR'] / 0.9 if fr and f['MR'] else np.nan,
              f['sep_libre'], '✔' if f['ok'] else '✘'] for p, f, fr in zip(puntos, fichas, frs)]
    st.dataframe(pd.DataFrame(filas, columns=['Punto', 'Lecho', 'Mu (t·m)', 'As req (cm²)', 'As usar (cm²)', 'Armado', 'As real (cm²)',
                                              'MR (t·m)', 'εt', 'FR NTC-2023', 'MR con FR(εt)', 'Sep. libre (cm)', 'OK']).round(4),
                 hide_index=True, width='stretch')
    bajos = [p['nombre'] for p, f, fr in zip(puntos, fichas, frs) if fr and f['MR'] and fr['FR'] < 0.9 and f['MR'] * fr['FR'] / 0.9 < p['Mu']]
    if bajos:
        st.warning('Con el FR de NTC-2023 por deformación (εt < εy + 0.003) el MR ya no alcanza en: ' + ', '.join(bajos)
                   + '. Agrega barras o aumenta la sección.')
    st.caption('FR NTC-2023: 0.65 si εt ≤ εy; 0.65 + 0.25(εt − εy)/0.003 en la transición; 0.9 si εt ≥ εy + 0.003 (εt = 0.003(d − c)/c). '
               'El MR de la tabla usa FR = 0.9 como en clase; la columna «MR con FR(εt)» lo corrige cuando la sección no es controlada por tensión.')
    for p, f in zip(puntos, fichas):
        with st.container(border=True):
            _desarrollo(f, f'{p["nombre"]}:  Mu = {p["Mu"]:.2f} t·m  ' + ('✔' if f['ok'] else '✘ revisa'))
    import matplotlib.pyplot as plt
    fig = pt.figura_puntos(fichas, b, h, rec, est_db, [p['nombre'].split(' (')[0] for p in puntos])
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    _comprobaciones(av.comprobaciones(env, puntos, fichas, b, fc, fy))

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
                   f'{d_c:.1f} cm más allá de donde la envolvente de Mu ya no excede el MR de las corridas.'
                   + (' Longitudes en proyección horizontal: en un segmento inclinado divide entre cos θ.' if inclinada else ''))
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
                        f'{"del apoyo " + str(s + 1) if tipo == "apoyo" else "del claro " + str(s + 1)})'
                        + (f'  ·  inclinada, Vu = V·cos θ' if abs(cos[z['e']] - 1) > 1e-9 else ''))
            for tit, tx in cf['tex']:
                st.markdown(f'*{tit}*')
                st.latex(tx)
            if cf['ok']:
                st.success(f'E {est}, {ramas} ramas @ {cf["s_usar"]} cm')
                if cf['s_usar'] < 10:
                    alt = p3.alternativas_estribo(z['Vu'] * 1000, b, d_c, fc, 4200.0, as_t, s_tope)
                    st.warning(f'@ {cf["s_usar"]} cm es muy cerrado para colar. '
                               + ('Alternativas con s ≥ 10 cm: ' + ' · '.join(o['texto'] for o in alt[:4]) if alt else
                                  'Ningún estribo #3/#4 de 2–4 ramas da s ≥ 10 cm: agranda la sección.'))
            else:
                st.error('Vu excede el límite de la sección: agranda b o h.')
        resumen.append([f'{z["x0"]:g} → {z["x1"]:g}', z['Vu'], as_t, cf['VcR'] / 1000, cf['s_max'], f'E {est}, {ramas}r @ {cf["s_usar"]} cm' if cf['ok'] else 'NO'])
    st.markdown('**Resumen de estribos por cortante**')
    st.dataframe(pd.DataFrame(resumen, columns=['Zona (m)', 'Vu (t)', 'As tens. (cm²)', 'VcR (t)', 'smáx (cm)', 'Estribos']).round(2),
                 hide_index=True, width='stretch')

    integ = p3.integridad_estructural(h, d_c, pt.DIAMETROS[var], est_db, s_tope)
    with st.container(border=True):
        st.markdown('##### Zona de confinamiento (integridad estructural)')
        for tit, tx in integ['tex']:
            st.latex(tx)
        if not usa_conf:
            st.caption('Desactivada: los estribos se dejan como salen por cortante.')
    tramos = av.tramos_estribos(env, zonas, seps, integ['Lconf'] / 100 if usa_conf else 0.0, integ['s_usar'])
    st.markdown('**Separación final a lo largo de la viga**')
    st.dataframe(pd.DataFrame([[f'{t["x0"]:.2f} → {t["x1"]:.2f}', t['x1'] - t['x0'], 'sí' if t['confinado'] else 'no',
                                f'E {est}, {ramas}r @ {t["s"]} cm' if t['s'] else 'NO']
                               for t in tramos], columns=['Tramo (m)', 'Longitud (m)', 'Zona 2h', 'Estribos']).round(2),
                 hide_index=True, width='stretch')

    st.markdown('#### 6 · Acero de piel (temperatura)')
    r_dibujo = r_val if r_fija else rec + est_db + pt.DIAMETROS[var] / 2
    piel = p3.acero_piel(b, h, r_dibujo, piel_var, piel_umbral, piel_s)
    for tit, tx in piel['tex']:
        st.latex(tx)
    for t in piel['lineas']:
        st.markdown(f'- {t}')
    if piel['aplica'] and piel['As_col'] < piel['As_ref']:
        st.caption(f'El área colocada ({piel["As_col"]:.2f} cm²) es menor que la referencia 0.003·b·h ({piel["As_ref"]:.2f} cm²); la hoja del '
                   'examen solo controla la separación. Si tu profesor pide el área, sube la varilla de piel.')

    st.markdown('#### 7 · Flechas (estado límite de servicio)')
    c1, c2, c3 = st.columns(3)
    fI = c1.number_input('Factor de I (agrietada)', 0.1, 1.0, 0.35, 0.05, key='v1_fI', help='Examen 3: I agrietada = 0.35·I.')
    fdif = c2.number_input('Factor de flecha diferida', 0.0, 4.0, 2.0, 0.5, key='v1_fdif',
                           help='Flecha total = (1 + factor)·δinst. En el examen: 3·δinst (factor 2).')
    lim_txt = c3.selectbox('Límite', list(av.LIMITES_FLECHA), key='v1_flim')
    w_serv = [cm[i] + pp_seg[i] + cv[i] for i in range(n)]
    P_serv = [nod['P CM (t)'][k] + nod['P CVmáx (t)'][k] for k in range(n + 1)]
    fl = av.revision_flechas(xn, tipos, w_serv, P_serv, b, h, fc, fI, fdif, lim_txt)
    st.latex(rf"E = 14000\sqrt{{f'_c}} = {fl['E']:,.0f}\ \text{{kg/cm}}^2;\quad I_g = \frac{{b h^3}}{{12}} = {fl['Ig']:,.0f};\quad "
             rf"I = {fI:g}\,I_g = {fl['I']:,.0f}\ \text{{cm}}^4;\quad \delta_{{total}} = {1 + fdif:g}\,\delta_{{inst}} \le \frac{{L}}{{{fl['div']}}} + {fl['extra']:g}\ \text{{cm}}")
    st.dataframe(pd.DataFrame([[f['tramo'], f['L'], f['x'], f['d_inst'], f['d_total'], f['d_perm'], '✔' if f['ok'] else '✘'] for f in fl['filas']],
                              columns=['Tramo', 'L (cm)', 'x de δmáx (m)', 'δinst (cm)', 'δtotal (cm)', 'δpermisible (cm)', 'OK']).round(4),
                 hide_index=True, width='stretch')
    st.caption('Cargas de servicio sin factorizar (CM + p.p. + CV). La app resuelve la flecha con la misma viga, sin ETABS. '
               'En un volado se compara con la L del volado.')

    st.markdown('#### 8 · Alzado del armado')
    _imagen(av.figura_alzado_viga(env, arm, zonas, seps, h, r_dibujo, var, est, tramos=tramos, piel=piel), 'alzado_viga')


def _comprobaciones(lista):
    """Comprobaciones automáticas con los datos capturados: cada una recalcula un resultado por otro camino."""
    malas = [c for c in lista if not c['ok']]
    with st.expander(('✔ Comprobaciones automáticas: todas pasan' if not malas else f'✘ Comprobaciones automáticas: {len(malas)} no pasan')
                     + f' ({len(lista)} revisiones)', expanded=bool(malas)):
        st.caption('Cada renglón recalcula un resultado por un camino distinto al del programa (equilibrio, estática del volado, '
                   'método de la cuantía q) con TUS datos. Si algo no coincide, aparece en rojo y el expansor se abre solo.')
        st.dataframe(pd.DataFrame([['✔' if c['ok'] else '✘', c['que'], c['programa'], c['referencia'], c['como']] for c in lista],
                                  columns=['', 'Qué se revisa', 'Programa', 'Otro camino', 'Cómo']).round(4),
                     hide_index=True, width='stretch')
        if malas:
            st.error('Hay resultados que no coinciden: revisa los datos (apoyos, cargas, sección) antes de usar los resultados.')


def _calculadora_losas():
    """Carga de losas a la viga por áreas tributarias (hoja «CORTANTE» de DISEÑO DE TRABES): trapecio en el lado largo del
    tablero, triángulo en el corto; w_viga = Σ(w·A)/L. El resultado se copia a la tabla de cargas por segmento."""
    with st.expander('Calculadora: carga de losas a la viga (áreas tributarias)'):
        st.caption('Cada renglón es un tablero de losa que se apoya en la viga. Lado largo → trapecio (2·largo − corto)·corto/4; '
                   'lado corto → triángulo corto²/4. Escribe la carga de la losa sin factorizar (t/m²) y copia el resultado a la tabla.')
        Lv = st.number_input('Longitud de la viga o del segmento (m)', 0.1, 50.0, 3.0, 0.1, key='v1_trib_L')
        base = pd.DataFrame({'Corto (m)': ['2.5', '3'], 'Largo (m)': ['3', '3'], 'Lado de la viga': ['largo', 'corto'],
                             'CM (t/m²)': ['0.512', '0.512'], 'CV (t/m²)': ['0.1', '0.1']})
        ed = st.data_editor(base, num_rows='dynamic', hide_index=True, width='stretch', key='v1_trib',
                            column_config={'Lado de la viga': st.column_config.SelectboxColumn('Lado de la viga', options=['largo', 'corto'], required=True),
                                           **{c: st.column_config.TextColumn(c) for c in ['Corto (m)', 'Largo (m)', 'CM (t/m²)', 'CV (t/m²)']}})
        ed = _numeros(ed.dropna(subset=['Corto (m)', 'Largo (m)']), ['Corto (m)', 'Largo (m)', 'CM (t/m²)', 'CV (t/m²)'])
        if ed is None or not len(ed):
            st.info('Captura al menos un tablero con números.')
            return
        filas = []
        for _, r in ed.iterrows():
            A = av.area_tributaria(r['Corto (m)'], r['Largo (m)'], r['Lado de la viga'])
            filas.append([r['Corto (m)'], r['Largo (m)'], r['Lado de la viga'], A, r['CM (t/m²)'] * A, r['CV (t/m²)'] * A])
        df = pd.DataFrame(filas, columns=['Corto', 'Largo', 'Lado', 'A trib. (m²)', 'CM·A (t)', 'CV·A (t)'])
        st.dataframe(df.round(4), hide_index=True, width='stretch')
        c1, c2 = st.columns(2)
        c1.metric('CM sobre la viga', f'{df["CM·A (t)"].sum() / Lv:.3f} t/m')
        c2.metric('CV sobre la viga', f'{df["CV·A (t)"].sum() / Lv:.3f} t/m')
