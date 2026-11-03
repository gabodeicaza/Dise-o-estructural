"""Pestaña «5 · Tercer parcial» de la app: mampostería, viento y concreto reforzado (motor en ame_parcial3.py)."""
import numpy as np
import pandas as pd
import streamlit as st

import ame_ntc as ntc
import ame_parcial3 as p3


def _pasos(lista):
    for t in lista:
        st.markdown(f'- {t}')


def _veredicto(ok, texto_ok='PASA', texto_no='NO PASA'):
    (st.success if ok else st.error)(texto_ok if ok else texto_no)


def _nums(txt, nombre):
    try:
        v = [float(x) for x in txt.replace(',', ' ').split()]
    except ValueError:
        raise ValueError(f'«{nombre}» debe ser una lista de números separados por espacios.')
    if not v:
        raise ValueError(f'Escribe «{nombre}».')
    return v


def _tabla(df, dec=3):
    st.dataframe(df, hide_index=True, width='stretch',
                 column_config={c: st.column_config.NumberColumn(format=f'%.{dec}f') for c in df.columns
                                if pd.api.types.is_float_dtype(df[c])})


# ------------------------------------------------------------------ mampostería
def _mamposteria():
    s_comp, s_cort, s_est, s_dist = st.tabs(['Carga vertical (PR)', 'Cortante por muro (VmR)', 'Fuerzas sísmicas (estático)',
                                             'Q, γ y otros límites'])
    with s_comp:
        c1, c2, c3 = st.columns(3)
        t = c1.number_input('t espesor (cm)', 1.0, 100.0, 14.0, key='mc_t')
        H = c2.number_input('H altura (cm)', 1.0, 2000.0, 350.0, key='mc_H')
        L = c3.number_input('L longitud (cm)', 1.0, 5000.0, 400.0, key='mc_L')
        c1, c2, c3 = st.columns(3)
        Pu = c1.number_input('Pu (kg) = 1.3 CM + 1.5 CVmáx', 0.0, 1e7, 8163.0, key='mc_Pu')
        fm = c2.number_input("f′m (kg/cm²)", 1.0, 200.0, 15.0, key='mc_fm')
        tipo = c3.selectbox('Tipo de muro', p3.TIPOS_MURO, index=1, key='mc_tipo')
        c1, c2, c3, c4 = st.columns(4)
        L2L1 = c1.number_input('L₂/L₁ de las losas', 0.0, 10.0, 1.0, key='mc_l2')
        cvcm = c2.number_input('CV/CM', 0.0, 10.0, 0.8, key='mc_cv')
        mov = c3.checkbox('Movimiento perpendicular restringido', True, key='mc_mov')
        vol = c3.checkbox('Sin volados ni cargas perpendiculares', True, key='mc_vol')
        b = c4.number_input('b apoyo de la losa (cm, 0 = t)', 0.0, 100.0, 0.0, key='mc_b')
        r = p3.muro_compresion(t, H, L, Pu, fm, tipo, L2L1, cvcm, mov, vol, b or None)
        cols = st.columns(3)
        cols[0].metric('FE', f'{r["FE"]:.4f}')
        cols[1].metric('PR (kg)', f'{r["PR"]:,.0f}')
        cols[2].metric('Pu (kg)', f'{Pu:,.0f}')
        _veredicto(r['pasa'])
        with st.expander('Procedimiento', expanded=True):
            _pasos(r['pasos'])
    with s_cort:
        c1, c2, c3 = st.columns(3)
        t = c1.number_input('t (cm)', 1.0, 100.0, 14.0, key='mv_t')
        L = c2.number_input('L (cm)', 1.0, 5000.0, 400.0, key='mv_L')
        H = c3.number_input('H (cm)', 1.0, 2000.0, 350.0, key='mv_H')
        c1, c2, c3 = st.columns(3)
        Pn = c1.number_input('Pn (kg) = CM + CVacc sin factor', 0.0, 1e7, 7520.0, key='mv_Pn')
        vm = c2.number_input("v′m (kg/cm²)", 0.1, 20.0, 2.0, key='mv_vm')
        Vu = c3.number_input('Vu (kg)', 0.0, 1e7, 5270.0, key='mv_Vu')
        r = p3.muro_cortante(t, L, H, Pn, vm, Vu=Vu)
        c1, c2, c3 = st.columns(3)
        c1.metric('f', f'{r["f"]:.4f}')
        c2.metric('VmR (kg)', f'{r["VmR"]:,.1f}')
        c3.metric('Vu (kg)', f'{Vu:,.1f}')
        _veredicto(r['pasa'])
        with st.expander('Procedimiento', expanded=True):
            _pasos(r['pasos'])
        st.markdown('**Entrepiso completo (varios muros en una dirección)**')
        df = st.data_editor(pd.DataFrame({'muro': ['A', 'C', 'D', 'E'], 't': [14.0] * 4, 'L': [150.0, 500.0, 400.0, 200.0],
                                          'H': [250.0] * 4, 'Pn': [12142.9, 20250.0, 16200.0, 8100.0],
                                          'Vu': [6020.0, 9960.0, 8340.0, 3250.0]}), num_rows='dynamic', key='mv_tabla',
                            hide_index=True, width='stretch')
        muros = [dict(nombre=str(f['muro']), t=f['t'], L=f['L'], H=f['H'], Pn=f['Pn'], Vu=f['Vu'])
                 for _, f in df.dropna().iterrows() if f['L'] > 0 and f['t'] > 0]
        if muros:
            rm = p3.revision_muros(muros)
            filas = pd.DataFrame(rm['filas']).rename(columns={'H_L': 'H/L', 'pasa': 'Pasa'})
            _tabla(filas, 4)
            st.caption(f'ΣVmR = {rm["sum_VmR"]:,.0f} kg  ·  ΣVu = {rm["sum_Vu"]:,.0f} kg')
            _veredicto(rm['entrepiso_pasa'] and rm['todos_pasan'], 'Pasan cada muro y el entrepiso',
                       'Algún muro o el entrepiso no pasa')
            for nombre, frac in rm['concentracion']:
                st.warning(f'Muro {nombre}: aporta {frac:.0%} de la resistencia del entrepiso (> 35 %). La NTC-S §5.7.1 '
                           'pide ×1.25 a sus fuerzas sísmicas; confírmalo con tu profesor.')
    with s_est:
        c1, c2, c3 = st.columns(3)
        Q = c1.number_input('Q', 1.0, 4.0, 2.0, 0.5, key='me_Q')
        estado = c2.selectbox('Estado', ['SV (R′ = R)', 'OI (Q = 1, R′ = 0.75R)'], key='me_est')
        FC = c3.number_input('FC', 0.5, 2.0, 1.1, key='me_fc')
        c1, c2, c3, c4 = st.columns(4)
        c_ = c1.number_input('c (g)', 0.01, 3.0, 0.6483, format='%.4f', key='me_c')
        k_ = c2.number_input('k', 0.05, 1.0, 0.505, format='%.3f', key='me_k')
        R0 = c3.number_input('R0', 1.0, 3.0, 2.0, key='me_R0')
        k1 = c4.number_input('k1', 0.5, 1.5, 1.0, key='me_k1')
        niv = st.data_editor(pd.DataFrame({'Nivel': ['2 (azotea)', '1'], 'W (t)': [70.0, 80.0], 'h (m)': [5.0, 2.5]}),
                             num_rows='dynamic', key='me_tabla', hide_index=True, width='stretch')
        niv = niv.dropna()
        if len(niv) >= 1 and (niv['W (t)'] > 0).all() and (niv['h (m)'] > 0).all():
            e = p3.estatico(niv['W (t)'], niv['h (m)'], c_, k_, Q, R0, k1, 'OI' if estado.startswith('OI') else 'SV', FC)
            tabla = pd.DataFrame({'Nivel': niv['Nivel'].values, 'W': niv['W (t)'].values, 'h': niv['h (m)'].values,
                                  'W·h': e['Wh'], 'F': e['F'], 'Fu': e['Fu'], 'Vu': e['Vu'], 'Fu·h': e['Mvu']})
            _tabla(tabla, 3)
            c1, c2, c3 = st.columns(3)
            c1.metric('c/(Q′R′)', f'{e["C"]:.5f}')
            c2.metric('VBu (t)', f'{e["VBu"]:.3f}')
            c3.metric('Mvu (t·m)', f'{e["MVU"]:.2f}')
            st.caption('Escribe los niveles de arriba hacia abajo. No aplica a Grupo A ni a estructuras fuertemente irregulares '
                       '(altura máxima 30 m regular / 20 m irregular; zona A 40 / 30 m).')
            with st.expander('Procedimiento'):
                _pasos(e['pasos'])
    with s_dist:
        est = st.selectbox('Estructuración (tabla 4.3.3)', list(p3.TABLA_433), index=1, key='md_est')
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input('Niveles sobre banqueta', 1, 30, 3, key='md_n')
        irr = c2.selectbox('Condiciones de irregularidad', [0, 1, 2, 3], key='md_irr')
        ft = c3.checkbox('Fuertemente irregular por torsión', key='md_ft')
        fe = c4.checkbox('Fuertemente irregular en elevación', key='md_fe')
        r = p3.distorsion_limite(est, int(n), int(irr), ft, fe)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Q', f'{r["Q"]:g}')
        c2.metric('γSV tabla', f'{r["gamma_SV"]:.4f}')
        c3.metric('γc', f'{r["gamma_c"]:g}')
        c4.metric('γSV corregida', f'{r["gamma_SV_corregida"]:.5f}')
        if r['gamma_OI'] is not None:
            st.caption(f'Ocupación inmediata: γOI = {r["gamma_OI"]:g} → corregida {r["gamma_OI_corregida"]:.5f}.')
        st.caption('Revisión SV: γ(del análisis sin FC) · 1.1 · Q · R′ ≤ γSV corregida. Limitación de daño: elástico × Ks, '
                   'sin FC, γ ≤ 0.002 (0.004 si los no estructurales están desligados).')
        st.markdown('**Ks y a_min**')
        c1, c2 = st.columns(2)
        Ts = c1.number_input('Ts (s)', 0.0, 5.0, 0.8, key='md_ts')
        Rp = c2.number_input("R′", 0.5, 5.0, 2.0, key='md_rp')
        c1, c2, c3 = st.columns(3)
        c1.metric('Ks', f'{ntc.ks_norma(Ts):.4f}')
        c2.metric('Zona', ntc.zona_ks(Ts))
        c3.metric('a_min', f'{p3.a_min(Ts, Rp):.4f}')
        st.markdown('**Excentricidad accidental** (ec. 2.3.2)')
        c1, c2 = st.columns(2)
        nn = c1.number_input('Número de niveles', 1, 40, 4, key='md_nn')
        bb = c2.number_input('b perpendicular al análisis (m)', 0.1, 500.0, 12.0, key='md_bb')
        ea = p3.excentricidades_accidentales(int(nn), bb)
        _tabla(pd.DataFrame({'Entrepiso': range(1, int(nn) + 1), 'ea (m)': ea}), 3)


# ------------------------------------------------------------------ viento
def _viento():
    c1, c2, c3 = st.columns(3)
    zona = c1.selectbox('Zona y velocidad regional VR (m/s)', list(p3.VR_ZONAS) + ['Otra (la capturo)'], key='vi_zona')
    if zona.startswith('Otra'):
        VR = c2.number_input('VR (m/s)', 1.0, 100.0, 30.08, key='vi_vr')
    else:
        grupo = c2.radio('Grupo', ['A (200 años)', 'B (50 años)', 'Temporal (10 años)'], index=1, key='vi_grupo')
        VR = float(p3.VR_ZONAS[zona][['A', 'B', 'T'].index(grupo[0])])
        c3.metric('VR aproximada', f'{VR:g} m/s')
        st.caption('Valores aproximados del apunte p. 27; el reglamento da VR exacta por colonia (Apéndice A). '
                   'Si el problema da VR, usa «Otra».')
    c1, c2, c3 = st.columns(3)
    terreno = c1.selectbox('Rugosidad del terreno', list(p3.TERRENOS), index=2, key='vi_ter')
    a0, d0, c0 = p3.TERRENOS[terreno]
    topo = c2.selectbox('Topografía', list(p3.TOPOGRAFIA), index=2, key='vi_topo')
    FT = p3.TOPOGRAFIA[topo]
    c3.metric('FT', f'{FT:g}')
    c1, c2, c3 = st.columns(3)
    alfa = c1.number_input('α', 0.01, 0.5, a0, format='%.3f', key=f'vi_a_{terreno}')
    delta = c2.number_input('δ (m)', 100.0, 600.0, d0, key=f'vi_d_{terreno}')
    cc = c3.number_input('c', 0.1, 2.0, c0, format='%.3f', key=f'vi_c_{terreno}')
    if terreno.startswith('R1'):
        st.caption('El α de R1 está ilegible en los apuntes (0.06?). Verifícalo en la NTC-Viento.')
    c1, c2, c3, c4 = st.columns(4)
    zt = c1.text_input('Alturas z de los niveles (m)', '3 6 9 12 15', key='vi_z')
    ancho = c2.number_input('Ancho de la cara expuesta (m)', 0.1, 500.0, 25.0, key='vi_ancho')
    largo = c3.number_input('Largo en el sentido del viento (m)', 0.1, 500.0, 35.0, key='vi_largo')
    FC = c4.number_input('FC', 0.5, 2.0, 1.1, key='vi_fc')
    try:
        z = _nums(zt, 'alturas z')
    except ValueError as e:
        st.error(str(e))
        return
    H = max(z)
    c1, c2 = st.columns(2)
    c1.metric('Tipo de estructura', p3.tipo_estructura(H, min(ancho, largo)).split(' · ')[0])
    c2.caption(f'H/A = {H / min(ancho, largo):.2f}: ' + p3.tipo_estructura(H, min(ancho, largo)).split(': ')[-1])
    caras = {'Barlovento': (0.8, False), 'Sotavento': (-0.4, True), 'Paredes laterales': (-0.8, True)}
    tabs = st.tabs(list(caras) + ['Techo plano', 'Servicio'])
    for tab, (nombre, (cp, cte)) in zip(tabs, caras.items()):
        with tab:
            cp = st.number_input(f'Cpe ({nombre})', -3.0, 3.0, cp, key=f'vi_cp_{nombre}')
            ancho_cara = largo if nombre == 'Paredes laterales' else ancho
            v = p3.viento_cara(z, ancho_cara, VR, alfa, delta, cc, FT, cp, cte, FC)
            _tabla(pd.DataFrame({'Nivel': range(1, len(z) + 1), 'z (m)': v['z'], 'Frz': v['Frz'], 'VD (m/s)': v['VD'],
                                 'pz (kg/m²)': v['pz'], 'AT (m²)': v['AT'], 'F (t)': v['F'], 'Fu (t)': v['Fu'],
                                 'Vu (t)': v['Vu'], 'Fu·z (t·m)': v['Mvu']}), 3)
            c1, c2 = st.columns(2)
            c1.metric('VBu (t)', f'{v["VBu"]:.2f}')
            c2.metric('Mvu (t·m)', f'{v["MVU"]:.2f}')
            if nombre == 'Sotavento':
                st.caption('La succión de sotavento jala en el mismo sentido del viento: para el cortante total del edificio '
                           'suma barlovento + sotavento (confírmalo con tu profesor).')
    with tabs[3]:
        cp = st.number_input('Cpe del techo', -3.0, 3.0, -0.8, key='vi_cp_techo')
        t = p3.techo_plano(largo, ancho, VR, alfa, delta, cc, H, FT, cp, FC)
        c1, c2, c3 = st.columns(3)
        c1.metric('pz (kg/m²)', f'{t["pz"]:.2f}')
        c2.metric('F (t)', f'{t["F"] / 1000:.2f}')
        c3.metric('Fu (t)', f'{t["Fu"] / 1000:.2f}')
        st.caption('Es succión (hacia arriba): se revisa contra CM + CVmín con FC = 0.9 (estabilidad).')
    with tabs[4]:
        st.markdown(f'- Desplazamiento máximo: Δmáx ≤ H/500 = {H * 100 / 500:.2f} cm\n'
                    '- Distorsión: γ ≤ 0.005 (sin elementos no estructurales) o γ ≤ 0.002 (hay elementos susceptibles a daño)')


# ------------------------------------------------------------------ concreto
def _concreto():
    s_flex, s_tant, s_cort, s_viga = st.tabs(['Flexión: As', 'Tanteos (eje neutro)', 'Cortante: estribos', 'Cargas y diagramas'])
    with s_flex:
        c1, c2, c3, c4 = st.columns(4)
        b = c1.number_input('b (cm)', 5.0, 300.0, 40.0, key='cf_b')
        h = c2.number_input('h (cm)', 10.0, 300.0, 80.0, key='cf_h')
        r = c3.number_input('r (cm)', 1.0, 20.0, 5.0, key='cf_r')
        Mu = c4.number_input('Mu (t·m)', 0.1, 5000.0, 35.03, key='cf_Mu')
        c1, c2 = st.columns(2)
        fc = c1.number_input("f′c (kg/cm²)", 100.0, 1000.0, 250.0, key='cf_fc')
        fy = c2.number_input('fy (kg/cm²)', 2000.0, 6000.0, 4200.0, key='cf_fy')
        res = p3.diseno_flexion(Mu, b, h, r, fc, fy)
        if not res['ok']:
            st.error('La sección no alcanza con acero subreforzado: cambia sección o concreto.')
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('As calculada', f'{res["As_calc"]:.2f} cm²')
            c2.metric('As de diseño', f'{res["As_dis"]:.2f} cm²')
            c3.metric('As,mín', f'{res["As_min"]:.2f} cm²')
            c4.metric('As,máx', f'{res["As_max"]:.2f} cm²')
            _tabla(pd.DataFrame(p3.proponer_armados(res['As_dis'])).assign(
                MR=lambda d: [p3.momento_resistente(a, b, res['d'], fc, fy) / 1e5 for a in d['As']]).rename(
                columns={'armado': 'Armado', 'As': 'As (cm²)', 'exceso': 'Exceso', 'MR': 'MR (t·m)'}), 2)
            st.caption('MR se verifica con el área REAL del armado. Si b ≥ 40 cm el estribo lleva 4 ramas.')
        with st.expander('Procedimiento', expanded=True):
            _pasos(res['pasos'])
    with s_tant:
        c1, c2, c3, c4 = st.columns(4)
        b = c1.number_input('b (cm)', 5.0, 300.0, 35.0, key='ct_b')
        d = c2.number_input('d (cm)', 5.0, 300.0, 55.0, key='ct_d')
        As = c3.number_input('As (cm²)', 0.1, 500.0, 11.94, key='ct_As')
        cp = c4.number_input('Tanteo de c (cm, 0 = resolver)', 0.0, 300.0, 25.0, key='ct_c')
        fc = c1.number_input("f′c", 100.0, 1000.0, 250.0, key='ct_fc')
        fy = c2.number_input('fy', 2000.0, 6000.0, 4200.0, key='ct_fy')
        lim = p3.limites_flexion(b, d, fc, fy)
        if cp > 0:
            e = p3.tanteo_eje_neutro(As, b, d, fc, fy, c_prop=cp)
            st.markdown(f'**Tanteo c = {cp:g} cm:** a = β1·c = {e["a"]:.2f} cm; C = a·b·f″c = {e["C"]:,.1f} kg; '
                        f'εs = 0.003(d − c)/c = {e["es"]:.5f} → {"fluye" if e["fluye"] else "NO fluye"}; fs = {e["fs"]:,.0f}; '
                        f'T = As·fs = {e["T"]:,.1f} kg → {"C > T: baja c" if e["C"] > e["T"] else "C < T: sube c"}.')
        e = p3.tanteo_eje_neutro(As, b, d, fc, fy)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('c (cm)', f'{e["c"]:.3f}')
        c2.metric('a (cm)', f'{e["a"]:.3f}')
        c3.metric('εs', f'{e["es"]:.5f}')
        c4.metric('MR (t·m)', f'{e["MR"] / 1e5:.2f}')
        st.caption(f'Equilibrio C = T = {e["T"]:,.1f} kg; el acero {"fluye → subreforzada" if e["fluye"] else "NO fluye → sobrerreforzada"}. '
                   f'Asb = {lim["Asb"]:.2f}, As,máx = {lim["As_max"]:.2f}, As,mín = {lim["As_min"]:.2f} cm² '
                   f'(con As = {As:g}: {"dentro de los límites" if lim["As_min"] <= As <= lim["As_max"] else "FUERA de los límites"}).')
    with s_cort:
        c1, c2, c3, c4 = st.columns(4)
        b = c1.number_input('b (cm)', 5.0, 300.0, 25.0, key='cc_b')
        h = c2.number_input('h (cm)', 10.0, 300.0, 50.0, key='cc_h')
        r = c3.number_input('r (cm)', 1.0, 20.0, 5.0, key='cc_r')
        L = c4.number_input('Claro L (cm)', 10.0, 3000.0, 500.0, key='cc_L')
        c1, c2, c3, c4 = st.columns(4)
        Vu = c1.number_input('Vu (t)', 0.1, 500.0, 14.83, key='cc_Vu')
        As = c2.number_input('As longitudinal (cm²)', 0.1, 500.0, 11.4, key='cc_As')
        est = c3.selectbox('Estribo', ['#3 (2 ramas)', '#3 (4 ramas)', '#4 (2 ramas)', '#4 (4 ramas)', '#2 (2 ramas)'], key='cc_est')
        fyest = c4.selectbox('fy del estribo', [4200.0, 2530.0], key='cc_fyest')
        c1, c2 = st.columns(2)
        fc = c1.number_input("f′c", 100.0, 1000.0, 250.0, key='cc_fc')
        av_bar = {'#2': 0.32, '#3': 0.71, '#4': 1.27}[est[:2]]
        Av = av_bar * (4 if '4 ramas' in est else 2)
        c2.metric('Av', f'{Av:.2f} cm²')
        res = p3.cortante_viga(Vu * 1000, b, h - r, h, L, fc, fyest, Av, As)
        if res['ok']:
            c1, c2, c3 = st.columns(3)
            c1.metric('VcR (kg)', f'{res["VcR"]:,.0f}')
            c2.metric('smáx (cm)', f'{res["s_max"]:.1f}')
            c3.metric('Estribos', f'{est[:2]} @ {res["s_usar"]} cm')
        else:
            st.error('Vu excede el límite 2.65·FR·b·d·√f′c: agranda la sección.')
        st.caption('Extremos de viga (L = 2h) en condición sísmica: integridad estructural s ≤ min{d/4; 8 db long.; 24 db est.; 30 cm}.')
        with st.expander('Procedimiento', expanded=True):
            _pasos(res['pasos'])
    with s_viga:
        c1, c2, c3 = st.columns(3)
        w = c1.number_input('w (t/m), ya factorizada o no', 0.01, 500.0, 3.55, key='cv_w')
        L = c2.number_input('L (m)', 0.5, 50.0, 5.0, key='cv_L')
        c3.caption('Carga última: wu = FC_CM·CM + FC_CV·CVmáx. Grupo B: 1.3 / 1.5; Grupo A: 1.5 / 1.7.')
        s = p3.viga_simple(w, L)
        e = p3.viga_doble_empotrada(w, L)
        _tabla(pd.DataFrame({'Apoyos': ['Simplemente apoyada', 'Doblemente empotrada'],
                             'Vu (t)': [s['Vu'], e['Vu']], 'Mu − (t·m)': [0.0, e['Mu_neg']],
                             'Mu + (t·m)': [s['Mu'], e['Mu_pos']]}), 3)
        st.markdown('**Wu de una viga (Grupo B/A)**')
        c1, c2, c3, c4 = st.columns(4)
        cm = c1.number_input('CM (t/m)', 0.0, 500.0, 1.0, key='cv_cm')
        cvm = c2.number_input('CVmáx (t/m)', 0.0, 500.0, 0.5, key='cv_cvm')
        pp = c3.number_input('Peso propio (t/m)', 0.0, 50.0, 0.768, key='cv_pp')
        grupo = c4.radio('Grupo', ['B', 'A'], horizontal=True, key='cv_grupo')
        fcm, fcv = (1.3, 1.5) if grupo == 'B' else (1.5, 1.7)
        st.metric(f'wu = {fcm}(CM + pp) + {fcv}·CVmáx', f'{fcm * (cm + pp) + fcv * cvm:.4f} t/m')


def mostrar():
    st.caption('Tercer parcial del Dr. Rivero (apuntes pp. 22–35): mampostería, viento y concreto reforzado. Cada módulo '
               'viene validado contra los ejemplos de tus apuntes. Las normas de material (Mampostería, Viento, Concreto) '
               'se toman de tus apuntes tal cual.')
    t_m, t_v, t_c = st.tabs(['Mampostería', 'Viento', 'Concreto reforzado'])
    with t_m:
        _mamposteria()
    with t_v:
        _viento()
    with t_c:
        _concreto()
