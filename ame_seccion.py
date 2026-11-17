"""Examen 3 · viga de concreto reforzado por secciones: varios puntos de flexión (cada uno con su Mu) y varios tramos de cortante
(cada uno con su Vu), y al final el armado de TODA la viga: varillas corridas, bastones, estribos por tramo, confinamiento y piel.

`calcular(d, puntos, tramos)` usa los mismos motores de las demás pantallas (ame_parcial3 y ame_porticos) y `bloques(res)` lo pasa al
formato de ame_detalle: la pantalla, el PDF, el Excel y el texto muestran lo mismo. Una sección sola es el caso de un punto y un tramo.
"""
import io
import math

import ame_detalle as det
import ame_parcial3 as p3
import ame_porticos as pt

DEFECTO = dict(b=35.0, h=70.0, r=5.0, fc=250.0, fy=4200.0, var='#8', est='#4', ramas=2, fy_est=4200.0, s_tope=25.0)


def _limpio(txt):
    return str(txt).replace('**', '')


def calcular(d, puntos, tramos):
    """d: b, h, r (cm); fc, fy (kg/cm²); var, est (#n); ramas; fy_est; s_tope (cm).
    puntos: [dict(punto, Mu (t·m), lecho 'superior'|'inferior', prop '4#8' o '', c_prop cm o 0)].
    tramos: [dict(tramo, Vu (t), As (cm², None = el mayor As real superior), s_prop cm o 0)]."""
    d = {**DEFECTO, **d}
    b, h, r, fc, fy = (float(d[k]) for k in ('b', 'h', 'r', 'fc', 'fy'))
    var, est, ramas = d['var'], d['est'], int(d['ramas'])
    fy_est, s_tope = float(d['fy_est']), float(d['s_tope'])
    if min(b, h, fc, fy, fy_est) <= 0:
        raise ValueError('Las dimensiones, f′c y fy deben ser positivos.')
    if r >= h / 2:
        raise ValueError('r debe ser menor que la mitad de h.')
    if not puntos:
        raise ValueError('Captura al menos un punto de flexión (Mu).')
    if not tramos:
        raise ValueError('Captura al menos un tramo de cortante (Vu).')
    if any(float(p['Mu']) < 0 for p in puntos) or any(float(t['Vu']) < 0 for t in tramos):
        raise ValueError('Mu y Vu no pueden ser negativos: captura sus valores absolutos.')
    db, est_db, a_est, ab = pt.DIAMETROS[var], pt.DIAMETROS[est], p3.VARILLAS[est], p3.VARILLAS[var]
    Av = a_est * ramas
    rec = max(r - est_db - db / 2, 1.5)
    Mu_max = max(float(p['Mu']) for p in puntos)
    Vu_max = max(float(t['Vu']) for t in tramos)
    viga = pt.ficha_viga(b, h, fc, fy, var, est, ramas, rec, r_fija=r, Mu_max=Mu_max, Vu_max=Vu_max)
    dd = h - r
    lim = p3.limites_flexion(b, dd, fc, fy)
    res = dict(d=d, b=b, h=h, r=r, fc=fc, fy=fy, var=var, est=est, ramas=ramas, fy_est=fy_est, s_tope=s_tope, Av=Av, db=db, ab=ab,
               est_db=est_db, rec=rec, viga=viga, d_ef=dd, lim=lim, Mu_max=Mu_max, Vu_max=Vu_max, error=None, avisos=[])

    pts = []
    for p in puntos:
        Mu = float(p['Mu'])
        ficha = pt.ficha_punto(Mu, b, h, fc, fy, var, rec=rec, est_db=est_db, r_fija=r, agrupar=True, lecho=p['lecho'],
                               propuesta=pt.parse_propuesta(p.get('prop', '')))
        e = dict(punto=str(p['punto']), lecho=p['lecho'], Mu=Mu, ficha=ficha, c_prop=float(p.get('c_prop') or 0), error=None,
                 alternativas=[], tanteo=None, tanteo_manual=None)
        if not ficha['As_real']:
            e['error'] = 'La sección no alcanza con acero subreforzado para este Mu: aumenta b, h o f′c.'
        else:
            e['alternativas'] = [dict(a, MR=p3.momento_resistente(a['As'], b, ficha['d'], fc, fy) / 1e5)
                                 for a in p3.proponer_armados(ficha['As_usar'])] if ficha['As_usar'] else []
            e['tanteo'] = p3.tanteo_eje_neutro(ficha['As_real'], b, ficha['d'], fc, fy)
            if e['c_prop'] > 0:
                e['tanteo_manual'] = p3.tanteo_eje_neutro(ficha['As_real'], b, ficha['d'], fc, fy, c_prop=e['c_prop'])
        pts.append(e)
    res['puntos'] = pts
    buenos = [e for e in pts if e['ficha']['As_real']]
    if not buenos:
        res.update(error='Ninguna sección alcanza con acero subreforzado: aumenta b, h o f′c.', ok=False, ok_flex=False, ok_cort=False,
                   tramos=[], corrido=None, integ=None, piel=None)
        return res

    # varillas corridas: máx(As,mín; 50 % del mayor As real), arriba y abajo; el resto son bastones
    As_max_real = max(e['ficha']['As_real'] for e in buenos)
    As_corr = max(lim['As_min'], 0.5 * As_max_real)
    n_corr = max(2, int(math.ceil(As_corr / ab - 1e-9)))
    bastones = [(e['punto'], e['ficha']['n'] - n_corr) for e in buenos if e['ficha']['n'] > n_corr]
    res['corrido'] = dict(As_max_real=As_max_real, As_corr=As_corr, n=n_corr, As=n_corr * ab, bastones=bastones)

    sup = [e['ficha']['As_real'] for e in buenos if e['lecho'] == 'superior']
    As_auto = max(sup) if sup else As_max_real
    res['As_auto'] = As_auto
    res['lim_cort'] = p3.limites_cortante_tex(b, dd, fc, Av, ramas, a_est)
    trs = []
    for t in tramos:
        Vu = float(t['Vu'])
        As_t = t.get('As')
        As_t = As_auto if As_t in (None, '') or float(As_t) <= 0 else float(As_t)
        cort = p3.cortante_ficha(Vu * 1000, b, dd, fc, fy_est, Av, As_t, s_tope)
        s_prop = float(t.get('s_prop') or 0)
        trs.append(dict(tramo=str(t['tramo']), Vu=Vu, As=As_t, cort=cort, s_prop=s_prop,
                        alt=(p3.alternativas_estribo(Vu * 1000, b, dd, fc, fy_est, As_t, s_tope) if cort['ok'] and cort['s_usar'] < 10 else []),
                        rev=(p3.revision_cortante(Vu * 1000, b, dd, fc, fy_est, Av, As_t, s_prop, s_tope) if s_prop > 0 else None)))
    res['tramos'] = trs
    res['integ'] = p3.integridad_estructural(h, dd, db, est_db, s_tope)
    res['piel'] = p3.acero_piel(b, h, r, est if est in ('#3', '#4') else '#3')
    res['ok_flex'] = bool(viga['ok_flexion'] and all(e['ficha']['ok'] for e in pts))
    res['ok_cort'] = bool(viga['ok_cortante'] and all(t['cort']['ok'] for t in trs) and all(t['rev']['ok'] for t in trs if t['rev']))
    res['ok'] = res['ok_flex'] and res['ok_cort']
    return res


def figura(res):
    """Cuadrícula con la sección de cada punto (lecho en tensión)."""
    buenos = [e for e in res['puntos'] if e['ficha']['As_real']]
    return pt.figura_puntos([e['ficha'] for e in buenos], res['b'], res['h'], res['rec'], res['est_db'], [f'{e["punto"]})' for e in buenos])


def _lineas_tanteo(e):
    return [f'c = {e["c"]:.3f} cm → a = β1·c = {e["a"]:.3f} cm; C = a·b·f″c = {e["C"]:,.1f} kg',
            f'εs = 0.003(d − c)/c = {e["es"]:.5f} → {"el acero fluye" if e["fluye"] else "el acero NO fluye"}; fs = {e["fs"]:,.0f} kg/cm²',
            f'T = As·fs = {e["T"]:,.1f} kg → {"C > T: baja c" if e["C"] > e["T"] + 1 else "C < T: sube c" if e["C"] < e["T"] - 1 else "C = T"}',
            f'MR = FR·T·(d − a/2) = {e["MR"] / 1e5:.2f} t·m']


def tabla_flexion(res):
    """(encabezados, filas) de la flexión punto por punto."""
    enc = ['Punto', 'Lecho', 'Mu (t·m)', 'As req (cm²)', 'Armado', 'As real (cm²)', 'MR (t·m)', 'Sep. libre (cm)', 'OK']
    filas = []
    for e in res['puntos']:
        f = e['ficha']
        if f['As_real']:
            filas.append([e['punto'], e['lecho'], e['Mu'], f['As_usar'], f['armado'], f['As_real'], f['MR'], f['sep_libre'],
                          '✔' if f['ok'] else '✘'])
        else:
            filas.append([e['punto'], e['lecho'], e['Mu'], '—', '—', '—', '—', '—', '✘'])
    return enc, filas


def tabla_cortante(res):
    enc = ['Tramo', 'Vu (t)', 'As tens. (cm²)', 'ρ', 'VcR (kg)', 'Vu−VcR (kg)', 's calc (cm)', 'smáx (cm)', 'Estribos']
    filas = []
    for t in res['tramos']:
        c = t['cort']
        filas.append([t['tramo'], t['Vu'], t['As'], c['rho'], c['VcR'], c['Vs'], c['s_calc'] if c['s_calc'] is not None else '—',
                      c['s_max'], f'E{res["est"]}, {res["ramas"]}r @ {c["s_usar"]} cm' if c['ok'] else 'NO'])
    return enc, filas


def resumen(res):
    """[(etiqueta, valor, detalle)] de la viga completa."""
    cor, piel, integ = res['corrido'], res['piel'], res['integ']
    items = [('Varillas corridas', f'{cor["n"]}{res["var"]} arriba y abajo',
              f'As = {cor["As"]:.2f} cm² (máx(As,mín; 50 % de {cor["As_max_real"]:.2f}) = {cor["As_corr"]:.2f})')]
    if cor['bastones']:
        items.append(('Bastones', ' · '.join(f'{p}) +{n}{res["var"]}' for p, n in cor['bastones']), 'se suman a las corridas en cada punto'))
    ests = {}
    for t in res['tramos']:
        ests.setdefault(f'E{res["est"]} {res["ramas"]}r @ {t["cort"]["s_usar"]} cm' if t['cort']['ok'] else 'NO PASA', []).append(t['tramo'])
    items.append(('Estribos', ' · '.join(f'{k} (tramo {", ".join(v)})' for k, v in ests.items()), f'Vu máx = {res["Vu_max"]:g} t'))
    items.append(('Confinamiento', f'2h = {integ["Lconf"]:g} cm', f'estribos @ {integ["s_usar"]} cm en cada extremo'))
    items.append(('Acero de piel', f'{piel["n_cara"]}{piel["var"]} por cara @ {piel["sep"]:.1f} cm' if piel['aplica'] else 'No se requiere',
                  f'As colocado {piel["As_col"]:.2f} cm²' if piel['aplica'] else 'h ≤ 60 cm'))
    return items


def bloques(res, con_figura=True):
    """Bloques de ame_detalle para el PDF, el Excel y el texto."""
    sub = (f'b × h = {res["b"]:g} × {res["h"]:g} cm · r = {res["r"]:g} cm · f′c = {res["fc"]:g} · fy = {res["fy"]:g} kg/cm² · '
           f'{len(res["puntos"])} punto(s) de flexión · {len(res.get("tramos") or [])} tramo(s) de cortante')
    B = [det.titulo('Viga de concreto reforzado', sub)]
    if res['error']:
        return B + [det.paso(1, 'Revisión de la sección', res['error'], 'Cambia las dimensiones o el concreto y vuelve a calcular.')]
    B.append(det.destacado('Armado de toda la viga', resumen(res)))
    malos = [t for t, ok in (('flexión', res['ok_flex']), ('cortante', res['ok_cort'])) if not ok]
    B.append(det.nota('La viga cumple: subreforzada en todos los puntos, el cortante no excede el límite y el armado alcanza.' if res['ok']
                      else 'La viga NO cumple una de las revisiones: ' + ', '.join(malos) + '.'))
    B.append(det.paso(1, 'Datos y revisiones de la viga', 'Materiales, peralte, límites de acero y cortante máximo.',
                      f'MR con As,máx contra el Mu mayor ({res["Mu_max"]:g} t·m) y V límite contra el Vu mayor ({res["Vu_max"]:g} t).'))
    B += [det.texto(_limpio(t)) for t in res['viga']['lineas']]
    B.append(det.paso(2, 'Flexión en cada punto', 'Cuadrática de equilibrio con FR = 0.9, As de diseño y número de varillas.',
                      'MR con el área real del armado y separación libre entre barras.'))
    enc, filas = tabla_flexion(res)
    B.append(det.tabla('Resumen de flexión', enc, filas, 2))
    for e in res['puntos']:
        B.append(det.texto(f'Punto {e["punto"]}) · Mu = {e["Mu"]:g} t·m · tensión {e["lecho"]}'))
        B += [det.texto(_limpio(t)) for t in (e['ficha']['lineas'] if not e['error'] else [e['error']])]
        if e['alternativas']:
            B.append(det.tabla('Otros armados posibles', ['Armado', 'As (cm²)', 'Exceso (cm²)', 'MR (t·m)'],
                               [[a['armado'], a['As'], a['exceso'], a['MR']] for a in e['alternativas']], 2))
    if con_figura:
        B.append(det.figura('Secciones con el armado a tensión', figura(res), 17))
    B.append(det.paso(3, 'Tanteo del eje neutro', 'Con el As real de cada punto se busca c hasta que C = T.',
                      'Se revisa que el acero fluya (subreforzada) y se obtiene MR.'))
    filas = [[e['punto'], e['ficha']['As_real'], e['tanteo']['c'], e['tanteo']['a'], e['tanteo']['es'], 'sí' if e['tanteo']['fluye'] else 'NO',
              e['tanteo']['MR'] / 1e5] for e in res['puntos'] if e['tanteo']]
    B.append(det.tabla('Equilibrio C = T', ['Punto', 'As real (cm²)', 'c (cm)', 'a (cm)', 'εs', 'Fluye', 'MR (t·m)'], filas, 4))
    for e in res['puntos']:
        if e['tanteo_manual']:
            B.append(det.texto(f'Punto {e["punto"]}) con el tanteo c = {e["c_prop"]:g} cm:'))
            B += [det.texto(t) for t in _lineas_tanteo(e['tanteo_manual'])]
    cor = res['corrido']
    B.append(det.paso(4, 'Varillas corridas y bastones', 'Corrido a todo lo largo, arriba y abajo, con ganchos al inicio y al final.',
                      'Los bastones completan el As real de cada punto y se anclan una distancia d más allá de donde ya no se requieren.'))
    B.append(det.texto(f'Mayor As real = {cor["As_max_real"]:.2f} cm²; 50 % = {0.5 * cor["As_max_real"]:.2f} cm²; As,mín = {res["lim"]["As_min"]:.2f} cm²'))
    B.append(det.texto(f'Corrido = máx(As,mín; 50 % As mayor) = {cor["As_corr"]:.2f} cm² → n = {cor["As_corr"]:.2f}/{res["ab"]} = '
                       f'{cor["As_corr"] / res["ab"]:.2f} → {cor["n"]}{res["var"]} ({cor["As"]:.2f} cm²) arriba y abajo'))
    B.append(det.texto('Bastones por punto: ' + (' · '.join(f'{p}) +{n}{res["var"]}' for p, n in cor['bastones']) if cor['bastones']
                                                  else 'ninguno, las corridas alcanzan en todos los puntos')))
    B.append(det.paso(5, 'Cortante: estribos por tramo', f'Estribo E{res["est"]} de {res["ramas"]} ramas (Av = {res["Av"]:.2f} cm²), '
                      f'fy = {res["fy_est"]:g} kg/cm², tope de separación {res["s_tope"]:g} cm.',
                      f'ρ con el As en tensión de cada tramo (si no lo das: el mayor As real superior, {res["As_auto"]:.2f} cm²).'))
    lc = p3.limites_cortante(res['b'], res['d_ef'], res['fc'])
    B.append(det.tabla('Límites (kg)', ['1.1·FR·b·d·√f′c', '1.25·FR·b·d·√f′c', '2.2·FR·b·d·√f′c', '0.25·FR·b·d·√f′c', '2.65·FR·b·d·√f′c'],
                       [[lc['L11'], lc['L125'], lc['L22'], lc['Vmin'], lc['Vmax']]], 0))
    enc, filas = tabla_cortante(res)
    B.append(det.tabla('Resumen de cortante', enc, filas, 4))
    for t in res['tramos']:
        B.append(det.texto(f'Tramo {t["tramo"]} · Vu = {t["Vu"]:g} t'))
        B += [det.texto(_limpio(x)) for x in t['cort']['lineas']]
        if not t['cort']['ok']:
            B.append(det.nota('Vu excede el límite de la sección: agranda b o h.'))
        if t['alt']:
            B.append(det.nota(f'@ {t["cort"]["s_usar"]} cm es muy cerrado para colar. Alternativas con s ≥ 10 cm: '
                              + ' · '.join(o['texto'] for o in t['alt'][:4])))
        if t['rev']:
            rv = t['rev']
            B.append(det.texto(f'Revisión de s = {t["s_prop"]:g} cm: VsR = {rv["VsR"] / 1000:.2f} t; VR = VcR + VsR = {rv["VR"] / 1000:.2f} t '
                               f'{"≥" if rv["ok_resistencia"] else "<"} Vu = {t["Vu"]:.2f} t; s {"≤" if rv["ok_separacion"] else ">"} '
                               f'smáx = {rv["s_max"]:.2f} cm → {"PASA" if rv["ok"] else "NO PASA"}'))
    B.append(det.paso(6, 'Zona de confinamiento y acero de piel', 'Integridad estructural: 2h desde la cara de cada apoyo.',
                      'Acero por temperatura en las caras laterales si h > 60 cm.'))
    B += [det.texto(t) for t in res['integ']['lineas']]
    B += [det.texto(t) for t in res['piel']['lineas']]
    return B


def pdf(res, proyecto='', autor=''):
    """PDF de la viga (bytes)."""
    import datetime

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import cm
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle

    import ame_pdf as ap
    ap._registrar_fuentes()
    st = ap._estilos()
    Ancho = letter[0] - 3.6 * cm
    buf = io.BytesIO()

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(ap.AZUL_CLARO)
        canvas.line(1.8 * cm, 1.5 * cm, letter[0] - 1.8 * cm, 1.5 * cm)
        canvas.setFont('DejaVu', 7.5)
        canvas.setFillColor(ap.AZUL_MED)
        canvas.drawString(1.8 * cm, 1.05 * cm, 'Viga de concreto reforzado' + (f' · {proyecto}' if proyecto else ''))
        canvas.drawRightString(letter[0] - 1.8 * cm, 1.05 * cm, f'Página {doc.page}')
        canvas.restoreState()

    doc = BaseDocTemplate(buf, pagesize=letter, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=2 * cm,
                          title='Viga de concreto reforzado', author=autor or 'AME')
    doc.addPageTemplates([PageTemplate(id='p', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')],
                                       onPage=pie)])
    bl = bloques(res)
    banda = Table([[Paragraph('<font color="white" size="19"><b>Viga de concreto reforzado</b></font><br/>'
                              f'<font color="#81A4CD" size="9">{ap._escapar(bl[0]["sub"])}</font>', st['base']),
                    Paragraph(f'<font color="#81A4CD" size="8.5">Proyecto: {ap._escapar(proyecto or "—")}<br/>'
                              f'Elaboró: {ap._escapar(autor or "—")}<br/>Fecha: {datetime.date.today().strftime("%d/%m/%Y")}</font>',
                              st['base'])]], colWidths=[Ancho * 0.7, Ancho * 0.3])
    banda.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), ap.NAVY), ('LEFTPADDING', (0, 0), (-1, -1), 14),
                               ('TOPPADDING', (0, 0), (-1, -1), 12), ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    E = [banda, Spacer(1, 8)] + ap._bloques(bl[1:], st, Ancho)
    try:
        doc.build(E)
    finally:
        for x in bl:
            if x['t'] == 'figura':
                import matplotlib.pyplot as plt
                plt.close(x['fig'])
    return buf.getvalue()


def excel(res):
    """Libro de Excel con el mismo procedimiento (bytes)."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = 'Viga'
    det.bloques_a_hoja(ws, bloques(res, con_figura=False))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def texto(res):
    """Procedimiento en texto plano (UTF-8)."""
    return det.bloques_a_texto(bloques(res, con_figura=False))
