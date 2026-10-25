"""Memoria de cálculo en PDF del Análisis Modal Espectral."""
import datetime
import io
import os
import re

import numpy as np
from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

import ame_detalle
import ame_sismico as ame

NAVY, AZUL_OSC, AZUL, AZUL_MED, AZUL_CLARO, PALE, BG = (colors.HexColor(c) for c in (
    '#0B2545', '#13315C', '#1D4E89', '#3E7CB1', '#81A4CD', '#DCE9F7', '#F1F6FC'))
ROJO = colors.HexColor('#B3261E')

_FUENTES_OK = False


def _registrar_fuentes():
    """DejaVu Sans (incluida con matplotlib): trae letras griegas, Ø, ², Σ, etc."""
    global _FUENTES_OK
    if _FUENTES_OK:
        return
    carpeta = os.path.join(get_data_path(), 'fonts', 'ttf')
    for nombre, archivo in (('DejaVu', 'DejaVuSans.ttf'), ('DejaVu-Bold', 'DejaVuSans-Bold.ttf'),
                            ('DejaVu-Italic', 'DejaVuSans-Oblique.ttf'), ('DejaVu-BoldItalic', 'DejaVuSans-BoldOblique.ttf')):
        pdfmetrics.registerFont(TTFont(nombre, os.path.join(carpeta, archivo)))
    pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold', italic='DejaVu-Italic',
                                  boldItalic='DejaVu-BoldItalic')
    _FUENTES_OK = True


def _estilos():
    base = ParagraphStyle('base', fontName='DejaVu', fontSize=9, leading=12.5, textColor=NAVY, alignment=TA_LEFT)
    return dict(
        base=base,
        h1=ParagraphStyle('h1', parent=base, fontName='DejaVu-Bold', fontSize=14, leading=18, textColor=AZUL,
                          spaceBefore=14, spaceAfter=6, keepWithNext=1),
        h2=ParagraphStyle('h2', parent=base, fontName='DejaVu-Bold', fontSize=10.5, leading=14, textColor=AZUL_OSC,
                          spaceBefore=8, spaceAfter=3, keepWithNext=1),
        h2s=ParagraphStyle('h2s', parent=base, fontName='DejaVu-Bold', fontSize=10.5, leading=14, textColor=AZUL_OSC,
                           spaceBefore=8, spaceAfter=3),
        nota=ParagraphStyle('nota', parent=base, fontSize=8, leading=11, textColor=AZUL_MED),
        formula=ParagraphStyle('formula', parent=base, fontName='DejaVu-Italic', leftIndent=14, textColor=AZUL_OSC,
                               spaceAfter=2),
        th=ParagraphStyle('th', parent=base, fontName='DejaVu-Bold', fontSize=8, leading=10, textColor=colors.white,
                          alignment=1),
    )


_ETIQUETA = re.compile(r'<(?!/?(?:b|i|u|font|sub|super|br)[\s>/])')


def _escapar(txt):
    """Escapa el '<' de las fórmulas (T<Ta) sin tocar las etiquetas de formato del Paragraph."""
    return _ETIQUETA.sub('&lt;', txt.replace('&', '&amp;'))


def _f(v, dec=4):
    if not isinstance(v, (int, float, np.number)):
        return v  # texto o Paragraph
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if abs(v) >= 1e5:
        return f'{v:,.1f}'
    return f'{v:.{dec}f}'


def _tabla(st, enc, filas, dec=4, anchos=None, fuente=8, zebra=True):
    datos = [[Paragraph(h, st['th']) for h in enc]] + [[_f(v, dec) for v in fila] for fila in filas]
    t = Table(datos, colWidths=anchos, repeatRows=1, hAlign='LEFT')
    estilo = [('FONTNAME', (0, 0), (-1, -1), 'DejaVu'), ('FONTSIZE', (0, 0), (-1, -1), fuente),
              ('TEXTCOLOR', (0, 1), (-1, -1), NAVY), ('BACKGROUND', (0, 0), (-1, 0), AZUL),
              ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
              ('GRID', (0, 0), (-1, -1), 0.4, AZUL_CLARO), ('TOPPADDING', (0, 0), (-1, -1), 3),
              ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if zebra:
        estilo += [('BACKGROUND', (0, i), (-1, i), BG) for i in range(2, len(datos), 2)]
    t.setStyle(TableStyle(estilo))
    return t


def _imagen(fig, ancho_cm):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160)
    buf.seek(0)
    w, h = fig.get_size_inches()
    ancho = ancho_cm * cm
    return Image(buf, width=ancho, height=ancho * h / w)


def _anchos(enc, filas, total):
    """Anchos de columna proporcionales al contenido (con un mínimo), para que las tablas anchas quepan."""
    pesos = []
    for j, h in enumerate(enc):
        largo = max([len(str(h))] + [len(str(f[j])) if not isinstance(f[j], float) else 12 for f in filas if j < len(f)] + [4])
        pesos.append(min(max(largo, 8), 34))
    suma = sum(pesos)
    return [total * w / suma for w in pesos]


def _sub(txt):
    """k_{c4} -> k<sub>c4</sub> para los símbolos de las matrices."""
    return re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', _escapar(str(txt)))


def _matriz_pdf(b, st, Ancho):
    datos = b['datos']
    n_f, n_c = datos.shape
    filas_l = b['filas'] or [str(i + 1) for i in range(n_f)]
    cols = b['cols'] or [str(j + 1) for j in range(n_c)]
    simb = b['simbolica'] is not None
    fuente = 9 if n_c <= 5 else (7.5 if n_c <= 7 else 6)
    if simb:
        celdas = [[Paragraph(_sub(b['simbolica'][i][j]) or '0', ParagraphStyle('c', parent=st['base'], fontSize=fuente, alignment=1))
                   for j in range(n_c)] for i in range(n_f)]
    else:
        celdas = [[_f(float(datos[i, j]), b['dec']) for j in range(n_c)] for i in range(n_f)]
    cab = [''] + [Paragraph(f'<b>{_escapar(c)}</b>', ParagraphStyle('cc', parent=st['base'], fontSize=7, alignment=1,
                                                                   textColor=AZUL_MED)) for c in cols]
    cuerpo = [[Paragraph(f'<b>{_escapar(filas_l[i])}</b>', ParagraphStyle('rr', parent=st['base'], fontSize=7,
                                                                         textColor=AZUL_MED))] + celdas[i] for i in range(n_f)]
    ancho_f = 1.4 * cm
    ancho_c = min((Ancho - ancho_f) / n_c, 3.6 * cm)
    t = Table([cab] + cuerpo, colWidths=[ancho_f] + [ancho_c] * n_c, hAlign='LEFT')
    t.setStyle(TableStyle([('FONTNAME', (1, 1), (-1, -1), 'DejaVu'), ('FONTSIZE', (1, 1), (-1, -1), fuente),
                           ('TEXTCOLOR', (1, 1), (-1, -1), NAVY), ('ALIGN', (1, 1), (-1, -1), 'RIGHT' if not simb else 'CENTER'),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2.5),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
                           ('LINEBEFORE', (1, 1), (1, -1), 1.4, AZUL), ('LINEAFTER', (-1, 1), (-1, -1), 1.4, AZUL),
                           ('LINEABOVE', (1, 1), (-1, 1), 0.4, AZUL_CLARO), ('LINEBELOW', (1, -1), (-1, -1), 0.4, AZUL_CLARO),
                           ('BACKGROUND', (1, 1), (-1, -1), BG)]))
    return t


def _bloques(bloques, st, Ancho):
    """Convierte los bloques de ame_detalle en elementos de reportlab."""
    P = lambda txt, est='base': Paragraph(_escapar(txt), st[est])
    E = []
    for b in bloques:
        t = b['t']
        if t == 'titulo':
            banda = Table([[Paragraph('<font color="white" size="20"><b>' + _escapar(b['texto']) + '</b></font>'
                                      + (f'<br/><font color="#81A4CD" size="10.5">{_escapar(b["sub"])}</font>' if b['sub'] else ''),
                                      st['base'])]], colWidths=[Ancho])
            banda.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), NAVY), ('LEFTPADDING', (0, 0), (-1, -1), 14),
                                       ('TOPPADDING', (0, 0), (-1, -1), 14), ('BOTTOMPADDING', (0, 0), (-1, -1), 14)]))
            E += [banda, Spacer(1, 10)]
        elif t == 'paso':
            cab = Table([[Paragraph(f'<font color="white" size="11"><b>PASO {b["n"]}</b></font>', st['base']),
                          Paragraph('<font size="13"><b>' + _escapar(b['texto']) + '</b></font>', ParagraphStyle(
                              'pt', parent=st['base'], textColor=AZUL_OSC, fontSize=13, leading=16))]],
                        colWidths=[2.2 * cm, Ancho - 2.2 * cm])
            cab.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, 0), AZUL), ('BACKGROUND', (1, 0), (1, 0), PALE),
                                     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 6),
                                     ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('LEFTPADDING', (0, 0), (-1, -1), 8)]))
            expl = Table([[Paragraph(_escapar(' '.join(b['expl'])), ParagraphStyle('ex', parent=st['base'], fontSize=9.5,
                                                                                 leading=13.5, textColor=AZUL_OSC))]],
                         colWidths=[Ancho])
            expl.setStyle(TableStyle([('LINEBEFORE', (0, 0), (0, 0), 3, AZUL_MED), ('LEFTPADDING', (0, 0), (-1, -1), 10),
                                      ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            E += [Spacer(1, 26), KeepTogether([cab, Spacer(1, 4), expl]), Spacer(1, 8)]
        elif t == 'seccion':
            ban = Table([[Paragraph('<font color="white" size="15"><b>' + _escapar(b['letra']) + '</b></font>', st['base']),
                          Paragraph('<font size="13"><b>' + _escapar(b['texto']) + '</b></font>', ParagraphStyle(
                              'sec', parent=st['base'], textColor=colors.white, fontSize=13, leading=16))]],
                        colWidths=[1.3 * cm, Ancho - 1.3 * cm])
            ban.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), NAVY), ('BACKGROUND', (0, 0), (0, 0), AZUL),
                                     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                                     ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                     ('LEFTPADDING', (1, 0), (1, 0), 12)]))
            E += [Spacer(1, 14), CondPageBreak(9 * cm), ban, Spacer(1, 10)]
        elif t == 'sub':
            E += [CondPageBreak(6 * cm), Spacer(1, 8), P(b['texto'], 'h2')]
        elif t == 'destacado':
            celdas = [[Paragraph('<b>' + _escapar(e_) + '</b>', ParagraphStyle('de', parent=st['nota'], textColor=NAVY)),
                       Paragraph('<font size="14"><b>' + _escapar(v_) + '</b></font>',
                                 ParagraphStyle('dv', parent=st['base'], textColor=AZUL, alignment=1, leading=17)),
                       Paragraph(_escapar(dt_), ParagraphStyle('dd', parent=st['nota'], alignment=1))]
                      for e_, v_, dt_ in b['items']]
            tt = Table(celdas, colWidths=[Ancho * 0.5, Ancho * 0.25, Ancho * 0.25])
            tt.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF4D6')),
                                    ('LINEBELOW', (0, 0), (-1, -2), 0.4, colors.white),
                                    ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#E3B341')),
                                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                    ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            E += [KeepTogether([P(b['titulo'], 'h2'), tt]), Spacer(1, 8)]
        elif t == 'figura':
            img = _imagen(b['fig'], b['ancho'])
            E += [CondPageBreak(img.drawHeight + 1.8 * cm), P(b['titulo'], 'h2'), img, Spacer(1, 10)]
        elif t == 'formula':
            E += [P(b['texto'], 'formula'), Spacer(1, 6)]
        elif t == 'matriz':
            tit = b['nombre'] + (f'   [{b["unidad"]}]' if b['unidad'] else '')
            E += [KeepTogether([P(tit, 'h2'), _matriz_pdf(b, st, Ancho)]), Spacer(1, 12)]
        elif t == 'tabla':
            filas = [[(Paragraph(_escapar(v), ParagraphStyle('tc', parent=st['base'], fontSize=7.5, leading=9.5, alignment=1))
                       if isinstance(v, str) and len(v) > 14 else v) for v in f] for f in b['filas']]
            fuente = 8 if len(b['enc']) <= 6 else 7
            E += ([Paragraph(_escapar(b['titulo']), st['h2s'] if len(b['filas']) > 8 else st['h2'])] if b['titulo'] else [])
            E += [_tabla(st, [_escapar(h) for h in b['enc']], filas, dec=b['dec'], fuente=fuente,
                         anchos=_anchos(b['enc'], b['filas'], Ancho)), Spacer(1, 12)]
        elif t in ('texto', 'nota'):
            E += [P(b['texto'], 'base' if t == 'texto' else 'nota'), Spacer(1, 6)]
    return E


def memoria_pdf(res, destino, proyecto='', autor='', metodo='ambos'):
    """Memoria de cálculo compacta, como las de los exámenes: la estructura arriba, las matrices K y M con el planteamiento
    de K, los periodos y formas modales, el espectro, las fuerzas por nivel y el método estático.
    `destino` es una ruta o un BytesIO. metodo: 'modal', 'estatico' o 'ambos'."""
    _registrar_fuentes()
    st = _estilos()
    p, n = res['p'], res['p']['n']
    P = lambda txt, est='base': Paragraph(_escapar(txt), st[est])
    Ancho = letter[0] - 3.6 * cm
    modal = metodo in ('ambos', 'modal')
    estat = metodo in ('ambos', 'estatico') and 'tabla_est' in res
    titulo_an = {(True, True): 'Análisis modal espectral y análisis estático', (True, False): 'Análisis modal espectral',
                 (False, True): 'Análisis estático'}[(modal, estat)]

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(AZUL_CLARO)
        canvas.line(1.8 * cm, 1.5 * cm, letter[0] - 1.8 * cm, 1.5 * cm)
        canvas.setFont('DejaVu', 7.5)
        canvas.setFillColor(AZUL_MED)
        canvas.drawString(1.8 * cm, 1.05 * cm, f'Memoria de cálculo · {titulo_an}' + (f' · {proyecto}' if proyecto else ''))
        canvas.drawRightString(letter[0] - 1.8 * cm, 1.05 * cm, f'Página {doc.page}')
        canvas.restoreState()

    doc = BaseDocTemplate(destino, pagesize=letter, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                          bottomMargin=2 * cm, title=f'Memoria de cálculo - {titulo_an}', author=autor or 'AME')
    doc.addPageTemplates([PageTemplate(id='p', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                                                             id='f')], onPage=pie)])
    E = []

    # ------------------------------------------------------------------ portada: la estructura arriba
    banda = Table([[Paragraph('<font color="white" size="19"><b>Memoria de cálculo</b></font><br/>'
                              f'<font color="#81A4CD" size="10.5">{_escapar(titulo_an)}</font>', st['base']),
                    Paragraph(f'<font color="#81A4CD" size="8.5">Proyecto: {_escapar(proyecto or "—")}<br/>'
                              f'Elaboró: {_escapar(autor or "—")}<br/>'
                              f'Fecha: {datetime.date.today().strftime("%d/%m/%Y")}</font>', st['base'])]],
                  colWidths=[Ancho * 0.62, Ancho * 0.38])
    banda.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), NAVY), ('LEFTPADDING', (0, 0), (-1, -1), 14),
                               ('TOPPADDING', (0, 0), (-1, -1), 12), ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    E += [banda, Spacer(1, 6)]
    if 'geom' in p:
        E.append(_imagen(ame.fig_estructura(p), 13.4))
    elif p.get('manual'):
        E.append(P('La estructura se definió con las matrices K y M (sin geometría).', 'nota'))

    # resumen de resultados
    fe = res.get('factor_escala')
    est_d = res['revision']['estados'][res['clave']]
    resumen = [['Periodo fundamental T₁', f'{res["T"][0]:.4f} s']]
    if modal:
        resumen += [['Vu en la base · modal espectral', f'{res["V_din"]:.3f} t'],
                    ['Σ Mvu · modal espectral', f'{res["tabla_din"]["Mvu"].sum():.2f} t·m']]
    if estat:
        resumen += [['Vu en la base · estático', f'{res["tabla_est"]["Vu"][-1]:.3f} t']]
        if modal:
            resumen += [['Factor de escala (V estático / V modal)', f'{fe:.4f}']]
    dmax = float(est_d['dist'].max())
    resumen += [[f'Distorsión máx. · {est_d["nombre"]}', f'{dmax:.5f}' + ('' if est_d['limite'] is None else
                                                                        f'  (límite {est_d["limite"]:g}: '
                                                                        + ('cumple' if dmax <= est_d['limite'] else 'NO cumple') + ')')]]
    # datos de entrada
    pares = [['Material', p.get('material', 'Concreto')],
             ['E', f'{p["E"]:,.0f} kg/cm²' if 'E' in p else 'K dada'],
             ['Grupo / intensidad', f'{p.get("subgrupo") or p["grupo"]} · {p.get("intensidad", "Base de diseño")}'],
             ['Estado límite', p['estado'] + (f' (Ks = {p["Ks"]:g})' if res['clave'] == 'DL' else '')],
             ['Q · k₁', f'{", ".join(f"{q:g}" for q in p["Q"])} · {p["k1"]:g}'],
             ['ζ · combinación', f'{p["zeta"]:g} · {p["combinacion"]}'],
             ['Espectro SASID', f'a₀ {p["a0"]:g} · c {p["c"]:g} · Tₐ {p["Ta"]:g} · T_b {p["Tb"]:g} · k {p["k"]:g} · T_s {p["Ts"]:g}'],
             ['Factor de carga FC', f'{p["factor_Fu"]:g}' if res['clave'] != 'DL' else '1 (sin factor de carga)']]
    t_res = _tabla(st, ['Resultado', 'Valor'], [[Paragraph(_escapar(a_), st['nota']), Paragraph(_escapar(b_), st['nota'])]
                                               for a_, b_ in resumen], anchos=[4.9 * cm, 3.9 * cm], fuente=8)
    td = Table([[Paragraph('<b>' + _escapar(a_) + '</b>', ParagraphStyle('dl', parent=st['nota'], textColor=AZUL)),
                 Paragraph(_escapar(str(b_)), st['nota'])] for a_, b_ in pares], colWidths=[2.7 * cm, 5.8 * cm])
    td.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), 0.3, AZUL_CLARO),
                            ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    lado = Table([[t_res, td]], colWidths=[Ancho * 0.51, Ancho * 0.49])
    lado.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0)]))
    E += [P('Resumen de resultados y datos de entrada', 'h2'), lado]

    # ------------------------------------------------------------------ A, B y C (mismos bloques que la página)
    E.extend(_bloques(ame_detalle.compacto(res, metodo), st, Ancho))

    doc.build(E)
    return destino
