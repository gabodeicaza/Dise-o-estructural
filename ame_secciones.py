"""Secciones transversales de columnas y vigas.

Sintaxis (dimensiones en cm; varias secciones separadas por espacio):
    25x50          rectangular b x h      (h = dimensión en la dirección X, b en la dirección Y)
    25x50e3        rectangular hueca, espesor de pared 3
    Ø60            circular
    Ø60e1.5        circular hueca (tubo), espesor 1.5
    H:30x2x50x1    perfil I armado: ancho de patín x espesor de patín x peralte total x espesor de alma
                   (el peralte va en la dirección X)
    T:50x10x60x15  sección T: ancho de ala x espesor de ala x peralte total x espesor de alma
    I:260000/A:1250            propiedades dadas: inercia (cm4) y área (cm2); opcional /I2:45000
                               para la inercia en la dirección Y

Para una VIGA, el peralte (h) es la dimensión vertical y se usa la inercia en X.
"""
import re

import numpy as np

_N = r'(\d+(?:\.\d+)?)'
_RECT = re.compile(rf'^{_N}x{_N}(?:e{_N})?$')
_CIRC = re.compile(rf'^[øo⌀d]{_N}(?:e{_N})?$')
_PERFIL = re.compile(rf'^([ht]):{_N}x{_N}x{_N}x{_N}$')
_GEN = re.compile(rf'^i:{_N}/a:{_N}(?:/i2:{_N})?$')

AYUDA_SECCIONES = ('Rectangular: 25x50 (b x h, h en la dirección X) · hueca: 25x50e3 · circular: Ø60 · tubo: Ø60e1.5 · '
                   'perfil I armado: H:30x2x50x1 · T: T:50x10x60x15 · propiedades: I:260000/A:1250')


def _tokens(texto):
    return [t for t in re.split(r'[\s,;]+', str(texto).strip()) if t]


def _compuesta(piezas):
    """Propiedades de una sección armada con rectángulos (cx, cy, dx, dy): A, Ix, Iy respecto al centroide.

    Ix = ∫ x² dA (flexión en el plano X-Z, la que resiste un sismo en X); Iy = ∫ y² dA.
    """
    A = sum(dx * dy for _, _, dx, dy in piezas)
    xc = sum(cx * dx * dy for cx, _, dx, dy in piezas) / A
    yc = sum(cy * dx * dy for _, cy, dx, dy in piezas) / A
    Ix = sum(dy * dx ** 3 / 12 + dx * dy * (cx - xc) ** 2 for cx, _, dx, dy in piezas)
    Iy = sum(dx * dy ** 3 / 12 + dx * dy * (cy - yc) ** 2 for _, cy, dx, dy in piezas)
    return A, Ix, Iy


def _sec(tipo, texto, A, Ix, Iy, dx, dy, **forma):
    return dict(tipo=tipo, texto=texto, A=A, Ix=Ix, Iy=Iy, dx=dx, dy=dy, forma=forma)


def parse_secciones(texto, nombre):
    """Devuelve una lista de secciones (dict con A, Ix, Iy, dx, dy, texto y datos para dibujarla)."""
    t = str(texto).lower().replace('×', 'x').replace('ø', 'ø')
    t = re.sub(r'\s*x\s*', 'x', t)
    t = re.sub(r'([øo⌀dht]:?)\s+(?=\d)', r'\1', t)
    t = re.sub(r':\s+', ':', t)
    out = []
    for tok in _tokens(t):
        if m := _RECT.match(tok):
            b, h = float(m.group(1)), float(m.group(2))
            e = float(m.group(3)) if m.group(3) else None
            if e is None:
                A, Ix, Iy = b * h, b * h ** 3 / 12, h * b ** 3 / 12
                out.append(_sec('rect', f'{b:g}x{h:g}', A, Ix, Iy, h, b, b=b, h=h))
            else:
                if 2 * e >= min(b, h):
                    raise ValueError(f'{nombre}: el espesor {e:g} es demasiado grande para la sección {b:g}x{h:g}.')
                bi, hi = b - 2 * e, h - 2 * e
                A = b * h - bi * hi
                out.append(_sec('hrect', f'{b:g}x{h:g}e{e:g}', A, (b * h ** 3 - bi * hi ** 3) / 12,
                                (h * b ** 3 - hi * bi ** 3) / 12, h, b, b=b, h=h, e=e))
        elif m := _CIRC.match(tok):
            D = float(m.group(1))
            e = float(m.group(2)) if m.group(2) else None
            if e is None:
                out.append(_sec('circ', f'Ø{D:g}', np.pi * D ** 2 / 4, np.pi * D ** 4 / 64, np.pi * D ** 4 / 64, D, D, D=D))
            else:
                if 2 * e >= D:
                    raise ValueError(f'{nombre}: el espesor {e:g} es demasiado grande para Ø{D:g}.')
                Di = D - 2 * e
                I = np.pi * (D ** 4 - Di ** 4) / 64
                out.append(_sec('hcirc', f'Ø{D:g}e{e:g}', np.pi * (D ** 2 - Di ** 2) / 4, I, I, D, D, D=D, e=e))
        elif m := _PERFIL.match(tok):
            letra, bf, tf, d, tw = m.group(1), *(float(g) for g in m.groups()[1:])
            if letra == 'h':
                if 2 * tf >= d:
                    raise ValueError(f'{nombre}: los patines de {tok} no caben en el peralte.')
                piezas = [(-(d - tf) / 2, 0, tf, bf), ((d - tf) / 2, 0, tf, bf), (0, 0, d - 2 * tf, tw)]
                A, Ix, Iy = _compuesta(piezas)
                out.append(_sec('H', f'H:{bf:g}x{tf:g}x{d:g}x{tw:g}', A, Ix, Iy, d, bf, bf=bf, tf=tf, d=d, tw=tw))
            else:
                if tf >= d:
                    raise ValueError(f'{nombre}: el ala de {tok} es más gruesa que el peralte.')
                piezas = [(d - tf / 2, 0, tf, bf), ((d - tf) / 2, 0, d - tf, tw)]
                A, Ix, Iy = _compuesta(piezas)
                out.append(_sec('T', f'T:{bf:g}x{tf:g}x{d:g}x{tw:g}', A, Ix, Iy, d, bf, bf=bf, tf=tf, d=d, tw=tw))
        elif m := _GEN.match(tok):
            I, A = float(m.group(1)), float(m.group(2))
            I2 = float(m.group(3)) if m.group(3) else I
            lado = float(np.sqrt(A))
            out.append(_sec('gen', f'I:{I:g}/A:{A:g}', A, I, I2, lado, lado))
        else:
            raise ValueError(f'{nombre}: la sección "{tok}" no es válida. {AYUDA_SECCIONES}')
    if not out:
        raise ValueError(f'{nombre}: falta la sección.')
    return out


def inercia(sec, direccion):
    """Inercia para un sismo en la dirección dada ('X' o 'Y')."""
    return sec['Ix'] if direccion == 'X' else sec['Iy']


def ancho_en_direccion(sec, direccion):
    """Dimensión de la sección paralela al sismo (para dibujarla en la elevación), en cm."""
    return sec['dx'] if direccion == 'X' else sec['dy']


def dibujar_seccion(ax, sec, cx, cy, fc='#DCE9F7', ec='#1D4E89'):
    """Dibuja la sección en planta (X horizontal, Y vertical) centrada en (cx, cy). Escala 1:1 en cm."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    f = sec['forma']
    kw = dict(fc=fc, ec=ec, lw=1.5)
    hueco = dict(fc='white', ec=ec, lw=1.2)
    t = sec['tipo']
    if t in ('rect', 'hrect'):
        ax.add_patch(Rectangle((cx - f['h'] / 2, cy - f['b'] / 2), f['h'], f['b'], **kw))
        if t == 'hrect':
            e = f['e']
            ax.add_patch(Rectangle((cx - f['h'] / 2 + e, cy - f['b'] / 2 + e), f['h'] - 2 * e, f['b'] - 2 * e, **hueco))
    elif t in ('circ', 'hcirc'):
        ax.add_patch(Circle((cx, cy), f['D'] / 2, **kw))
        if t == 'hcirc':
            ax.add_patch(Circle((cx, cy), f['D'] / 2 - f['e'], **hueco))
    elif t == 'H':
        bf, tf, d, tw = f['bf'], f['tf'], f['d'], f['tw']
        for sx in (-1, 1):
            ax.add_patch(Rectangle((cx + sx * (d - tf) / 2 - tf / 2, cy - bf / 2), tf, bf, **kw))
        ax.add_patch(Rectangle((cx - (d - 2 * tf) / 2, cy - tw / 2), d - 2 * tf, tw, **kw))
    elif t == 'T':
        bf, tf, d, tw = f['bf'], f['tf'], f['d'], f['tw']
        A = bf * tf + (d - tf) * tw
        xc = ((d - tf / 2) * bf * tf + ((d - tf) / 2) * (d - tf) * tw) / A
        ax.add_patch(Rectangle((cx + d - tf - xc, cy - bf / 2), tf, bf, **kw))
        ax.add_patch(Rectangle((cx - xc, cy - tw / 2), d - tf, tw, **kw))
    else:  # propiedades dadas: se dibuja un cuadro equivalente con línea punteada
        a = sec['dx']
        ax.add_patch(Rectangle((cx - a / 2, cy - a / 2), a, a, fc='white', ec=ec, lw=1.2, ls='--'))
