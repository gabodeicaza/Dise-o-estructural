"""Interfaz gráfica del Análisis Modal Espectral. Ejecutar:  python ame_app.py"""
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ame_sismico as ame

# Escala de azules
NAVY, AZUL_OSC, AZUL, AZUL_MED, AZUL_CLARO, PALE, BG, BLANCO = (
    '#0B2545', '#13315C', '#1D4E89', '#3E7CB1', '#81A4CD', '#DCE9F7', '#F1F6FC', '#FFFFFF')
FUENTE = 'Segoe UI'

TIPOS = ['Grupo B', 'Grupo A - sismo base', 'Grupo A - sismo infrecuente']
COLS_NIVEL = [('h', 'h entrepiso (m)'), ('ncol', 'Nº columnas'), ('b', 'b (cm)'), ('hs', 'h sección (cm)'),
              ('nart', 'Nº articuladas\nen la base'), ('w', 'Carga w (t/m)'), ('L', 'Longitud L (m)'),
              ('W', 'o Peso W (t)')]


def estilos(root):
    st = ttk.Style(root)
    st.theme_use('clam')
    st.configure('.', font=(FUENTE, 10), background=BG, foreground=NAVY)
    st.configure('TFrame', background=BG)
    st.configure('Card.TFrame', background=BLANCO)
    st.configure('Card.TLabel', background=BLANCO, foreground=NAVY)
    st.configure('Muted.TLabel', background=BLANCO, foreground=AZUL_MED, font=(FUENTE, 9))
    st.configure('Head.TLabel', background=PALE, foreground=AZUL_OSC, font=(FUENTE, 9, 'bold'), anchor='center')
    st.configure('CardTitle.TLabel', background=BLANCO, foreground=AZUL_OSC, font=(FUENTE, 11, 'bold'))
    st.configure('KpiNum.TLabel', background=BLANCO, foreground=AZUL, font=(FUENTE, 20, 'bold'))
    st.configure('KpiTit.TLabel', background=BLANCO, foreground=AZUL_MED, font=(FUENTE, 9, 'bold'))
    for w in ('TEntry', 'TCombobox'):
        st.configure(w, fieldbackground=BLANCO, bordercolor=AZUL_CLARO, lightcolor=AZUL_CLARO,
                     darkcolor=AZUL_CLARO, padding=4)
        st.map(w, bordercolor=[('focus', AZUL)], lightcolor=[('focus', AZUL)])
    st.configure('TCombobox', arrowcolor=AZUL, background=PALE)
    st.map('TCombobox', fieldbackground=[('readonly', BLANCO)], selectbackground=[('readonly', BLANCO)],
           selectforeground=[('readonly', NAVY)])
    st.configure('Accent.TButton', background=AZUL, foreground=BLANCO, font=(FUENTE, 11, 'bold'),
                 padding=(22, 9), borderwidth=0)
    st.map('Accent.TButton', background=[('active', AZUL_OSC), ('pressed', NAVY)])
    st.configure('Soft.TButton', background=PALE, foreground=AZUL_OSC, padding=(14, 8), borderwidth=0)
    st.map('Soft.TButton', background=[('active', AZUL_CLARO)])
    st.configure('TNotebook', background=BG, borderwidth=0)
    st.configure('TNotebook.Tab', background=PALE, foreground=AZUL_OSC, padding=(20, 9), font=(FUENTE, 10, 'bold'),
                 borderwidth=0)
    st.map('TNotebook.Tab', background=[('selected', AZUL)], foreground=[('selected', BLANCO)])
    st.configure('Treeview', background=BLANCO, fieldbackground=BLANCO, foreground=NAVY, rowheight=28,
                 bordercolor=AZUL_CLARO, borderwidth=1)
    st.configure('Treeview.Heading', background=AZUL, foreground=BLANCO, font=(FUENTE, 10, 'bold'), relief='flat',
                 padding=6)
    st.map('Treeview.Heading', background=[('active', AZUL_OSC)])
    st.map('Treeview', background=[('selected', AZUL_CLARO)], foreground=[('selected', NAVY)])


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.inner = ttk.Frame(self.canvas, padding=14)
        win = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.inner.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._rueda))
        self.canvas.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))

    def _rueda(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120), 'units')


def card(parent, titulo):
    """Devuelve (marco exterior, marco interior) con franja azul superior."""
    outer = tk.Frame(parent, bg=BLANCO, highlightbackground=AZUL_CLARO, highlightthickness=1)
    tk.Frame(outer, bg=AZUL, height=4).pack(fill='x')
    inner = ttk.Frame(outer, style='Card.TFrame', padding=12)
    inner.pack(fill='both', expand=True)
    ttk.Label(inner, text=titulo, style='CardTitle.TLabel').grid(row=0, column=0, columnspan=6, sticky='w',
                                                                 pady=(0, 8))
    return outer, inner


def tabla(parent, columnas, anchos):
    marco = ttk.Frame(parent)
    t = ttk.Treeview(marco, columns=columnas, show='headings', height=4)
    for c, w in zip(columnas, anchos):
        t.heading(c, text=c)
        t.column(c, width=w, anchor='center')
    t.tag_configure('par', background=BG)
    t.pack(fill='both', expand=True)
    return marco, t


def llenar(t, filas, decimales=4):
    filas = list(filas)
    t.delete(*t.get_children())
    for i, f in enumerate(filas):
        txt = [str(int(v)) if j == 0 else f'{v:.{decimales}f}' for j, v in enumerate(f)]
        t.insert('', 'end', values=txt, tags=('par',) if i % 2 else ())
    t.configure(height=max(len(filas), 2))


class App:
    def __init__(self, root):
        self.root = root
        self.res = None
        self.filas_nivel = []
        root.title('Análisis Modal Espectral')
        root.geometry('1220x860')
        root.configure(bg=BG)
        estilos(root)

        cab = tk.Frame(root, bg=NAVY)
        cab.pack(fill='x')
        tk.Label(cab, text='Análisis Modal Espectral', bg=NAVY, fg=BLANCO,
                 font=(FUENTE, 20, 'bold')).pack(anchor='w', padx=22, pady=(14, 0))
        tk.Label(cab, text='Marcos de cortante · fuerzas sísmicas por nivel, cortante basal y factor de escala',
                 bg=NAVY, fg=AZUL_CLARO, font=(FUENTE, 10)).pack(anchor='w', padx=22, pady=(0, 14))

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True, padx=10, pady=10)
        self._pestana_datos()
        self._pestana_resultados()
        self.poner(ame.EJEMPLO)

    # ------------------------------------------------------------------ datos
    def _pestana_datos(self):
        sf = ScrollFrame(self.nb)
        self.nb.add(sf, text='  1 · Datos  ')
        f = sf.inner
        f.columnconfigure((0, 1, 2), weight=1, uniform='c')

        # Estructura
        outer, inner = card(f, 'Estructura (niveles de abajo hacia arriba)')
        outer.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 12))
        self.tbl = ttk.Frame(inner, style='Card.TFrame')
        self.tbl.grid(row=1, column=0, columnspan=6, sticky='ew')
        for j, (_, et) in enumerate([('n', 'Nivel')] + COLS_NIVEL):
            ttk.Label(self.tbl, text=et, style='Head.TLabel', padding=5).grid(row=0, column=j, sticky='nsew',
                                                                             padx=1, pady=1)
            self.tbl.columnconfigure(j, weight=1)
        bt = ttk.Frame(inner, style='Card.TFrame')
        bt.grid(row=2, column=0, columnspan=6, sticky='w', pady=(8, 4))
        ttk.Button(bt, text='+ Agregar nivel', style='Soft.TButton', command=self.agregar_nivel).pack(side='left')
        ttk.Button(bt, text='− Quitar último', style='Soft.TButton',
                   command=self.quitar_nivel).pack(side='left', padx=6)
        ttk.Label(inner, style='Muted.TLabel', wraplength=1050, text=(
            '“h sección” es la dimensión de la columna en la dirección del sismo (I = b·h³/12). '
            'Las columnas articuladas en la base (normalmente solo en el nivel 1) usan 3EI/h³ en vez de 12EI/h³. '
            'El peso del nivel es w × L; si escribes W (t) se usa ese valor.')
                  ).grid(row=3, column=0, columnspan=6, sticky='w')

        self.v = {k: tk.StringVar() for k in ('E', 'fc', 'tipo', 'Q', 'k1', 'a0', 'c', 'Ta', 'Tb', 'k', 'Ts',
                                              'factor_Fu', 'mult_V')}

        def campo(inner, fila, et, clave, ancho=10, valores=None, solo_lectura=False, col=0):
            ttk.Label(inner, text=et, style='Card.TLabel').grid(row=fila, column=col, sticky='w', pady=3, padx=(0, 8))
            if valores:
                w = ttk.Combobox(inner, textvariable=self.v[clave], values=valores, width=ancho,
                                 state='readonly' if solo_lectura else 'normal')
            else:
                w = ttk.Entry(inner, textvariable=self.v[clave], width=ancho)
            w.grid(row=fila, column=col + 1, sticky='w', pady=3)

        outer, inner = card(f, 'Tipo de edificación y sismo')
        outer.grid(row=1, column=0, sticky='nsew', padx=(0, 6))
        campo(inner, 1, 'Tipo', 'tipo', 26, TIPOS, True)
        campo(inner, 2, 'Q', 'Q', 26, ['1', '1.5', '2', '3', '4'])
        campo(inner, 3, 'k1', 'k1', 8)
        ttk.Label(inner, style='Muted.TLabel', wraplength=300, text=(
            'Q admite varios valores separados por espacio (se usa el primero para las fuerzas). '
            'Grupo A: base → R′ = 0.75R; infrecuente → R′ = R.')).grid(row=4, column=0, columnspan=3, sticky='w',
                                                                      pady=(6, 0))

        outer, inner = card(f, 'Concreto y opciones')
        outer.grid(row=1, column=1, sticky='nsew', padx=6)
        campo(inner, 1, 'E (kg/cm²)', 'E', 10)
        campo(inner, 2, "f'c (kg/cm²)", 'fc', 10)
        campo(inner, 3, 'Factor Fu', 'factor_Fu', 10)
        campo(inner, 4, 'Multiplicar V estático por', 'mult_V', 10)
        ttk.Label(inner, style='Muted.TLabel', wraplength=300, text=(
            "Si dejas E vacío se calcula con E = 14000·√f'c.")).grid(row=5, column=0, columnspan=3, sticky='w',
                                                                       pady=(6, 0))

        outer, inner = card(f, 'Espectro del sitio (de SASID)')
        outer.grid(row=1, column=2, sticky='nsew', padx=(6, 0))
        for i, (et, c) in enumerate([('a0 (cm/s²)', 'a0'), ('c (cm/s²)', 'c'), ('Ta (s)', 'Ta'),
                                     ('Tb (s)', 'Tb'), ('k', 'k'), ('Ts (s)', 'Ts')]):
            campo(inner, 1 + i % 3, et, c, 8, col=(i // 3) * 2)

        barra = ttk.Frame(f)
        barra.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(16, 0))
        ttk.Button(barra, text='Calcular', style='Accent.TButton', command=self.calcular).pack(side='left')
        ttk.Button(barra, text='Cargar ejemplo', style='Soft.TButton',
                   command=lambda: self.poner(ame.EJEMPLO)).pack(side='left', padx=(14, 4))
        ttk.Button(barra, text='Guardar proyecto', style='Soft.TButton', command=self.guardar).pack(side='left', padx=4)
        ttk.Button(barra, text='Cargar proyecto', style='Soft.TButton', command=self.cargar).pack(side='left', padx=4)

    def agregar_nivel(self, datos=None):
        i = len(self.filas_nivel) + 1
        vars_ = {k: tk.StringVar(value=(datos or {}).get(k, '')) for k, _ in COLS_NIVEL}
        ttk.Label(self.tbl, text=f'{i}' + (' (base)' if i == 1 else ''), style='Card.TLabel',
                  anchor='center').grid(row=i, column=0, sticky='ew')
        ents = []
        for j, (k, _) in enumerate(COLS_NIVEL, 1):
            e = ttk.Entry(self.tbl, textvariable=vars_[k], width=9, justify='center')
            e.grid(row=i, column=j, padx=2, pady=2, sticky='ew')
            ents.append(e)
        self.filas_nivel.append((vars_, ents, self.tbl.grid_slaves(row=i, column=0)[0]))

    def quitar_nivel(self, forzar=False):
        if len(self.filas_nivel) > 1 or forzar:
            _, ents, lbl = self.filas_nivel.pop()
            lbl.destroy()
            for e in ents:
                e.destroy()

    def leer(self):
        d = {k: v.get() for k, v in self.v.items()}
        d['niveles'] = [{k: v.get() for k, v in fila.items()} for fila, _, _ in self.filas_nivel]
        return d

    def poner(self, d):
        while self.filas_nivel:
            self.quitar_nivel(forzar=True)
        for r in d['niveles']:
            self.agregar_nivel(r)
        for k, v in self.v.items():
            v.set(d.get(k, ''))

    def guardar(self):
        ruta = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('Proyecto', '*.json')])
        if ruta:
            Path(ruta).write_text(json.dumps(self.leer(), indent=2, ensure_ascii=False), encoding='utf-8')

    def cargar(self):
        ruta = filedialog.askopenfilename(filetypes=[('Proyecto', '*.json')])
        if ruta:
            try:
                self.poner(json.loads(Path(ruta).read_text(encoding='utf-8')))
            except (ValueError, KeyError):
                messagebox.showerror('Proyecto no válido', 'No se pudo leer ese archivo de proyecto.')

    # ------------------------------------------------------------- resultados
    def _pestana_resultados(self):
        sf = ScrollFrame(self.nb)
        self.nb.add(sf, text='  2 · Resultados  ')
        f = sf.inner
        f.columnconfigure(0, weight=1)

        top = ttk.Frame(f)
        top.grid(row=0, column=0, sticky='ew')
        top.columnconfigure((0, 1, 2, 3), weight=1, uniform='k')
        self.kpi = {}
        for i, (clave, tit) in enumerate([('T', 'PERIODO FUNDAMENTAL'), ('Vd', 'CORTANTE BASAL DINÁMICO'),
                                          ('Ve', 'CORTANTE BASAL ESTÁTICO'), ('fe', 'FACTOR DE ESCALA')]):
            outer, inner = card(top, '')
            inner.winfo_children()[0].destroy()
            outer.grid(row=0, column=i, sticky='nsew', padx=(0 if i == 0 else 6, 0 if i == 3 else 6))
            ttk.Label(inner, text=tit, style='KpiTit.TLabel').grid(row=0, column=0, sticky='w')
            num = ttk.Label(inner, text='—', style='KpiNum.TLabel')
            num.grid(row=1, column=0, sticky='w')
            sub = ttk.Label(inner, text='', style='Muted.TLabel')
            sub.grid(row=2, column=0, sticky='w')
            self.kpi[clave] = (num, sub)

        ttk.Button(f, text='Exportar a Excel', style='Accent.TButton', command=self.exportar).grid(
            row=1, column=0, sticky='e', pady=12)

        self.sub = ttk.Notebook(f)
        self.sub.grid(row=2, column=0, sticky='nsew')

        p1 = ttk.Frame(self.sub, padding=12)
        self.sub.add(p1, text=' Fuerzas por nivel ')
        ttk.Label(p1, text='Análisis dinámico (modal espectral)', font=(FUENTE, 11, 'bold'),
                  foreground=AZUL_OSC).pack(anchor='w', pady=(0, 6))
        m, self.t_din = tabla(p1, ['Nivel', 'h (m)', 'F (t)', 'Fu (t)', 'Vu (t)', 'Mvu (t·m)'], [90] + [150] * 5)
        m.pack(fill='x')
        ttk.Label(p1, text='Análisis estático (para el factor de escala)', font=(FUENTE, 11, 'bold'),
                  foreground=AZUL_OSC).pack(anchor='w', pady=(16, 6))
        m, self.t_est = tabla(p1, ['Nivel', 'h (m)', 'Wi (t)', 'Wihi (t·m)', 'Fi (t)', 'Fu (t)', 'Vu (t)', 'Mvu (t·m)'],
                              [90] + [110] * 7)
        m.pack(fill='x')

        p2 = ttk.Frame(self.sub, padding=12)
        self.sub.add(p2, text=' Modos ')
        m, self.t_mod = tabla(p2, ['Modo', 'T (s)', 'f (Hz)', 'ω (rad/s)', 'γ', "Sa/(Q'R') (cm/s²)", "Q·R'"],
                              [80, 120, 120, 120, 120, 180, 120])
        m.pack(fill='x')

        p3 = ttk.Frame(self.sub, padding=6)
        self.sub.add(p3, text=' Detalle completo ')
        self.txt = tk.Text(p3, font=('Consolas', 10), wrap='none', height=30, bg=BLANCO, fg=NAVY,
                           relief='flat', highlightbackground=AZUL_CLARO, highlightthickness=1)
        sx = ttk.Scrollbar(p3, orient='horizontal', command=self.txt.xview)
        self.txt.configure(xscrollcommand=sx.set)
        sx.pack(side='bottom', fill='x')
        self.txt.pack(fill='both', expand=True)

        self.p_mod = ttk.Frame(self.sub)
        self.sub.add(self.p_mod, text=' Formas modales ')
        self.p_esp = ttk.Frame(self.sub)
        self.sub.add(self.p_esp, text=' Espectro ')

    @staticmethod
    def _fig(frame, fig):
        for w in frame.winfo_children():
            w.destroy()
        cv = FigureCanvasTkAgg(fig, master=frame)
        cv.draw()
        cv.get_tk_widget().pack(fill='both', expand=True)

    def calcular(self):
        try:
            res = ame.calcular(ame.preparar_estructura(self.leer()))
        except ValueError as e:
            messagebox.showerror('Revisa los datos', str(e))
            return
        except Exception as e:  # p. ej. matriz singular por datos incoherentes
            messagebox.showerror('No se pudo calcular', f'{type(e).__name__}: {e}')
            return
        self.res = res
        p = res['p']
        T1 = res['T'][0]
        self.kpi['T'][0].config(text=f'{T1:.4f} s')
        self.kpi['T'][1].config(text=f'{p["n"]} niveles · γ₁ = {res["gam"][0]:.3f}')
        self.kpi['Vd'][0].config(text=f'{res["V_din"]:.3f} t')
        self.kpi['Vd'][1].config(text=f'Σ Mvu = {res["tabla_din"]["Mvu"].sum():.2f} t·m')
        self.kpi['Ve'][0].config(text=f'{res["V_est"]:.3f} t')
        self.kpi['Ve'][1].config(text=f'Sa/(Q′R′) máx = {res["Fmax_est"]:.4f} g')
        fe = res['factor_escala']
        self.kpi['fe'][0].config(text=f'{fe:.4f}')
        self.kpi['fe'][1].config(text='Escalar fuerzas dinámicas' if fe > 1 else 'El dinámico ya cumple (≤ 1)')

        t = res['tabla_din']
        llenar(self.t_din, zip(t['nivel'], t['h'], t['F'], t['Fu'], t['Vu'], t['Mvu']))
        e = res['tabla_est']
        llenar(self.t_est, zip(e['nivel'], e['h'], e['Wi'], e['hWi'], e['Fi'], e['Fu'], e['Vu'], e['Mvu']))
        n = p['n']
        qr = {fila[0]: fila[-1] for fila in res['QR'][::len(p['Q'])]}  # Q·R' con la primera Q
        llenar(self.t_mod, [[j + 1, res['T'][j], res['f'][j], res['w'][j], res['gam'][j],
                             res['Sa_modal'][j, 0], qr[j + 1]] for j in range(n)])
        self.txt.delete('1.0', 'end')
        self.txt.insert('1.0', ame.reporte_texto(res))
        self._fig(self.p_mod, ame.fig_modal(res))
        self._fig(self.p_esp, ame.fig_espectro(res))
        self.nb.select(1)

    def exportar(self):
        if not self.res:
            messagebox.showinfo('Sin resultados', 'Primero pulsa Calcular.')
            return
        ruta = filedialog.asksaveasfilename(defaultextension='.xlsx', initialfile='resultados_AME.xlsx',
                                            filetypes=[('Excel', '*.xlsx')])
        if not ruta:
            return
        try:
            ame.exportar_excel(self.res, ruta)
        except PermissionError:
            messagebox.showerror('Error', 'No se pudo escribir el archivo (¿está abierto en Excel?).')
            return
        messagebox.showinfo('Exportado', f'Resultados guardados en:\n{ruta}')


def gui():
    try:  # nitidez en pantallas de alta resolución (Windows)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    gui()
