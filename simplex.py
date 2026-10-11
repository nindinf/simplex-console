#!/usr/bin/env python3
"""
Kalkulator Simplex (maksimasi) untuk n variabel dan m kendala.

- Kendala boleh <=, >= atau =  (>= dan = dikerjakan dengan metode Big-M)
- Grafik daerah layak untuk n = 2 (matplotlib, opsional)


"""
from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional

DECIMAL = False  # tampilan angka: pecahan (False) atau desimal (True)
MAX_ITER = 200
_SUB = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")


def sub(k) -> str:
    """Angka subskrip: 12 -> ₁₂ (dipakai antarmuka)."""
    return str(k).translate(_SUB)


# --------------------------------------------------------------------------
# Angka: a + b*M, dengan M = penalti Big-M (bilangan positif sangat besar)
# --------------------------------------------------------------------------
def fr(q: Fraction) -> str:
    """Format Fraction sesuai mode tampilan."""
    if DECIMAL:
        s = f"{float(q):.4f}".rstrip("0").rstrip(".")
        if s in ("", "-0"):
            s = "0"
        return s.replace(".", ",")
    return str(q)


class BM:
    """Bilangan a + b*M. Baris kendala selalu murni (b = 0); hanya baris Z yang memuat M."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fraction(a)
        self.b = Fraction(b)

    def __add__(self, o):
        return BM(self.a + o.a, self.b + o.b)

    def __sub__(self, o):
        return BM(self.a - o.a, self.b - o.b)

    def __mul__(self, o):  # paling banyak satu operand memuat M
        return BM(self.a * o.a, self.a * o.b + self.b * o.a)

    def __truediv__(self, o):  # pembagi (pivot) selalu murni
        return BM(self.a / o.a, self.b / o.a)

    def __neg__(self):
        return BM(-self.a, -self.b)

    def key(self):
        return (self.b, self.a)  # M jauh lebih besar dari konstanta apa pun

    def __lt__(self, o):
        return self.key() < o.key()

    def __eq__(self, o):
        return self.key() == o.key()

    def __hash__(self):
        return hash(self.key())

    def is_zero(self):
        return self.a == 0 and self.b == 0

    def is_neg(self):
        return self.b < 0 or (self.b == 0 and self.a < 0)

    def is_pos(self):
        return self.b > 0 or (self.b == 0 and self.a > 0)

    def pure(self):
        return self.b == 0

    def __str__(self):
        if self.b == 0:
            return fr(self.a)
        if self.b == 1:
            bs = "M"
        elif self.b == -1:
            bs = "-M"
        else:
            bs = fr(self.b) + "M"
            if not DECIMAL and self.b.denominator != 1:
                bs = "(" + fr(self.b) + ")M"
        if self.a == 0:
            return bs
        return bs + (" - " + fr(-self.a) if self.a < 0 else " + " + fr(self.a))

    __repr__ = __str__


def parse_number(s: str) -> Fraction:
    """'3', '-2,5', '1/2', '0.75' -> Fraction"""
    s = s.strip().replace(",", ".").replace("−", "-")
    if "/" in s:
        p, q = s.split("/", 1)
        return Fraction(p.strip()) / Fraction(q.strip())
    return Fraction(s)


# --------------------------------------------------------------------------
# Model soal
# --------------------------------------------------------------------------
@dataclass
class Problem:
    n: int
    m: int
    c: List[Fraction]
    A: List[List[Fraction]]
    b: List[Fraction]
    rel: List[str]  # '<=', '>=', '='
    # versi asli (sebelum nilai kanan negatif dibalik)
    oA: List[List[Fraction]] = field(default_factory=list)
    oB: List[Fraction] = field(default_factory=list)
    oRel: List[str] = field(default_factory=list)
    flipped: List[bool] = field(default_factory=list)


def make_problem(c, A, b, rel) -> Problem:
    """Bentuk standar: nilai kanan harus >= 0. Jika negatif, kedua ruas dikali -1 dan tanda dibalik."""
    n, m = len(c), len(b)
    flip = {"<=": ">=", ">=": "<=", "=": "="}
    P = Problem(n, m, list(c), [], [], [], oA=[list(r) for r in A], oB=list(b), oRel=list(rel))
    for i in range(m):
        if b[i] < 0:
            P.A.append([-v for v in A[i]])
            P.b.append(-b[i])
            P.rel.append(flip[rel[i]])
            P.flipped.append(True)
        else:
            P.A.append(list(A[i]))
            P.b.append(b[i])
            P.rel.append(rel[i])
            P.flipped.append(False)
    return P


# --------------------------------------------------------------------------
# Mesin simplex (tabel + Big-M)
# --------------------------------------------------------------------------
@dataclass
class Iter:
    T: list
    basis: list
    enter: int
    enter_ties: list
    ratios: list
    leave: int
    leave_ties: list
    min_ratio: Optional[BM]
    best: BM
    unbounded: bool = False
    pivot: Optional[BM] = None
    ops: list = field(default_factory=list)
    T2: Optional[list] = None
    basis2: Optional[list] = None
    leave_var: int = -1


@dataclass
class Solution:
    P: Problem
    names: list
    art_start: int
    col_s: list
    col_a: list
    adj: list
    T0: list
    basis0: list
    iters: list
    status: str  # optimal | unbounded | infeasible | maxiter
    T: list
    basis: list
    x: list
    s: list
    shadow: list
    bad_art: list
    pts: list
    zs: list

    @property
    def cols(self):
        return len(self.names)

    @property
    def Z(self):
        return self.T[-1][-1]


def solve(P: Problem) -> Solution:
    n, m = P.n, P.m
    names = [f"x{j + 1}" for j in range(n)]
    col_s, col_a = [-1] * m, [-1] * m
    for i in range(m):
        if P.rel[i] != "=":
            col_s[i] = len(names)
            names.append(f"s{i + 1}")
    art_start = len(names)
    for i in range(m):
        if P.rel[i] != "<=":
            col_a[i] = len(names)
            names.append(f"a{i + 1}")
    cols = len(names)

    T, basis0 = [], []
    for i in range(m):
        row = [BM() for _ in range(cols + 1)]
        for j in range(n):
            row[j] = BM(P.A[i][j])
        if P.rel[i] == "<=":
            row[col_s[i]] = BM(1)
        if P.rel[i] == ">=":
            row[col_s[i]] = BM(-1)
        if P.rel[i] != "<=":
            row[col_a[i]] = BM(1)
        row[cols] = BM(P.b[i])
        T.append(row)
        basis0.append(col_s[i] if P.rel[i] == "<=" else col_a[i])

    z = [BM() for _ in range(cols + 1)]
    for j in range(n):
        z[j] = BM(-P.c[j])
    for i in range(m):
        if col_a[i] >= 0:
            z[col_a[i]] = BM(0, 1)  # +M pada kolom variabel buatan
    adj = []  # variabel basis harus bernilai 0 di baris Z: Z - M * (baris yang memuat a_i)
    for i in range(m):
        if col_a[i] < 0:
            continue
        f = z[col_a[i]]
        neu = [z[j] - f * T[i][j] for j in range(cols + 1)]
        adj.append((i, f, z[:], neu))
        z = neu
    T.append(z)

    T0 = [r[:] for r in T]
    basis = basis0[:]
    iters: List[Iter] = []
    status = "optimal"
    guard = 0
    while True:
        guard += 1
        if guard > MAX_ITER:
            status = "maxiter"
            break
        enter, best = -1, None
        for j in range(art_start):  # variabel buatan tidak boleh masuk basis
            v = T[m][j]
            if v.is_neg() and (best is None or v < best):
                best, enter = v, j
        if enter < 0:
            break
        enter_ties = [j for j in range(art_start) if T[m][j] == best]
        ratios, leave, min_r = [], -1, None
        for i in range(m):
            a = T[i][enter]
            if a.is_pos():
                r = T[i][cols] / a
                ratios.append(r)
                if min_r is None or r < min_r:
                    min_r, leave = r, i
            else:
                ratios.append(None)
        leave_ties = [i for i in range(m) if leave >= 0 and ratios[i] is not None and ratios[i] == min_r]
        it = Iter([r[:] for r in T], basis[:], enter, enter_ties, ratios, leave, leave_ties, min_r, best)
        if leave < 0:
            it.unbounded = True
            iters.append(it)
            status = "unbounded"
            break
        pivot = T[leave][enter]
        it.pivot = pivot
        new_row = [x / pivot for x in T[leave]]
        ops = [("pivot", leave, None, T[leave][:], new_row)]
        newT = [r[:] for r in T]
        newT[leave] = new_row
        for i in range(m + 1):
            if i == leave:
                continue
            f = T[i][enter]
            if f.is_zero():
                ops.append(("same", i, f, T[i][:], T[i][:]))
                continue
            neu = [T[i][j] - f * new_row[j] for j in range(cols + 1)]
            newT[i] = neu
            ops.append(("elim", i, f, T[i][:], neu))
        it.leave_var = basis[leave]
        T = newT
        basis = basis[:]
        basis[leave] = enter
        it.ops, it.T2, it.basis2 = ops, [r[:] for r in T], basis[:]
        iters.append(it)

    bad_art = []
    if status == "optimal":
        bad_art = [i for i in range(m) if basis[i] >= art_start and T[i][cols].is_pos()]
        if bad_art:
            status = "infeasible"

    x = [T[basis.index(j)][cols] if j in basis else BM() for j in range(n)]
    s, shadow = [], []
    for k in range(m):
        if col_s[k] < 0:
            s.append(None)
            shadow.append(None)
            continue
        s.append(T[basis.index(col_s[k])][cols] if col_s[k] in basis else BM())
        sh = T[m][col_s[k]]
        shadow.append(sh if P.rel[k] == "<=" else -sh)

    snaps = [(T0, basis0)] + [(it.T2, it.basis2) for it in iters if it.T2 is not None]
    pts = []
    for tab, bas in snaps:
        v = [0.0] * n
        for i, bj in enumerate(bas):
            if bj < n:
                v[bj] = float(tab[i][cols].a)
        pts.append(v)
    zs = [float(tab[m][cols].a) if tab[m][cols].pure() else None for tab, _ in snaps]
    return Solution(P, names, art_start, col_s, col_a, adj, T0, basis0, iters, status, T, basis, x, s, shadow, bad_art, pts, zs)


# --------------------------------------------------------------------------
# Tampilan teks
# --------------------------------------------------------------------------
def expr(coefs, names) -> str:
    out = ""
    for a, nm in zip(coefs, names):
        if isinstance(a, BM):
            a = a.a
        if a == 0:
            continue
        neg, ab = a < 0, abs(a)
        if ab == 1:
            c = ""
        elif not DECIMAL and ab.denominator != 1:
            c = f"({fr(ab)})"
        else:
            c = fr(ab)
        t = c + nm
        out = ("-" if neg else "") + t if not out else out + (" - " if neg else " + ") + t
    return out or "0"


def vec(row) -> str:
    return "[ " + "  ".join(str(v) for v in row[:-1]) + " | " + str(row[-1]) + " ]"


def canonical(sol: "Solution", names=None):
    """Bentuk baku simpleks (canonical form).
    Mengembalikan (kalimat penjelasan, baris Z, daftar persamaan kendala)."""
    names = names or sol.names
    P, n, m = sol.P, sol.P.n, sol.P.m

    def ab(q):
        s = fr(abs(q))
        return f"({s})" if (not DECIMAL and abs(q).denominator != 1) else s

    def full(coefs):
        out = ""
        for a, nm in zip(coefs, names):
            t = ab(a) + nm
            out = (("-" if a < 0 else "") + t) if not out else out + (" - " if a < 0 else " + ") + t
        return out

    sl = [names[sol.col_s[i]] for i in range(m) if P.rel[i] == "<="]
    su = [names[sol.col_s[i]] for i in range(m) if P.rel[i] == ">="]
    ar = [names[c] for c in sol.col_a if c >= 0]
    has_ge, has_eq = ">=" in P.rel, "=" in P.rel
    types = ">= dan =" if (has_ge and has_eq) else (">=" if has_ge else "=")
    if not ar:
        sentence = (f"Karena semua kendala bertanda <=, kita hanya perlu menambahkan variabel slack ({', '.join(sl)}) "
                    "tanpa perlu menambahkan variabel buatan (Big-M).")
    else:
        parts = []
        if sl:
            parts.append(f"variabel slack ({', '.join(sl)}) pada kendala <=")
        if su:
            parts.append(f"variabel surplus ({', '.join(su)}) pada kendala >=")
        parts.append(f"variabel buatan ({', '.join(ar)}) pada kendala {types}")
        sentence = (f"Karena ada kendala bertanda {types}, kita menambahkan " + ", ".join(parts) +
                    ". Variabel buatan dikenai penalti M (metode Big-M), yaitu bilangan positif yang sangat besar, "
                    "agar terdorong keluar dari basis.")
    z = "Z"
    for j in range(n):
        z += (" + " if P.c[j] < 0 else " - ") + ab(P.c[j]) + names[j]
    for i in range(m):
        if sol.col_s[i] >= 0:
            z += f" - 0{names[sol.col_s[i]]}"
    for a in ar:
        z += f" + M{a}"
    z += " = 0"
    lines = []
    for i in range(m):
        si = names[sol.col_s[i]] if sol.col_s[i] >= 0 else ""
        ai = names[sol.col_a[i]] if sol.col_a[i] >= 0 else ""
        tail = {"<=": f" + {si}", ">=": f" - {si} + {ai}", "=": f" + {ai}"}[P.rel[i]]
        lines.append(f"{full(P.A[i])}{tail} = {fr(P.b[i])}")
    return sentence, z, lines


def tableau(T, basis, names, pivot=None, ratios=None):
    m, cols = len(basis), len(T[0]) - 1
    head = ["Basis"] + names + ["NK"] + (["Rasio"] if ratios is not None else [])
    rows = []
    for i in range(m + 1):
        r = ["Z" if i == m else names[basis[i]]]
        for j in range(cols + 1):
            t = str(T[i][j])
            if pivot == (i, j):
                t = f"[{t}]"  # angka kunci diberi tanda kurung siku
            r.append(t)
        if ratios is not None:
            r.append("" if i == m else ratios[i])
        rows.append(r)
    w = [max(len(head[j]), *(len(r[j]) for r in rows)) for j in range(len(head))]
    fmt_row = lambda r: " | ".join(r[0].ljust(w[0]) if j == 0 else r[j].rjust(w[j]) for j in range(len(r)))
    rule = "-+-".join("-" * x for x in w)
    print("  " + fmt_row(head))
    print("  " + rule)
    for i, r in enumerate(rows):
        if i == m:
            print("  " + rule)
        print("  " + fmt_row(r))


def step(no, title):
    print(f"\nLangkah {no}: {title}")


def report(sol: Solution):
    P, n, m, names = sol.P, sol.P.n, sol.P.m, sol.names
    cols, xn = sol.cols, sol.names[: sol.P.n]
    bar = "=" * 74
    print(bar)
    print("SOAL")
    print("  Maksimumkan Z =", expr(P.c, xn))
    for i in range(m):
        print(f"  K{i + 1}: {expr(P.oA[i], xn)} {P.oRel[i]} {fr(P.oB[i])}")
    print("  semua variabel >= 0")
    print(bar)

    # ---- Langkah 1 ----
    step(1, "Ubah ke bentuk standar dan bentuk baku simpleks")
    print("  Bentuk standar: Z dimaksimumkan, semua variabel >= 0, semua nilai kanan >= 0.")
    for i in range(m):
        print(f"  K{i + 1}: {expr(P.oA[i], xn)} {P.oRel[i]} {fr(P.oB[i])}")
        if P.flipped[i]:
            print(f"      nilai kanan negatif, kedua ruas dikali -1 -> {expr(P.A[i], xn)} {P.rel[i]} {fr(P.b[i])}")
    print("  " + ", ".join(xn) + " >= 0")
    sentence, zline, lines = canonical(sol)
    print("\n  Bentuk Baku Simpleks (Canonical Form)")
    print(textwrap.fill(sentence, width=74, initial_indent="  ", subsequent_indent="  "))
    print("   * Baris Z    : " + zline)
    for i, ln in enumerate(lines, 1):
        print(f"   * Kendala {i}  : {ln}")
    print("  Basis awal:", ", ".join(names[j] for j in sol.basis0))

    # ---- Langkah 2 ----
    step(2, "Susun tabel simplex awal")
    for i, f, old, neu in sol.adj:
        print(f"  Baris Z baru = baris Z lama - ({f}) x baris {names[sol.basis0[i]]}")
        print("    lama:", vec(old))
        print("    baru:", vec(neu))
    tableau(sol.T0, sol.basis0, names)
    negs = [names[j] for j in range(sol.art_start) if sol.T0[m][j].is_neg()]
    print("  ->", f"Baris Z masih memuat koefisien negatif ({', '.join(negs)}): lanjut iterasi." if negs else "Tidak ada koefisien Z negatif: tabel awal sudah optimal.")

    # ---- Iterasi ----
    for k, it in enumerate(sol.iters, 1):
        print("\n" + "-" * 74)
        print(f"ITERASI {k}")
        print("-" * 74)
        Zr = it.T[m]
        step(3, "Tentukan kolom kunci (koefisien Z paling negatif)")
        print("  Baris Z:", ", ".join(f"{names[j]}={Zr[j]}" for j in range(sol.art_start)))
        tie = f" (sama dengan {', '.join(names[j] for j in it.enter_ties)}; dipilih paling kiri)" if len(it.enter_ties) > 1 else ""
        print(f"  Paling negatif {it.best} pada {names[it.enter]} -> {names[it.enter]} masuk basis{tie}")

        step(4, "Tentukan baris kunci (rasio NK / kolom kunci terkecil)")
        rat_txt = []
        for i in range(m):
            a, nk, r = it.T[i][it.enter], it.T[i][cols], it.ratios[i]
            if r is None:
                print(f"  {names[it.basis[i]]}: entri kolom kunci {a} (<= 0), tidak dihitung")
                rat_txt.append("-")
            else:
                print(f"  {names[it.basis[i]]}: {nk} / {a} = {r}")
                rat_txt.append(str(r))
        if it.unbounded:
            print(f"  -> Semua entri kolom {names[it.enter]} <= 0, tidak ada rasio.")
            print(f"     {names[it.enter]} bisa naik tanpa batas, Z juga: soal TIDAK TERBATAS. Berhenti.")
            break
        tie = " (rasio minimum sama: degenerasi, dipilih baris paling atas)" if len(it.leave_ties) > 1 else ""
        print(f"  -> Rasio minimum {it.min_ratio} pada baris {names[it.basis[it.leave]]}: {names[it.basis[it.leave]]} keluar{tie}")

        step(5, "Tentukan angka kunci (pivot)")
        print(f"  Angka kunci = {it.pivot}  (baris {names[it.basis[it.leave]]}, kolom {names[it.enter]})")
        tableau(it.T, it.basis, names, pivot=(it.leave, it.enter), ratios=rat_txt)

        step(6, "Hitung baris-baris baru")
        print(f"  {names[it.enter]} menggantikan {names[it.leave_var]} di basis.")
        for kind, i, f, old, neu in it.ops:
            nm = "Z" if i == m else names[it.basis[i]]
            if kind == "pivot":
                print(f"  Baris kunci baru ({names[it.enter]}) = baris {nm} lama / {it.pivot}")
            elif kind == "same":
                print(f"  Baris {nm} tidak berubah (entri kolom kunci sudah 0)")
            else:
                print(f"  Baris {nm} baru = baris {nm} lama - ({f}) x baris kunci baru")
            print("    lama:", vec(old))
            if kind != "same":
                print("    baru:", vec(neu))
        tableau(it.T2, it.basis2, names)
        print(f"  Z sekarang = {it.T2[m][cols]}")

        step(7, "Uji optimalitas")
        negs = [f"{names[j]} ({it.T2[m][j]})" for j in range(sol.art_start) if it.T2[m][j].is_neg()]
        if negs:
            print(f"  Baris Z masih punya koefisien negatif: {', '.join(negs)} -> kembali ke langkah 3.")
        else:
            print("  Semua koefisien baris Z >= 0 -> solusi OPTIMAL, iterasi berhenti.")

    # ---- Kesimpulan ----
    print("\n" + bar)
    print("KESIMPULAN")
    print(bar)
    if sol.status == "unbounded":
        print("Tidak ada solusi optimal berhingga: Z dapat naik tanpa batas.")
    elif sol.status == "maxiter":
        print(f"Dihentikan setelah {MAX_ITER} iterasi tanpa optimum (kemungkinan cycling).")
    elif sol.status == "infeasible":
        bad = ", ".join(f"{names[sol.basis[i]]} = {sol.T[i][cols]}" for i in sol.bad_art)
        print(f"Variabel buatan masih positif di basis ({bad}): kendala saling bertentangan, TIDAK ADA SOLUSI LAYAK.")
    else:
        print(f"Z maksimum = {sol.Z}")
        for j in range(n):
            print(f"  {names[j]} = {sol.x[j]}  ({'basis' if j in sol.basis else 'non-basis'})")
        sym = {"<=": "habis terpakai", ">=": "tepat di batas minimum"}
        for i in range(m):
            if sol.s[i] is None:
                print(f"  K{i + 1} (=): terpenuhi tepat")
                continue
            used = sol.s[i].is_zero()
            lab = sym[P.rel[i]] if used else ("masih ada sisa" if P.rel[i] == "<=" else "melebihi batas minimum")
            extra = f", harga bayangan {sol.shadow[i]}" if used else ""
            print(f"  K{i + 1} ({P.rel[i]}): {lab}  (s{i + 1} = {sol.s[i]}{extra})")
        print("  Harga bayangan = perubahan Z bila nilai kanan kendala naik 1 satuan.")
    print(f"Jumlah iterasi: {sum(1 for it in sol.iters if it.T2 is not None)}")


# --------------------------------------------------------------------------
# Grafik (matplotlib, opsional)
# --------------------------------------------------------------------------
def _style_axes(ax, theme):
    if theme != "dark":
        return
    bg, fg = "#171A21", "#E5E7EB"
    ax.figure.set_facecolor(bg)
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_color("#3A4150")
    ax.tick_params(colors=fg)
    for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
        lab.set_color(fg)
    ax.grid(color="#2A2F3A", alpha=0.8)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor(bg)
        leg.get_frame().set_edgecolor("#3A4150")
        for t in leg.get_texts():
            t.set_color(fg)


def build_figure(sol: Solution, theme: str = "light", pyplot: bool = False):
    """Buat grafik matplotlib (daerah layak untuk n = 2, selain itu grafik Z per iterasi).
    pyplot=False memakai Figure langsung (aman untuk aplikasi web seperti Streamlit)."""
    ink = "white" if theme == "dark" else "black"

    def new_fig(size):
        if pyplot:
            import matplotlib.pyplot as plt

            return plt.figure(figsize=size)
        from matplotlib.figure import Figure

        return Figure(figsize=size)

    P, n = sol.P, sol.P.n
    if n == 2:
        fig = new_fig((5.4, 4.3))
        ax = fig.subplots()
        a = [(float(r[0]), float(r[1])) for r in P.A]
        b = [float(v) for v in P.b]
        xm = ym = 0.0
        for (a1, a2), bb in zip(a, b):
            if a1 > 1e-12:
                xm = max(xm, bb / a1)
            if a2 > 1e-12:
                ym = max(ym, bb / a2)
        for p in sol.pts:
            xm, ym = max(xm, p[0]), max(ym, p[1])
        xm = xm or ym or 1.0
        ym = ym or xm
        XM, YM = xm * 1.15, ym * 1.15

        hp = []  # setengah bidang a1*x + a2*y <= r
        for (a1, a2), bb, rel in zip(a, b, P.rel):
            if rel in ("<=", "="):
                hp.append((a1, a2, bb))
            if rel in (">=", "="):
                hp.append((-a1, -a2, -bb))
        hp += [(-1, 0, 0), (0, -1, 0), (1, 0, XM), (0, 1, YM)]
        verts = []
        for i in range(len(hp)):
            for j in range(i + 1, len(hp)):
                p, q = hp[i], hp[j]
                det = p[0] * q[1] - p[1] * q[0]
                if abs(det) < 1e-12:
                    continue
                x = (p[2] * q[1] - p[1] * q[2]) / det
                y = (p[0] * q[2] - p[2] * q[0]) / det
                if all(h[0] * x + h[1] * y <= h[2] + 1e-7 * (1 + abs(h[2])) for h in hp):
                    if not any(abs(v[0] - x) < 1e-7 and abs(v[1] - y) < 1e-7 for v in verts):
                        verts.append((x, y))
        if verts:
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)
            verts.sort(key=lambda v: math.atan2(v[1] - cy, v[0] - cx))
            ax.fill([v[0] for v in verts], [v[1] for v in verts], alpha=0.25, color="orange", label="Daerah layak")

        def line_pts(a1, a2, r):
            if abs(a2) > 1e-12:
                xs = [0, XM]
                return xs, [(r - a1 * x) / a2 for x in xs]
            if abs(a1) > 1e-12:
                return [r / a1] * 2, [0, YM]
            return None  # garis merosot (semua koefisien nol)

        for i, ((a1, a2), bb) in enumerate(zip(a, b)):
            lp = line_pts(a1, a2, bb)
            if lp is None:
                continue
            xs, ys = lp
            ax.plot(xs, ys, lw=1.6, label=f"K{i + 1}: {expr(P.A[i], ['x1', 'x2'])} {P.rel[i]} {fr(P.b[i])}")
        if sol.status == "optimal":
            lp = line_pts(float(P.c[0]), float(P.c[1]), float(sol.Z.a))
            if lp is not None:
                ax.plot(lp[0], lp[1], "--", color="crimson", lw=1.6, label=f"Garis tujuan Z = {fr(sol.Z.a)}")
        path = []
        for p in sol.pts:
            if not path or abs(path[-1][0] - p[0]) > 1e-9 or abs(path[-1][1] - p[1]) > 1e-9:
                path.append(p)
        if path:
            ax.plot([p[0] for p in path], [p[1] for p in path], "-o", ms=4, color=ink, lw=2, label="Jalur simplex")
        for p in path:
            ax.annotate(f"({p[0]:g}, {p[1]:g})", p, textcoords="offset points", xytext=(5, 5), fontsize=7, color=ink)
        if sol.status == "optimal" and path:
            ax.plot(*path[-1], "o", ms=9, color="gold", mec=ink, zorder=5)
        ax.set_xlim(0, XM)
        ax.set_ylim(0, YM)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Daerah layak dan jalur simplex")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)
    else:
        fig = new_fig((8.5, 3.3))
        ax1, ax2 = fig.subplots(1, 2)
        pts = [(i, z) for i, z in enumerate(sol.zs) if z is not None]
        if pts:
            ax1.plot([p[0] for p in pts], [p[1] for p in pts], "-o", lw=2)
            for p in pts:
                ax1.annotate(f"{p[1]:g}", p, textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=ink)
        ax1.set_xlabel("Iterasi")
        ax1.set_ylabel("Z")
        ax1.set_title("Nilai Z per iterasi")
        ax1.grid(alpha=0.3)
        vals = [float(v.a) for v in sol.x]
        ax2.bar(sol.names[:n], vals)
        for i, v in enumerate(vals):
            ax2.annotate(f"{v:g}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=ink)
        ax2.set_title("Variabel keputusan pada solusi akhir")
    fig.tight_layout()
    for a in fig.axes:
        _style_axes(a, theme)
    return fig


def plot(sol: Solution, save: Optional[str] = None):
    try:
        import matplotlib

        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(Grafik dilewati: matplotlib belum terpasang. Pasang dengan: pip install matplotlib)")
        return
    fig = build_figure(sol, "light", pyplot=True)
    if save:
        fig.savefig(save, dpi=150)
        print(f"\nGrafik disimpan ke {save}")
    else:
        try:
            plt.show()
        except Exception as e:  # lingkungan tanpa layar
            print(f"\n(Grafik tidak bisa ditampilkan: {e}. Coba --save-plot grafik.png)")


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
SIGNS = {"<=": "<=", "=<": "<=", "≤": "<=", ">=": ">=", "=>": ">=", "≥": ">=", "=": "=", "==": "="}


def ask_int(prompt, lo=1, hi=20):
    while True:
        try:
            v = int(input(prompt).strip())
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Masukkan bilangan bulat {lo}..{hi}.")


def ask_numbers(prompt, count):
    while True:
        try:
            vals = [parse_number(t) for t in input(prompt).split()]
            if len(vals) == count:
                return vals
            print(f"  Harus tepat {count} angka.")
        except (ValueError, ZeroDivisionError):
            print("  Ada isian yang bukan angka. Contoh: 3  -2,5  1/2")


def ask_problem() -> Problem:
    print("== Maksimasi dengan metode simplex ==")
    n = ask_int("Jumlah variabel (n): ")
    m = ask_int("Jumlah kendala (m): ")
    c = ask_numbers(f"Koefisien fungsi tujuan Maks Z ({n} angka dipisah spasi): ", n)
    A, b, rel = [], [], []
    print(f"Tiap kendala: {n} koefisien, tanda (<=, >=, =), nilai kanan.  Contoh: 3 2 <= 18")
    i = 0
    while i < m:
        line = re.sub(r"(<=|=<|>=|=>|≤|≥|==|=)", r" \1 ", input(f"  Kendala {i + 1}: "))
        tok = line.split()
        try:
            sg = [t for t in tok if t in SIGNS]
            if len(sg) != 1 or len(tok) != n + 2 or tok[n] not in SIGNS:
                raise ValueError
            A.append([parse_number(t) for t in tok[:n]])
            rel.append(SIGNS[tok[n]])
            b.append(parse_number(tok[n + 1]))
            i += 1
        except (ValueError, ZeroDivisionError):
            print(f"  Format salah. Tulis {n} angka, satu tanda, satu angka. Contoh: 3 2 <= 18")
    return make_problem(c, A, b, rel)


def demo_problem() -> Problem:
    F = Fraction
    return make_problem([F(3), F(5)], [[F(1), F(0)], [F(0), F(2)], [F(3), F(2)]], [F(4), F(12), F(18)], ["<=", "<=", "<="])


def main():
    global DECIMAL
    ap = argparse.ArgumentParser(description="Kalkulator simplex maksimasi (<=, >=, =)")
    ap.add_argument("--demo", action="store_true", help="pakai contoh Maks Z = 3x1 + 5x2")
    ap.add_argument("--decimal", action="store_true", help="tampilkan angka sebagai desimal")
    ap.add_argument("--no-plot", action="store_true", help="jangan buat grafik")
    ap.add_argument("--save-plot", metavar="FILE", help="simpan grafik ke file (png/pdf) tanpa menampilkan jendela")
    args = ap.parse_args()
    DECIMAL = args.decimal

    P = demo_problem() if args.demo else ask_problem()
    sol = solve(P)
    report(sol)
    if not args.no_plot:
        plot(sol, args.save_plot)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nDibatalkan.")
        sys.exit(1)
