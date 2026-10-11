"""
Simplex Console (Streamlit)

Antarmuka untuk kalkulator simplex maksimasi: n variabel, kendala <=, >=, =

"""
from __future__ import annotations

import io
import re
import time
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st

import simplex as sx

st.set_page_config(page_title="Simplex Console", layout="wide")

REL = ["<=", ">=", "="]
CLASSIC = {"c": ["0", "0"], "A": [["0", "0"], ["0", "0"], ["0", "0"]], "b": ["0", "0", "0"]}
SEVEN = [
    
]

st.markdown(
    """
<style>
.pill{display:inline-block;font-family:monospace;font-weight:700;letter-spacing:.08em;
      padding:.25rem .7rem;border-radius:6px;border:1px solid #F5A524;color:#F5A524}
.pill.bad{border-color:#F87171;color:#F87171}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helper ----
def pretty(name: str) -> str:
    """x1 -> x₁, s2 -> s₂, a3 -> a₃"""
    return re.sub(r"\d+", lambda m: sx.sub(m.group()), name)


def default_tables(n: int, m: int):
    cols = [f"x{j + 1}" for j in range(n)]
    obj = pd.DataFrame([[CLASSIC["c"][j] if j < 2 else "0" for j in range(n)]], columns=cols)
    rows = []
    for i in range(m):
        a = [CLASSIC["A"][i][j] if (i < 3 and j < 2) else "0" for j in range(n)]
        rows.append(a + ["<="] + [CLASSIC["b"][i] if i < 3 else "0"])
    con = pd.DataFrame(rows, columns=cols + ["Tanda", "RK"])
    return obj, con


def parse_tables(obj: pd.DataFrame, con: pd.DataFrame, n: int, m: int):
    try:
        c = [sx.parse_number(str(obj.iloc[0][f"x{j + 1}"])) for j in range(n)]
        A = [[sx.parse_number(str(con.iloc[i][f"x{j + 1}"])) for j in range(n)] for i in range(m)]
        b = [sx.parse_number(str(con.iloc[i]["RK"])) for i in range(m)]
        rel = [str(con.iloc[i]["Tanda"]).strip() for i in range(m)]
        if any(r not in REL for r in rel):
            return None, "Kolom Tanda harus berisi <=, >=, atau =."
    except (ValueError, ZeroDivisionError, KeyError, IndexError):
        return None, "Ada isian yang bukan angka. Gunakan angka, pecahan seperti 1/2, atau desimal seperti 0,5."
    return sx.make_problem(c, A, b, rel), None


def current_theme() -> str:
    try:
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return st.get_option("theme.base") or "dark"


def tab_df(T, basis, names, pivot=None, ratios=None):
    """Tabel simplex sebagai DataFrame berwarna: baris/kolom kunci diarsir, angka kunci ditebalkan."""
    m = len(basis)
    idx = [names[basis[i]] if i < m else "Z" for i in range(m + 1)]
    cols = names + ["RK"]
    data = [[str(T[i][j]) for j in range(len(T[0]))] for i in range(m + 1)]
    if ratios is not None:
        cols = cols + ["Rasio"]
        for i in range(m + 1):
            data[i].append(ratios[i] if i < m else "")
    df = pd.DataFrame(data, index=idx, columns=cols)
    df.index.name = "Basis"

    def paint(d):
        css = pd.DataFrame("", index=d.index, columns=d.columns)
        css.iloc[m, :] = "font-weight:700"
        if pivot is not None:
            r, c = pivot
            css.iloc[r, :] = "background-color:rgba(245,165,36,0.18)"
            css.iloc[:, c] = "background-color:rgba(245,165,36,0.18)"
            css.iloc[r, c] = "background-color:rgba(245,165,36,0.65);color:#14161A;font-weight:700"
        return css

    return df.style.apply(paint, axis=None)


def show_metrics(items):
    for k in range(0, len(items), 5):
        row = items[k:k + 5]
        for col, (lab, val) in zip(st.columns(len(row)), row):
            col.metric(lab, val)


# --------------------------------------------------------------- pengerjaan -
def render_setup(sol: sx.Solution, pn):
    P, n, m = sol.P, sol.P.n, sol.P.m
    xn = pn[:n]
    arts = [pn[c] for c in sol.col_a if c >= 0]
    with st.expander("Langkah 1 dan 2: bentuk standar dan tabel awal", expanded=True):
        st.markdown("**Langkah 1 · Ubah ke bentuk standar dan bentuk baku simpleks**")
        st.markdown(
            "Bentuk standar: Z dimaksimumkan, semua variabel ≥ 0, dan setiap ruas kanan ≥ 0. "
            "Kendala dengan ruas kanan negatif dikali -1 sehingga tandanya berbalik."
        )
        lines = [f"Maksimumkan Z = {sx.expr(P.c, xn)}"]
        for i in range(m):
            lines.append(f"K{sx.sub(i + 1)}: {sx.expr(P.oA[i], xn)} {P.oRel[i]} {sx.fr(P.oB[i])}")
            if P.flipped[i]:
                lines.append(f"    ruas kanan negatif, kedua ruas dikali -1: {sx.expr(P.A[i], xn)} {P.rel[i]} {sx.fr(P.b[i])}")
        lines.append(", ".join(xn) + " >= 0")
        st.code("\n".join(lines), language=None)
        st.markdown("##### Bentuk baku simpleks (canonical form)")
        sentence, zline, clines = sx.canonical(sol, pn)
        st.markdown(sentence)
        st.markdown("\n".join([f"- **Baris Z:** `{zline}`"] + [f"- **Kendala {i}:** `{ln}`" for i, ln in enumerate(clines, 1)]))
        if arts:
            st.caption("Suku M pada baris Z berasal dari fungsi tujuan Z = (fungsi semula) - M(" + " + ".join(arts) + ").")
        st.caption("Basis awal: " + ", ".join(pn[j] for j in sol.basis0))

        st.markdown("**Langkah 2 · Susun tabel simplex awal**")
        st.markdown("Baris Z berisi koefisien tujuan dengan tanda dibalik; RK adalah ruas kanan.")
        if sol.adj:
            st.markdown("Variabel basis harus punya koefisien 0 di baris Z, jadi baris Z dikurangi M × baris yang memuat variabel buatan:")
            ops = []
            for i, f, old, neu in sol.adj:
                ops.append(f"Baris Z baru = baris Z lama - ({f}) x baris {pn[sol.basis0[i]]}")
                ops.append("  lama: " + sx.vec(old))
                ops.append("  baru: " + sx.vec(neu))
            st.code("\n".join(ops), language=None)
        st.dataframe(tab_df(sol.T0, sol.basis0, pn))
        negs = [pn[j] for j in range(sol.art_start) if sol.T0[m][j].is_neg()]
        if negs:
            st.info(f"Baris Z masih memuat koefisien negatif ({', '.join(negs)}), jadi solusi belum optimal dan iterasi dimulai.")
        else:
            st.success("Baris Z tidak memuat koefisien negatif, jadi tabel awal sudah optimal.")


def render_iter(sol: sx.Solution, k: int, it: sx.Iter, pn):
    m, cols = sol.P.m, sol.cols
    with st.expander(f"Iterasi {k}: langkah 3 sampai 7", expanded=(k == 1)):
        st.markdown("**Langkah 3 · Tentukan kolom kunci**")
        st.markdown("Kolom dengan koefisien Z paling negatif dipilih, karena menaikkan variabel itu menambah Z paling cepat.")
        st.markdown("  ".join(f"`{pn[j]} = {it.T[m][j]}`" for j in range(sol.art_start)))
        tie = (f" Ada koefisien yang sama ({', '.join(pn[j] for j in it.enter_ties)}); dipilih yang paling kiri."
               if len(it.enter_ties) > 1 else "")
        st.markdown(f"Paling negatif `{it.best}` pada `{pn[it.enter]}`, jadi **{pn[it.enter]} masuk basis**.{tie}")

        st.markdown("**Langkah 4 · Tentukan baris kunci**")
        st.markdown("Bagi RK dengan entri kolom kunci (hanya yang positif). Rasio terkecil menentukan variabel yang keluar.")
        rat_lines, rat_txt = [], []
        for i in range(m):
            a, rk, r = it.T[i][it.enter], it.T[i][cols], it.ratios[i]
            if r is None:
                rat_lines.append(f"{pn[it.basis[i]]}: entri kolom kunci {a} (<= 0), tidak dihitung")
                rat_txt.append("-")
            else:
                rat_lines.append(f"{pn[it.basis[i]]}: {rk} / {a} = {r}")
                rat_txt.append(str(r))
        st.code("\n".join(rat_lines), language=None)
        if it.unbounded:
            st.error(
                f"Semua entri kolom {pn[it.enter]} bernilai ≤ 0, jadi tidak ada rasio. {pn[it.enter]} bisa naik tanpa batas "
                "dan Z ikut naik tanpa batas. Soal ini tidak terbatas (unbounded); perhitungan berhenti."
            )
            return
        tie = " Rasio minimum sama pada beberapa baris (degenerasi); dipilih baris paling atas." if len(it.leave_ties) > 1 else ""
        st.markdown(f"Rasio minimum `{it.min_ratio}` pada baris `{pn[it.basis[it.leave]]}`, jadi **{pn[it.basis[it.leave]]} keluar**.{tie}")

        st.markdown("**Langkah 5 · Tentukan angka kunci (pivot)**")
        st.markdown(f"Angka kunci = `{it.pivot}`, titik temu baris `{pn[it.basis[it.leave]]}` dan kolom `{pn[it.enter]}`.")
        st.dataframe(tab_df(it.T, it.basis, pn, pivot=(it.leave, it.enter), ratios=rat_txt))

        st.markdown("**Langkah 6 · Hitung baris-baris baru**")
        st.markdown(f"{pn[it.enter]} menggantikan {pn[it.leave_var]} di basis. Baris kunci dibagi angka kunci, lalu baris lain "
                    "dikurangi kelipatan baris kunci baru sampai entri kolom kunci menjadi 0.")
        ops = []
        for kind, i, f, old, neu in it.ops:
            nm = "Z" if i == m else pn[it.basis[i]]
            if kind == "pivot":
                ops.append(f"Baris kunci baru ({pn[it.enter]}) = baris {nm} lama / {it.pivot}")
            elif kind == "same":
                ops.append(f"Baris {nm} tidak berubah (entri kolom kunci sudah 0)")
            else:
                ops.append(f"Baris {nm} baru = baris {nm} lama - ({f}) x baris kunci baru")
            ops.append("  lama: " + sx.vec(old))
            if kind != "same":
                ops.append("  baru: " + sx.vec(neu))
        st.code("\n".join(ops), language=None)
        st.dataframe(tab_df(it.T2, it.basis2, pn))
        st.caption(f"Z sekarang = {it.T2[m][cols]}")

        st.markdown("**Langkah 7 · Uji optimalitas**")
        negs = [f"{pn[j]} ({it.T2[m][j]})" for j in range(sol.art_start) if it.T2[m][j].is_neg()]
        if negs:
            st.warning(f"Baris Z masih punya koefisien negatif: {', '.join(negs)}. Solusi belum optimal, kembali ke langkah 3.")
        else:
            st.success("Semua koefisien baris Z sudah ≥ 0. Tidak ada variabel yang bisa menaikkan Z lagi, jadi solusi tabel ini optimal.")


def render_conclusion(sol: sx.Solution, pn):
    P, n, m = sol.P, sol.P.n, sol.P.m
    st.subheader("Kesimpulan")
    if sol.status == "unbounded":
        st.error("Tidak ada solusi optimal berhingga: nilai Z dapat dinaikkan tanpa batas.")
        return
    if sol.status == "maxiter":
        st.error(f"Perhitungan dihentikan setelah {sx.MAX_ITER} iterasi tanpa optimum. Soal ini kemungkinan berputar (cycling).")
        return
    if sol.status == "infeasible":
        bad = ", ".join(f"{pn[sol.basis[i]]} = {sol.T[i][sol.cols]}" for i in sol.bad_art)
        st.error(f"Variabel buatan masih positif di basis ({bad}). Kendala-kendalanya saling bertentangan, jadi tidak ada solusi layak.")
        return
    st.success(f"Z maksimum = {sol.Z} pada " + ", ".join(f"{pn[j]} = {sol.x[j]}" for j in range(n)))
    rows = [{"Item": pn[j], "Nilai": str(sol.x[j]), "Keterangan": "basis" if j in sol.basis else "non-basis"} for j in range(n)]
    for i in range(m):
        if sol.s[i] is None:
            rows.append({"Item": f"Kendala {i + 1} (=)", "Nilai": "-", "Keterangan": "terpenuhi tepat"})
            continue
        used = sol.s[i].is_zero()
        if P.rel[i] == "<=":
            lab = "habis terpakai" if used else "masih ada sisa"
        else:
            lab = "tepat di batas minimum" if used else "melebihi batas minimum"
        extra = f", harga bayangan {sol.shadow[i]}" if used else ""
        rows.append({"Item": f"Kendala {i + 1} ({P.rel[i]})", "Nilai": f"{pn[sol.col_s[i]]} = {sol.s[i]}", "Keterangan": lab + extra})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Harga bayangan = perubahan Z jika ruas kanan kendala itu naik 1 satuan.")


# --------------------------------------------------------------- halaman ----
st.title("Simplex Console")
st.caption("Pemandu Langkah Interaktif")

with st.sidebar:
    st.header("Masukan")
    n = int(st.number_input("Jumlah variabel (n)", min_value=1, max_value=8, value=2, step=1))
    m = int(st.number_input("Jumlah kendala (m)", min_value=1, max_value=8, value=3, step=1))
    mode = st.radio("Tampilan angka", ["Pecahan", "Desimal"], horizontal=True)
sx.DECIMAL = mode == "Desimal"

# soal awal untuk ukuran (n, m) yang baru dipilih
if st.session_state.get("dims") != (n, m):
    o0, c0 = default_tables(n, m)
    P0, _ = parse_tables(o0, c0, n, m)
    st.session_state["dims"] = (n, m)
    st.session_state["problem"] = P0

st.subheader("Soal")
obj0, con0 = default_tables(n, m)
with st.form("soal"):
    st.caption("Fungsi tujuan: Maksimumkan Z")
    obj_df = st.data_editor(
        obj0, key=f"obj_{n}_{m}", hide_index=True, num_rows="fixed",
        column_config={f"x{j + 1}": st.column_config.TextColumn(pretty(f"x{j + 1}")) for j in range(n)},
    )
    st.caption("Kendala: pilih tanda ≤ (<=), ≥ (>=) atau = pada kolom Tanda. Isian boleh pecahan (1/2) atau desimal (0,5).")
    con_df = st.data_editor(
        con0, key=f"con_{n}_{m}", hide_index=True, num_rows="fixed",
        column_config={
            **{f"x{j + 1}": st.column_config.TextColumn(pretty(f"x{j + 1}")) for j in range(n)},
            "Tanda": st.column_config.SelectboxColumn("Tanda", options=REL, required=True),
            "RK": st.column_config.TextColumn("Ruas kanan"),
        },
    )
    submitted = st.form_submit_button("Hitung", type="primary")

if submitted:
    P_new, err = parse_tables(obj_df, con_df, n, m)
    if err:
        st.error(err)
    else:
        st.session_state["problem"] = P_new

P = st.session_state.get("problem")
if P is None:
    st.stop()

t0 = time.perf_counter()
sol = sx.solve(P)
ms = (time.perf_counter() - t0) * 1000
pn = [pretty(x) for x in sol.names]
K = sum(1 for it in sol.iters if it.T2 is not None)
ok = sol.status == "optimal"
label = {"optimal": "OPTIMAL", "unbounded": "TIDAK TERBATAS", "infeasible": "TIDAK LAYAK", "maxiter": "BERHENTI"}[sol.status]

st.markdown(
    f'<span class="pill{"" if ok else " bad"}">STATUS: {label}</span> &nbsp; '
    f'<span style="opacity:.7"> {"" if ms < 1 else round(ms)} </span>',
    unsafe_allow_html=True,
)
items = [("Z maksimum", str(sol.Z) if ok else ("∞" if sol.status == "unbounded" else "-"))]
items += [(pn[j], str(sol.x[j]) if ok else "-") for j in range(P.n)]
items.append(("Iterasi", str(K)))
show_metrics(items)

st.subheader("Visualisasi grafik")
if P.n != 2:
    st.caption(f"n = {P.n}: grafik daerah layak hanya untuk 2 variabel, jadi ditampilkan grafik Z per iterasi dan nilai variabel.")
try:
    # n = 2: grafik ditaruh di kolom kiri agar tidak memenuhi layar; selain itu selebar layar tetapi pendek
    holder = st.columns([5, 3])[0] if P.n == 2 else st.container()
    with holder:
        st.pyplot(sx.build_figure(sol, theme=current_theme()), use_container_width=True)
except ImportError:
    st.warning("matplotlib belum terpasang: pip install matplotlib")

st.subheader("Proses Pemodelan")
st.markdown("\n".join(f"{i}. {t}" for i, t in enumerate(SEVEN, 1)))
st.caption("Langkah 3 sampai 7 diulang pada setiap iterasi.")
render_setup(sol, pn)
for k, it in enumerate(sol.iters, 1):
    render_iter(sol, k, it, pn)
render_conclusion(sol, pn)

buf = io.StringIO()
with redirect_stdout(buf):
    sx.report(sol)
st.download_button("Unduh pengerjaan (.txt)", buf.getvalue(), file_name="pengerjaan_simplex.txt", mime="text/plain")
