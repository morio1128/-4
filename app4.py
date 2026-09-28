# app.py の冒頭
import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # ← この行を追加（グラフの日本語文字化け防止）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="パイプライン型ADC 入門", layout="wide")

VREF = 2.0  # 基準電圧（固定）。入力レンジは 0 ～ Vref
KW = [
    "全体構造",
    "残差・残差増幅器 (MDAC)",
    "パイプライン処理",
    "時差・誤差補正 (DEC)",
]
TH_LO, TH_HI = 0.75, 1.25  # 1.5bit Sub-ADC のしきい値
LAST_LO, LAST_HI = 0.5, 1.5  # 最終stage（MDACなし）のしきい値


# ===================== 計算部 =====================
def weight(k, n):
    """Stage k の出力 D_k がコードに加算されるときの重み（DECの1bitずらし加算）"""
    return 1 if k == n else 2 ** (n - 1 - k)


def run_adc(vin, n, off=0.0):
    """1.5bit/stageモデル。D=0,1,2。残差 Vres = 2*(Vin - D*Vref/4) = 2Vin - D
    off: Sub-ADC比較器のオフセット[V]（全stageのしきい値を同量ずらす）"""
    v = min(max(vin, 0.0), VREF)
    rows, m = [], 0
    for k in range(1, n + 1):
        last = k == n
        lo, hi = (LAST_LO, LAST_HI) if last else (TH_LO, TH_HI)
        d = 0 if v < lo + off else (1 if v < hi + off else 2)
        vdac = d * VREF / 4
        vsub = v - vdac
        vres = np.nan if last else 2 * vsub
        rows.append(
            dict(stage=k, vin=v, d=d, vdac=vdac, vsub=vsub, vres=vres)
        )
        m += d * weight(k, n)
        v = vres
    return rows, min(m, 2**n - 1), m


def transfer(x, off=0.0):
    d = np.where(x < TH_LO + off, 0, np.where(x < TH_HI + off, 1, 2))
    return 2 * x - d


# ===================== 全体回路ブロック図 =====================
def draw_block(n, focus):
    ON, OFF, ALL = "#ffd54f", "#e3eaf2", "#c8e6c9"
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    W, G, X0 = 2.0, 0.4, 1.9
    xe = X0 + n * (W + G)
    ax.set_xlim(-0.8, xe + 3.2)
    ax.set_ylim(-0.4, 4.4)
    ov, mdac, pipe, dec = (focus == KW[i] for i in range(4))

    def box(x, y, w, h, txt, on):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03",
                fc=ALL if ov else (ON if on else OFF),
                ec="#345",
                lw=1.5,
            )
        )
        ax.text(
            x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9
        )

    def arr(p, q):
        ax.annotate(
            "", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.4)
        )

    ax.text(-0.75, 3.15, "入力 Vin", fontsize=11)
    arr((-0.3, 3.0), (0.2, 3.0))
    box(0.2, 2.4, 1.2, 1.2, "S/H\n(クロック)", pipe)
    arr((1.4, 3.0), (X0, 3.0))
    for k in range(1, n + 1):
        x = X0 + (k - 1) * (W + G)
        last = k == n
        txt = (
            f"stage{k}\nサブADC + DAC\n-> 減算 -> 2倍 (MDAC)"
            if not last
            else f"stage{k}\nサブADCのみ\n(最終stage)"
        )
        box(x, 2.4, W, 1.2, txt, pipe or (mdac and not last))
        if not last:
            arr((x + W, 3.0), (x + W + G, 3.0))
            ax.text(x + W + G / 2, 3.12, "残差 Vres", ha="center", fontsize=8)
        arr((x + W / 2, 2.4), (x + W / 2, 1.5))
        ax.text(x + W / 2 + 0.05, 1.85, f"D{k}", fontsize=9)
        box(
            x + 0.1,
            0.7,
            W - 0.2,
            0.8,
            f"シフトレジスタ\n({n - k} クロック遅延)",
            dec,
        )
        ax.plot([x + W / 2] * 2, [0.7, 0.3], c="#345", lw=1.4)
    ax.plot([X0 + W / 2, xe], [0.3, 0.3], c="#345", lw=1.4)
    arr((xe, 0.3), (xe + 0.3, 0.3))
    box(xe + 0.3, -0.1, 1.9, 1.2, "デジタル誤差補正\n(DEC)", dec)
    arr((xe + 2.2, 0.5), (xe + 2.7, 0.5))
    ax.text(xe + 2.75, 0.5, f"出力 Dout\n({n} ビット)", va="center", fontsize=10)
    label = ["全体構造", "残差・残差増幅器", "パイプライン処理", "シフトレジスタ・DEC"][
        KW.index(focus)
    ]
    ax.set_title(
        f"パイプライン型ADC ブロック図 (N = {n})  ハイライト: {label}",
        fontsize=11,
    )
    return fig


# ===================== 0. 全体構造 =====================
def view_overview(rows, code, n):
    st.subheader("全体構造")
    st.markdown(
        "サイドバーでキーワードを選ぶと、対応するブロックがハイライトされます。\n\n"
        "1. **残差・MDAC**：各stageが粗く判定し、その分を引いた**残差を2倍**して次stageへ渡す。\n"
        "2. **パイプライン処理**：各stageのS/Hが値を保持し、**全stageが別サンプルを同時に処理**する。\n"
        "3. **時差・誤差補正 (DEC)**：stageごとに確定時刻が違うビットを**シフトレジスタで揃え**、"
        "**1bitずらして加算**して比較器誤差を補正しながら1つのコードにする。"
    )
    df = (
        pd.DataFrame(rows)
        .rename(
            columns={
                "stage": "stage",
                "vin": "入力 Vin[V]",
                "d": "判定 D",
                "vdac": "DAC出力[V]",
                "vsub": "差分 Vin-DAC[V]",
                "vres": "出力 Vout(残差×2)[V]",
            }
        )
        .set_index("stage")
    )
    st.dataframe(df.round(3))
    st.caption(
        "最終stageはMDACを持たず、Sub-ADCの判定のみを出力します（Vout は NaN）。"
    )


# ===================== 1. 残差・MDAC =====================
def view_mdac(rows, n):
    st.subheader("残差・残差増幅器 (MDAC)")
    steps = [
        "① Vin入力",
        "② Sub-ADC判定",
        "③ DAC減算（残差）",
        "④ ×2増幅（Vout）",
    ]
    c1, c2 = st.columns(2)
    k = c1.slider("解析するステージ（最終stageはMDACなし）", 1, n - 1, 1)
    step = c2.radio("計算ステップ", steps, horizontal=True)
    si = steps.index(step)
    s = rows[k - 1]
    cmp_txt = (
        "Vin < 0.75 V"
        if s["d"] == 0
        else "0.75 V ≤ Vin < 1.25 V" if s["d"] == 1 else "Vin ≥ 1.25 V"
    )
    msgs = [
        f"① 入力：stage {k} の Vin = **{s['vin']:.3f} V**",
        f"② Sub-ADC（しきい値 0.75 V / 1.25 V）：{cmp_txt} → **D{k} = {s['d']}**",
        f"③ DAC出力 = D × Vref/4 = {s['vdac']:.3f} V、減算：Vin − DAC = **{s['vsub']:.3f} V**（残差）",
        f"④ MDACで2倍増幅：Vout = 2 × {s['vsub']:.3f} = **{s['vres']:.3f} V** → 次stageのVinへ",
    ]
    for i in range(si):
        st.caption(msgs[i])
    st.success(msgs[si])

    x = np.linspace(0, VREF, 800)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(x, transfer(x), c="#1565c0", label="Vout = 2Vin - D*Vref/2")
    for t in TH_LO, TH_HI:
        a1.axvline(t, c="r", ls=":", lw=1)
    a1.plot([], [], c="r", ls=":", label="サブADC しきい値")
    ys = [0, 0, s["vsub"], s["vres"]][si]
    a1.plot([s["vin"]], [ys], "o", c="orange", ms=11, zorder=5)
    if si >= 2:
        a1.plot([s["vin"]] * 2, [0, ys], c="orange", lw=1)
    a1.set(
        xlabel="入力電圧 Vin [V]",
        ylabel="電圧 [V]",
        xlim=(0, 2),
        ylim=(-0.1, 2.1),
        title=f"stage {k} : 鋸歯状伝達特性 (残差プロット)",
    )
    a1.legend(fontsize=8, loc="upper left")
    a1.grid(alpha=0.3)

    labels = ["入力 Vin", "DAC出力", "差分 (Vin-DAC)", "出力 Vout (2倍)"]
    vals = [s["vin"], s["vdac"], s["vsub"], s["vres"]]
    cols = ["#90caf9", "#a5d6a7", "#ffcc80", "#ef9a9a"]
    for i in range(4):
        a2.bar(labels[i], vals[i] if i <= si else 0, color=cols[i])
        if i <= si:
            a2.text(
                i, vals[i] + 0.03, f"{vals[i]:.2f}", ha="center", fontsize=9
            )
    a2.set(
        ylim=(0, 2.2),
        ylabel="電圧 [V]",
        title=f"stage {k} : 各ノードの電圧変化",
    )
    a2.axhline(VREF, c="r", ls=":")
    st.pyplot(fig)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(12, 2.8))
    idx = list(range(1, n))
    vs = [rows[i - 1]["vres"] for i in idx]
    ax.bar(idx, vs, color=["orange" if i == k else "#90caf9" for i in idx])
    ax.axhline(VREF, c="r", ls=":")
    ax.set(
        xlabel="stage",
        ylabel="出力電圧 Vout [V]",
        ylim=(0, 2.3),
        title="各stageの出力電圧（残差）",
    )
    ax.set_xticks(idx)
    for i, v in zip(idx, vs):
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=8)
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption(
        "残差は常に 0 ～ Vref に収まるため、次stageは同じ回路で同じ処理を繰り返せます。"
    )


# ===================== 2. パイプライン処理 =====================
def view_pipe(vin, n):
    st.subheader("パイプライン処理")
    ns, names = 4, ["A", "B", "C", "D"]
    vins = [
        min(max(v, 0), VREF)
        for v in [vin, vin * 0.6 + 0.3, (vin + 0.9) % 2, 2 - vin * 0.5]
    ]
    res = [run_adc(v, n)[0] for v in vins]
    t = st.slider("クロックサイクル t", 1, n + ns - 1, 1)
    data = {}
    for c in range(1, t + 1):
        row = {
            "S/H (入力)": (
                f"サンプル {names[c-1]} ({vins[c-1]:.2f}V)"
                if c <= ns
                else "—"
            )
        }
        for k in range(1, n + 1):
            s = c - k + 1
            row[f"stage{k}"] = (
                f"サンプル{names[s-1]}: D{k}={res[s-1][k-1]['d']}"
                if 1 <= s <= ns
                else "—"
            )
        s = c - n + 1
        row["変換完了"] = f"サンプル{names[s-1]} 完了" if 1 <= s <= ns else "—"
        data[f"クロック {c}"] = row
    df = pd.DataFrame(data).T
    st.dataframe(
        df.style.apply(
            lambda r: (
                ["background-color:#ffd54f"] * len(r)
                if r.name == f"クロック {t}"
                else [""] * len(r)
            ),
            axis=1,
        )
    )
    a, b, c = st.columns(3)
    a.metric("レイテンシ", f"{n} クロック")
    b.metric("スループット", "1 サンプル / クロック")
    c.metric("同時処理サンプル数", f"最大 {n}")
    st.markdown(
        "- **レイテンシ**：入力されてから結果が出るまでの遅延。stage数Nに比例し、**Nクロック**かかります。\n"
        "- **スループット**：各stageのS/H（レジスタ）が値を保持するので、全stageが別サンプルを並列処理でき、"
        "**定常状態では毎クロック1サンプル**を出力できます。\n"
        "- stage数を増やすと分解能は上がりますが、増えるのは**レイテンシだけ**でスループットは変わりません。"
    )


# ===================== 3. 時差・誤差補正 (DEC) =====================
def view_dec(vin, n):
    st.subheader("時差・誤差補正 (DEC)")
    c1, c2 = st.columns(2)
    t = c1.slider(
        "クロック（サンプルAがstage1に入ってからの経過）", 1, n, n
    )
    off = c2.slider(
        "比較器オフセット [V]（全stageのSub-ADCしきい値をずらす）",
        -0.4,
        0.4,
        0.0,
        0.05,
    )
    rows, code, raw = run_adc(vin, n, off)
    _, ideal, _ = run_adc(vin, n, 0.0)

    st.markdown(
        "##### ① タイムアライメント（stage k のビットはクロック k で確定 → N−k クロック遅延）"
    )
    cell = {}
    for k in range(1, n + 1):
        nm = f"D{k}" + (
            "(MSB側)" if k == 1 else "(LSB側)" if k == n else ""
        )
        d = rows[k - 1]["d"]
        if t < k:
            cell[nm] = "— (未確定)"
        else:
            pos, dly = t - k, n - k
            cell[nm] = (
                f"{d}  ✔整列"
                if pos >= dly
                else f"{d}  (遅延中 {pos}/{dly})"
            )
    st.dataframe(pd.DataFrame([cell], index=[f"クロック {t}"]))
    st.dataframe(
        pd.DataFrame({
            "ビット": [f"D{k}" for k in range(1, n + 1)],
            "確定クロック": list(range(1, n + 1)),
            "遅延stage数(N-k)": [n - k for k in range(1, n + 1)],
            "出力クロック": [n] * n,
        })
        .set_index("ビット")
        .T
    )

    st.markdown(
        "##### ② DEC：各stageの出力（D=0,1,2）を **1bitずつずらして加算**"
    )
    if t < n:
        st.info(
            f"クロック {n} で全ビットが揃い、加算が行われます。スライダーを進めてください。"
        )
    else:
        w = n + 1
        lines = []
        for r in rows:
            k = r["stage"]
            sh = int(np.log2(weight(k, n)))
            lines.append(f"D{k} = {r['d']}   {format(r['d'] << sh, f'0{w}b')}")
        lines.append("-" * (9 + w))
        lines.append(
            f"合計      {format(raw, f'0{w}b')}  = {raw}"
            + ("  → 上限にクリップ" if raw != code else "")
        )
        st.code("\n".join(lines))
        st.caption(
            "隣り合うstageのビットが1bit重なって足される（冗長性）ことで、"
            "前stageの判定ミスを後stageの残差が打ち消します。最下位2stageは同じ重みです。"
        )

        lsb = VREF / 2**n
        vrec = code * lsb
        err = min(max(vin, 0), VREF) - vrec
        a, b, c, d = st.columns(4)
        a.metric("Dout (2進)", f"{code:0{n}b}")
        b.metric("Dout (10進)", f"{code} / {2**n - 1}")
        c.metric("再構成電圧", f"{vrec:.3f} V")
        d.metric(
            "量子化誤差",
            f"{err:+.3f} V",
            help=f"1 LSB = Vref/2^N = {lsb:.4f} V",
        )
        st.caption(
            f"1 LSB = {lsb:.4f} V。理想の量子化誤差は ±0.5 LSB（±{lsb/2:.4f} V）以内"
            "（最上位コード付近はクリップにより最大1 LSB）。"
        )
        if abs(code - ideal) <= 1:
            st.success(
                f"✅ オフセット {off:+.2f} V があっても、DEC後のコード {code} は理想値 {ideal} と"
                "ほぼ一致（±1 LSB以内）。冗長性による誤差補正が働いています。"
            )
            st.caption("※ 最終stageのしきい値自体のずれにより、±1 LSB程度は残ります。")
        else:
            st.error(
                f"❌ コード {code} が理想値 {ideal} から大きくずれました。"
                "残差が次stageの入力範囲を超え、補正可能範囲（目安 ±0.25 V ＝ Vref/8）を超えています。"
            )

    x = np.linspace(0, VREF, 800)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(x, transfer(x, off), c="#1565c0")
    ax.axhspan(0, VREF, color="green", alpha=0.08, label="次stageの入力許容範囲")
    for th in TH_LO + off, TH_HI + off:
        ax.axvline(th, c="r", ls=":", lw=1)
    ax.set(
        xlabel="入力電圧 Vin [V]",
        ylabel="出力電圧 Vout [V]",
        xlim=(0, 2),
        ylim=(-0.3, 2.3),
        title=f"比較器オフセット {off:+.2f} V 時の伝達特性",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "しきい値がずれても残差は 0 ～ Vref に収まる間は、後stageが正しく変換し直せます。"
        "ずれが大きくなって残差が緑の範囲外に出ると、補正できません。"
    )


# ===================== メイン =====================
st.title("🔧 パイプライン型AD変換器の動作原理")
st.caption("高専4年 電気・電子・情報系向け ／ 1.5bit/stageモデル")

with st.sidebar:
    st.header("フォーカス切替")
    focus = st.radio("表示内容", KW)
    st.header("パラメータ")
    vin = st.slider("アナログ入力電圧 Vin [V]", 0.0, 2.0, 1.37, 0.01)
    n = st.slider("ステージ数 N", 3, 6, 4)
    st.number_input("基準電圧 Vref [V]（固定）", value=VREF, disabled=True)
    rows, code, _ = run_adc(vin, n)
    st.metric("出力コード", f"{code:0{n}b}", f"{code}（10進）")

fig = draw_block(n, focus)
st.pyplot(fig)
plt.close(fig)
st.divider()

if focus == KW[0]:
    view_overview(rows, code, n)
elif focus == KW[1]:
    view_mdac(rows, n)
elif focus == KW[2]:
    view_pipe(vin, n)
else:
    view_dec(vin, n)


# ============================================================================
# 電圧の流れ・情報の流れと詳細解説ブロック
# ============================================================================

V_COL, I_COL, G_COL = (
    "#1565c0",
    "#d84315",
    "#b0b8c0",
)  # 電圧=青 / 情報=赤 / 未通過=灰


def _arrow(
    ax, p, q, state, kind, label=None, lpos=(0.0, 0.12), head=True, fs=8.5
):
    """state: future(未通過) / done(通過済) / active(いま流れている)"""
    col = G_COL if state == "future" else (V_COL if kind == "V" else I_COL)
    lw = {"future": 1.2, "done": 2.4, "active": 4.5}[state]
    ax.annotate(
        "",
        xy=q,
        xytext=p,
        arrowprops=dict(
            arrowstyle="-|>" if head else "-",
            lw=lw,
            color=col,
            mutation_scale=14,
            linestyle="--" if state == "future" else "-",
            shrinkA=0,
            shrinkB=0,
        ),
    )
    if label and state != "future":
        ax.text(
            (p[0] + q[0]) / 2 + lpos[0],
            (p[1] + q[1]) / 2 + lpos[1],
            label,
            ha="center",
            va="bottom",
            fontsize=fs,
            color=col,
            fontweight="bold",
        )


def _first(steps):
    first = {}
    for i, s in enumerate(steps):
        for e in s["edges"]:
            first.setdefault(e, i)
    return first


def _state(eid, first, idx):
    if eid not in first or first[eid] > idx:
        return "future"
    return "active" if first[eid] == idx else "done"


def _legend(ax, loc="upper right"):
    ax.plot([], [], c=V_COL, lw=3, label="電圧 (アナログ)")
    ax.plot([], [], c=I_COL, lw=3, label="情報 (デジタル)")
    ax.plot([], [], c=G_COL, lw=1.5, ls="--", label="未到達")
    ax.legend(loc=loc, fontsize=8, ncol=3, frameon=False)


# ---------- ① 全体の流れ：ステップ定義 ----------
def _build_steps(rows, n, vin, code, raw):
    S = []
    S.append(
        dict(
            kind="V",
            edges=["in"],
            boxes=["sh"],
            title="Vin → S/H",
            text=(
                f"アナログ電圧 Vin = **{vin:.3f} V** が S/H"
                " に入り、クロックに合わせて**サンプリング**されます。電圧はキャパシタの電荷として保持されます。"
            ),
        )
    )
    S.append(
        dict(
            kind="V",
            edges=["sh_s1"],
            boxes=["sh", "s1"],
            title="S/H → stage1",
            text=(
                f"保持された **{rows[0]['vin']:.3f} V** が stage1"
                " の入力になります。"
            ),
        )
    )
    for k in range(1, n):
        r = rows[k - 1]
        S.append(
            dict(
                kind="I",
                edges=[f"d{k}"],
                boxes=[f"s{k}"],
                title=f"stage{k}：Sub-ADCが判定 → D{k}",
                text=(
                    f"Sub-ADC が Vin = {r['vin']:.3f} V を しきい値 0.75 V /"
                    " 1.25 V と比較し、"
                    f"**D{k} = {r['d']}**（デジタル情報）を出力。この D{k} は"
                    " ①DACへ ②シフトレジスタへ の2方向に流れます。"
                ),
            )
        )
        S.append(
            dict(
                kind="V",
                edges=[f"v{k}"],
                boxes=[f"s{k}", f"s{k+1}"],
                title=f"stage{k}：DAC減算 → ×2 → stage{k+1}へ",
                text=(
                    f"DAC が D{k} から **{r['vdac']:.3f} V** を作り、Σ で Vin −"
                    f" DAC = {r['vin']:.3f} − {r['vdac']:.3f} = {r['vsub']:.3f}"
                    " V（残差）。MDAC が ×2 して **Vres ="
                    f" {r['vres']:.3f} V** を stage{k+1} へ渡します。"
                ),
            )
        )
    r = rows[n - 1]
    S.append(
        dict(
            kind="I",
            edges=[f"d{n}"],
            boxes=[f"s{n}"],
            title=f"stage{n}（最終stage）：判定のみ",
            text=(
                f"最終stageは Vin = {r['vin']:.3f} V をしきい値 0.5 V / 1.5 V"
                f" で判定して **D{n} = {r['d']}** を出すだけ。"
                "MDACはありません。→ これで全ビットが出そろいました。"
            ),
        )
    )
    S.append(
        dict(
            kind="I",
            edges=[f"r{k}" for k in range(1, n + 1)],
            boxes=[f"r{k}" for k in range(1, n + 1)],
            title="シフトレジスタで時間同期",
            text=(
                "stagekのビットは k クロック目に確定済み。早く確定したビットほど長く待たせ（stagekは"
                " N−k クロック遅延）、全ビットを同じクロックに揃えます。"
            ),
        )
    )
    S.append(
        dict(
            kind="I",
            edges=["dec"],
            boxes=["dec"],
            title="DEC：1bitずらして加算",
            text=(
                "DEC が各Dを重み（2^(N-1-k)）付きで、**1bitずつ重ねて加算**します。"
                f"合計 = **{raw}**"
                + (
                    f"（上限にクリップして {code}）"
                    if raw != code
                    else ""
                )
                + "。"
            ),
        )
    )
    S.append(
        dict(
            kind="I",
            edges=["out"],
            boxes=["dec"],
            title="Dout 出力",
            text=(
                f"最終出力 **Dout = {code:0{n}b}**（10進 {code}）。ここまでが1サンプル分の電圧・情報の旅です。"
            ),
        )
    )
    return S


def draw_flow(n, rows, vin, code, raw, steps, idx):
    first = _first(steps)
    cur = set(steps[idx]["boxes"])
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    W, G, X0 = 2.0, 0.4, 1.9
    xe = X0 + n * (W + G)
    ax.set_xlim(-0.8, xe + 3.4)
    ax.set_ylim(-0.4, 4.6)

    def box(bid, x, y, w, h, txt):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03",
                lw=1.5,
                ec="#345",
                fc="#ffd54f" if bid in cur else "#e3eaf2",
            )
        )
        ax.text(
            x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9
        )

    def edge(eid, p, q, kind, label=None, lpos=(0, 0.12), head=True):
        _arrow(ax, p, q, _state(eid, first, idx), kind, label, lpos, head)

    ax.text(-0.75, 3.15, "入力 Vin", fontsize=11)
    box("sh", 0.2, 2.4, 1.2, 1.2, "S/H")
    edge("in", (-0.3, 3.0), (0.2, 3.0), "V")
    edge(
        "sh_s1",
        (1.4, 3.0),
        (X0, 3.0),
        "V",
        f"{rows[0]['vin']:.2f}V",
        (0, 0.1),
    )
    for k in range(1, n + 1):
        r = rows[k - 1]
        x = X0 + (k - 1) * (W + G)
        last = k == n
        box(
            f"s{k}",
            x,
            2.4,
            W,
            1.2,
            (
                f"stage{k}\nサブADC+DAC\n->減算->2倍"
                if not last
                else f"stage{k}\nサブADCのみ"
            ),
        )
        if not last:
            edge(
                f"v{k}",
                (x + W, 3.0),
                (x + W + G, 3.0),
                "V",
                f"{r['vres']:.2f}V",
                (0, 0.1),
            )
        edge(
            f"d{k}",
            (x + W / 2, 2.4),
            (x + W / 2, 1.5),
            "I",
            f"D{k}={r['d']}",
            (0.45, -0.1),
        )
        box(
            f"r{k}",
            x + 0.1,
            0.7,
            W - 0.2,
            0.8,
            f"シフトレジスタ\n({n - k} クロック)",
        )
        edge(f"r{k}", (x + W / 2, 0.7), (x + W / 2, 0.3), "I", head=False)
    edge(
        "dec",
        (X0 + W / 2, 0.3),
        (xe + 0.3, 0.3),
        "I",
        f"合計={raw}",
        (0, 0.06),
    )
    box("dec", xe + 0.3, -0.1, 1.9, 1.2, "DEC\n(シフト&加算)")
    edge(
        "out",
        (xe + 2.2, 0.5),
        (xe + 2.7, 0.5),
        "I",
        f"Dout={code:0{n}b}",
        (0.45, 0.1),
    )
    _legend(ax)
    ax.set_title(
        f"信号の流れ: ステップ {idx + 1} / {len(steps)}", fontsize=11
    )
    return fig


# ---------- ② stage1の中身（拡大） ----------
def draw_stage(k, r, sub):
    E = [
        (
            "in",
            (0.2, 2.0),
            (1.0, 2.0),
            "V",
            f"Vin={r['vin']:.3f}V",
            (0.2, 0.1),
            False,
            0,
        ),
        ("br", (1.0, 2.0), (1.0, 3.7), "V", None, (0, 0), False, 0),
        ("sum_in", (1.0, 2.0), (5.4, 2.0), "V", "Vin (+)", (0, 0.1), True, 0),
        ("cmp", (1.0, 3.7), (1.6, 3.7), "V", None, (0, 0), True, 1),
        (
            "d_out",
            (2.7, 4.2),
            (2.7, 4.9),
            "I",
            f"D{k}={r['d']} -> シフトレジスタへ",
            (1.3, -0.1),
            True,
            1,
        ),
        (
            "d_dac",
            (3.8, 3.7),
            (5.0, 3.7),
            "I",
            f"D={r['d']}",
            (0, 0.1),
            True,
            2,
        ),
        (
            "dac_out",
            (5.8, 3.2),
            (5.8, 2.4),
            "V",
            f"DAC={r['vdac']:.2f}V (-)",
            (1.0, -0.1),
            True,
            2,
        ),
        (
            "sum_amp",
            (6.15, 2.0),
            (7.2, 2.0),
            "V",
            f"{r['vsub']:.3f}V",
            (0, 0.1),
            True,
            3,
        ),
        (
            "amp_out",
            (8.4, 2.0),
            (9.7, 2.0),
            "V",
            f"Vres={r['vres']:.3f}V",
            (0, 0.1),
            True,
            4,
        ),
    ]
    first = {e[0]: e[7] for e in E}
    act = {1: "sadc", 2: "dac", 3: "sum", 4: "amp"}.get(sub)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.8, 5.3)

    def fc(b):
        return "#ffd54f" if act == b else "#e3eaf2"

    ax.add_patch(
        FancyBboxPatch(
            (1.6, 3.2),
            2.2,
            1.0,
            boxstyle="round,pad=0.03",
            fc=fc("sadc"),
            ec="#345",
            lw=1.5,
        )
    )
    ax.text(
        2.7,
        3.7,
        "サブADC\n(比較器 2個)",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.add_patch(
        FancyBboxPatch(
            (5.0, 3.2),
            1.6,
            1.0,
            boxstyle="round,pad=0.03",
            fc=fc("dac"),
            ec="#345",
            lw=1.5,
        )
    )
    ax.text(5.8, 3.7, "DAC", ha="center", va="center", fontsize=10)
    ax.add_patch(Circle((5.8, 2.0), 0.35, fc=fc("sum"), ec="#345", lw=1.5))
    ax.text(5.8, 2.0, "Σ", ha="center", va="center", fontsize=13)
    ax.add_patch(
        Polygon(
            [(7.2, 1.3), (7.2, 2.7), (8.4, 2.0)],
            fc=fc("amp"),
            ec="#345",
            lw=1.5,
        )
    )
    ax.text(7.55, 2.0, "2倍増幅", ha="center", va="center", fontsize=11)
    for eid, p, q, kind, lab, lp, head, st_no in E:
        state = (
            "future"
            if first[eid] > sub
            else ("active" if first[eid] == sub else "done")
        )
        _arrow(ax, p, q, state, kind, lab, lp, head, fs=9)
    _legend(ax, "lower right")
    ax.set_title(f"stage {k} の内部構造 (MDACstage)", fontsize=11)
    return fig


# ---------- ③ スイッチトキャパシタMDAC ----------
def draw_sc(phase):
    p1, p2 = phase == 1, phase == 2
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axis("off")
    ax.set_xlim(-0.3, 9.2)
    ax.set_ylim(0.9, 5.4)

    def wire(pts, active):
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            c=V_COL if active else G_COL,
            lw=3.5 if active else 1.5,
            solid_capstyle="round",
        )

    ax.text(-0.25, 3.2, "入力 Vin", fontsize=11)
    ax.text(-0.25, 1.8, "DAC出力 Vdac", fontsize=11)
    wire([(0.4, 3.2), (1.0, 3.2)], p1)
    wire([(1.8, 3.2), (2.4, 3.2), (2.4, 2.5)], p1)
    wire([(0.4, 1.8), (1.0, 1.8)], p2)
    wire([(1.8, 1.8), (2.4, 1.8), (2.4, 2.5)], p2)
    for y, name, on in (3.2, "スイッチ S1 (φ1)", p1), (
        1.8,
        "スイッチ S2 (φ2)",
        p2,
    ):
        ax.add_patch(
            FancyBboxPatch(
                (1.0, y - 0.35),
                0.8,
                0.7,
                boxstyle="round,pad=0.02",
                fc="#a5d6a7" if on else "#eceff1",
                ec="#345",
            )
        )
        ax.text(
            1.4,
            y,
            name + ("\nON (導通)" if on else "\nOFF (遮断)"),
            ha="center",
            va="center",
            fontsize=7,
        )
    wire([(2.4, 2.5), (3.0, 2.5)], True)
    for x in 3.0, 3.3:
        ax.plot([x, x], [2.1, 2.9], c="#345", lw=3)
    wire([(3.3, 2.5), (4.2, 2.5), (4.2, 2.9), (4.6, 2.9)], True)
    ax.add_patch(
        Polygon(
            [(4.6, 1.7), (4.6, 3.3), (6.6, 2.5)],
            fc="#e3eaf2",
            ec="#345",
            lw=1.5,
        )
    )
    ax.text(4.72, 2.83, "-", fontsize=12)
    ax.text(4.72, 2.05, "+", fontsize=12)
    ax.plot([4.6, 4.2], [2.1, 2.1], c="#345", lw=1.5)
    ax.text(3.75, 2.0, "接地 (GND)", fontsize=8)
    wire([(4.2, 2.9), (4.2, 4.0)], p2)
    for y in 4.0, 4.25:
        ax.plot([3.9, 4.5], [y, y], c="#345", lw=3)
    wire([(4.2, 4.25), (4.2, 4.9), (7.4, 4.9), (7.4, 2.5)], p2)
    wire([(6.6, 2.5), (8.6, 2.5)], p2)
    ax.text(8.7, 2.45, "出力 Vout", fontsize=11)
    ax.text(3.15, 3.05, "Cs", ha="center", fontsize=10)
    ax.text(4.75, 4.05, "Cf", fontsize=10)
    ax.text(3.0, 3.5, "X: 仮想接地", fontsize=8)
    if p1:
        ax.text(5.0, 4.4, "Cf はリセット（放電状態）", fontsize=8)
    ax.set_title(
        "スイッチトキャパシタ MDAC  -  "
        + ("φ1: サンプリング期間" if p1 else "φ2: 増幅期間"),
        fontsize=11,
    )
    return fig


# ---------- ④ 2相クロックとパイプライン ----------
def draw_timing(n, ns=3):
    names, cols = ["A", "B", "C"], ["#90caf9", "#a5d6a7", "#ffcc80"]
    T = n + 2 * (ns - 1) + 1
    rows_n = n + 2
    fig, ax = plt.subplots(figsize=(12, 0.55 * rows_n + 1.2))
    for h in range(T):
        for i, on in enumerate((h % 2 == 0, h % 2 == 1)):
            if on:
                ax.add_patch(
                    plt.Rectangle(
                        (h, rows_n - 1 - i + 0.15),
                        1,
                        0.7,
                        fc="#cfd8dc",
                        ec="#345",
                    )
                )
    for k in range(1, n + 1):
        y = rows_n - 1 - (k + 1)
        for s in range(ns):
            for j, tag in (k - 1 + 2 * s, "サンプル"), (k + 2 * s, "増幅"):
                ax.add_patch(
                    plt.Rectangle(
                        (j, y + 0.1),
                        1,
                        0.8,
                        fc=cols[s],
                        ec="#345",
                        hatch="" if tag == "サンプル" else "//",
                    )
                )
                ax.text(
                    j + 0.5,
                    y + 0.5,
                    f"サンプル{names[s]}\n{tag}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    ax.set_xlim(0, T)
    ax.set_ylim(0, rows_n)
    ax.set_yticks([rows_n - 0.5 - i for i in range(rows_n)])
    ax.set_yticklabels(
        ["φ1", "φ2"] + [f"stage{k}" for k in range(1, n + 1)]
    )
    ax.set_xticks(range(T + 1))
    ax.set_xlabel("半クロック (フェーズ)")
    ax.set_title(
        "2相動作: stage k増幅中にstage k+1がサンプリング", fontsize=11
    )
    return fig


# ============================ 追加UI ============================
st.divider()
st.header("🔍 信号の流れと詳細メカニズム")
st.caption(
    "矢印を1本ずつたどりながら、**電圧（アナログ）＝青**と**情報（デジタル）＝赤**が"
    "どの順番で流れるかを確認できます。"
)

_rw, _cd, _raw = run_adc(vin, n)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧭 全体の流れ（矢印）",
    "🔬 stageの中身",
    "⚡ MDACの電気的動作",
    "⏱ 2相クロックとパイプライン",
    "📝 3大要素の詳細解説",
])

with tab1:
    steps = _build_steps(_rw, n, vin, _cd, _raw)
    i = (
        st.slider(
            "ステップ（矢印を1本ずつたどる）",
            1,
            len(steps),
            1,
            key="flow_step",
        )
        - 1
    )
    figf = draw_flow(n, _rw, vin, _cd, _raw, steps, i)
    st.pyplot(figf)
    plt.close(figf)
    icon = (
        "🔵 電圧（アナログ）の流れ"
        if steps[i]["kind"] == "V"
        else "🔴 情報（デジタル）の流れ"
    )
    st.info(f"**ステップ {i + 1}：{steps[i]['title']}** （{icon}）\n\n{steps[i]['text']}")
    with st.expander("全ステップの一覧（電圧と情報の遷移順）"):
        for j, s in enumerate(steps):
            st.markdown(
                f"{'🔵' if s['kind'] == 'V' else '🔴'} **{j + 1}. {s['title']}**"
            )
    st.caption(
        "ポイント：電圧（青）は左から右へ1stageずつ受け渡され、"
        "情報（赤）は各stageから下へ取り出されてシフトレジスタ → DEC → Dout へ集まります。"
    )

with tab2:
    c1, c2 = st.columns(2)
    kk = c1.slider("拡大するステージ", 1, n - 1, 1, key="zoom_k")
    subs = [
        "① Vinが2方向へ分岐",
        "② Sub-ADC判定",
        "③ DACが電圧を生成",
        "④ Σで減算",
        "⑤ ×2アンプで残差増幅",
    ]
    sname = c2.radio("流れの段階", subs, key="zoom_sub")
    si2 = subs.index(sname)
    r = _rw[kk - 1]
    figs = draw_stage(kk, r, si2)
    st.pyplot(figs)
    plt.close(figs)
    tx = [
        (
            f"入力 **{r['vin']:.3f} V** が、①Sub-ADC"
            " へ向かう枝と ②Σ の＋入力へ向かう枝に**分岐**します（電圧は同じ）。"
        ),
        (
            "Sub-ADC の2個の比較器が 0.75 V / 1.25 V と比較 → **D"
            f"{kk} = {r['d']}**。ここからデジタル情報（赤）が生まれます。"
        ),
        (
            f"D{kk} = {r['d']} に応じて DAC が **{r['vdac']:.3f} V**（= D ×"
            " Vref/4）を出力。デジタル→アナログへ戻る点です。"
        ),
        (
            f"Σ で **{r['vin']:.3f} − {r['vdac']:.3f} = {r['vsub']:.3f}"
            " V**。粗い判定で表せた分を引き、細かい残りだけ（残差）にします。"
        ),
        (
            f"アンプが×2 → **Vres = {r['vres']:.3f} V**。範囲が 0～Vref"
            " に戻るので、次stageが同じ回路で処理できます。"
        ),
    ]
    st.info(tx[si2])
    st.caption(
        "1つのstageの中で、電圧は「分岐 → 引き算 → 2倍」と流れ、情報は「比較 → D →"
        " DAC／シフトレジスタ」と流れます。"
    )

with tab3:
    ph = st.radio(
        "クロック相",
        ["φ1：サンプル期間", "φ2：増幅期間"],
        horizontal=True,
        key="sc_phase",
    )
    pn = 1 if ph.startswith("φ1") else 2
    figc = draw_sc(pn)
    st.pyplot(figc)
    plt.close(figc)
    r1 = _rw[0]
    cs, cf = 2.0, 1.0  # Cs/Cf = 2 → 利得2
    if pn == 1:
        st.info(
            f"**φ1**：スイッチS1がON。Cs の下側に Vin = {r1['vin']:.3f} V"
            " が加わり、Cs に電荷 "
            f"**Q = Cs × Vin = {cs * r1['vin']:.3f}**（Cf=1"
            " に規格化）が蓄えられます。Cf は放電（リセット）。"
            "この時点で入力電圧が『電荷』として保持されます（＝S/H動作）。"
        )
    else:
        st.info(
            f"**φ2**：スイッチS1がOFF、S2がON。Cs の下側が Vdac ="
            f" {r1['vdac']:.3f} V に切り替わります。"
            f"Cs に残せる電荷は Cs × Vdac = {cs * r1['vdac']:.3f}。差の **"
            f"{cs * (r1['vin'] - r1['vdac']):.3f}** が"
            f"オペアンプを通って Cf に移り、**Vout = {r1['vres']:.3f} V**"
            " になります。"
        )
    st.latex(
        r"C_s V_{in} = C_s V_{dac} + C_f V_{out} \implies V_{out} = \frac{C_s}{C_f} (V_{in} - V_{dac})"
    )
    st.markdown(
        r"""
        **原理のポイント（電荷保存の法則）**：
        1. **$\phi_1$（サンプリング相）**: スイッチにより $C_s$ に電荷 $Q_1 = C_s V_{in}$ が蓄積されます。
        2. **$\phi_2$（増幅相）**: $C_s$ の端子が $V_{dac}$ に接続され、溢れた電荷がオペアンプの帰還キャパシタ $C_f$ へ移動します（$Q_2 = C_s V_{dac} + C_f V_{out}$）。
        3. 電荷保存則 $Q_1 = Q_2$ より、$C_s / C_f = 2$ となるよう容量比を設計することで **$V_{out} = 2(V_{in} - V_{dac})$** の演算回路（MDAC）を精度よく実現できます。
        """
    )

with tab4:
    figt = draw_timing(n)
    st.pyplot(figt)
    plt.close(figt)
    st.markdown(
        """
        ##### 2相ノンオーバーラップクロック ($\phi_1, \phi_2$) による並列動作
        - パイプライン型ADCでは、奇数ステージと偶数ステージが**互い違い（インターリーブ）のクロックフェーズ**で動作します。
        - **$\phi_1$ フェーズ**: 奇数ステージがサンプリングを行い、同時に偶数ステージが増幅・出力（次ステージへ伝達）を行います。
        - **$\phi_2$ フェーズ**: 奇数ステージが増幅・出力を行い、同時に偶数ステージがサンプリングを行います。
        - 前ステージが出力を確定させている間に次ステージがその値を読み取るため、クロックごとに連続してデータをパイプライン処理できます。
        """
    )

with tab5:
    st.markdown(
        """
        ### 💡 パイプライン型ADCの3大要素解説

        #### 1. 残差・残差増幅器（MDAC: Multiplying DAC）
        - **役割**: サブADCでの判定結果（$D$）からアナログ電圧（$V_{dac}$）を出力し、入力 $V_{in}$ との差分（残差）を2倍に増幅して次ステージへ転送します。
        - **メリット**: 残差を2倍に伸ばして次ステージへ引き渡すことで、すべてのステージで**全く同じ入力レンジ（$0 \sim V_{ref}$）と判定閾値（$0.75\text{V}, 1.25\text{V}$）を再利用**できます。

        #### 2. パイプライン処理とサンプル＆ホールド（S/H）
        - **役割**: 各ステージがサンプルホールド機能（スイッチトキャパシタ）を備え、前段から受け取った電圧を保持しながら独立して動作します。
        - **メリット**: 分解能 $N$（ステージ数）を増やしても、定常状態での**スループット（1秒あたりのサンプル出力数）は低下しません**。1クロック毎に新しい1サンプルの変換結果が出力されます（※ただし全体出力までに $N$ クロックのレイテンシが発生します）。

        #### 3. 冗長構成とデジタル誤差補正（DEC: Digital Error Correction）
        - **役割**: 1.5bit/stageの冗長判定（$D=0, 1, 2$）を行い、隣り合うステージの判定ビットを1bit重なるようにシフト＆加算（Shift & Add）します。また、各ステージの確定時間差をシフトレジスタでアライメントします。
        - **メリット**: 比較器（コンパレータ）のオフセットやノイズにより中間ステージで誤判定が発生しても、**次ステージの残差処理とDEC加算によって判定ミスが完全に打ち消されます**。これにより、高精度・高コストな比較器を使用せずに高速動作を実現できます。
        """
    )
