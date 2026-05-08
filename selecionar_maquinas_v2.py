"""
Selecção Inteligente de Máquinas v2 — Alibaba Cluster Trace v2018
===================================================================
Filosofia: diversidade > qualidade máxima.
Amostra TODOS os perfis comportamentais (idle, bursty, stable, high-load).

    python selecionar_maquinas_v2.py

Dependências: pip install pandas tqdm numpy
"""

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc: print(f"  {desc}...", flush=True)
        return iterable


# ══════════════════════════════════════════════════════════
#  CONFIGURAÇÃO — editar aqui
# ══════════════════════════════════════════════════════════

# ── Quantas máquinas por dataset ──
N_TOTAL = 100                   # máquinas a seleccionar

# ── Datasets a gerar (True/False para activar/desactivar) ──
GENERATE_FORECASTING = True     # previsão temporal de métricas
GENERATE_ANOMALY     = False     # deteção de anomalias
GENERATE_GENERAL     = False     # dataset geral (todas as 100)

# ── Composição por perfil para cada dataset ──
# Percentagens por perfil comportamental (devem somar ~100)
# Os perfis são: stable_high, stable_mid, stable_low,
#                bursty, idle_with_spikes, trending, erratic

FORECASTING_MIX = {
    # Precisa de séries regulares com padrões aprendíveis
    "stable_high":      15,   # carga consistente alta — fácil de prever
    "stable_mid":       15,   # carga consistente média
    "stable_low":       10,   # carga consistente baixa
    "bursty":           25,   # picos frequentes — o que queremos prever
    "idle_with_spikes": 15,   # contraste idle/activo
    "trending":         10,   # tendências crescentes/decrescentes
    "erratic":          10,   # difíceis de prever — testa limites do modelo
}

ANOMALY_MIX = {
    # Precisa de contraste normal vs anómalo
    "stable_high":      10,
    "stable_mid":       10,
    "stable_low":       10,   # baseline "normal"
    "bursty":           20,   # anomalias frequentes
    "idle_with_spikes": 25,   # spikes raros = anomalias claras
    "trending":         10,   # drift pode ser anomalia lenta
    "erratic":          15,   # comportamento imprevisível
}

# ── Critérios mínimos de qualidade (eliminatórios) ──
MIN_POINTS = 200                # mínimo absoluto de registos
MIN_DURATION_S = 3600 * 12      # mínimo 12 horas de dados

# ── Paths ──
OUTPUT_DIR = Path("alibaba_processado")
CHUNK_SIZE = 300_000

DATA_CANDIDATES = [
    Path("alibaba_data"), Path("data"), Path("."),
    Path("datasets_cloud/3_Alibaba_v2018"),
    Path("datasets_cloud/3_Alibaba_v2018/alibaba_data"),
]

# ── Schema machine_usage (9 colunas) ──
SCHEMA = [
    "machine_id", "time_stamp",
    "cpu_util_percent", "mem_util_percent",
    "mem_gps", "mkpi",
    "net_in", "net_out", "disk_io_percent",
]


def find_csv(name):
    for d in DATA_CANDIDATES:
        if not d.exists(): continue
        for p in [f"{name}.csv", f"**/{name}*.csv"]:
            m = list(d.glob(p))
            if m: return m[0]
    return None


# ══════════════════════════════════════════════════════════
#  PASS 1: PERFILAR TODAS AS MÁQUINAS (streaming)
# ══════════════════════════════════════════════════════════

def profile_all_machines(csv_path):
    """
    Scan completo com features ricas:
    - Básicas: n_points, duration, médias, stds
    - Burstiness: spike_count, p95/p50 ratio
    - Estabilidade: coefficient of variation
    - Gaps: max_gap, n_long_gaps
    - Distribuição de carga: % tempo em idle/low/mid/high
    """
    print(f"\n{'='*62}")
    print(f" PASS 1: Perfilar todas as máquinas (features ricas)")
    print(f"{'='*62}")
    print(f"  Ficheiro: {csv_path.name} ({csv_path.stat().st_size/1024/1024/1024:.2f} GB)")

    # Verificar formato
    with open(csv_path, "r") as f:
        for i in range(2):
            line = f.readline().rstrip()
            fields = line.split(",")
            print(f"  Linha {i}: [{len(fields)} campos] {line[:100]}")

    # Acumuladores por máquina
    # Para features ricas, guardamos histogramas e percentis em vez de só soma/soma²
    accum = defaultdict(lambda: {
        "count": 0,
        "ts_min": float("inf"), "ts_max": float("-inf"),
        # CPU: guardar todos os valores para percentis (em buckets para poupar RAM)
        "cpu_hist": np.zeros(101, dtype=np.int64),  # histogram [0..100]
        "cpu_sum": 0.0, "cpu_sq_sum": 0.0, "cpu_valid": 0,
        # MEM
        "mem_hist": np.zeros(101, dtype=np.int64),
        "mem_sum": 0.0, "mem_sq_sum": 0.0, "mem_valid": 0,
        # NET, DISK
        "net_in_sum": 0.0, "net_out_sum": 0.0, "disk_sum": 0.0,
        "net_valid": 0, "disk_valid": 0,
        # MKPI (cache misses)
        "mkpi_sum": 0.0, "mkpi_valid": 0,
        # Gaps
        "ts_prev": None, "max_gap": 0.0,
        "gap_sum": 0.0, "gap_count": 0,
        "n_long_gaps": 0,  # gaps > 1 hora
        # Spikes: transições de <30% para >70% em CPU
        "prev_cpu_low": False, "spike_count": 0,
        # Tendência: acumular para regressão linear simples
        "ts_cpu_pairs_n": 0,
        "ts_sum": 0.0, "ts_sq_sum": 0.0,
        "ts_cpu_sum": 0.0,
    })

    total_lines = 0
    reader = pd.read_csv(
        csv_path, names=SCHEMA, header=None,
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for chunk in tqdm(reader, desc="  Scanning"):
        total_lines += len(chunk)

        chunk["time_stamp"] = pd.to_numeric(chunk["time_stamp"], errors="coerce")
        chunk["cpu_util_percent"] = pd.to_numeric(chunk["cpu_util_percent"], errors="coerce")
        chunk["mem_util_percent"] = pd.to_numeric(chunk["mem_util_percent"], errors="coerce")
        chunk["net_in"] = pd.to_numeric(chunk["net_in"], errors="coerce")
        chunk["net_out"] = pd.to_numeric(chunk["net_out"], errors="coerce")
        chunk["disk_io_percent"] = pd.to_numeric(chunk["disk_io_percent"], errors="coerce")
        chunk["mkpi"] = pd.to_numeric(chunk["mkpi"], errors="coerce")

        for mid, grp in chunk.groupby("machine_id"):
            a = accum[mid]
            a["count"] += len(grp)

            ts = grp["time_stamp"].dropna().values
            cpu = grp["cpu_util_percent"].dropna().values
            mem = grp["mem_util_percent"].dropna().values
            net_i = grp["net_in"].dropna().values
            net_o = grp["net_out"].dropna().values
            disk = grp["disk_io_percent"].dropna().values
            mkpi = grp["mkpi"].dropna().values

            # Filtrar ranges
            cpu = cpu[(cpu >= 0) & (cpu <= 100)]
            mem = mem[(mem >= 0) & (mem <= 100)]
            disk = disk[(disk >= 0) & (disk <= 100)]

            # ── Timestamps ──
            if len(ts) > 0:
                a["ts_min"] = min(a["ts_min"], float(ts.min()))
                a["ts_max"] = max(a["ts_max"], float(ts.max()))
                ts_sorted = np.sort(ts)

                if len(ts_sorted) > 1:
                    diffs = np.diff(ts_sorted)
                    a["max_gap"] = max(a["max_gap"], float(diffs.max()))
                    a["gap_sum"] += float(diffs.sum())
                    a["gap_count"] += len(diffs)
                    a["n_long_gaps"] += int((diffs > 3600).sum())

                if a["ts_prev"] is not None:
                    ig = float(ts_sorted[0]) - a["ts_prev"]
                    if ig > 0:
                        a["max_gap"] = max(a["max_gap"], ig)
                        a["gap_sum"] += ig
                        a["gap_count"] += 1
                        if ig > 3600:
                            a["n_long_gaps"] += 1
                a["ts_prev"] = float(ts_sorted[-1])

            # ── CPU ──
            if len(cpu) > 0:
                a["cpu_sum"] += float(cpu.sum())
                a["cpu_sq_sum"] += float((cpu ** 2).sum())
                a["cpu_valid"] += len(cpu)
                # Histogram
                cpu_int = np.clip(cpu, 0, 100).astype(int)
                for v in cpu_int:
                    a["cpu_hist"][v] += 1
                # Spikes (transição low→high)
                for v in cpu:
                    is_low = v < 30
                    if a["prev_cpu_low"] and v > 70:
                        a["spike_count"] += 1
                    a["prev_cpu_low"] = is_low
                # Trend (regressão linear CPU ~ tempo)
                if len(ts) == len(grp) and len(cpu) == len(grp):
                    valid_mask = grp["time_stamp"].notna().values & grp["cpu_util_percent"].notna().values
                    t_vals = grp["time_stamp"].values[valid_mask].astype(float)
                    c_vals = grp["cpu_util_percent"].values[valid_mask].astype(float)
                    c_vals = c_vals[(c_vals >= 0) & (c_vals <= 100)]
                    n_valid = min(len(t_vals), len(c_vals))
                    if n_valid > 0:
                        t_vals = t_vals[:n_valid]
                        c_vals = c_vals[:n_valid]
                        a["ts_sum"] += float(t_vals.sum())
                        a["ts_sq_sum"] += float((t_vals ** 2).sum())
                        a["ts_cpu_sum"] += float((t_vals * c_vals).sum())
                        a["ts_cpu_pairs_n"] += n_valid

            # ── MEM ──
            if len(mem) > 0:
                a["mem_sum"] += float(mem.sum())
                a["mem_sq_sum"] += float((mem ** 2).sum())
                a["mem_valid"] += len(mem)
                mem_int = np.clip(mem, 0, 100).astype(int)
                for v in mem_int:
                    a["mem_hist"][v] += 1

            # ── NET, DISK, MKPI ──
            if len(net_i) > 0:
                a["net_in_sum"] += float(net_i.sum())
                a["net_valid"] += len(net_i)
            if len(net_o) > 0:
                a["net_out_sum"] += float(net_o.sum())
            if len(disk) > 0:
                a["disk_sum"] += float(disk.sum())
                a["disk_valid"] += len(disk)
            if len(mkpi) > 0:
                a["mkpi_sum"] += float(mkpi.sum())
                a["mkpi_valid"] += len(mkpi)

    print(f"\n  Total linhas: {total_lines:,}")
    print(f"  Máquinas: {len(accum):,}")

    # ── Converter para DataFrame com features ricas ──
    rows = []
    for mid, a in accum.items():
        nc = a["cpu_valid"]
        nm = a["mem_valid"]
        dur = a["ts_max"] - a["ts_min"] if a["ts_max"] > a["ts_min"] else 0

        # Médias e stds
        cpu_mean = a["cpu_sum"] / nc if nc > 0 else 0
        cpu_var = (a["cpu_sq_sum"] / nc - cpu_mean**2) if nc > 1 else 0
        cpu_std = max(0, cpu_var) ** 0.5
        mem_mean = a["mem_sum"] / nm if nm > 0 else 0
        mem_var = (a["mem_sq_sum"] / nm - mem_mean**2) if nm > 1 else 0
        mem_std = max(0, mem_var) ** 0.5

        # Percentis de CPU via histograma
        h = a["cpu_hist"]
        total_h = h.sum()
        if total_h > 0:
            cumsum = np.cumsum(h) / total_h
            cpu_p5  = np.searchsorted(cumsum, 0.05)
            cpu_p25 = np.searchsorted(cumsum, 0.25)
            cpu_p50 = np.searchsorted(cumsum, 0.50)
            cpu_p75 = np.searchsorted(cumsum, 0.75)
            cpu_p95 = np.searchsorted(cumsum, 0.95)
            cpu_p99 = np.searchsorted(cumsum, 0.99)
        else:
            cpu_p5 = cpu_p25 = cpu_p50 = cpu_p75 = cpu_p95 = cpu_p99 = 0

        # Distribuição de carga (% tempo em cada faixa)
        if total_h > 0:
            pct_idle = float(h[0:5].sum()) / total_h * 100       # [0, 5)
            pct_low  = float(h[5:30].sum()) / total_h * 100      # [5, 30)
            pct_mid  = float(h[30:70].sum()) / total_h * 100     # [30, 70)
            pct_high = float(h[70:101].sum()) / total_h * 100    # [70, 100]
        else:
            pct_idle = pct_low = pct_mid = pct_high = 0

        # Burstiness: p95/p50 ratio (alto = bursty)
        burstiness = cpu_p95 / max(cpu_p50, 1)

        # Coefficient of variation
        cpu_cv = cpu_std / max(cpu_mean, 0.01)

        # Trend: slope da regressão linear CPU ~ tempo
        n_pairs = a["ts_cpu_pairs_n"]
        if n_pairs > 10:
            t_mean = a["ts_sum"] / n_pairs
            c_mean = a["cpu_sum"] / nc if nc > 0 else 0
            cov_tc = a["ts_cpu_sum"] / n_pairs - t_mean * c_mean
            var_t = a["ts_sq_sum"] / n_pairs - t_mean**2
            slope = cov_tc / max(var_t, 1e-10)
            # Normalizar: slope em %/hora
            trend_per_hour = slope * 3600
        else:
            trend_per_hour = 0

        avg_interval = a["gap_sum"] / a["gap_count"] if a["gap_count"] > 0 else 0

        rows.append({
            "machine_id": mid,
            # Básicas
            "n_points": a["count"],
            "duration_s": round(dur),
            "duration_days": round(dur / 86400, 2),
            "avg_interval_s": round(avg_interval),
            # CPU
            "cpu_mean": round(cpu_mean, 2),
            "cpu_std": round(cpu_std, 2),
            "cpu_cv": round(cpu_cv, 3),
            "cpu_p5": cpu_p5, "cpu_p25": cpu_p25, "cpu_p50": cpu_p50,
            "cpu_p75": cpu_p75, "cpu_p95": cpu_p95, "cpu_p99": cpu_p99,
            # MEM
            "mem_mean": round(mem_mean, 2),
            "mem_std": round(mem_std, 2),
            # NET, DISK, MKPI
            "net_in_mean": round(a["net_in_sum"] / max(a["net_valid"], 1), 2),
            "net_out_mean": round(a["net_out_sum"] / max(a["net_valid"], 1), 2),
            "disk_mean": round(a["disk_sum"] / max(a["disk_valid"], 1), 2),
            "mkpi_mean": round(a["mkpi_sum"] / max(a["mkpi_valid"], 1), 2),
            # Distribuição de carga
            "pct_idle": round(pct_idle, 1),
            "pct_low": round(pct_low, 1),
            "pct_mid": round(pct_mid, 1),
            "pct_high": round(pct_high, 1),
            # Comportamento
            "burstiness": round(burstiness, 2),
            "spike_count": a["spike_count"],
            "trend_per_hour": round(trend_per_hour, 4),
            # Gaps
            "max_gap_hours": round(a["max_gap"] / 3600, 1),
            "n_long_gaps": a["n_long_gaps"],
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
#  PASS 2: CLASSIFICAR PERFIS COMPORTAMENTAIS
# ══════════════════════════════════════════════════════════

def classify_profiles(prof):
    """
    Atribui um perfil comportamental a cada máquina.
    Não é filtragem — é categorização. Nenhuma máquina é descartada.
    """
    prof = prof.copy()
    prof["profile"] = "unclassified"

    # Ordem importa: regras mais específicas primeiro
    # Cada máquina recebe o PRIMEIRO perfil que match

    # ── stable_high: carga alta e consistente ──
    mask_sh = (prof["cpu_mean"] >= 50) & (prof["cpu_cv"] < 0.5)
    prof.loc[mask_sh & (prof["profile"] == "unclassified"), "profile"] = "stable_high"

    # ── stable_mid: carga média e consistente ──
    mask_sm = (prof["cpu_mean"].between(15, 50)) & (prof["cpu_cv"] < 0.6)
    prof.loc[mask_sm & (prof["profile"] == "unclassified"), "profile"] = "stable_mid"

    # ── stable_low: carga baixa mas presente (não idle) ──
    mask_sl = (prof["cpu_mean"].between(5, 15)) & (prof["cpu_cv"] < 0.8)
    prof.loc[mask_sl & (prof["profile"] == "unclassified"), "profile"] = "stable_low"

    # ── bursty: alta variação com picos frequentes ──
    mask_b = (prof["burstiness"] >= 3) | (prof["spike_count"] >= 10) | (prof["cpu_cv"] >= 1.0)
    prof.loc[mask_b & (prof["profile"] == "unclassified"), "profile"] = "bursty"

    # ── idle_with_spikes: maioritariamente idle mas com actividade ocasional ──
    mask_is = (prof["pct_idle"] >= 60) & (prof["spike_count"] >= 1)
    prof.loc[mask_is & (prof["profile"] == "unclassified"), "profile"] = "idle_with_spikes"

    # ── trending: slope significativo (>0.5% por hora) ──
    mask_t = (abs(prof["trend_per_hour"]) > 0.5)
    prof.loc[mask_t & (prof["profile"] == "unclassified"), "profile"] = "trending"

    # ── erratic: alta variação sem padrão claro ──
    mask_e = (prof["cpu_cv"] >= 0.7)
    prof.loc[mask_e & (prof["profile"] == "unclassified"), "profile"] = "erratic"

    # ── restantes: idle puro ──
    prof.loc[prof["profile"] == "unclassified", "profile"] = "idle_pure"

    return prof


# ══════════════════════════════════════════════════════════
#  PASS 3: AMOSTRAGEM ESTRATIFICADA
# ══════════════════════════════════════════════════════════

def stratified_sample(prof, mix_config, n_total, dataset_name):
    """
    Amostra N máquinas respeitando a composição por perfil.
    Se um perfil não tem máquinas suficientes, redistribui para outros.
    Dentro de cada perfil, selecciona as de melhor qualidade (mais pontos, menos gaps).
    """
    print(f"\n  Sampling para '{dataset_name}' ({n_total} máquinas):")

    # Filtro mínimo de qualidade
    eligible = prof[
        (prof["n_points"] >= MIN_POINTS) &
        (prof["duration_s"] >= MIN_DURATION_S)
    ].copy()

    print(f"    Elegíveis (qualidade mínima): {len(eligible)} / {len(prof)}")

    # Score de qualidade dentro de cada perfil (para desempate)
    eligible["quality"] = (
        np.log1p(eligible["n_points"]) * 2 +
        eligible["duration_days"] * 1 -
        eligible["n_long_gaps"] * 0.5
    )

    # Calcular quotas por perfil
    selected = []
    remaining_quota = 0

    # Primeiro: atribuir quotas baseadas no mix
    allocations = {}
    for profile, pct in mix_config.items():
        pool = eligible[eligible["profile"] == profile]
        desired = max(1, round(n_total * pct / 100))
        actual = min(desired, len(pool))
        allocations[profile] = {"desired": desired, "actual": actual, "pool": len(pool)}
        remaining_quota += desired - actual

        if actual > 0:
            picks = pool.nlargest(actual, "quality")
            selected.append(picks)

    # Redistribuir quotas não preenchidas para perfis com excesso
    if remaining_quota > 0:
        already_selected_ids = set()
        if selected:
            already_selected_ids = set(pd.concat(selected)["machine_id"])

        remaining_pool = eligible[~eligible["machine_id"].isin(already_selected_ids)]
        if len(remaining_pool) > 0:
            extra = remaining_pool.nlargest(min(remaining_quota, len(remaining_pool)), "quality")
            selected.append(extra)

    if not selected:
        print(f"    ✗ Nenhuma máquina seleccionada!")
        return pd.DataFrame()

    result = pd.concat(selected).drop_duplicates("machine_id").head(n_total)

    # Relatório
    print(f"    Total seleccionadas: {len(result)}")
    for profile in sorted(mix_config.keys()):
        n_sel = len(result[result["profile"] == profile])
        n_pool = allocations.get(profile, {}).get("pool", 0)
        desired = allocations.get(profile, {}).get("desired", 0)
        fill = "✓" if n_sel >= desired else f"({n_sel}/{desired})"
        print(f"      {profile:<22} {n_sel:>3} seleccionadas  (pool: {n_pool:>4})  {fill}")

    # Perfis extra (de redistribuição)
    for profile in result["profile"].unique():
        if profile not in mix_config:
            n_sel = len(result[result["profile"] == profile])
            print(f"      {profile:<22} {n_sel:>3} (redistribuição)")

    return result


# ══════════════════════════════════════════════════════════
#  PASS 4: EXTRAIR SÉRIES PARA DISCO
# ══════════════════════════════════════════════════════════

def extract_to_disk(csv_path, machine_ids, output_path, profiles_map):
    """Extrai séries completas com coluna de perfil, streaming."""
    selected_set = set(machine_ids)
    total = 0
    header_written = False

    if output_path.exists():
        output_path.unlink()

    reader = pd.read_csv(
        csv_path, names=SCHEMA, header=None,
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for chunk in tqdm(reader, desc=f"  Extracting → {output_path.name}"):
        filtered = chunk[chunk["machine_id"].isin(selected_set)].copy()
        if len(filtered) == 0:
            continue

        # Limpar
        for col in ["cpu_util_percent", "mem_util_percent", "mem_gps",
                     "mkpi", "net_in", "net_out", "disk_io_percent"]:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
        for col in ["cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"]:
            filtered.loc[filtered[col] < 0, col] = np.nan
            filtered.loc[filtered[col] > 100, col] = np.nan

        # Adicionar perfil
        filtered["machine_profile"] = filtered["machine_id"].map(profiles_map)

        filtered.to_csv(
            output_path,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
        )
        header_written = True
        total += len(filtered)

    return total


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print(" ALIBABA v2018 — Selecção v2 (diversidade + perfis)")
    print("=" * 62)

    csv_path = find_csv("machine_usage")
    if csv_path is None:
        print("✗ machine_usage.csv não encontrado!")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════
    # PASS 1: PROFILE
    # ════════════════════════════════════════════════════════

    prof = profile_all_machines(csv_path)

    # ════════════════════════════════════════════════════════
    # PASS 2: CLASSIFY
    # ════════════════════════════════════════════════════════

    print(f"\n{'='*62}")
    print(f" PASS 2: Classificação de perfis comportamentais")
    print(f"{'='*62}")

    prof = classify_profiles(prof)

    # Guardar perfis completos
    prof_path = OUTPUT_DIR / "perfil_todas_maquinas_v2.csv"
    prof.to_csv(prof_path, index=False)
    print(f"\n  ✓ Perfis: {prof_path}")

    # Relatório de perfis
    print(f"\n  Distribuição de perfis ({len(prof)} máquinas):")
    vc = prof["profile"].value_counts()
    for profile, cnt in vc.items():
        pct = cnt / len(prof) * 100
        bar = "█" * max(1, int(pct / 2))

        sub = prof[prof["profile"] == profile]
        cpu_info = f"CPU {sub['cpu_mean'].mean():>5.1f}% ± {sub['cpu_std'].mean():>4.1f}"
        pts_info = f"{sub['n_points'].median():>6,.0f} pts"

        print(f"    {profile:<22} {cnt:>4} ({pct:>5.1f}%) {bar}")
        print(f"    {'':22} {cpu_info}  {pts_info}")

    # Estatísticas globais
    print(f"\n  Top 5 por burstiness:")
    for _, r in prof.nlargest(5, "burstiness").iterrows():
        print(f"    {str(r['machine_id'])[:15]:<16} burst={r['burstiness']:>5.1f}  "
              f"spikes={r['spike_count']:>3}  CPU={r['cpu_mean']:>5.1f}% ± {r['cpu_std']:>4.1f}")

    print(f"\n  Top 5 por trend:")
    for _, r in prof.nlargest(5, "trend_per_hour").iterrows():
        print(f"    {str(r['machine_id'])[:15]:<16} trend={r['trend_per_hour']:>+6.2f}%/h  "
              f"CPU={r['cpu_mean']:>5.1f}%  {r['duration_days']:.1f}d")

    # ════════════════════════════════════════════════════════
    # PASS 3: SAMPLE + PASS 4: EXTRACT
    # ════════════════════════════════════════════════════════

    print(f"\n{'='*62}")
    print(f" PASS 3+4: Amostragem e extracção")
    print(f"{'='*62}")

    datasets_generated = []

    # ── GENERAL (todas as 100, diversidade máxima) ──
    if GENERATE_GENERAL:
        general_mix = {p: round(100 * cnt / len(prof))
                       for p, cnt in vc.items() if cnt > 0}
        # Garantir que soma ~100
        total_pct = sum(general_mix.values())
        if total_pct > 0:
            general_mix = {k: max(1, round(v * 100 / total_pct))
                           for k, v in general_mix.items()}

        sel_general = stratified_sample(prof, general_mix, N_TOTAL, "general")
        if len(sel_general) > 0:
            profiles_map = dict(zip(sel_general["machine_id"], sel_general["profile"]))
            out = OUTPUT_DIR / "dataset_general.csv"
            n = extract_to_disk(csv_path, sel_general["machine_id"].tolist(), out, profiles_map)
            size = out.stat().st_size / 1024 / 1024
            print(f"\n  ✓ {out.name}: {n:,} linhas, {size:.1f} MB")
            datasets_generated.append(("general", out, n, len(sel_general)))

            sel_general.to_csv(OUTPUT_DIR / "selecao_general.csv", index=False)

    # ── FORECASTING ──
    if GENERATE_FORECASTING:
        sel_fc = stratified_sample(prof, FORECASTING_MIX, N_TOTAL, "forecasting")
        if len(sel_fc) > 0:
            profiles_map = dict(zip(sel_fc["machine_id"], sel_fc["profile"]))
            out = OUTPUT_DIR / "dataset_forecasting.csv"
            n = extract_to_disk(csv_path, sel_fc["machine_id"].tolist(), out, profiles_map)
            size = out.stat().st_size / 1024 / 1024
            print(f"\n  ✓ {out.name}: {n:,} linhas, {size:.1f} MB")
            datasets_generated.append(("forecasting", out, n, len(sel_fc)))

            sel_fc.to_csv(OUTPUT_DIR / "selecao_forecasting.csv", index=False)

    # ── ANOMALY ──
    if GENERATE_ANOMALY:
        sel_an = stratified_sample(prof, ANOMALY_MIX, N_TOTAL, "anomaly")
        if len(sel_an) > 0:
            profiles_map = dict(zip(sel_an["machine_id"], sel_an["profile"]))
            out = OUTPUT_DIR / "dataset_anomaly.csv"
            n = extract_to_disk(csv_path, sel_an["machine_id"].tolist(), out, profiles_map)
            size = out.stat().st_size / 1024 / 1024
            print(f"\n  ✓ {out.name}: {n:,} linhas, {size:.1f} MB")
            datasets_generated.append(("anomaly", out, n, len(sel_an)))

            sel_an.to_csv(OUTPUT_DIR / "selecao_anomaly.csv", index=False)

    # ════════════════════════════════════════════════════════
    # RESUMO FINAL
    # ════════════════════════════════════════════════════════

    print(f"\n{'='*62}")
    print(f" RESUMO")
    print(f"{'='*62}")

    print(f"\n  Datasets gerados:")
    for name, path, n_lines, n_machines in datasets_generated:
        size = path.stat().st_size / 1024 / 1024
        print(f"    {name:<15} {n_machines:>3} máquinas  {n_lines:>10,} linhas  {size:>6.1f} MB")

    print(f"\n  Colunas em cada dataset:")
    print(f"    {SCHEMA + ['machine_profile']}")

    print(f"""
  Ficheiros de selecção (perfil de cada máquina escolhida):
    selecao_general.csv
    selecao_forecasting.csv
    selecao_anomaly.csv

  Uso em pandas:
    df = pd.read_csv('alibaba_processado/dataset_forecasting.csv')

    # Ver perfis presentes
    df['machine_profile'].value_counts()

    # Filtrar por perfil
    bursty = df[df['machine_profile'] == 'bursty']

    # Uma máquina
    m1 = df[df['machine_id'] == df['machine_id'].unique()[0]]
    m1 = m1.sort_values('time_stamp')

  Perfis completos de TODAS as máquinas:
    perfil_todas_maquinas_v2.csv
""")


if __name__ == "__main__":
    main()
