"""Emparejamiento Sistema ↔ Banco y exportación de 'Sistema + Banco.csv' con Conciliado=1.

Fases:
- F0 (join exacto por hash): Id banco (opcional), Cuenta, Código, Fecha (día), Monto (2 dec), Referencia idéntica.
- F1 (join exacto sin Referencia): mismos hashes; calcula similitud de referencia solo para candidatos.
- Fallback exact (por grupos): filtro por tolerancias y similitud dentro de cada grupo.
- Relajado: misma cuenta; score compuesto por banco, código, monto, fecha y similitud.

Rendimiento:
- Joins hash vectorizados resuelven la mayoría de exactos rápidamente.
- La similitud de referencia se calcula solo en candidatos filtrados (batch).
- Evita reutilizar contrapartes con asignación codiciosa por mayor score.

Uso:
- pip install pandas numpy python-dateutil rapidfuzz
- python build_pairs.py
"""

from __future__ import annotations
import sys
import math
from dataclasses import dataclass
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from rapidfuzz.fuzz import token_set_ratio

BASE_DIR = Path(__file__).resolve().parent
SISTEMA_CSV = BASE_DIR / "Sistema.csv"
BANCO_CSV = BASE_DIR / "Banco.csv"
OUTPUT_CSV = BASE_DIR / "Sistema + Banco.csv"

# Esquema base consistente con build_sistema.py y build_banco.py
SCHEMA_BASE = ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"]

# Parámetros de matching y pesos del score
@dataclass
class MatchParams:
    """Parámetros de matching y pesos del score.
    Attributes:
        amount_tolerance: Tolerancia absoluta para monto (misma moneda).
        date_window_days: Ventana de días para fecha (abs).
        ref_similarity_threshold: Umbral de similitud [0,1] para referencia.
        require_same_bank_exact: Exigir mismo Id banco en exactos F0/F1.
        require_same_trans_exact: Exigir mismo Código en exactos (bloqueo).
        w1..w6: Pesos para score del match relajado.
        log_interval_* / show_progress: Control de logging y progreso.
        relaxed_require_same_bank: Si True, el relajado exige mismo Id banco como llave.
        relaxed_require_same_code: Si True, el relajado exige mismo Código como llave.
        relaxed_max_estimated_pairs: Umbral duro para abortar relajado si el estimado supera este valor.
        max_accounts_relaxed: Límite opcional de cuentas a procesar en relajado (debug).
    """
    amount_tolerance: float = 0.01  # misma moneda
    date_window_days: int = 1       # días
    ref_similarity_threshold: float = 0.90
    require_same_bank_exact: bool = True
    require_same_trans_exact: bool = True
    # Pesos para score relajado
    w1_same_account: float = 1.0
    w2_same_bank: float = 1.0
    w3_amount: float = 1.0
    w4_time: float = 1.0
    w5_ref: float = 1.0
    w6_same_trans: float = 1.0
    # Logging / progreso
    log_interval_groups: int = 50
    log_interval_rows: int = 10000
    show_progress: bool = True
    # NUEVO: controles relajado
    relaxed_require_same_bank: bool = True
    relaxed_require_same_code: bool = True
    relaxed_max_estimated_pairs: int = 200_000_000
    max_accounts_relaxed: Optional[int] = None

P = MatchParams()

def _to_float(x) -> Optional[float]:
    """Convierte strings de monto a float; tolera comas, símbolos y vacíos."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        # Remover símbolos y reintentar
        s2 = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
        try:
            return float(s2)
        except Exception:
            return None

def _to_date(x) -> Optional[pd.Timestamp]:
    """Parsea fechas a Timestamp o None si inválida."""
    if pd.isna(x) or str(x).strip() == "":
        return None
    if isinstance(x, (pd.Timestamp, )):
        return pd.to_datetime(x)
    for dayfirst in (True, False):
        try:
            return pd.to_datetime(dtparser.parse(str(x).strip(), dayfirst=dayfirst))
        except Exception:
            continue
    return None

def _norm_text(s: object) -> str:
    """Normaliza texto a mayúsculas y recorta espacios; vacío si NaN."""
    if pd.isna(s):
        return ""
    return str(s).strip().upper()

def _abs_days(a: pd.Timestamp, b: pd.Timestamp) -> Optional[int]:
    """Diferencia absoluta en días entre dos timestamps (o None)."""
    if a is None or b is None:
        return None
    return abs((a - b).days)

def _ref_sim(a: str, b: str) -> float:
    """Similitud de referencias en [0,1] usando token_set_ratio (RapidFuzz)."""
    if not a or not b:
        return 0.0
    return token_set_ratio(a, b) / 100.0

def _prepare(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prepara DataFrame para matching: normaliza tipos, crea claves de hash y un índice único."""
    # Mantener solo columnas base
    df = df[SCHEMA_BASE].copy()
    # Tipos y normalización light (ya vienen normalizados de los builds)
    df["Id banco"] = df["Id banco"].map(_norm_text).astype("category")
    df["Cuenta bancaria"] = df["Cuenta bancaria"].map(lambda x: "" if pd.isna(x) else "".join(ch for ch in str(x) if ch.isdigit())).astype("category")
    df["Referencia"] = df["Referencia"].map(_norm_text)
    df["Fecha_dt"] = df["Fecha"].map(_to_date)
    df["Monto_f"] = df["Monto"].map(_to_float)
    df["Código"] = df["Código"].map(_norm_text).astype("category")
    # Claves preformateadas para joins hash (evita float==)
    df["Fecha_key"] = df["Fecha_dt"].dt.strftime("%Y-%m-%d")
    df["Monto_key"] = df["Monto_f"].map(lambda v: None if pd.isna(v) else f"{v:.2f}")
    # Índice único para no perder referencia
    df[f"{prefix}_idx"] = np.arange(len(df), dtype=np.int64)
    return df

def _fast_exact_merge(df_s: pd.DataFrame, df_b: pd.DataFrame, params: MatchParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join exacto vectorizado (F0/F1) y retorno de matches + remanente del sistema.
    Retorna:
        match: DataFrame [s_idx,b_idx,score] sin reutilización tras greedy.
        no_used_s: DataFrame del sistema que quedó sin asignar tras F0/F1.
    """
    # Filtrar válidos
    df_s = df_s[~df_s["Monto_f"].isna() & df_s["Fecha_dt"].notna()].copy()
    df_b = df_b[~df_b["Monto_f"].isna() & df_b["Fecha_dt"].notna()].copy()

    key_base = ["Cuenta bancaria", "Código", "Fecha_key", "Monto_key"]
    if params.require_same_bank_exact:
        key = ["Id banco"] + key_base
    else:
        key = key_base

    # F0: join incluyendo Referencia
    key_ref = key + ["Referencia"]
    cols_keep_s = ["s_idx", "Referencia"]
    cols_keep_b = ["b_idx", "Referencia"]
    m0 = df_s[key_ref + ["s_idx"]].merge(
        df_b[key_ref + ["b_idx"]],
        how="inner",
        on=key_ref,
        suffixes=("_s", "_b")
    )
    m0 = m0[["s_idx", "b_idx"]].drop_duplicates()
    m0["score"] = 200.0  # score alto para exactos
    used_s = set(m0["s_idx"].tolist())
    used_b = set(m0["b_idx"].tolist())

    # F1: join por claves exactas sin Referencia (similitud para candidatos)
    s1 = df_s[~df_s["s_idx"].isin(used_s)][key + cols_keep_s].copy()
    b1 = df_b[~df_b["b_idx"].isin(used_b)][key + cols_keep_b].copy()
    if len(s1) == 0 or len(b1) == 0:
        return m0, df_s[~df_s["s_idx"].isin(used_s)].copy()

    cand = s1.merge(b1, how="inner", on=key, suffixes=("_s", "_b"))
    # Si no hay candidatos, terminar
    if cand.empty:
        return m0, df_s[~df_s["s_idx"].isin(used_s)].copy()

    # Calcular similitud SOLO para filas cand
    # Batch para no saturar memoria
    batch = 500_000
    ok_rows = []
    for i in range(0, len(cand), batch):
        chunk = cand.iloc[i:i + batch].copy()
        sim = [ _ref_sim(a, b) for a, b in zip(chunk["Referencia_s"].tolist(), chunk["Referencia_b"].tolist()) ]
        sim = np.array(sim, dtype=float)
        mask = sim >= params.ref_similarity_threshold
        if mask.any():
            part = chunk.loc[mask, ["s_idx", "b_idx"]].copy()
            part["score"] = 150.0 + 10.0 * sim[mask]
            ok_rows.append(part)
    m1 = pd.concat(ok_rows, ignore_index=True) if ok_rows else pd.DataFrame(columns=["s_idx","b_idx","score"])

    # Unir F0 + F1
    match = pd.concat([m0, m1], ignore_index=True) if not m1.empty else m0
    # Greedy para evitar reutilizar
    match = _greedy_assign(match, "s_idx", "b_idx", "score")

    used_s2 = set(match["s_idx"].tolist())
    no_used_s = df_s[~df_s["s_idx"].isin(used_s2)].copy()
    return match, no_used_s

def _greedy_assign(candidates: pd.DataFrame, left_key: str, right_key: str, score_col: str) -> pd.DataFrame:
    """Asigna pares evitando reutilizar elementos, priorizando mayor score (codicioso)."""
    if candidates.empty:
        return candidates

    cand = candidates.sort_values(by=[score_col], ascending=False).copy()
    used_left = set()
    used_right = set()
    rows = []
    for row in cand.itertuples(index=False):
        l = getattr(row, left_key)
        r = getattr(row, right_key)
        sc = getattr(row, score_col)
        if l in used_left or r in used_right:
            continue
        used_left.add(l)
        used_right.add(r)
        rows.append(row)

    return pd.DataFrame(rows, columns=cand.columns)

def _build_exact_candidates(df_s: pd.DataFrame, df_b: pd.DataFrame, params: MatchParams) -> pd.DataFrame:
    """Genera candidatos exactos por grupos (fallback) aplicando tolerancias y similitud."""
    start_time = time.perf_counter()
    # Bloqueo por (Id banco?, Cuenta, Código)
    # Si se requiere mismo banco, el bloque usa (Id banco, Cuenta, Código); si no, (Cuenta, Código)
    if params.require_same_bank_exact:
        s_blocks = df_s.groupby(["Id banco", "Cuenta bancaria", "Código"], observed=False)
        b_blocks = df_b.groupby(["Id banco", "Cuenta bancaria", "Código"], observed=False)
        common_keys = set(s_blocks.groups.keys()).intersection(b_blocks.groups.keys())
        block_keys = [("Id banco", "Cuenta bancaria", "Código"), common_keys]
    else:
        s_blocks = df_s.groupby(["Cuenta bancaria", "Código"], observed=False)
        b_blocks = df_b.groupby(["Cuenta bancaria", "Código"], observed=False)
        common_keys = set(s_blocks.groups.keys()).intersection(b_blocks.groups.keys())
        block_keys = [("Cuenta bancaria", "Código"), common_keys]

    records = []
    groups_total = len(common_keys)
    groups_done = 0
    rows_scanned = 0
    pair_checks = 0
    candidates_valid = 0
    key_cols, keys = block_keys
    if params.show_progress:
        print(f"[INFO] (Exacto) Grupos comunes: {groups_total:,}")
    for key in keys:
        s_g = df_s.loc[s_blocks.groups[key]]
        b_g = df_b.loc[b_blocks.groups[key]]
        groups_done += 1

        # Para cada fila del sistema, filtrar candidatos por monto y fecha
        b_amount = b_g["Monto_f"].to_numpy()
        b_dates = b_g["Fecha_dt"].to_numpy()
        b_refs = b_g["Referencia"].to_numpy()
        b_idxs = b_g["s_idx"].to_numpy() if "s_idx" in b_g.columns else b_g["b_idx"].to_numpy()  # use b_idx
        if "b_idx" in b_g.columns:
            b_idxs = b_g["b_idx"].to_numpy()
        else:
            # si no existe, créalo de la posición
            b_idxs = np.arange(len(b_g), dtype=np.int64)

        for s_row in s_g.itertuples(index=False):
            if params.require_same_trans_exact and s_row._asdict().get("Código", "") != "":
                # Código es parte del bloque, ya igual
                pass
            s_amt = getattr(s_row, "Monto_f")
            s_date = getattr(s_row, "Fecha_dt")
            s_ref = getattr(s_row, "Referencia")
            s_idx = getattr(s_row, "s_idx")

            if s_amt is None or s_date is None:
                continue

            # Filtro por monto y fecha
            amt_diff = np.abs(b_amount - s_amt)
            ok_amt = amt_diff <= params.amount_tolerance

            # Fecha
            days_diff = np.array([np.nan if (pd.isna(d) or s_date is None) else abs((d - s_date).days) for d in b_dates])
            ok_date = days_diff <= params.date_window_days

            mask = ok_amt & ok_date
            if not mask.any():
                continue

            # Similitud de referencia SOLO para índices con mask=True
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            sim_vals = np.array([_ref_sim(s_ref, b_refs[j]) for j in idx])
            ok_ref = sim_vals >= params.ref_similarity_threshold
            if not ok_ref.any():
                continue

            # Stats
            rows_scanned += 1
            pair_checks += idx.size
            candidates_valid += int(ok_ref.sum())

            amt_term_all = 1.0 - np.minimum(amt_diff[idx] / max(params.amount_tolerance, 1e-9), 1.0)
            time_term_all = 1.0 - np.minimum(days_diff[idx] / max(params.date_window_days, 1), 1.0)
            score = 100.0 + 10.0 * amt_term_all + 10.0 * time_term_all + 10.0 * sim_vals
            for j_sub, okv in enumerate(ok_ref):
                if okv:
                    j = idx[j_sub]
                    records.append((s_idx, b_idxs[j], float(score[j_sub])))

            # Log por filas
            if params.show_progress and rows_scanned % max(1, params.log_interval_rows) == 0:
                elapsed = time.perf_counter() - start_time
                rate = rows_scanned / max(elapsed, 1e-9)
                print(f"[PROGRESS][Exacto] Filas sist eval: {rows_scanned:,} | Checks: {pair_checks:,} | Candidatos válidos: {candidates_valid:,} | Records: {len(records):,} | {rate:,.0f} filas/s")

        # Log por grupos
        if params.show_progress and groups_done % max(1, params.log_interval_groups) == 0:
            elapsed = time.perf_counter() - start_time
            pct = (groups_done / max(groups_total, 1)) * 100.0
            print(f"[PROGRESS][Exacto] Grupos: {groups_done:,}/{groups_total:,} ({pct:0.1f}%) | Records: {len(records):,} | Tiempo: {elapsed:0.1f}s")

    if not records:
        return pd.DataFrame(columns=["s_idx", "b_idx", "score"])

    return pd.DataFrame(records, columns=["s_idx", "b_idx", "score"])

def _int_cents(v: float) -> Optional[int]:
    if v is None or pd.isna(v):
        return None
    try:
        return int(round(float(v) * 100))
    except Exception:
        return None

def _build_relaxed_candidates(df_s: pd.DataFrame, df_b: pd.DataFrame, params: MatchParams) -> pd.DataFrame:
    """Genera candidatos relajados por CUENTA en streaming, sin joins globales.

    Estrategia:
    - Para cada cuenta presente en ambos lados:
      - Expandir SOLO lado Sistema con offsets de fecha +/- window y monto +/- tol (en centavos) para construir llaves discretas.
      - Hacer merge hash con el lado Banco (sin expandir) por llaves: Cuenta + Fecha_key + Monto_cents (+ opcionalmente Banco/Código).
      - Puntuar y asignar de forma codiciosa dentro de la cuenta (sin conflictos entre cuentas).
    - Devuelve DataFrame con columnas [s_idx, b_idx, score].
    """
    import math
    start_time = time.perf_counter()

    # Filtrar válidos
    s = df_s[~df_s["Monto_f"].isna() & df_s["Fecha_dt"].notna()].copy()
    b = df_b[~df_b["Monto_f"].isna() & df_b["Fecha_dt"].notna()].copy()
    if s.empty or b.empty:
        return pd.DataFrame(columns=["s_idx", "b_idx", "score"])

    # Preparar llaves discretas en BANCO (no expandimos Banco)
    b = b.copy()
    b["Fecha_key"] = b["Fecha_dt"].dt.strftime("%Y-%m-%d")
    b["Monto_cents"] = (b["Monto_f"] * 100).round().astype("Int64")

    # Definir llaves de join
    join_keys = ["Cuenta bancaria", "Fecha_key", "Monto_cents"]
    if params.relaxed_require_same_bank:
        join_keys.append("Id banco")
    if params.relaxed_require_same_code:
        join_keys.append("Código")

    # Estimación temprana de combinaciones (abort temprano si excede umbral)
    # Cuenta agrupada (o Cuenta+Banco+Código si se exige) y multiplicar tamaños por factor de expansión
    group_keys = ["Cuenta bancaria"]
    if params.relaxed_require_same_bank:
        group_keys.append("Id banco")
    if params.relaxed_require_same_code:
        group_keys.append("Código")

    s_counts = s.groupby(group_keys, observed=False).size()
    b_counts = b.groupby(group_keys, observed=False).size()
    common_groups = s_counts.index.intersection(b_counts.index)
    if len(common_groups) == 0:
        return pd.DataFrame(columns=["s_idx", "b_idx", "score"])

    # Factor de expansión: (2*date_window+1) * (2*cent_tol+1)
    cent_tol = max(0, int(round(params.amount_tolerance * 100)))
    exp_factor = (2 * params.date_window_days + 1) * (2 * cent_tol + 1)
    # Estimado superior
    est_pairs = int(sum((int(s_counts[g]) * int(b_counts[g]) * exp_factor) for g in common_groups))
    if est_pairs > params.relaxed_max_estimated_pairs:
        raise MemoryError(
            f"[ABORT RELAJADO] Estimado de pares {est_pairs:,} excede el umbral "
            f"relaxed_max_estimated_pairs={params.relaxed_max_estimated_pairs:,}. "
            f"Reduce tolerancias/ventanas o activa relaxed_require_same_*."
        )

    # Procesamiento por CUENTA (sin joins globales)
    accounts_s = s["Cuenta bancaria"].astype(str)
    accounts_b = b["Cuenta bancaria"].astype(str)
    common_accounts = sorted(set(accounts_s.unique()).intersection(accounts_b.unique()))

    results: List[pd.DataFrame] = []
    processed_accounts = 0
    total_accounts = len(common_accounts)
    # Precalcular offsets
    date_offsets = list(range(-params.date_window_days, params.date_window_days + 1))
    cent_offsets = list(range(-cent_tol, cent_tol + 1))

    # Reducir columnas para join / score
    s_base_cols = ["s_idx", "Cuenta bancaria", "Id banco", "Código", "Referencia", "Fecha_dt", "Monto_f"]
    b_base_cols = ["b_idx", "Cuenta bancaria", "Id banco", "Código", "Referencia", "Fecha_dt", "Monto_f", "Fecha_key", "Monto_cents"]

    for acct in common_accounts:
        if params.max_accounts_relaxed is not None and processed_accounts >= params.max_accounts_relaxed:
            break

        s_g = s.loc[s["Cuenta bancaria"] == acct, s_base_cols].copy()
        b_g = b.loc[b["Cuenta bancaria"] == acct, b_base_cols].copy()
        if s_g.empty or b_g.empty:
            processed_accounts += 1
            continue

        # Opcionalmente restringir por banco/código antes de expandir si se exigen
        if params.relaxed_require_same_bank:
            # Procesar por banco para reducir combinaciones
            bank_groups = sorted(set(s_g["Id banco"].unique()).intersection(set(b_g["Id banco"].unique())))
        else:
            bank_groups = [None]

        for bank in bank_groups:
            s_gb = s_g if bank is None else s_g[s_g["Id banco"] == bank]
            b_gb = b_g if bank is None else b_g[b_g["Id banco"] == bank]
            if s_gb.empty or b_gb.empty:
                continue

            if params.relaxed_require_same_code:
                code_groups = sorted(set(s_gb["Código"].unique()).intersection(set(b_gb["Código"].unique())))
            else:
                code_groups = [None]

            for code in code_groups:
                s_gbc = s_gb if code is None else s_gb[s_gb["Código"] == code]
                b_gbc = b_gb if code is None else b_gb[b_gb["Código"] == code]
                if s_gbc.empty or b_gbc.empty:
                    continue

                # Expandir S: offsets de fecha y centavos
                frames = []
                # Precalcular Monto_cents base en S
                s_m_cents = (s_gbc["Monto_f"] * 100).round().astype("Int64")
                for doff in date_offsets:
                    s_tmp = s_gbc.copy()
                    s_tmp["Fecha_key"] = (s_tmp["Fecha_dt"] + pd.to_timedelta(doff, unit="D")).dt.strftime("%Y-%m-%d")
                    # Variantes de centavos (expandir por offsets)
                    for coff in cent_offsets:
                        st = s_tmp.copy()
                        st["Monto_cents"] = (s_m_cents + coff).astype("Int64")
                        frames.append(st[["s_idx", "Cuenta bancaria", "Id banco", "Código", "Referencia", "Fecha_dt", "Monto_f", "Fecha_key", "Monto_cents"]])
                s_exp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["s_idx","Cuenta bancaria","Id banco","Código","Referencia","Fecha_dt","Monto_f","Fecha_key","Monto_cents"])
                if s_exp.empty:
                    continue

                # Merge por llaves discretas
                k = ["Cuenta bancaria", "Fecha_key", "Monto_cents"]
                if params.relaxed_require_same_bank:
                    k.append("Id banco")
                if params.relaxed_require_same_code:
                    k.append("Código")

                cand = s_exp.merge(
                    b_gbc[["b_idx", "Cuenta bancaria", "Id banco", "Código", "Referencia", "Fecha_dt", "Monto_f", "Fecha_key", "Monto_cents"]],
                    how="inner",
                    on=k,
                    suffixes=("_s", "_b")
                )
                if cand.empty:
                    continue

                # Puntuar
                ref_sim = np.array([_ref_sim(a, b) for a, b in zip(cand["Referencia_s"].tolist(), cand["Referencia_b"].tolist())], dtype=float)
                same_account = 1.0  # ya estamos por cuenta
                same_bank = (cand["Id banco_s"].values == cand["Id banco_b"].values).astype(float)
                same_trans = (cand["Código_s"].values == cand["Código_b"].values).astype(float)
                adiff = (cand["Monto_f_s"].values - cand["Monto_f_b"].values)
                adiff = np.abs(adiff)
                tdiff = np.abs((cand["Fecha_dt_s"].values - cand["Fecha_dt_b"].values).astype("timedelta64[D]").astype(int))
                amt_term = 1.0 - np.minimum(adiff / max(params.amount_tolerance, 1e-9), 1.0)
                time_term = 1.0 - np.minimum(tdiff / max(params.date_window_days, 1), 1.0)
                ref_term = np.where(ref_sim >= params.ref_similarity_threshold, ref_sim, 0.0)
                score = (
                    params.w1_same_account * same_account +
                    params.w2_same_bank * same_bank +
                    params.w6_same_trans * same_trans +
                    params.w3_amount * amt_term +
                    params.w4_time * time_term +
                    params.w5_ref * ref_term
                )

                ok = score > 0
                if not np.any(ok):
                    continue

                outp = cand.loc[ok, ["s_idx", "b_idx"]].copy()
                outp["score"] = score[ok]
                # Greedy por cuenta/banco/código (subconjunto pequeño)
                assigned = _greedy_assign(outp, "s_idx", "b_idx", "score")
                if not assigned.empty:
                    results.append(assigned)

        processed_accounts += 1
        if params.show_progress and processed_accounts % max(1, params.log_interval_groups) == 0:
            elapsed = time.perf_counter() - start_time
            pct = (processed_accounts / max(total_accounts, 1)) * 100.0
            print(f"[PROGRESS][Relajado streaming] Cuentas: {processed_accounts:,}/{total_accounts:,} ({pct:0.1f}%) | Tiempo: {elapsed:0.1f}s")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["s_idx","b_idx","score"])

def build_pairs() -> pd.DataFrame:
    """Punto de entrada: lee Sistema/Banco, ejecuta F0/F1, fallback y relajado; exporta salida y estadísticas."""
    start_time_total = time.perf_counter()
    if not SISTEMA_CSV.exists() or not BANCO_CSV.exists():
        print("[ERROR] Faltan archivos Sistema.csv o Banco.csv")
        sys.exit(1)

    print(f"[INFO] Leyendo {SISTEMA_CSV.name} y {BANCO_CSV.name}")
    df_s_raw = pd.read_csv(SISTEMA_CSV, dtype=str)
    df_b_raw = pd.read_csv(BANCO_CSV, dtype=str)
    print(f"[INFO] Tamaños entrada -> Sistema: {len(df_s_raw):,} | Banco: {len(df_b_raw):,}")

    df_s = _prepare(df_s_raw, "s")
    df_b = _prepare(df_b_raw, "b")

    # Filtrar filas con monto o fecha inválidos (mantener por consistencia)
    df_s = df_s[~df_s["Monto_f"].isna() & df_s["Fecha_dt"].notna()].reset_index(drop=True)
    df_b = df_b[~df_b["Monto_f"].isna() & df_b["Fecha_dt"].notna()].reset_index(drop=True)
    print(f"[INFO] Filtrado válido -> Sistema: {len(df_s):,} | Banco: {len(df_b):,}")

    # NUEVO: Fase 0/1 por join vectorizado
    print("[INFO] Join vectorizado para exactos...")
    match_exact_fast, s_left_after_fast = _fast_exact_merge(df_s, df_b, P)
    print(f"[INFO] Matches exactos por join: {len(match_exact_fast):,}")

    used_s = set(match_exact_fast["s_idx"].tolist())
    used_b = set(match_exact_fast["b_idx"].tolist())

    # Fallback exact (optimizado) SOLO sobre no usados (mucho menor)
    s_rem = df_s[~df_s["s_idx"].isin(used_s)].reset_index(drop=True)
    b_rem = df_b[~df_b["b_idx"].isin(used_b)].reset_index(drop=True)
    cand_exact = _build_exact_candidates(s_rem, b_rem, P)
    match_exact_rest = _greedy_assign(cand_exact, "s_idx", "b_idx", "score")
    print(f"[INFO] Matches exactos (fallback): {len(match_exact_rest):,}")

    # Unir exactos totales
    match_exact = pd.concat([match_exact_fast, match_exact_rest], ignore_index=True)

    # Marcar usados tras exactos
    used_s = set(match_exact["s_idx"].tolist())
    used_b = set(match_exact["b_idx"].tolist())

    # Fase 2 (relajado streaming por cuenta) con abort temprano si se estima muy grande
    s_left = df_s[~df_s["s_idx"].isin(used_s)].reset_index(drop=True)
    b_left = df_b[~df_b["b_idx"].isin(used_b)].reset_index(drop=True)
    print(f"[INFO] Sobrantes para relajado — Sistema: {len(s_left):,}, Banco: {len(b_left):,}")
    try:
        cand_relaxed = _build_relaxed_candidates(s_left, b_left, P)
    except MemoryError as me:
        print(str(me))
        print("[INFO] Se omite fase relajada por riesgo de memoria. Continúo solo con matches exactos.")
        cand_relaxed = pd.DataFrame(columns=["s_idx","b_idx","score"])

    match_relaxed = _greedy_assign(cand_relaxed, "s_idx", "b_idx", "score") if not cand_relaxed.empty else cand_relaxed
    print(f"[INFO] Matches relajados asignados: {len(match_relaxed):,}")

    # Unir matches
    matches = pd.concat([match_exact, match_relaxed], ignore_index=True)

    # Preparar salida Sistema + Banco (Conciliado=1)
    s_cols = [c for c in SCHEMA_BASE]
    b_cols = [c for c in SCHEMA_BASE]

    df_s_mat = df_s.set_index("s_idx").loc[matches["s_idx"]][s_cols].reset_index(drop=True)
    df_b_mat = df_b.set_index("b_idx").loc[matches["b_idx"]][b_cols].reset_index(drop=True)

    out = pd.concat([df_s_mat, df_b_mat], axis=1)
    out["Conciliado"] = 1

    # Renombrar columnas a duplicados como en el ejemplo (mismo header repetido)
    # Internamente están en orden: [SIS 6] + [BAN 6] + Conciliado
    out.columns = (
        ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"] +
        ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"] +
        ["Conciliado"]
    )

    # Exportar
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    elapsed_total = time.perf_counter() - start_time_total
    print(f"[OK] Exportado: {OUTPUT_CSV} ({len(out)} filas)")
    print(f"[STATS] Tiempo total: {elapsed_total:0.2f}s | Matches exactos: {len(match_exact)} | Matches relajados: {len(match_relaxed)} | Matches totales: {len(matches)}")

    # Previsualización
    print("\n[PREVIEW] Primeras 10 filas Sistema + Banco (Conciliado=1):")
    print(out.head(10).to_string(index=False))
    return out

if __name__ == "__main__":
    build_pairs()