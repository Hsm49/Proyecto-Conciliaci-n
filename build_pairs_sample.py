from __future__ import annotations
import argparse
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from rapidfuzz.fuzz import token_set_ratio

BASE_DIR = Path(__file__).resolve().parent
SISTEMA_CSV = BASE_DIR / "Sistema.csv"
BANCO_CSV = BASE_DIR / "Banco.csv"
OUTPUT_CSV = BASE_DIR / "Sistema + Banco (sample).csv"

SCHEMA_BASE = ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"]

@dataclass
class MatchParams:
    amount_tolerance: float = 0.01
    date_window_days: int = 1
    ref_similarity_threshold: float = 0.90
    require_same_bank_exact: bool = True
    require_same_trans_exact: bool = True

def _to_float(x) -> Optional[float]:
    if pd.isna(x):
        return None
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        s2 = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
        try:
            return float(s2)
        except Exception:
            return None

def _to_date(x) -> Optional[pd.Timestamp]:
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
    if pd.isna(s):
        return ""
    return str(s).strip().upper()

def _ref_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return token_set_ratio(a, b) / 100.0

def _prepare(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df[SCHEMA_BASE].copy()
    df["Id banco"] = df["Id banco"].map(_norm_text)
    df["Cuenta bancaria"] = df["Cuenta bancaria"].map(lambda x: "" if pd.isna(x) else "".join(ch for ch in str(x) if ch.isdigit()))
    df["Referencia"] = df["Referencia"].map(_norm_text)
    df["Fecha_dt"] = df["Fecha"].map(_to_date)
    df["Monto_f"] = df["Monto"].map(_to_float)
    df["Código"] = df["Código"].map(_norm_text)
    df[f"{prefix}_idx"] = np.arange(len(df), dtype=np.int64)
    return df

def build_pairs_sample(params: MatchParams, limit: int) -> pd.DataFrame:
    start_time = time.perf_counter()
    if not SISTEMA_CSV.exists() or not BANCO_CSV.exists():
        raise FileNotFoundError("Faltan Sistema.csv o Banco.csv")

    print(f"[INFO] Leyendo archivos de entrada...", flush=True)
    df_s_raw = pd.read_csv(SISTEMA_CSV, dtype=str)
    df_b_raw = pd.read_csv(BANCO_CSV, dtype=str)
    print(f"[INFO] Sistema: {len(df_s_raw):,} filas | Banco: {len(df_b_raw):,} filas", flush=True)

    df_s = _prepare(df_s_raw, "s")
    df_b = _prepare(df_b_raw, "b")

    # Filtrar filas con monto o fecha inválidos
    df_s = df_s[~df_s["Monto_f"].isna() & df_s["Fecha_dt"].notna()].reset_index(drop=True)
    df_b = df_b[~df_b["Monto_f"].isna() & df_b["Fecha_dt"].notna()].reset_index(drop=True)
    print(f"[INFO] Filtrado válido -> Sistema: {len(df_s):,} | Banco: {len(df_b):,}", flush=True)

    # Bloqueo por claves exactas
    if params.require_same_bank_exact:
        s_groups = df_s.groupby(["Id banco", "Cuenta bancaria", "Código"])
        b_groups = df_b.groupby(["Id banco", "Cuenta bancaria", "Código"])
        common_keys = set(s_groups.groups.keys()).intersection(b_groups.groups.keys())
    else:
        s_groups = df_s.groupby(["Cuenta bancaria", "Código"])
        b_groups = df_b.groupby(["Cuenta bancaria", "Código"])
        common_keys = set(s_groups.groups.keys()).intersection(b_groups.groups.keys())

    used_s = set()
    used_b = set()
    matches = []

    # Indicadores de progreso
    groups_total = len(common_keys)
    groups_done = 0
    rows_scanned = 0
    pair_checks = 0  # número de comparaciones sistema x candidatos banco
    candidates_valid = 0  # candidatos que pasan monto+fecha+referencia
    last_log_time = time.perf_counter()
    log_interval_groups = getattr(params, "_log_interval_groups", 20)
    log_interval_rows = getattr(params, "_log_interval_rows", 10000)
    show_progress = getattr(params, "_show_progress", True)

    # Iteración con corte temprano al alcanzar 'limit'
    print(f"[INFO] Grupos comunes para matching exacto: {groups_total:,}", flush=True)
    for key in common_keys:
        s_g = df_s.loc[s_groups.groups[key]].copy()
        b_g = df_b.loc[b_groups.groups[key]].copy()
        groups_done += 1

        # Precalcular vectores del grupo banco
        b_amount = b_g["Monto_f"].to_numpy()
        b_dates = b_g["Fecha_dt"].to_numpy()
        b_refs = b_g["Referencia"].to_numpy()
        b_idxs = b_g["b_idx"].to_numpy()

        for s_row in s_g.itertuples(index=False):
            s_idx = getattr(s_row, "s_idx")
            if s_idx in used_s:
                continue

            s_amt = getattr(s_row, "Monto_f")
            s_date = getattr(s_row, "Fecha_dt")
            s_ref = getattr(s_row, "Referencia")

            if s_amt is None or s_date is None:
                continue

            # Filtro por monto y fecha dentro de ventana
            amt_diff = np.abs(b_amount - s_amt)
            ok_amt = amt_diff <= params.amount_tolerance
            days_diff = np.array([np.nan if (pd.isna(d) or s_date is None) else abs((d - s_date).days) for d in b_dates])
            ok_date = days_diff <= params.date_window_days
            mask = ok_amt & ok_date
            if not mask.any():
                continue

            # Similitud de referencia
            sim = np.array([_ref_sim(s_ref, br) for br in b_refs])
            ok_ref = sim >= params.ref_similarity_threshold
            ok = np.where(mask & ok_ref)[0]
            if ok.size == 0:
                continue

            # Stats
            rows_scanned += 1
            pair_checks += len(b_amount)
            candidates_valid += int(ok.size)

            # Puntuar y elegir el mejor candidato aún no usado
            amt_term = 1.0 - np.minimum(amt_diff / max(params.amount_tolerance, 1e-9), 1.0)
            time_term = 1.0 - np.minimum(days_diff / max(params.date_window_days, 1), 1.0)
            score = 100.0 + 10.0 * amt_term + 10.0 * time_term + 10.0 * sim

            # Ordenar candidatos por score desc y tomar el primero disponible
            order = ok[np.argsort(score[ok])[::-1]]
            chosen = None
            for j in order:
                b_idx = int(b_idxs[j])
                if b_idx in used_b:
                    continue
                chosen = b_idx
                break

            if chosen is None:
                continue

            used_s.add(s_idx)
            used_b.add(chosen)
            matches.append((s_idx, chosen))

            if len(matches) >= limit:
                break
            # Progreso por filas
            if show_progress and rows_scanned % max(1, log_interval_rows) == 0:
                elapsed = time.perf_counter() - start_time
                rate = rows_scanned / max(elapsed, 1e-9)
                print(f"[PROGRESS] Filas sistema evaluadas: {rows_scanned:,} | Checks: {pair_checks:,} | Matches: {len(matches):,} | {rate:,.0f} filas/s", flush=True)
        if len(matches) >= limit:
            break
        # Progreso por grupos
        if show_progress and groups_done % max(1, log_interval_groups) == 0:
            elapsed = time.perf_counter() - start_time
            pct = (groups_done / max(groups_total, 1)) * 100.0
            print(f"[PROGRESS] Grupos: {groups_done:,}/{groups_total:,} ({pct:0.1f}%) | Matches: {len(matches):,} | Tiempo: {elapsed:0.1f}s", flush=True)

    if not matches:
        print("[INFO] No se encontraron emparejamientos exactos en el límite solicitado.")
        return pd.DataFrame(columns=["Id banco","Cuenta bancaria","Referencia","Fecha","Monto","Código",
                                     "Id banco","Cuenta bancaria","Referencia","Fecha","Monto","Código","Conciliado"])

    # Construir salida con Conciliado=1
    s_cols = [c for c in SCHEMA_BASE]
    b_cols = [c for c in SCHEMA_BASE]

    s_part = df_s.set_index("s_idx").loc[[m[0] for m in matches]][s_cols].reset_index(drop=True)
    b_part = df_b.set_index("b_idx").loc[[m[1] for m in matches]][b_cols].reset_index(drop=True)

    out = pd.concat([s_part, b_part], axis=1)
    out["Conciliado"] = 1
    out.columns = (
        ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"] +
        ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"] +
        ["Conciliado"]
    )

    # Exportar
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    elapsed_total = time.perf_counter() - start_time
    print(f"[OK] Exportado sample (exact-only, limit={limit}): {OUTPUT_CSV} ({len(out)} filas)")
    print(f"[STATS] Tiempo total: {elapsed_total:0.2f}s | Filas sistema evaluadas: {rows_scanned:,} | Checks: {pair_checks:,} | Candidatos válidos: {candidates_valid:,} | Matches: {len(matches):,}")
    print("\n[PREVIEW] Primeras 10 filas (sample exact):")
    print(out.head(10).to_string(index=False))
    return out

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Emparejamiento exacto con límite (sample)")
    ap.add_argument("--limit", type=int, default=500, help="Número máximo de emparejamientos exactos a generar")
    ap.add_argument("--amount-tol", type=float, default=0.01, help="Tolerancia de monto")
    ap.add_argument("--date-window", type=int, default=1, help="Ventana de fechas en días")
    ap.add_argument("--ref-threshold", type=float, default=0.90, help="Umbral de similitud de referencia [0,1]")
    ap.add_argument("--no-require-same-bank", action="store_true", help="No exigir Id banco igual en exactos")
    ap.add_argument("--no-require-same-code", action="store_true", help="No exigir Código igual en exactos (se sigue bloqueando por cuenta y opcionalmente banco)")
    ap.add_argument("--log-interval-groups", type=int, default=20, help="Cada cuántos grupos imprimir progreso")
    ap.add_argument("--log-interval-rows", type=int, default=10000, help="Cada cuántas filas de sistema evaluadas imprimir progreso")
    ap.add_argument("--no-progress", action="store_true", help="Desactivar mensajes periódicos de progreso")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    params = MatchParams(
        amount_tolerance=args.amount_tol,
        date_window_days=args.date_window,
        ref_similarity_threshold=args.ref_threshold,
        require_same_bank_exact=not args.no_require_same_bank,
        require_same_trans_exact=not args.no_require_same_code,
    )
    # pasar parámetros de logging como atributos internos
    setattr(params, "_log_interval_groups", args.log_interval_groups)
    setattr(params, "_log_interval_rows", args.log_interval_rows)
    setattr(params, "_show_progress", not args.no_progress)
    build_pairs_sample(params, limit=args.limit)