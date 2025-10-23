from __future__ import annotations
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dtparser
from decimal import Decimal, InvalidOperation

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Set Datos Conciliacion"
OUTPUT_CSV = BASE_DIR / "Banco.csv"

SCHEMA_BASE = ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"]

BANK_TABLE = "PS_BANK_STMT_TBL"

COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "PS_BANK_STMT_TBL": {
        "BNK_ID_NBR": "Id banco",
        "BANK_ACCOUNT_NUM": "Cuenta bancaria",
        "RECON_REF_ID": "Referencia",
        "RECON_BANK_DT": "Fecha",
        "RECON_TRAN_AMT": "Monto",
        "RECON_TRANS_CODE": "Código",
    }
}

@dataclass
class ReadResult:
    table: str
    df: pd.DataFrame
    source: str  # filepath or file+sheet


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = _strip_accents(s).upper()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_account(x: object) -> str:
    if pd.isna(x):
        return ""
    # Manejar notación científica o tipos numéricos de Excel
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        try:
            s = format(Decimal(str(x)), "f")
            return re.sub(r"\D+", "", s)
        except Exception:
            return re.sub(r"\D+", "", str(x))
    s = str(x).strip()
    sci_pat = re.compile(r"^\s*\d+(?:\.\d+)?[eE][+\-]?\d+\s*$")
    if sci_pat.match(s):
        try:
            s = format(Decimal(s), "f")
        except (InvalidOperation, ValueError):
            pass
    # Mantener solo dígitos, conservando ceros a la izquierda
    s = re.sub(r"\D+", "", s)
    return s


def _parse_date(x: object) -> Optional[pd.Timestamp]:
    if pd.isna(x) or (isinstance(x, str) and x.strip() == ""):
        return None
    if isinstance(x, (pd.Timestamp, )):
        return pd.to_datetime(x)
    s = str(x).strip()
    for dayfirst in (True, False):
        try:
            dt = dtparser.parse(s, dayfirst=dayfirst)
            return pd.to_datetime(dt)
        except Exception:
            continue
    return None


def _norm_date_to_iso(x: object) -> str:
    dt = _parse_date(x)
    return "" if dt is None else dt.strftime("%Y-%m-%d")


def _parse_amount(x: object) -> Optional[float]:
    if pd.isna(x) or (isinstance(x, str) and x.strip() == ""):
        return None
    s = str(x).strip()
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) in (1, 2):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _read_csv(fp: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(fp, encoding=enc, dtype=str)
        except Exception:
            continue
    return pd.read_csv(fp, dtype=str)


def _read_any(filepath: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    ext = filepath.suffix.lower()
    if ext == ".csv":
        return _read_csv(filepath)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, sheet_name=sheet or 0, dtype=str)
    if ext == ".parquet":
        return pd.read_parquet(filepath)
    if ext == ".tsv":
        return pd.read_csv(filepath, sep="\t", dtype=str)
    raise ValueError(f"Extensión no soportada: {ext}")


def _find_sources(input_dir: Path, base_name: str) -> List[Tuple[str, Path, Optional[str]]]:
    """
    Devuelve lista de (table_name, filepath, sheet_name|None) para TODAS las variantes, e.g. PS_BANK_STMT_TBL_*
    Coincide por nombre de archivo y por nombre de hoja en Excel.
    """
    results: List[Tuple[str, Path, Optional[str]]] = []
    if not input_dir.exists():
        return results

    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet", ".tsv"}]
    pat = re.compile(re.escape(base_name), flags=re.IGNORECASE)

    for fp in files:
        # 1) Coincidencia por nombre de archivo
        if pat.search(fp.stem):
            results.append((base_name, fp, None))

        # 2) Coincidencia por nombre de hoja en Excel
        if fp.suffix.lower() in (".xlsx", ".xls"):
            try:
                xl = pd.ExcelFile(fp)
                for sheet_name in xl.sheet_names:
                    if pat.search(str(sheet_name)):
                        results.append((base_name, fp, str(sheet_name)))
            except Exception:
                continue

    # Deduplicar por (tabla, archivo, hoja)
    unique: List[Tuple[str, Path, Optional[str]]] = []
    seen = set()
    for tbl, fp, sheet in results:
        key = (tbl, fp.resolve(), sheet or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append((tbl, fp, sheet))
    return unique


def _rename_and_project(df: pd.DataFrame, table: str) -> pd.DataFrame:
    mapping = COLUMN_MAP[table]
    cols_inter = {src: dst for src, dst in mapping.items() if src in df.columns}
    df = df.rename(columns=cols_inter)

    # Asegurar columnas del esquema base
    for col in SCHEMA_BASE:
        if col not in df.columns:
            df[col] = pd.NA

    # Normalización
    df["Id banco"] = df["Id banco"].apply(_norm_text)
    df["Cuenta bancaria"] = df["Cuenta bancaria"].apply(_norm_account)
    df["Referencia"] = df["Referencia"].apply(_norm_text)
    df["Fecha"] = df["Fecha"].apply(_norm_date_to_iso)
    df["Monto"] = df["Monto"].apply(_parse_amount)
    df["Código"] = df["Código"].apply(_norm_text)

    df_std = df[SCHEMA_BASE].copy()
    df_std.insert(0, "source_table", table)
    return df_std


def build_banco() -> pd.DataFrame:
    sources = _find_sources(INPUT_DIR, BANK_TABLE)
    if not sources:
        print(f"[ADVERTENCIA] No se encontraron archivos para {BANK_TABLE} en: {INPUT_DIR}")
        return pd.DataFrame(columns=["source_table"] + SCHEMA_BASE)

    frames: List[pd.DataFrame] = []
    for table, fp, sheet in sources:
        try:
            df_raw = _read_any(fp, sheet)
            df_std = _rename_and_project(df_raw, table)
            frames.append(df_std)
            src_label = f"{fp.name}" if sheet is None else f"{fp.name}#{sheet}"
            print(f"[OK] Cargado {table} desde {src_label} ({len(df_std)} filas)")
        except Exception as ex:
            print(f"[ERROR] Falló lectura/mapeo de {table} en {fp}: {ex}")

    if not frames:
        return pd.DataFrame(columns=["source_table"] + SCHEMA_BASE)

    df_all = pd.concat(frames, ignore_index=True)

    # Deduplicar
    df_all = df_all.drop_duplicates(subset=SCHEMA_BASE + ["source_table"]).reset_index(drop=True)

    # Orden sugerido
    df_all = df_all.sort_values(by=["source_table", "Id banco", "Cuenta bancaria", "Fecha", "Referencia"], na_position="last").reset_index(drop=True)

    return df_all


def main():
    print(f"[INFO] Carpeta de entrada: {INPUT_DIR}")
    df_bnk = build_banco()
    if df_bnk.empty:
        print("[INFO] No hay datos para exportar.")
        sys.exit(0)

    # Previsualización
    print("\n[PREVIEW] Primeras 10 filas de Banco:")
    print(df_bnk.head(10).to_string(index=False))

    # Exportar CSV
    df_out = df_bnk.copy()
    df_out["Monto"] = df_out["Monto"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Exportado Banco a: {OUTPUT_CSV} ({len(df_out)} filas)")


if __name__ == "__main__":
    main()