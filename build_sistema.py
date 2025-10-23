"""Construcción de la hoja/archivo 'Sistema' a partir de múltiples tablas de PeopleSoft.

Este script:
- Detecta todas las variantes de las tablas del sistema (PS_BNK_RCN_TRAN, PS_CASH_FLOW_TR, PS_PAYMENT_TBL) en 'Set Datos Conciliacion'.
- Normaliza columnas al esquema base: Id banco, Cuenta bancaria, Referencia, Fecha, Monto, Código.
- Aplica limpieza de texto, fechas y montos; preserva ceros de cuentas.
- Deduplica y exporta 'Sistema.csv'.

Entradas:
- Carpeta: ./Set Datos Conciliacion
  - Archivos CSV/XLSX/Parquet/TSV cuyos nombres u hojas contengan los nombres base de tabla.

Salida:
- ./Sistema.csv (UTF-8 BOM)

Uso:
- pip install pandas python-dateutil openpyxl pyarrow
- python build_sistema.py
"""

from __future__ import annotations
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dtparser

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Set Datos Conciliacion"
OUTPUT_CSV = BASE_DIR / "Sistema.csv"

SCHEMA_BASE = ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"]

SYSTEM_TABLES = ["PS_BNK_RCN_TRAN", "PS_CASH_FLOW_TR", "PS_PAYMENT_TBL"]

COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "PS_BNK_RCN_TRAN": {
        "BNK_ID_NBR": "Id banco",
        "BANK_ACCOUNT_NUM": "Cuenta bancaria",
        "TRAN_REF_ID": "Referencia",
        "TRAN_DT": "Fecha",
        "TRAN_AMT": "Monto",
        "RECON_TRANS_CODE": "Código",
    },
    "PS_CASH_FLOW_TR": {
        "BNK_ID_NBR": "Id banco",
        "BANK_ACCOUNT_NUM": "Cuenta bancaria",
        "TR_SOURCE_ID": "Referencia",
        "BUSINESS_DATE": "Fecha",
        "AMOUNT": "Monto",
        "PYMNT_METHOD": "Código",
    },
    "PS_PAYMENT_TBL": {
        # "Id banco" puede no existir en export; se dejará vacío si no está
        "BANK_ACCOUNT_NUM": "Cuenta bancaria",
        # "Referencia" podría no estar disponible; se dejará vacío si no está
        "PYMNT_DT": "Fecha",
        "PYMNT_AMT": "Monto",
        "PYMNT_METHOD": "Código",
    },
}

@dataclass
class ReadResult:
    """Resultado de lectura de una fuente detectada.
    Attributes:
        table: Nombre base de la tabla (e.g., 'PS_BNK_RCN_TRAN').
        df: DataFrame cargado de la fuente.
        source: Ruta de archivo y, si aplica, nombre de hoja 'archivo.xlsx#Hoja'.
    """
    table: str
    df: pd.DataFrame
    source: str  # filepath or file+sheet


def _strip_accents(s: str) -> str:
    """Elimina acentos del string usando normalización Unicode."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _norm_text(x: object) -> str:
    """Normaliza texto: trim, mayúsculas, sin acentos y colapsa espacios."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = _strip_accents(s).upper()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_account(x: object) -> str:
    """Normaliza la cuenta bancaria dejando solo dígitos y conservando ceros a la izquierda."""
    if pd.isna(x):
        return ""
    s = re.sub(r"\D+", "", str(x))
    return s


def _parse_date(x: object) -> Optional[pd.Timestamp]:
    """Parsea fechas tolerando formatos comunes (dd/mm y mm/dd) y valores de Excel."""
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
    """Devuelve la fecha en formato ISO 'YYYY-MM-DD' o vacío si no es válida."""
    dt = _parse_date(x)
    return "" if dt is None else dt.strftime("%Y-%m-%d")


def _parse_amount(x: object) -> Optional[float]:
    """Convierte montos a float manejando separadores de miles/decimales y símbolos."""
    if pd.isna(x) or (isinstance(x, str) and x.strip() == ""):
        return None
    s = str(x).strip()
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", ".") if len(parts[-1]) in (1, 2) else s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _read_csv(fp: Path) -> pd.DataFrame:
    """Lee CSV probando codificaciones comunes para evitar errores por BOM/acentos."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(fp, encoding=enc, dtype=str)
        except Exception:
            continue
    return pd.read_csv(fp, dtype=str)


def _read_any(filepath: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    """Lee archivos CSV/XLSX/Parquet/TSV y retorna DataFrame de strings."""
    ext = filepath.suffix.lower()
    if ext in (".csv",):
        return _read_csv(filepath)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, sheet_name=sheet or 0, dtype=str)
    if ext in (".parquet",):
        return pd.read_parquet(filepath)
    if ext in (".tsv",):
        return pd.read_csv(filepath, sep="\t", dtype=str)
    raise ValueError(f"Extensión no soportada: {ext}")


def _find_sources(input_dir: Path, table_names: List[str]) -> List[Tuple[str, Path, Optional[str]]]:
    """Encuentra TODAS las fuentes que contengan el nombre base de cada tabla (archivos u hojas).
    Retorna tuplas (nombre_tabla, ruta_archivo, hoja|None).
    """
    results: List[Tuple[str, Path, Optional[str]]] = []
    if not input_dir.exists():
        return results

    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet", ".tsv"}]

    for fp in files:
        for tbl in table_names:
            pat = re.compile(re.escape(tbl), flags=re.IGNORECASE)
            # 1) Coincidencia por nombre de archivo (mantener TODOS los matches)
            if pat.search(fp.stem):
                results.append((tbl, fp, None))

            # 2) Coincidencia por nombre de hoja dentro de Excel (todas las que hagan match)
            if fp.suffix.lower() in (".xlsx", ".xls"):
                try:
                    xl = pd.ExcelFile(fp)
                    for sheet_name in xl.sheet_names:
                        if pat.search(str(sheet_name)):
                            results.append((tbl, fp, str(sheet_name)))
                except Exception:
                    continue

    # Deduplicar por (tabla, archivo, hoja) para evitar repeticiones
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
    """Renombra columnas según COLUMN_MAP, asegura el esquema base y normaliza valores."""
    mapping = COLUMN_MAP[table]
    # Renombrar intersección
    cols_inter = {src: dst for src, dst in mapping.items() if src in df.columns}
    df = df.rename(columns=cols_inter)

    # Asegurar columnas del esquema base
    for col in SCHEMA_BASE:
        if col not in df.columns:
            df[col] = pd.NA

    # Normalizaciones
    df["Id banco"] = df["Id banco"].apply(_norm_text)
    df["Cuenta bancaria"] = df["Cuenta bancaria"].apply(_norm_account)
    df["Referencia"] = df["Referencia"].apply(_norm_text)
    df["Fecha"] = df["Fecha"].apply(_norm_date_to_iso)
    df["Monto"] = df["Monto"].apply(_parse_amount)
    df["Código"] = df["Código"].apply(_norm_text)

    # Si Id banco quedó vacío para PS_PAYMENT_TBL, mantener como vacío
    if table == "PS_PAYMENT_TBL":
        # Asegurar tipo texto para "Id banco" y dejar "" donde NA
        df["Id banco"] = df["Id banco"].fillna("").astype(str)

    # Proyección al esquema base
    df_std = df[SCHEMA_BASE].copy()
    df_std.insert(0, "source_table", table)
    return df_std


def build_sistema() -> pd.DataFrame:
    """Construye el DataFrame unificado del Sistema aplicando mapeo, normalización y deduplicación."""
    sources = _find_sources(INPUT_DIR, SYSTEM_TABLES)
    if not sources:
        print(f"[ADVERTENCIA] No se encontraron archivos para {SYSTEM_TABLES} en: {INPUT_DIR}")
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
    """Punto de entrada: construye y exporta Sistema.csv, mostrando una previsualización."""
    print(f"[INFO] Carpeta de entrada: {INPUT_DIR}")
    df_sis = build_sistema()
    if df_sis.empty:
        print("[INFO] No hay datos para exportar.")
        sys.exit(0)

    # Previsualización
    print("\n[PREVIEW] Primeras 10 filas de Sistema:")
    print(df_sis.head(10).to_string(index=False))

    # Exportar CSV
    df_out = df_sis.copy()
    # Convertir monto a string con punto decimal para salida consistente
    df_out["Monto"] = df_out["Monto"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Exportado Sistema a: {OUTPUT_CSV} ({len(df_out)} filas)")


if __name__ == "__main__":
    main()