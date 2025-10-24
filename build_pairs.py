"""Interfaz de consola para el proceso de conciliación por pasos.

Paso 1: Combina Sistema.csv y Banco.csv en un archivo único sin emparejamiento
Paso 2: Realiza SOLO el emparejamiento/matching de pares

Uso:
- python build_pairs.py --step 1  # Genera archivo combinado
- python build_pairs.py --step 2  # Realiza matching simple
- python build_pairs.py --help    # Muestra ayuda
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dateutil import parser as dtparser
from rapidfuzz.fuzz import token_set_ratio

BASE_DIR = Path(__file__).resolve().parent
SISTEMA_CSV = BASE_DIR / "Sistema.csv"
BANCO_CSV = BASE_DIR / "Banco.csv"
COMBINED_CSV = BASE_DIR / "Sistema + Banco (combined).csv"
MATCHED_CSV = BASE_DIR / "Sistema + Banco (matched).csv"

SCHEMA_BASE = ["Id banco", "Cuenta bancaria", "Referencia", "Fecha", "Monto", "Código"]

def print_banner():
    """Muestra banner del programa."""
    print("=" * 60)
    print("    SISTEMA DE CONCILIACIÓN BANCARIA - PROCESAMIENTO POR PASOS")
    print("=" * 60)

def check_input_files() -> bool:
    """Verifica que existan los archivos de entrada."""
    missing = []
    if not SISTEMA_CSV.exists():
        missing.append(str(SISTEMA_CSV))
    if not BANCO_CSV.exists():
        missing.append(str(BANCO_CSV))
    
    if missing:
        print(f"[ERROR] Archivos faltantes: {', '.join(missing)}")
        print("Ejecuta primero build_sistema.py y build_banco.py")
        return False
    return True

def step1_combine_files():
    """Paso 1: Combina Sistema.csv y Banco.csv sin emparejamiento."""
    print("\n[PASO 1] Combinando archivos Sistema.csv y Banco.csv...")
    
    if not check_input_files():
        return False
    
    start_time = time.perf_counter()
    
    # Leer archivos
    print(f"[INFO] Leyendo {SISTEMA_CSV.name}...")
    df_sistema = pd.read_csv(SISTEMA_CSV, dtype=str)
    print(f"[INFO] Sistema: {len(df_sistema):,} filas")
    
    print(f"[INFO] Leyendo {BANCO_CSV.name}...")
    df_banco = pd.read_csv(BANCO_CSV, dtype=str)
    print(f"[INFO] Banco: {len(df_banco):,} filas")
    
    # Preparar estructura combinada como en tu ejemplo
    print("[INFO] Preparando estructura combinada...")
    
    # Crear filas de Sistema (lado izquierdo con datos, lado derecho vacío)
    sistema_rows = []
    for _, row in df_sistema.iterrows():
        sistema_row = list(row[SCHEMA_BASE]) + [""] * len(SCHEMA_BASE) + [0]  # Conciliado = 0
        sistema_rows.append(sistema_row)
    
    # Crear filas de Banco (lado izquierdo vacío, lado derecho con datos)
    banco_rows = []
    for _, row in df_banco.iterrows():
        banco_row = [""] * len(SCHEMA_BASE) + list(row[SCHEMA_BASE]) + [0]  # Conciliado = 0
        banco_rows.append(banco_row)
    
    # Combinar todo
    all_rows = sistema_rows + banco_rows
    
    # Crear DataFrame final
    columns = SCHEMA_BASE + SCHEMA_BASE + ["Conciliado"]
    output_df = pd.DataFrame(all_rows, columns=columns)
    
    # Exportar
    output_df.to_csv(COMBINED_CSV, index=False, encoding="utf-8-sig")
    
    elapsed = time.perf_counter() - start_time
    print(f"\n[OK] Archivo combinado creado: {COMBINED_CSV}")
    print(f"[INFO] Total de filas: {len(output_df):,}")
    print(f"[INFO] - Filas Sistema: {len(df_sistema):,}")
    print(f"[INFO] - Filas Banco: {len(df_banco):,}")
    print(f"[INFO] Tiempo: {elapsed:.2f}s")
    
    # Previsualización
    print("\n[PREVIEW] Primeras 5 filas del archivo combinado:")
    print(output_df.head(5).to_string(index=False))
    
    return True

def _to_float(x) -> Optional[float]:
    """Convierte string a float."""
    if pd.isna(x) or str(x).strip() == "":
        return None
    try:
        return float(str(x).replace(",", ""))
    except:
        return None

def _to_date(x) -> Optional[pd.Timestamp]:
    """Convierte string a fecha."""
    if pd.isna(x) or str(x).strip() == "":
        return None
    for dayfirst in (True, False):
        try:
            return pd.to_datetime(dtparser.parse(str(x).strip(), dayfirst=dayfirst))
        except:
            continue
    return None

def _norm_text(s: object) -> str:
    """Normaliza texto."""
    if pd.isna(s):
        return ""
    return str(s).strip().upper()

def _ref_similarity(a: str, b: str) -> float:
    """Calcula similitud entre referencias."""
    if not a or not b:
        return 0.0
    return token_set_ratio(a, b) / 100.0

def step2_match_pairs(amount_tol: float = 0.01, date_window: int = 1, ref_threshold: float = 0.85):
    """Paso 2: Realiza SOLO el matching/emparejamiento de pares."""
    print(f"\n[PASO 2] Realizando matching de pares...")
    print(f"[PARAMS] Tolerancia monto: {amount_tol}, Ventana fecha: {date_window} días, Umbral referencia: {ref_threshold}")
    
    if not COMBINED_CSV.exists():
        print(f"[ERROR] No existe {COMBINED_CSV}")
        print("Ejecuta primero: python build_pairs.py --step 1")
        return False
    
    start_time = time.perf_counter()
    
    # Leer archivo combinado
    print(f"[INFO] Leyendo {COMBINED_CSV.name}...")
    df_combined = pd.read_csv(COMBINED_CSV, dtype=str)
    print(f"[INFO] Total filas: {len(df_combined):,}")
    
    # Separar Sistema y Banco basado en datos no vacíos
    # Sistema: primeras 6 columnas tienen datos, segundas 6 están vacías
    sistema_mask = (
        df_combined.iloc[:, 0].notna() & (df_combined.iloc[:, 0] != "") &
        df_combined.iloc[:, 1].notna() & (df_combined.iloc[:, 1] != "")
    ) & (
        (df_combined.iloc[:, 6].isna() | (df_combined.iloc[:, 6] == "")) &
        (df_combined.iloc[:, 7].isna() | (df_combined.iloc[:, 7] == ""))
    )
    
    # Banco: segundas 6 columnas tienen datos, primeras 6 están vacías  
    banco_mask = (
        df_combined.iloc[:, 6].notna() & (df_combined.iloc[:, 6] != "") &
        df_combined.iloc[:, 7].notna() & (df_combined.iloc[:, 7] != "")
    ) & (
        (df_combined.iloc[:, 0].isna() | (df_combined.iloc[:, 0] == "")) &
        (df_combined.iloc[:, 1].isna() | (df_combined.iloc[:, 1] == ""))
    )
    
    df_sistema = df_combined[sistema_mask].copy()
    df_banco = df_combined[banco_mask].copy()
    
    print(f"[INFO] Filas Sistema identificadas: {len(df_sistema):,}")
    print(f"[INFO] Filas Banco identificadas: {len(df_banco):,}")
    
    if df_sistema.empty or df_banco.empty:
        print("[WARNING] No hay datos suficientes para emparejamiento")
        return False
    
    # Preparar datos para matching (usar solo las columnas relevantes)
    print("[INFO] Preparando datos para matching...")
    
    # Extraer y normalizar datos de Sistema
    s_data = df_sistema.iloc[:, :6].copy()
    s_data.columns = SCHEMA_BASE
    s_data["s_idx"] = df_sistema.index
    s_data["Fecha_dt"] = s_data["Fecha"].apply(_to_date)
    s_data["Monto_f"] = s_data["Monto"].apply(_to_float)
    s_data["Referencia"] = s_data["Referencia"].apply(_norm_text)
    s_data["Cuenta bancaria"] = s_data["Cuenta bancaria"].apply(lambda x: "".join(c for c in str(x) if c.isdigit()) if pd.notna(x) else "")
    
    # Extraer y normalizar datos de Banco
    b_data = df_banco.iloc[:, 6:12].copy()
    b_data.columns = SCHEMA_BASE
    b_data["b_idx"] = df_banco.index
    b_data["Fecha_dt"] = b_data["Fecha"].apply(_to_date)
    b_data["Monto_f"] = b_data["Monto"].apply(_to_float)
    b_data["Referencia"] = b_data["Referencia"].apply(_norm_text)
    b_data["Cuenta bancaria"] = b_data["Cuenta bancaria"].apply(lambda x: "".join(c for c in str(x) if c.isdigit()) if pd.notna(x) else "")
    
    # Filtrar válidos (que tengan monto y fecha)
    s_valid = s_data[s_data["Monto_f"].notna() & s_data["Fecha_dt"].notna()].copy()
    b_valid = b_data[b_data["Monto_f"].notna() & b_data["Fecha_dt"].notna()].copy()
    
    print(f"[INFO] Sistema válido para matching: {len(s_valid):,}")
    print(f"[INFO] Banco válido para matching: {len(b_valid):,}")
    
    # Realizar matching simple
    print("[INFO] Buscando matches...")
    matches = []
    used_banco = set()
    processed = 0
    
    for _, s_row in s_valid.iterrows():
        processed += 1
        if processed % 1000 == 0:
            print(f"[PROGRESS] Procesadas {processed:,}/{len(s_valid):,} filas Sistema, matches: {len(matches):,}")
        
        # Buscar candidatos en la misma cuenta bancaria
        candidates = b_valid[
            (b_valid["Cuenta bancaria"] == s_row["Cuenta bancaria"]) &
            (~b_valid["b_idx"].isin(used_banco))
        ]
        
        if candidates.empty:
            continue
        
        best_match = None
        best_score = 0
        
        for _, b_row in candidates.iterrows():
            # Verificar tolerancia de monto
            amt_diff = abs(s_row["Monto_f"] - b_row["Monto_f"])
            if amt_diff > amount_tol:
                continue
            
            # Verificar ventana de fecha
            date_diff = abs((s_row["Fecha_dt"] - b_row["Fecha_dt"]).days)
            if date_diff > date_window:
                continue
            
            # Verificar similitud de referencia
            ref_sim = _ref_similarity(s_row["Referencia"], b_row["Referencia"])
            if ref_sim < ref_threshold:
                continue
            
            # Calcular score simple
            score = 100 + (1 - amt_diff/max(amount_tol, 0.01)) * 10 + ref_sim * 10
            
            if score > best_score:
                best_score = score
                best_match = b_row["b_idx"]
        
        if best_match is not None:
            matches.append((s_row["s_idx"], best_match, best_score))
            used_banco.add(best_match)
    
    print(f"[INFO] Matches encontrados: {len(matches)}")
    
    # Aplicar matches SOLO copiando datos del banco a sistema y marcando conciliado
    print("[INFO] Aplicando matches...")
    result_df = df_combined.copy()
    
    for s_idx, b_idx, score in matches:
        # Copiar datos del banco a la fila del sistema (columnas 6-11)
        for i in range(6):
            result_df.iloc[s_idx, 6 + i] = df_combined.iloc[b_idx, 6 + i]
        
        # Marcar como conciliado
        result_df.iloc[s_idx, -1] = 1
        
        # Eliminar la fila del banco (ya fue emparejada)
        result_df = result_df.drop(b_idx)
    
    # Exportar resultado
    result_df.to_csv(MATCHED_CSV, index=False, encoding="utf-8-sig")
    
    elapsed = time.perf_counter() - start_time
    print(f"\n[OK] Matching completado: {MATCHED_CSV}")
    print(f"[INFO] Total filas resultado: {len(result_df):,}")
    print(f"[INFO] Matches realizados: {len(matches):,}")
    print(f"[INFO] Tiempo: {elapsed:.2f}s")
    
    # Previsualización
    matched_rows = result_df[result_df["Conciliado"] == 1]
    if not matched_rows.empty:
        print(f"\n[PREVIEW] Primeras 3 filas emparejadas:")
        print(matched_rows.head(3).to_string(index=False))
    
    unmatched_sistema = len(result_df[(result_df.iloc[:, 0].notna()) & (result_df.iloc[:, 0] != "") & (result_df["Conciliado"] == 0)])
    unmatched_banco = len(result_df[(result_df.iloc[:, 6].notna()) & (result_df.iloc[:, 6] != "") & (result_df["Conciliado"] == 0)])
    
    print(f"\n[STATS] Sin emparejar -> Sistema: {unmatched_sistema:,}, Banco: {unmatched_banco:,}")
    
    return True

def main():
    """Función principal con interfaz de consola."""
    parser = argparse.ArgumentParser(
        description="Procesamiento de conciliación bancaria por pasos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python build_pairs.py --step 1                    # Crear archivo combinado
  python build_pairs.py --step 2                    # Realizar matching
  python build_pairs.py --step 2 --amount-tol 0.05  # Con tolerancia personalizada
        """
    )
    
    parser.add_argument("--step", type=int, choices=[1, 2], required=True,
                       help="Paso a ejecutar: 1=Combinar archivos, 2=Matching")
    
    # Parámetros para paso 2
    parser.add_argument("--amount-tol", type=float, default=0.01,
                       help="Tolerancia para diferencias de monto (default: 0.01)")
    parser.add_argument("--date-window", type=int, default=1,
                       help="Ventana de días para fechas (default: 1)")
    parser.add_argument("--ref-threshold", type=float, default=0.85,
                       help="Umbral de similitud para referencias 0-1 (default: 0.85)")
    
    args = parser.parse_args()
    
    print_banner()
    
    success = False
    if args.step == 1:
        success = step1_combine_files()
    elif args.step == 2:
        success = step2_match_pairs(
            amount_tol=args.amount_tol,
            date_window=args.date_window,
            ref_threshold=args.ref_threshold
        )
    
    if success:
        print(f"\n[SUCCESS] Paso {args.step} completado exitosamente")
        if args.step == 1:
            print(f"Siguiente paso: python build_pairs.py --step 2")
    else:
        print(f"\n[ERROR] Paso {args.step} falló")
        sys.exit(1)

if __name__ == "__main__":
    main()