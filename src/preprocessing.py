"""
Preprocessing utilities for DANE birth records dataset (Nacimientos 2018).

DANE PESO_NAC encoding (categorical codes, NOT actual grams):
    1: < 1 000 g     (extreme low birth weight)
    2: 1 000-1 499 g (very low birth weight)
    3: 1 500-1 999 g (low birth weight)
    4: 2 000-2 499 g (low birth weight)
    5: 2 500-2 999 g (normal)
    6: 3 000-3 499 g (normal)
    7: 3 500-3 999 g (normal)
    8: 4 000-4 499 g (large)
    9: >= 4 500 g / Sin informacion

BAJO_PESO = 1 when PESO_NAC in {1, 2, 3, 4}  (birth weight < 2 500 g)
CESAREA   = 1 when TIPO_PARTO == 2
"""

from pathlib import Path

import pandas as pd

# ── Encoding constants ─────────────────────────────────────────────────────────

LOW_WEIGHT_CODES: set[int] = {1, 2, 3, 4}

TARGET_CESAREA = "CESAREA"
TARGET_BAJO_PESO = "BAJO_PESO"

# ── Column groups ──────────────────────────────────────────────────────────────

COLUMNS_TO_DROP = [
    "OTRO_SIT",        # 99.7% null — no information value
    "FECHA_NACM",      # 47.4% null — not predictive
    "COD_MUNIC",       # too granular; use COD_DPTO / CODPTORE
    "CODMUNRE",        # duplicate of COD_MUNIC for residence
    "APGAR1",          # measured after birth
    "APGAR2",          # measured after birth
    "IDHEMOCLAS",      # blood type — marginal predictive value
    "IDFACTORRH",      # blood type — marginal predictive value
    "EST_CIVM",        # civil status — administrative, not clinical
    "NIV_EDUM",        # education level — secondary predictor
    "ULTCURMAD",       # last grade — secondary predictor
    "CODPRES",         # healthcare provider — too granular
    "NIV_EDUP",        # father education — secondary predictor
    "ULTCURPAD",       # father last grade — secondary predictor
    "T_GES_AGRU_CIE",  # ICD-10 grouped version of T_GES — redundant
    "IDPERTET",        # ethnic group — sensitivity concern
    "ANO",             # year — constant (2018)
    "MES",             # month of birth — not clinically relevant
    "SEG_SOCIAL",      # social security — redundant with IDCLASADMI
    "SIT_PARTO",       # delivery situation — administrative
    "ATEN_PAR",        # birth attendant — not available pre-birth
    "TALLA_NAC",       # birth length — measured at birth, correlated with PESO_NAC
    "AREANAC",         # birth area — use AREA_RES (residence) instead
]

NUMERIC_FEATURES = ["EDAD_MADRE", "T_GES", "NUMCONSUL", "N_HIJOSV", "N_EMB"]
CATEGORICAL_FEATURES = ["AREA_RES", "IDCLASADMI", "SEXO", "MUL_PARTO", "CODPTORE"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── I/O helpers ────────────────────────────────────────────────────────────────


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw DANE nacimientos CSV with proper encoding."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"Raw dataset loaded — {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ── Cleaning pipeline ──────────────────────────────────────────────────────────


def _drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=present)


def _impute_administrative_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill geographic/administrative nulls with -1 (unknown category)."""
    for col in ["IDCLASADMI", "CODPTORE", "AREA_RES"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric conversion for columns with mixed types."""
    for col in ["PESO_NAC", "EDAD_MADRE", "T_GES", "NUMCONSUL", "N_HIJOSV", "N_EMB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records with invalid administrative codes.

    DANE categorical encodings (all integer codes, NOT actual measurements):
    - PESO_NAC  : 1-9 (weight groups)
    - EDAD_MADRE: 1-9 (age groups; 1=<15y, 2=15-19y, 3=20-24y, ... 9=50+y)
    - T_GES     : 1-6, 9 (gestational week groups; 4=37-41w is most common)
    - TIPO_PARTO: 1=espontaneo, 2=cesarea, 3=instrumental
    """
    n0 = len(df)

    # Valid delivery types (1: spontaneous, 2: C-section, 3: instrumental)
    if "TIPO_PARTO" in df.columns:
        df = df[df["TIPO_PARTO"].isin([1, 2, 3])]

    # Valid DANE categorical weight codes 1-9
    if "PESO_NAC" in df.columns:
        df = df[df["PESO_NAC"].between(1, 9)]

    # Valid DANE maternal age codes 1-9 (NOT actual years)
    if "EDAD_MADRE" in df.columns:
        df = df[df["EDAD_MADRE"].between(1, 9)]

    # Valid sex codes
    if "SEXO" in df.columns:
        df = df[df["SEXO"].isin([1, 2])]

    # Exclude N_EMB code 99 (sin informacion)
    if "N_EMB" in df.columns:
        df = df[df["N_EMB"] != 99]

    n1 = len(df)
    print(f"After validity filters — {n1:,} rows retained ({n0 - n1:,} removed)")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full cleaning pipeline to raw birth records."""
    df = df.copy()
    # Drop row with single null PROFESION before dropping the column
    if "PROFESION" in df.columns:
        df = df.dropna(subset=["PROFESION"])
        df = df.drop(columns=["PROFESION"])
    df = _drop_irrelevant_columns(df)
    df = _impute_administrative_nulls(df)
    df = _coerce_numeric_columns(df)
    df = _filter_valid_records(df)
    return df.reset_index(drop=True)


# ── Feature engineering ────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables and clinically meaningful derived features.

    EDAD_MADRE DANE codes: 1=<15y, 2=15-19y, 3=20-24y, 4=25-29y,
                           5=30-34y, 6=35-39y, 7=40-44y, 8=45-49y, 9=50+y
    T_GES DANE codes     : 2=22-27w, 3=28-36w, 4=37-41w, 5=42+w, 9=unknown
    """
    df = df.copy()

    # Target 1: C-section (1) vs other delivery (0)
    df[TARGET_CESAREA] = (df["TIPO_PARTO"] == 2).astype(int)

    # Target 2: Low birth weight — PESO_NAC code <= 4 means < 2 500 g
    df[TARGET_BAJO_PESO] = df["PESO_NAC"].isin(LOW_WEIGHT_CODES).astype(int)

    # High-risk age: code 1 (< 15y) or codes 7-9 (>= 40y)
    df["EDAD_RIESGO"] = df["EDAD_MADRE"].isin({1, 7, 8, 9}).astype(int)

    # Preterm birth: T_GES codes 2-3 (<37 weeks)
    if "T_GES" in df.columns:
        df["PREMATURO"] = df["T_GES"].isin({1, 2, 3}).astype(int)

    # First pregnancy
    df["PRIMIGESTA"] = (df["N_EMB"] == 1).astype(int)

    return df


# ── Full pipeline ──────────────────────────────────────────────────────────────


def build_dataset(raw_path: str | Path) -> pd.DataFrame:
    """End-to-end: load raw CSV -> clean -> engineer features."""
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    print(
        f"\nTarget distribution:\n"
        f"  {TARGET_CESAREA:12s} — {df[TARGET_CESAREA].mean():.1%} positive\n"
        f"  {TARGET_BAJO_PESO:12s} — {df[TARGET_BAJO_PESO].mean():.1%} positive"
    )
    return df
