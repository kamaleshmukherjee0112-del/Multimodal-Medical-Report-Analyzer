import pandas as pd
import re
from typing import List

# --------------------------------------------------
# Constants
# --------------------------------------------------

CLINICAL_TEXT_VALUES = [
    "positive", "negative", "reactive", "non reactive",
    "detected", "not detected",
    "present", "absent",
    "nil", "normal", "abnormal"
]

COMMON_TEST_ABBREV = {
    "hb", "rbc", "wbc", "esr", "plt", "ast", "alt",
    "ldl", "hdl", "vldl", "tsh", "t3", "t4",
    "hba1c", "bun", "creatinine"
}

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def looks_clinical_text(val: str) -> bool:
    v = val.lower().strip()
    return any(k in v for k in CLINICAL_TEXT_VALUES)


def parse_float_safe(text):
    """
    Extract first numeric value from text safely.
    """
    try:
        return float(re.findall(r"[-+]?\d*\.?\d+", str(text))[0])
    except:
        return None


def parse_reference_range(ref):
    """
    Safely parse numeric reference ranges like:
    '40-50', '83 - 101', '00 - 02', '150000 - 410000'
    """
    nums = re.findall(r"\d+\.?\d*", str(ref))

    if len(nums) >= 2:
        low = float(nums[0])
        high = float(nums[1])

        # Safety: handle reversed OCR cases
        if low > high:
            low, high = high, low

        return low, high

    return None, None


# --------------------------------------------------
# Main cleaner
# --------------------------------------------------

def clean_lab_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OCR/Tabula lab table into a normalized format.
    Adds deterministic numeric flagging: Low / Normal / High.
    """

    if df is None or df.empty:
        return pd.DataFrame(
            columns=["test_name", "value", "unit", "reference_range", "flag"]
        )

    # 1️⃣ Drop completely empty rows & columns
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.dropna(axis=1, how="all")

    # 2️⃣ Header inference
    if isinstance(df.columns[0], (int, float)):
        header_row = df.iloc[0].fillna("").astype(str)
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = header_row

    # 3️⃣ Normalize column names
    col_map = {}
    for c in df.columns:
        c_norm = str(c).strip().lower()
        if any(k in c_norm for k in ["test", "investigation", "parameter", "analyte"]):
            col_map[c] = "test_name"
        elif any(k in c_norm for k in ["result", "value", "observed"]):
            col_map[c] = "value"
        elif "unit" in c_norm:
            col_map[c] = "unit"
        elif any(k in c_norm for k in ["ref", "range", "interval", "normal"]):
            col_map[c] = "reference_range"
        elif any(k in c_norm for k in ["flag", "remark"]):
            col_map[c] = "flag"
        else:
            col_map[c] = None

    df = df.rename(columns={o: n for o, n in col_map.items() if n})
    df = df[[c for c in df.columns if c in [
        "test_name", "value", "unit", "reference_range", "flag"
    ]]]

    if "test_name" not in df.columns:
        return pd.DataFrame(
            columns=["test_name", "value", "unit", "reference_range", "flag"]
        )

    # 4️⃣ Clean test names
    df["test_name"] = df["test_name"].astype(str).str.strip()
    df = df[df["test_name"] != ""]

    # 5️⃣ Remove repeated header rows inside table
    header_like = df["test_name"].str.lower().isin(
        ["test", "investigation", "parameter", "analyte"]
    )
    df = df[~header_like]

    # 6️⃣ Junk filtering
    junk_keywords: List[str] = [
        "diagnostic", "clinic", "hospital", "address",
        "phone", "mobile", "doctor", "consultant",
        "report id", "patient name", "sample",
        "generated on", "page", "thank you"
    ]

    name_lower = df["test_name"].str.lower()
    mask_junk_kw = name_lower.str.contains("|".join(junk_keywords), regex=True)

    # 6️⃣ Decide if row looks clinical
    value_str = df["value"].astype(str) if "value" in df.columns else pd.Series([""] * len(df))

    if "reference_range" in df.columns:
        ref_str = df["reference_range"].astype(str)
    else:
        ref_str = pd.Series([""] * len(df))

    has_digit = (
        value_str.str.contains(r"\d", regex=True) |
        ref_str.str.contains(r"\d", regex=True) |
        df["test_name"].str.contains(r"\d", regex=True)
    )

    clinical_text = value_str.apply(looks_clinical_text)
    short_but_valid = df["test_name"].str.lower().isin(COMMON_TEST_ABBREV)

    keep_row = has_digit | clinical_text | short_but_valid
    df = df[keep_row]


    # 7️⃣ Final cleanup of fields
    for col in ["value", "unit", "reference_range", "flag"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(str).str.strip()

    # 8️⃣ Basic numeric flagging (Low / Normal / High)
    for idx, row in df.iterrows():
        value = parse_float_safe(row.get("value"))
        low, high = parse_reference_range(row.get("reference_range"))

        # If comparison not possible, leave flag unchanged
        if value is None or low is None or high is None:
            continue

        if value < low:
            df.at[idx, "flag"] = "Low"
        elif value > high:
            df.at[idx, "flag"] = "High"
        else:
            df.at[idx, "flag"] = "Normal"

    return df.reset_index(drop=True)