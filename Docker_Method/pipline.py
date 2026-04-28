# 核心输入（必须项）
CORE_FEATURES = [
    "RIDAGEYR", "RIAGENDR", "BMXBMI",
    "BPXSY1", "BPXDI1",
    "LBXSCR", "URXUMA", "URXUCR",
    "LBXGLU", "DIQ010", "BPQ020"
]

# 辅助输入（可补充性能）
AUX_FEATURES = [
    "RIDRETH1", "LBXTC", "LBXTR", "LBXSNA",
    "LBXSCA", "LBXSCL", "LBXHGB", "MCQ160B",
    "SMQ020", "PAQ605"
]

# 其他特征（仅在完整 NHANES 时使用）
OPTIONAL_FEATURES = [
    "INDHHINC", "LBXGH", "LBXSPH", "LBXRBC"
]
def compute_egfr(scr, age, sex):
    kappa = 0.7 if sex == 'F' else 0.9
    alpha = -0.329 if sex == 'F' else -0.411
    sex_factor = 1.018 if sex == 'F' else 1
    return 141 * min(scr/kappa, 1)**alpha * max(scr/kappa, 1)**-1.209 * (0.993**age) * sex_factor

def label_ckd(df):
    df["eGFR"] = df.apply(lambda r: compute_egfr(r["LBXSCR"], r["RIDAGEYR"], 'F' if r["RIAGENDR"]==2 else 'M'), axis=1)
    df["UACR"] = df["URXUMA"] / df["URXUCR"]
    df["CKD"] = ((df["eGFR"] < 60) | (df["UACR"] >= 30)).astype(int)
    return df
