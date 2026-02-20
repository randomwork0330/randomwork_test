"""
_3_TMPRSS2ERG/tmprss2erg_50_50.tsv 기반 GGA 파일 생성
==========================================================
단계:
  1) ERG와 absPearsonR >= 0.3 인 유전자 추출 (ERG 포함)
  2) 추출된 유전자들 사이의 모든 쌍(pair)에 대해 absPearsonR 계산
  3) absPearsonR >= 0.3 인 쌍만 남겨 gene_gene_asso.tsv 생성
  4) 저장 경로: _3_TMPRSS2ERG/__A__gene_gene_asso.tsv
"""

import pandas as pd
import numpy as np
from itertools import combinations
import time

TSV_IN  = (r"C:\Users\user\Desktop\CNN_CVPR\__DATAAn__\random_work_test"
           r"\_3_TMPRSS2ERG\tmprss2erg_50_50.tsv")
OUT_DIR = r"C:\Users\user\Desktop\CNN_CVPR\__DATAAn__\random_work_test\_3_TMPRSS2ERG"
OUT_FILE = OUT_DIR + r"\__A__gene_gene_asso.tsv"

THRESHOLD = 0.3   # absolute Pearson R 컷오프

# ── 1. 데이터 로드 ──────────────────────────────────────────────────────────
print("데이터 로드 중...")
t0 = time.time()
df_raw = pd.read_csv(TSV_IN, sep="\t", index_col=0)  # shape: (19928, 100)
print(f"  원본 shape: {df_raw.shape}  ({time.time()-t0:.1f}s)")

# 행=유전자, 열=샘플 → 수치 행렬 (float32 절약)
expr = df_raw.astype(np.float32)

# ── 2. ERG 벡터 추출 및 전체 유전자와 absPearsonR 계산 ──────────────────────
print("\n[Step 1] ERG와 모든 유전자 간 absPearsonR 계산...")
if "ERG" not in expr.index:
    raise ValueError("ERG 유전자가 데이터에 없습니다!")

erg_vec = expr.loc["ERG"].values  # shape: (100,)

# 행렬 연산으로 한 번에 Pearson R 계산
#   - 각 행을 평균 0, 표준편차 1로 표준화 후 내적 / n
X = expr.values  # (19928, 100)
n_samples = X.shape[1]

# 행별 표준화 (zscore)
mean_ = X.mean(axis=1, keepdims=True)
std_  = X.std(axis=1, keepdims=True)
std_[std_ == 0] = 1e-8  # 분산=0 보호

X_z = (X - mean_) / std_   # (19928, 100)

erg_z = (erg_vec - erg_vec.mean()) / (erg_vec.std() + 1e-8)  # (100,)

# Pearson R = (X_z @ erg_z) / n_samples
pearson_r = (X_z @ erg_z) / n_samples          # (19928,)
abs_r     = np.abs(pearson_r)

# ERG와 absPearsonR >= 0.3 인 유전자
genes_all  = expr.index.tolist()
mask_erg   = abs_r >= THRESHOLD
erg_corr_genes = [g for g, m in zip(genes_all, mask_erg) if m]

print(f"  ERG와 absPearsonR >= {THRESHOLD}: {len(erg_corr_genes)}개 유전자")

# ── 3. 추출된 유전자들 사이의 모든 쌍 absPearsonR 계산 ──────────────────────
print(f"\n[Step 2] {len(erg_corr_genes)}개 유전자 간 pairwise absPearsonR 계산...")
t1 = time.time()

# 서브 행렬 추출
sub_expr = expr.loc[erg_corr_genes].values.astype(np.float64)  # (N, 100)
N = len(erg_corr_genes)

# 전체 표준화
sub_mean = sub_expr.mean(axis=1, keepdims=True)
sub_std  = sub_expr.std(axis=1, keepdims=True)
sub_std[sub_std == 0] = 1e-8
sub_z = (sub_expr - sub_mean) / sub_std  # (N, 100)

# 전체 쌍 상관계수 행렬 = sub_z @ sub_z.T / n_samples
# N이 크면 메모리 주의 → N x N 행렬 생성
print(f"  N={N} -> corr matrix: {N}x{N} ~{N*N*8/1e6:.1f} MB")

corr_matrix = (sub_z @ sub_z.T) / n_samples  # (N, N), Pearson R
abs_corr    = np.abs(corr_matrix)

print(f"  상관행렬 계산 완료  ({time.time()-t1:.1f}s)")

# ── 4. 상위 삼각 행렬에서 absPearsonR >= 0.3 쌍 추출 ────────────────────────
print(f"\n[Step 3] absPearsonR >= {THRESHOLD} 인 쌍 추출 중...")
t2 = time.time()

# 자기 자신(대각선) 제외, 상위 삼각
rows_i, cols_j = np.triu_indices(N, k=1)
abs_vals = abs_corr[rows_i, cols_j]
raw_vals  = corr_matrix[rows_i, cols_j]

mask_pair = abs_vals >= THRESHOLD
pair_i    = rows_i[mask_pair]
pair_j    = cols_j[mask_pair]
pair_r    = raw_vals[mask_pair]
pair_absr = abs_vals[mask_pair]

print(f"  총 가능한 쌍: {len(rows_i):,}개")
print(f"  absPearsonR >= {THRESHOLD} 쌍: {len(pair_i):,}개  ({time.time()-t2:.1f}s)")

# ── 5. 결과 DataFrame 구성 ─────────────────────────────────────────────────
gene_arr = np.array(erg_corr_genes)

result_df = pd.DataFrame({
    "gene_A":         gene_arr[pair_i],
    "gene_B":         gene_arr[pair_j],
    "pearsonR":       np.round(pair_r, 6),
    "absPearsonR":    np.round(pair_absr, 6),
})

# absPearsonR 내림차순 정렬
result_df = result_df.sort_values("absPearsonR", ascending=False).reset_index(drop=True)

# ── 6. 저장 ────────────────────────────────────────────────────────────────
result_df.to_csv(OUT_FILE, sep="\t", index=False)

print(f"\n{'='*60}")
print(f"저장 완료: {OUT_FILE}")
print(f"{'='*60}")
print(f"  ERG 상관 유전자 수 : {len(erg_corr_genes):,}개 (absPearsonR≥{THRESHOLD})")
print(f"  최종 쌍 수         : {len(result_df):,}개")
print(f"  총 소요시간        : {time.time()-t0:.1f}초")
print(f"\n  absPearsonR 분포:")
print(f"    최대: {result_df['absPearsonR'].max():.4f}")
print(f"    평균: {result_df['absPearsonR'].mean():.4f}")
print(f"    중앙: {result_df['absPearsonR'].median():.4f}")
print(f"\n  상위 10개 쌍:")
print(result_df.head(10).to_string(index=False))
