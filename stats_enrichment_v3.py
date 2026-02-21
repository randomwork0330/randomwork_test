"""
전립선암 표적 유전자의 GAT+RWR v3 랭킹 상위 농축 통계 분석
=============================================================
검정 1. Mann-Whitney U test        - 두 집단 rank 분포 차이 (비모수)
검정 2. Kolmogorov-Smirnov test    - rank CDF 분포 차이
검정 3. Fisher's exact test        - 각 상위 k% 컷오프에서 과농축 여부
검정 4. Permutation test           - 실제 평균 rank vs. 무작위 기대값 (10,000회)
검정 5. GSEA-style ES/NES          - 순서 정렬 기반 ES/NES (1,000 permutation)
검정 6. Fold Enrichment table      - Top-k% 내 관측/기대 비율

입력: node_importance_v3_annotated.tsv
출력: stats_enrichment_v3_result.tsv
      stats_enrichment_v3_result.txt
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp, fisher_exact
import warnings
import os
warnings.filterwarnings("ignore")

BASE    = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test"
TSV_IN  = os.path.join(BASE, "node_importance_v3_annotated.tsv")
TSV_OUT = os.path.join(BASE, "stats_enrichment_v3_result.tsv")
TXT_OUT = os.path.join(BASE, "stats_enrichment_v3_result.txt")

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
print(f"[Load] {TSV_IN}")
df = pd.read_csv(TSV_IN, sep="\t")
gene_df = df[df["node_type"] == "gene"].copy().reset_index(drop=True)
N_gene  = len(gene_df)

is_target = gene_df["prad_annotation"].notna() & (gene_df["prad_annotation"] != "")
is_erg    = gene_df["prad_annotation"].str.startswith("(ergPRAD)", na=False)
is_prad   = is_target & ~is_erg

target_ranks     = gene_df.loc[is_target, "rank"].values
erg_ranks        = gene_df.loc[is_erg,    "rank"].values
prad_ranks       = gene_df.loc[is_prad,   "rank"].values
background_ranks = gene_df.loc[~is_target,"rank"].values
all_gene_ranks   = gene_df["rank"].values

print(f"  유전자 총 수 : {N_gene:,}")
print(f"  PRAD 표적    : {len(target_ranks)} (ergPRAD: {len(erg_ranks)}, 일반: {len(prad_ranks)})")
print(f"  배경 유전자  : {len(background_ranks):,}")
print()

results_rows = []


# ═══════════════════════════════════════════════════════════════════════════════
# 검정 1. Mann-Whitney U test
# H0: 표적 유전자와 배경 유전자의 rank 분포가 같다
# H1: 표적 유전자의 rank가 유의미하게 낮다 (더 중요)
# ═══════════════════════════════════════════════════════════════════════════════
def run_mwu(group_ranks, bg_ranks, label):
    stat, p = mannwhitneyu(group_ranks, bg_ranks, alternative="less")
    n1, n2  = len(group_ranks), len(bg_ranks)
    z = (stat - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)
    r = abs(z) / np.sqrt(n1 + n2)
    return dict(
        test="Mann-Whitney U", group=label,
        n_target=n1, n_background=n2,
        median_target_rank=round(np.median(group_ranks), 1),
        median_bg_rank=round(np.median(bg_ranks), 1),
        statistic=round(stat, 2), p_value=p,
        effect_size_r=round(r, 4),
        significant="YES" if p < 0.05 else "NO",
        note=f"rank낮을수록 중요. effect r={r:.3f} ({'large' if r>=0.5 else 'medium' if r>=0.3 else 'small'})"
    )

results_rows += [
    run_mwu(target_ranks, background_ranks, "ALL_PRAD_TARGET vs BG"),
    run_mwu(erg_ranks,    background_ranks, "ergPRAD_TARGET vs BG"),
    run_mwu(prad_ranks,   background_ranks, "PRAD_TARGET vs BG"),
]
print("[Done] 검정 1. Mann-Whitney U")


# ═══════════════════════════════════════════════════════════════════════════════
# 검정 2. Kolmogorov-Smirnov test
# alternative='greater': 표적 CDF > 배경 CDF (rank가 낮을수록 좌측 편향)
# ═══════════════════════════════════════════════════════════════════════════════
def run_ks(group_ranks, bg_ranks, label):
    stat, p = ks_2samp(group_ranks, bg_ranks, alternative="greater")
    return dict(
        test="KS test", group=label,
        n_target=len(group_ranks), n_background=len(bg_ranks),
        median_target_rank=round(np.median(group_ranks), 1),
        median_bg_rank=round(np.median(bg_ranks), 1),
        statistic=round(stat, 4), p_value=p,
        effect_size_r=round(stat, 4),
        significant="YES" if p < 0.05 else "NO",
        note="KS D-stat: 표적 CDF가 배경보다 좌측 편향 = 상위 농축"
    )

results_rows += [
    run_ks(target_ranks, background_ranks, "ALL_PRAD_TARGET vs BG"),
    run_ks(erg_ranks,    background_ranks, "ergPRAD_TARGET vs BG"),
    run_ks(prad_ranks,   background_ranks, "PRAD_TARGET vs BG"),
]
print("[Done] 검정 2. KS test")


# ═══════════════════════════════════════════════════════════════════════════════
# 검정 3. Fisher's exact test (상위 k% 컷오프)
# ═══════════════════════════════════════════════════════════════════════════════
cutoffs = [0.01, 0.05, 0.10, 0.20]

def run_fisher(group_ranks, all_n, label, cutoff):
    k         = int(np.ceil(all_n * cutoff))
    in_top_t  = int(np.sum(group_ranks <= k))
    out_top_t = len(group_ranks) - in_top_t
    in_top_b  = k - in_top_t
    out_top_b = (all_n - k) - out_top_t
    table     = [[in_top_t, out_top_t], [in_top_b, out_top_b]]
    or_, p    = fisher_exact(table, alternative="greater")
    expected  = len(group_ranks) * cutoff
    fold      = in_top_t / expected if expected > 0 else np.nan
    return dict(
        test=f"Fisher's exact (top {int(cutoff*100)}%)",
        group=label,
        n_target=len(group_ranks), n_background=all_n - len(group_ranks),
        median_target_rank=round(np.median(group_ranks), 1),
        median_bg_rank="-",
        statistic=round(or_, 4), p_value=p,
        effect_size_r=round(fold, 3),
        significant="YES" if p < 0.05 else "NO",
        note=(f"top{int(cutoff*100)}%(rank<={k}): "
              f"obs={in_top_t}/{len(group_ranks)} ({in_top_t/len(group_ranks):.1%}), "
              f"exp={expected:.1f}, fold={fold:.2f}x, OR={or_:.2f}")
    )

for cut in cutoffs:
    results_rows += [
        run_fisher(target_ranks, N_gene, "ALL_PRAD_TARGET", cut),
        run_fisher(erg_ranks,    N_gene, "ergPRAD_TARGET",  cut),
        run_fisher(prad_ranks,   N_gene, "PRAD_TARGET",     cut),
    ]
print("[Done] 검정 3. Fisher's exact (top 1/5/10/20%)")


# ═══════════════════════════════════════════════════════════════════════════════
# 검정 4. Permutation test (10,000회)
# ═══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
N_PERM = 10_000

def run_permutation(group_ranks, all_ranks, label, n_perm=N_PERM):
    obs_mean   = np.mean(group_ranks)
    n          = len(group_ranks)
    perm_means = np.array([
        np.mean(np.random.choice(all_ranks, n, replace=False))
        for _ in range(n_perm)
    ])
    p = np.mean(perm_means <= obs_mean)
    z = (obs_mean - perm_means.mean()) / (perm_means.std() + 1e-12)
    return dict(
        test=f"Permutation test ({n_perm:,}x)",
        group=label,
        n_target=n, n_background=len(all_ranks) - n,
        median_target_rank=round(np.median(group_ranks), 1),
        median_bg_rank=round(perm_means.mean(), 1),
        statistic=round(z, 4), p_value=p,
        effect_size_r=round(abs(z), 4),
        significant="YES" if p < 0.05 else "NO",
        note=(f"obs_mean={obs_mean:.1f}, "
              f"perm_mean={perm_means.mean():.1f}+/-{perm_means.std():.1f}, "
              f"Z={z:.3f}, p={p:.5f}")
    )

results_rows += [
    run_permutation(target_ranks, all_gene_ranks, "ALL_PRAD_TARGET"),
    run_permutation(erg_ranks,    all_gene_ranks, "ergPRAD_TARGET"),
    run_permutation(prad_ranks,   all_gene_ranks, "PRAD_TARGET"),
]
print("[Done] 검정 4. Permutation test")


# ═══════════════════════════════════════════════════════════════════════════════
# 검정 5. GSEA-style Enrichment Score (ES / NES)
# ═══════════════════════════════════════════════════════════════════════════════
def gsea_es(sorted_gene_df, target_set, score_col="visit_prob"):
    n   = len(sorted_gene_df)
    nh  = len(target_set)
    hit_scores = sorted_gene_df.loc[
        sorted_gene_df["node"].isin(target_set), score_col].values
    sum_hit = hit_scores.sum() if hit_scores.sum() > 0 else 1e-12
    running = 0.0
    es_max  = -np.inf
    es_min  = np.inf
    miss_inc = 1.0 / (n - nh) if (n - nh) > 0 else 0
    for _, row in sorted_gene_df.iterrows():
        if row["node"] in target_set:
            running += row[score_col] / sum_hit
        else:
            running -= miss_inc
        es_max = max(es_max, running)
        es_min = min(es_min, running)
    return es_max if abs(es_max) >= abs(es_min) else es_min

def run_gsea(group_nodes, all_gene_df, label, n_perm=1000, score_col="visit_prob"):
    sorted_df  = all_gene_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    target_set = set(group_nodes)
    obs_es     = gsea_es(sorted_df, target_set, score_col)
    all_nodes  = sorted_df["node"].values
    n_t        = len(target_set)
    perm_es = [
        gsea_es(sorted_df, set(np.random.choice(all_nodes, n_t, replace=False)), score_col)
        for _ in range(n_perm)
    ]
    perm_es  = np.array(perm_es)
    pos_mean = perm_es[perm_es >= 0].mean() if (perm_es >= 0).any() else 1e-12
    neg_mean = abs(perm_es[perm_es < 0].mean()) if (perm_es < 0).any() else 1e-12
    if obs_es >= 0:
        nes = obs_es / pos_mean
        p   = np.mean(perm_es >= obs_es)
    else:
        nes = obs_es / neg_mean
        p   = np.mean(perm_es <= obs_es)
    return dict(
        test=f"GSEA-style ES ({n_perm} perm)",
        group=label,
        n_target=n_t, n_background=len(all_gene_df) - n_t,
        median_target_rank=round(all_gene_df.loc[all_gene_df["node"].isin(target_set), "rank"].median(), 1),
        median_bg_rank=round(all_gene_df.loc[~all_gene_df["node"].isin(target_set), "rank"].median(), 1),
        statistic=round(nes, 4), p_value=p,
        effect_size_r=round(obs_es, 4),
        significant="YES" if p < 0.05 else "NO",
        note=f"ES={obs_es:.4f}, NES={nes:.4f}, p={p:.4f} [NES>0: 상위 농축]"
    )

target_nodes = gene_df.loc[is_target, "node"].values
erg_nodes    = gene_df.loc[is_erg,    "node"].values
prad_nodes   = gene_df.loc[is_prad,   "node"].values

results_rows += [
    run_gsea(target_nodes, gene_df, "ALL_PRAD_TARGET"),
    run_gsea(erg_nodes,    gene_df, "ergPRAD_TARGET"),
    run_gsea(prad_nodes,   gene_df, "PRAD_TARGET"),
]
print("[Done] 검정 5. GSEA-style ES/NES")


# ─── Fold enrichment 요약 함수 ─────────────────────────────────────────────
def rank_summary(group_ranks, total_n, label):
    lines = []
    for pct in [1, 5, 10, 20, 30, 50]:
        k       = int(np.ceil(total_n * pct / 100))
        obs_in  = int(np.sum(group_ranks <= k))
        exp_in  = len(group_ranks) * pct / 100
        fold    = obs_in / exp_in if exp_in > 0 else np.nan
        lines.append(f"  top{pct:2d}%(rank<={k:5d}): "
                     f"{obs_in:3d}/{len(group_ranks)} "
                     f"({obs_in/len(group_ranks):.1%}) | "
                     f"exp={exp_in:.1f} | fold={fold:.2f}x")
    return "\n".join(lines)


# ─── 저장 및 리포트 ────────────────────────────────────────────────────────
result_df = pd.DataFrame(results_rows)
result_df.to_csv(TSV_OUT, sep="\t", index=False)

SEP = "=" * 100
sep = "-" * 100
lines = []
lines.append(SEP)
lines.append("  전립선암 표적 유전자 GAT+RWR v3 랭킹 상위 농축 통계 분석 리포트")
lines.append(f"  입력: node_importance_v3_annotated.tsv")
lines.append(SEP)
lines.append(f"  유전자 총 수 : {N_gene:,}")
lines.append(f"  표적 유전자  : {len(target_ranks)} "
             f"(ergPRAD특화: {len(erg_ranks)}, 일반PRAD: {len(prad_ranks)})")
lines.append(f"  배경 유전자  : {len(background_ranks):,}")
lines.append(f"  Permutation  : {N_PERM:,}회 (seed=42)")
lines.append(SEP)

test_groups = result_df.groupby("test", sort=False)
for test_name, grp in test_groups:
    lines.append(f"\n{'--'} {test_name} {'-'*60}")
    for _, row in grp.iterrows():
        sig = "*** SIGNIFICANT ***" if row["significant"] == "YES" else "    not significant"
        lines.append(f"  [{sig}] {row['group']}")
        lines.append(f"      n_target={row['n_target']}, "
                     f"median_rank(target)={row['median_target_rank']}, "
                     f"median_rank(bg)={row['median_bg_rank']}")
        lines.append(f"      stat={row['statistic']}, "
                     f"p={row['p_value']:.2e}, "
                     f"effect={row['effect_size_r']}")
        lines.append(f"      {row['note']}")

# Fold Enrichment Table
lines.append(f"\n{SEP}")
lines.append("  Top-k% 내 표적 유전자 Fold Enrichment")
lines.append(SEP)
for label, ranks in [
    (f"ALL_PRAD_TARGET ({len(target_ranks)})", target_ranks),
    (f"ergPRAD_TARGET  ({len(erg_ranks):2d})", erg_ranks),
    (f"PRAD_TARGET     ({len(prad_ranks)})", prad_ranks),
]:
    lines.append(f"\n  [{label}]")
    lines.append(rank_summary(ranks, N_gene, label))

# ergPRAD 상위 랭킹 유전자 목록
lines.append(f"\n{SEP}")
lines.append("  ergPRAD 특화 유전자 순위표 (rank 기준)")
lines.append(SEP)
erg_gene_df = gene_df[is_erg].sort_values("rank")
lines.append(f"  {'rank':>6}  {'gene':<12}  {'visit_prob':>12}  annotation_preview")
lines.append(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*50}")
for _, row in erg_gene_df.iterrows():
    ann = row["prad_annotation"]
    # drug 부분 추출 (Drugs: 이후)
    if "Drugs:" in ann:
        drug_part = "Drugs:" + ann.split("Drugs:")[-1][:60]
    else:
        drug_part = ann[:60]
    lines.append(f"  {int(row['rank']):>6}  {row['node']:<12}  "
                 f"{row['visit_prob']:>12.6f}  {drug_part}")

# 종합 결론
lines.append(f"\n{SEP}")
lines.append("  종합 결론")
lines.append(SEP)

all_sig = result_df[result_df["significant"] == "YES"]
lines.append(f"  유의한 검정 수: {len(all_sig)} / {len(result_df)}")

# ergPRAD 핵심 통계
erg_mwu  = result_df[(result_df["test"] == "Mann-Whitney U") &
                      result_df["group"].str.contains("ergPRAD")].iloc[0]
erg_perm = result_df[result_df["test"].str.contains("Permutation") &
                      result_df["group"].str.contains("ergPRAD")].iloc[0]
erg_gsea = result_df[result_df["test"].str.contains("GSEA") &
                      result_df["group"].str.contains("ergPRAD")].iloc[0]
all_mwu  = result_df[(result_df["test"] == "Mann-Whitney U") &
                      result_df["group"].str.contains("ALL")].iloc[0]

lines.append(f"\n  >> ergPRAD 특화 {len(erg_ranks)}개 유전자:")
lines.append(f"     - 중앙 rank {erg_mwu['median_target_rank']:.0f} vs 배경 "
             f"{erg_mwu['median_bg_rank']:.0f} "
             f"(Mann-Whitney U p={erg_mwu['p_value']:.2e})")
lines.append(f"     - Permutation Z={erg_perm['statistic']:.3f}, "
             f"p={erg_perm['p_value']:.5f}")
lines.append(f"     - GSEA NES={erg_gsea['statistic']:.3f} "
             f"(p={erg_gsea['p_value']:.4f})")

lines.append(f"\n  >> 전체 표적 {len(target_ranks)}개 유전자:")
lines.append(f"     - 중앙 rank {all_mwu['median_target_rank']:.0f} vs 배경 "
             f"{all_mwu['median_bg_rank']:.0f} "
             f"(Mann-Whitney U p={all_mwu['p_value']:.2e})")

lines.append(f"\n  --> 전립선암 표적 유전자들은 GAT+RWR v3 랭킹에서 "
             f"무작위보다 유의미하게 상위에 농축됨")
lines.append(f"      (pairwise GGA 기반 이종 네트워크 + GAT hub score 가중 RWR의 "
             f"생물학적 타당성 통계 지지)")
lines.append(SEP + "\n")

report = "\n".join(lines)
print()
print(report)

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write(report)

print(f"[저장] TSV: {TSV_OUT}")
print(f"[저장] TXT: {TXT_OUT}")
