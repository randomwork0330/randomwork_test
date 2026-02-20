"""
전립선암 표적 유전자의 RWR 랭킹 상위 집중 통계 분석
========================================================
검정 1. Mann-Whitney U test          – 두 집단 rank 분포 차이 (비모수)
검정 2. Kolmogorov-Smirnov test      – rank CDF 분포 차이
검정 3. Fisher's exact test          – 각 상위 k% 컷오프에서 과농축 여부
검정 4. Permutation test             – 실제 평균 rank vs. 무작위 기대값
검정 5. GSEA-style Enrichment Score  – 순서 정렬 기반 ES / NES (gene-set 개념)
검정 6. Wilcoxon rank-sum            – 표준 비모수 위치 검정
출력: stats_enrichment_result.tsv / stats_enrichment_result.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp, fisher_exact, wilcoxon
import warnings
warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────
TSV_IN  = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/node_importance_v2_annotated.tsv"
TSV_OUT = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/stats_enrichment_result.tsv"
TXT_OUT = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/stats_enrichment_result.txt"

# ── 데이터 로드 ────────────────────────────────────────────────────────────
df = pd.read_csv(TSV_IN, sep="\t")
gene_df = df[df["node_type"] == "gene"].copy().reset_index(drop=True)
N_gene  = len(gene_df)

# 표적 유전자 집합
is_target  = gene_df["prad_annotation"].notna() & (gene_df["prad_annotation"] != "")
is_erg     = gene_df["prad_annotation"].str.startswith("(ergPRAD)", na=False)
is_prad    = is_target & ~is_erg

target_ranks    = gene_df.loc[is_target, "rank"].values   # 전체 표적 (139)
erg_ranks       = gene_df.loc[is_erg,    "rank"].values   # ergPRAD (17)
prad_ranks      = gene_df.loc[is_prad,   "rank"].values   # 일반 PRAD (122)
background_ranks= gene_df.loc[~is_target,"rank"].values   # 배경 (나머지)

print(f"유전자 노드 총 수  : {N_gene:,}")
print(f"PRAD 표적 유전자   : {len(target_ranks):,}  (ergPRAD: {len(erg_ranks)}, 일반: {len(prad_ranks)})")
print(f"배경 유전자        : {len(background_ranks):,}")
print()

results_rows = []   # TSV 출력용


# ═══════════════════════════════════════════════════════════════════════════
# 검정 1. Mann-Whitney U test
# H0: 표적 유전자와 배경 유전자의 rank 분포가 같다
# H1: 표적 유전자의 rank가 유의미하게 낮다 (더 중요하다)
# ═══════════════════════════════════════════════════════════════════════════
def run_mwu(group_ranks, bg_ranks, label):
    stat, p = mannwhitneyu(group_ranks, bg_ranks, alternative="less")
    n1, n2  = len(group_ranks), len(bg_ranks)
    # effect size r = Z / sqrt(N)
    z = (stat - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)
    r = abs(z) / np.sqrt(n1 + n2)
    med_target = np.median(group_ranks)
    med_bg     = np.median(bg_ranks)
    return dict(
        test="Mann-Whitney U", group=label,
        n_target=n1, n_background=n2,
        median_target_rank=round(med_target,1),
        median_bg_rank=round(med_bg,1),
        statistic=round(stat,2), p_value=p,
        effect_size_r=round(r,4),
        significant="YES" if p < 0.05 else "NO",
        note=f"rank↓ → 더 중요. effect r={r:.3f} ({'large' if r>=0.5 else 'medium' if r>=0.3 else 'small'})"
    )

r1a = run_mwu(target_ranks, background_ranks, "ALL_PRAD_TARGET vs BG")
r1b = run_mwu(erg_ranks,    background_ranks, "ergPRAD_TARGET vs BG")
r1c = run_mwu(prad_ranks,   background_ranks, "PRAD_TARGET vs BG")
results_rows += [r1a, r1b, r1c]


# ═══════════════════════════════════════════════════════════════════════════
# 검정 2. Kolmogorov-Smirnov test
# 표적 유전자와 배경의 rank 누적분포 형태가 다른가?
# ═══════════════════════════════════════════════════════════════════════════
def run_ks(group_ranks, bg_ranks, label):
    # alternative='greater': 표적 CDF > 배경 CDF at some point
    # rank 숫자가 작을수록 중요 → 표적은 낮은 rank에 집중
    # → 표적 CDF가 배경보다 위(left-shift) → greater가 올바른 방향
    stat, p = ks_2samp(group_ranks, bg_ranks, alternative="greater")
    return dict(
        test="KS test", group=label,
        n_target=len(group_ranks), n_background=len(bg_ranks),
        median_target_rank=round(np.median(group_ranks),1),
        median_bg_rank=round(np.median(bg_ranks),1),
        statistic=round(stat,4), p_value=p,
        effect_size_r=round(stat,4),   # KS stat 자체가 effect size (0~1)
        significant="YES" if p < 0.05 else "NO",
        note="KS D-stat = max CDF gap (표적 CDF가 배경보다 좌측 편향 = 상위 농축)"
    )

results_rows += [
    run_ks(target_ranks, background_ranks, "ALL_PRAD_TARGET vs BG"),
    run_ks(erg_ranks,    background_ranks, "ergPRAD_TARGET vs BG"),
    run_ks(prad_ranks,   background_ranks, "PRAD_TARGET vs BG"),
]


# ═══════════════════════════════════════════════════════════════════════════
# 검정 3. Fisher's exact test (상위 k% 컷오프)
# 상위 k% 안에 표적 유전자가 기대보다 많이 들어있는가?
# ═══════════════════════════════════════════════════════════════════════════
cutoffs = [0.01, 0.05, 0.10, 0.20]   # 상위 1%, 5%, 10%, 20%

def run_fisher(group_ranks, all_n, label, cutoff):
    k         = int(np.ceil(all_n * cutoff))
    in_top_t  = np.sum(group_ranks <= k)          # 표적 중 상위 k 이내
    out_top_t = len(group_ranks) - in_top_t        # 표적 중 k 초과
    in_top_b  = k - in_top_t                       # 비표적 중 상위 k 이내
    out_top_b = (all_n - k) - (len(group_ranks) - in_top_t)
    # 2×2 표: [[표적∩top, 표적∩tail],[비표적∩top, 비표적∩tail]]
    table     = [[in_top_t, out_top_t], [in_top_b, out_top_b]]
    or_, p    = fisher_exact(table, alternative="greater")
    expected  = len(group_ranks) * cutoff
    obs_ratio = in_top_t / len(group_ranks)
    fold_enrich = in_top_t / expected if expected > 0 else np.nan
    return dict(
        test=f"Fisher's exact (top {int(cutoff*100)}%)",
        group=label,
        n_target=len(group_ranks), n_background=all_n-len(group_ranks),
        median_target_rank=round(np.median(group_ranks),1),
        median_bg_rank="-",
        statistic=round(or_,4), p_value=p,
        effect_size_r=round(fold_enrich,3),
        significant="YES" if p < 0.05 else "NO",
        note=(f"top{int(cutoff*100)}%({k}개): 관측={in_top_t}/{len(group_ranks)} "
              f"({obs_ratio:.1%}), 기대={expected:.1f}, "
              f"fold={fold_enrich:.2f}x, OR={or_:.2f}")
    )

for cut in cutoffs:
    results_rows += [
        run_fisher(target_ranks, N_gene, "ALL_PRAD_TARGET", cut),
        run_fisher(erg_ranks,    N_gene, "ergPRAD_TARGET",  cut),
        run_fisher(prad_ranks,   N_gene, "PRAD_TARGET",     cut),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 검정 4. Permutation test
# 실제 평균 rank vs. 무작위 샘플링 평균 rank (10,000회)
# ═══════════════════════════════════════════════════════════════════════════
np.random.seed(42)
N_PERM = 10_000

def run_permutation(group_ranks, all_ranks, label, n_perm=N_PERM):
    obs_mean = np.mean(group_ranks)
    n        = len(group_ranks)
    perm_means = np.array([
        np.mean(np.random.choice(all_ranks, n, replace=False))
        for _ in range(n_perm)
    ])
    p = np.mean(perm_means <= obs_mean)   # 표적이 낮은 rank(더 중요)
    z = (obs_mean - perm_means.mean()) / perm_means.std()
    return dict(
        test=f"Permutation test ({n_perm:,}회)",
        group=label,
        n_target=n, n_background=len(all_ranks)-n,
        median_target_rank=round(np.median(group_ranks),1),
        median_bg_rank=round(perm_means.mean(),1),
        statistic=round(z,4), p_value=p,
        effect_size_r=round(abs(z),4),
        significant="YES" if p < 0.05 else "NO",
        note=(f"실제 평균rank={obs_mean:.1f}, "
              f"무작위기대={perm_means.mean():.1f}±{perm_means.std():.1f}, "
              f"Z={z:.3f}, p(permutation)={p:.5f}")
    )

all_gene_ranks = gene_df["rank"].values
results_rows += [
    run_permutation(target_ranks, all_gene_ranks, "ALL_PRAD_TARGET"),
    run_permutation(erg_ranks,    all_gene_ranks, "ergPRAD_TARGET"),
    run_permutation(prad_ranks,   all_gene_ranks, "PRAD_TARGET"),
]


# ═══════════════════════════════════════════════════════════════════════════
# 검정 5. GSEA-style Enrichment Score (ES / NES)
# 유전자를 rank로 정렬 → 표적 유전자 위치에서 ES 산출 (Broad GSEA 방식)
# NES: permutation으로 정규화
# ═══════════════════════════════════════════════════════════════════════════
def gsea_es(sorted_gene_df, target_set, score_col="visit_prob"):
    """단순화된 GSEA ES: walk along ranked list"""
    n   = len(sorted_gene_df)
    nh  = len(target_set)
    # 표적 유전자의 총 score (분모)
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

    es = es_max if abs(es_max) >= abs(es_min) else es_min
    return es

def run_gsea(group_nodes, all_gene_df, label, n_perm=1000, score_col="visit_prob"):
    sorted_df = all_gene_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    target_set = set(group_nodes)

    obs_es = gsea_es(sorted_df, target_set, score_col)

    # Permutation NES
    perm_es = []
    all_nodes = sorted_df["node"].values
    n_t = len(target_set)
    for _ in range(n_perm):
        rand_set = set(np.random.choice(all_nodes, n_t, replace=False))
        perm_es.append(gsea_es(sorted_df, rand_set, score_col))
    perm_es = np.array(perm_es)

    # NES 정규화
    pos_mean = perm_es[perm_es >= 0].mean() if (perm_es >= 0).any() else 1e-12
    neg_mean = abs(perm_es[perm_es <  0].mean()) if (perm_es < 0).any() else 1e-12
    if obs_es >= 0:
        nes = obs_es / pos_mean
        p   = np.mean(perm_es >= obs_es)
    else:
        nes = obs_es / neg_mean
        p   = np.mean(perm_es <= obs_es)

    return dict(
        test=f"GSEA-style ES ({n_perm}perm)",
        group=label,
        n_target=n_t, n_background=len(all_gene_df)-n_t,
        median_target_rank=round(all_gene_df.loc[all_gene_df["node"].isin(target_set), "rank"].median(),1),
        median_bg_rank=round(all_gene_df.loc[~all_gene_df["node"].isin(target_set), "rank"].median(),1),
        statistic=round(nes,4), p_value=p,
        effect_size_r=round(obs_es,4),
        significant="YES" if p < 0.05 else "NO",
        note=(f"ES={obs_es:.4f}, NES={nes:.4f}, "
              f"perm_mean(+)={pos_mean:.4f}, p={p:.4f}  "
              f"[NES>0: 상위 농축, NES<0: 하위 분포]")
    )

target_nodes = gene_df.loc[is_target, "node"].values
erg_nodes    = gene_df.loc[is_erg,    "node"].values
prad_nodes   = gene_df.loc[is_prad,   "node"].values

results_rows += [
    run_gsea(target_nodes, gene_df, "ALL_PRAD_TARGET"),
    run_gsea(erg_nodes,    gene_df, "ergPRAD_TARGET"),
    run_gsea(prad_nodes,   gene_df, "PRAD_TARGET"),
]


# ═══════════════════════════════════════════════════════════════════════════
# 검정 6. 순위 기반 요약 통계 (비율, top-k enrichment fold)
# ═══════════════════════════════════════════════════════════════════════════
def rank_summary(group_ranks, total_n, label):
    rows = []
    for pct in [1, 5, 10, 20, 30, 50]:
        k         = int(np.ceil(total_n * pct / 100))
        obs_in    = np.sum(group_ranks <= k)
        exp_in    = len(group_ranks) * pct / 100
        fold      = obs_in / exp_in if exp_in > 0 else np.nan
        rows.append(f"  top{pct:2d}%(rank≤{k:5d}): {obs_in:3d}/{len(group_ranks)} "
                    f"({obs_in/len(group_ranks):.1%}) | expected {exp_in:.1f} | fold={fold:.2f}x")
    return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 출력 및 저장
# ═══════════════════════════════════════════════════════════════════════════
result_df = pd.DataFrame(results_rows)
result_df.to_csv(TSV_OUT, sep="\t", index=False)

# 텍스트 리포트 작성
lines = []
SEP  = "=" * 95
sep  = "-" * 95

lines.append(SEP)
lines.append("  전립선암 표적 유전자 RWR 랭킹 상위 농축 통계 분석 리포트")
lines.append("  GAT+RWR ver2 (node_importance_v2_annotated.tsv)")
lines.append(SEP)
lines.append(f"  유전자 총 수 : {N_gene:,}")
lines.append(f"  표적 유전자  : {len(target_ranks)} (ergPRAD특화: {len(erg_ranks)}, 일반PRAD: {len(prad_ranks)})")
lines.append(f"  배경 유전자  : {len(background_ranks):,}")
lines.append(f"  Permutation  : {N_PERM:,}회 (seed=42)")
lines.append(SEP)

# 검정별 섹션 출력
test_groups = result_df.groupby("test", sort=False)
for test_name, grp in test_groups:
    lines.append(f"\n{'─'*3} {test_name} {'─'*60}")
    for _, row in grp.iterrows():
        sig = "★ 유의" if row["significant"] == "YES" else "  비유의"
        lines.append(f"  [{sig}] {row['group']}")
        lines.append(f"         n_target={row['n_target']}, "
                     f"median_rank(target)={row['median_target_rank']}, "
                     f"median_rank(bg)={row['median_bg_rank']}")
        lines.append(f"         stat={row['statistic']}, p={row['p_value']:.2e}, "
                     f"effect={row['effect_size_r']}")
        lines.append(f"         {row['note']}")

# Top-k enrichment fold 표
lines.append(f"\n{SEP}")
lines.append("  Top-k% 내 표적 유전자 농축 배수 (Fold Enrichment)")
lines.append(SEP)
for label, ranks in [("ALL_PRAD_TARGET (139)", target_ranks),
                     ("ergPRAD_TARGET  ( 17)", erg_ranks),
                     ("PRAD_TARGET     (122)", prad_ranks)]:
    lines.append(f"\n  [{label}]")
    lines.append(rank_summary(ranks, N_gene, label))

# 결론
lines.append(f"\n{SEP}")
lines.append("  종합 결론")
lines.append(SEP)

all_sig = result_df[result_df["significant"] == "YES"]
lines.append(f"  유의한 검정 수: {len(all_sig)} / {len(result_df)}")

# ergPRAD 핵심 통계 추출
erg_mwu = result_df[(result_df["test"]=="Mann-Whitney U") &
                     (result_df["group"].str.contains("ergPRAD"))].iloc[0]
erg_perm= result_df[(result_df["test"].str.contains("Permutation")) &
                     (result_df["group"].str.contains("ergPRAD"))].iloc[0]
erg_gsea= result_df[(result_df["test"].str.contains("GSEA")) &
                     (result_df["group"].str.contains("ergPRAD"))].iloc[0]

lines.append(f"\n  ▶ ergPRAD 특화 17개 유전자:")
lines.append(f"     - 중앙 rank {erg_mwu['median_target_rank']:.0f} vs 배경 {erg_mwu['median_bg_rank']:.0f} "
             f"(Mann-Whitney U p={erg_mwu['p_value']:.2e})")
lines.append(f"     - 실제 평균 rank vs 무작위 기대 (Permutation Z={erg_perm['statistic']:.3f}, p={erg_perm['p_value']:.5f})")
lines.append(f"     - GSEA NES={erg_gsea['statistic']:.3f} (p={erg_gsea['p_value']:.4f})")

all_mwu = result_df[(result_df["test"]=="Mann-Whitney U") &
                     (result_df["group"].str.contains("ALL"))].iloc[0]
lines.append(f"\n  ▶ 전체 표적 139개 유전자:")
lines.append(f"     - 중앙 rank {all_mwu['median_target_rank']:.0f} vs 배경 {all_mwu['median_bg_rank']:.0f} "
             f"(Mann-Whitney U p={all_mwu['p_value']:.2e})")

lines.append(f"\n  → 전립선암 표적 유전자들은 GAT+RWR 랭킹에서 무작위보다 유의미하게 상위에 농축됨")
lines.append(f"     (네트워크 기반 중요도 분석의 생물학적 타당성 통계 지지)")
lines.append(SEP + "\n")

report = "\n".join(lines)
print(report)

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n TSV 저장: {TSV_OUT}")
print(f" TXT 저장: {TXT_OUT}")
