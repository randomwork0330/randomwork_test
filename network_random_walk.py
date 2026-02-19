"""
Network Construction & Random Walk with Restart (RWR) Node Importance
======================================================================
gga.tsv / ppi.tsv / dgt.tsv 를 통합한 이종 네트워크에서
확률적 RWR 시뮬레이션으로 가장 중요한 노드를 추론합니다.

[정규화 전략]
- 엣지 타입(gga / ppi / dgt)별로 독립적으로 Min-Max 정규화
- 타입 간 스케일 편향 제거 후 동일 0~1 척도로 통일
- gga+ppi 중복 엣지는 정규화된 값의 가중 평균 (각 타입 신뢰도 반영)
- 워크 이동 확률은 현재 노드 기준 이웃 weight 합=1 로 local softmax
"""

import pandas as pd
import numpy as np
import networkx as nx
import random
from collections import Counter


# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────
def load_data(base_path="./"):
    gga = pd.read_csv(base_path + "gga.tsv", sep="\t")
    ppi = pd.read_csv(base_path + "ppi.tsv", sep="\t")
    dgt = pd.read_csv(base_path + "dgt.tsv", sep="\t")
    print(f"[Load] GGA: {len(gga)} | PPI: {len(ppi)} | DGT: {len(dgt)} edges")
    return gga, ppi, dgt


# ──────────────────────────────────────────────
# 2. 엣지 타입별 Min-Max 정규화
# ──────────────────────────────────────────────
def minmax_normalize(series):
    """
    각 엣지 타입 내부에서 독립적으로 Min-Max 정규화.
    mn == mx (모든 값 동일)이면 전부 1.0 반환.
    """
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


# ──────────────────────────────────────────────
# 3. 이종 네트워크 구성
# ──────────────────────────────────────────────
def build_heterogeneous_network(gga, ppi, dgt):
    """
    세 데이터를 통합한 이종 그래프 구성.

    엣지 weight 처리:
    - gga / ppi / dgt 각각 독립 Min-Max 정규화 후 삽입
    - gga+ppi 중복 엣지: 신뢰도 가중 평균 (gga=0.4, ppi=0.6)
      → PPI가 실험적 근거로 일반적으로 더 신뢰도 높음
    """
    G = nx.Graph()

    # --- 타입별 Min-Max 정규화 ---
    gga = gga.copy()
    ppi = ppi.copy()
    dgt = dgt.copy()
    gga["norm_w"] = minmax_normalize(gga["assoScore"])
    ppi["norm_w"] = minmax_normalize(ppi["ppiScore"])
    dgt["norm_w"] = minmax_normalize(dgt["evidenceScore"])

    # GGA 엣지 삽입
    for _, row in gga.iterrows():
        G.add_node(row["geneA"], node_type="gene")
        G.add_node(row["geneB"], node_type="gene")
        G.add_edge(row["geneA"], row["geneB"],
                   weight=row["norm_w"],
                   raw_weight=row["assoScore"],
                   edge_type="gga")

    # PPI 엣지 삽입 (중복 시 가중 평균 병합)
    GGA_W, PPI_W = 0.4, 0.6  # 타입별 신뢰도 가중치
    for _, row in ppi.iterrows():
        G.add_node(row["geneA"], node_type="gene")
        G.add_node(row["geneB"], node_type="gene")
        if G.has_edge(row["geneA"], row["geneB"]):
            # 기존 gga weight와 ppi weight를 신뢰도 가중 평균
            gga_w = G[row["geneA"]][row["geneB"]]["weight"]
            merged = GGA_W * gga_w + PPI_W * row["norm_w"]
            G[row["geneA"]][row["geneB"]]["weight"]    = merged
            G[row["geneA"]][row["geneB"]]["edge_type"] = "gga+ppi"
        else:
            G.add_edge(row["geneA"], row["geneB"],
                       weight=row["norm_w"],
                       raw_weight=row["ppiScore"],
                       edge_type="ppi")

    # DGT 엣지 삽입
    for _, row in dgt.iterrows():
        G.add_node(row["drug"], node_type="drug")
        G.add_node(row["gene"], node_type="gene")
        G.add_edge(row["drug"], row["gene"],
                   weight=row["norm_w"],
                   raw_weight=row["evidenceScore"],
                   edge_type="dgt")

    gene_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
    print(f"[Network] Nodes: {G.number_of_nodes()} "
          f"(Gene: {len(gene_nodes)}, Drug: {len(drug_nodes)})")
    print(f"[Network] Edges: {G.number_of_edges()}")

    # 엣지 weight 분포 확인
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    print(f"[Network] Edge weight | min={min(weights):.4f} "
          f"mean={np.mean(weights):.4f} max={max(weights):.4f}")
    return G


# ──────────────────────────────────────────────
# 4. 확률적 Random Walk with Restart 시뮬레이션
# ──────────────────────────────────────────────
def simulate_random_walk(G, n_walks=1000, walk_length=100,
                         restart_prob=0.15, seed=42):
    """
    실제 확률적 RWR 시뮬레이션.

    매 스텝:
      - restart_prob 확률 → 시작 노드로 복귀 (teleport)
      - 나머지 확률      → 이웃 weight 비례 확률로 다음 노드 선택
                           (local softmax: 현재 노드 이웃 weight 합=1)
    모든 노드를 시작점으로 n_walks번 반복 → 방문 횟수 집계.
    """
    random.seed(seed)
    np.random.seed(seed)

    visit_count = Counter()
    nodes       = list(G.nodes())
    total_walks = len(nodes) * n_walks
    done        = 0
    log_step    = max(total_walks // 10, 1)

    print(f"[RWR] 총 {total_walks:,}번 워크 시뮬레이션 시작 "
          f"(restart_prob={restart_prob}, walk_length={walk_length})")

    for start_node in nodes:
        for _ in range(n_walks):
            current = start_node

            for _ in range(walk_length):
                visit_count[current] += 1

                # teleport
                if random.random() < restart_prob:
                    current = start_node
                    continue

                neighbors = list(G.neighbors(current))
                if not neighbors:
                    current = start_node
                    continue

                # local softmax: 이웃 weight 합=1 로 전이 확률 산출
                # Min-Max 정규화로 weight=0인 엣지 존재 가능 → epsilon 추가로 NaN 방지
                w = np.array([G[current][nb]["weight"] for nb in neighbors],
                             dtype=float)
                w += 1e-8          # epsilon: zero-weight 엣지도 최소 이동 확률 보장
                w /= w.sum()
                current = np.random.choice(neighbors, p=w)

            done += 1

        # 10% 단위 진행률 출력
        if done % log_step < n_walks:
            print(f"  [{done/total_walks*100:3.0f}%] {done:,}/{total_walks:,} walks")

    print(f"[RWR] 완료 | 총 방문 횟수: {sum(visit_count.values()):,}")
    return visit_count


# ──────────────────────────────────────────────
# 5. 결과 출력
# ──────────────────────────────────────────────
def print_results(G, visit_count, top_k=10):
    total       = sum(visit_count.values())
    visit_prob  = {n: c / total for n, c in visit_count.items()}
    sorted_nodes = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "="*70)
    print(f"    [RWR 방문 횟수 기반 노드 중요도 Top {top_k}]")
    print("="*70)
    print(f"{'Rank':<5} {'Node':<15} {'Type':<6} "
          f"{'방문횟수':>10} {'방문확률':>10} {'Degree':>7}")
    print("-"*70)

    for rank, (node, cnt) in enumerate(sorted_nodes[:top_k], 1):
        ntype  = G.nodes[node].get("node_type", "?")
        prob   = visit_prob[node]
        degree = G.degree(node)
        print(f"{rank:<5} {node:<15} {ntype:<6} "
              f"{cnt:>10,} {prob:>10.4f} {degree:>7}")

    print("="*70)

    top_node, top_cnt = sorted_nodes[0]
    top_type = G.nodes[top_node].get("node_type", "?")
    neighbors = list(G.neighbors(top_node))
    print(f"\n★ 가장 중요한 노드 : [{top_node}] (type={top_type})")
    print(f"  방문 횟수        : {top_cnt:,}회  ({visit_prob[top_node]*100:.2f}%)")
    print(f"  연결 이웃 ({len(neighbors)}개)  : {', '.join(neighbors)}")

    return visit_prob, sorted_nodes


# ──────────────────────────────────────────────
# 6. 결과 저장
# ──────────────────────────────────────────────
def save_results(G, visit_count, visit_prob, base_path):
    rows = [
        {
            "node":        node,
            "node_type":   G.nodes[node].get("node_type", "?"),
            "visit_count": visit_count.get(node, 0),
            "visit_prob":  round(visit_prob.get(node, 0.0), 6),
            "degree":      G.degree(node),
        }
        for node in G.nodes()
    ]
    df = (pd.DataFrame(rows)
            .sort_values("visit_count", ascending=False)
            .reset_index(drop=True))
    df.index += 1  # rank 1부터 시작

    out = base_path + "node_importance.tsv"
    df.to_csv(out, sep="\t", index_label="rank")
    print(f"[Saved] {out}")


# ──────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    BASE_PATH = "C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/"

    gga, ppi, dgt = load_data(BASE_PATH)
    G             = build_heterogeneous_network(gga, ppi, dgt)

    visit_count             = simulate_random_walk(
        G,
        n_walks      = 1000,   # 노드당 워크 수
        walk_length  = 100,    # 워크당 최대 스텝
        restart_prob = 0.15,   # teleport 확률
        seed         = 42
    )

    visit_prob, sorted_nodes = print_results(G, visit_count, top_k=10)
    save_results(G, visit_count, visit_prob, BASE_PATH)
