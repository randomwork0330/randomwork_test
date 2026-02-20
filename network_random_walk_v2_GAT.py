"""
Network RWR ver2: Graph Attention Network (GAT) + Random Walk with Restart
==========================================================================
ver1 대비 변경점:
  - GAT(Edge-Aware, 2-layer, 8-head)로 각 엣지의 attention weight 학습
  - 학습된 attention을 RWR 전이 확률로 사용 (α=0.7 혼합)
  - 자기지도 학습: Link Prediction BCE Loss + degree 정규화 항
  - 출력: node_importance_v2.tsv (+ gat_hub_score 컬럼)

[파이프라인]
  데이터 로드 + 네트워크 구성 (ver1 동일)
    → 그래프 → Tensor 변환 (X, edge_index, edge_attr)
    → GAT 학습 (200 epochs, patience=20)
    → Attention 추출 → 전이 확률 혼합 캐시 구성
    → RWR 시뮬레이션 (멀티프로세싱, n_workers=12)
    → 결과 저장

[GAT 아키텍처]
  Layer1: EdgeAwareGATLayer(8→16, 8-head, concat) → (N, 128)
  Layer2: EdgeAwareGATLayer(128→16, 8-head, mean) → (N, 16)
  Attention: e_ij = LeakyReLU(a^T·[W·h_i||W·h_j||W_e·edge_feat_ij])
             alpha_ij = softmax over src neighbors
"""

import os
import glob
import time
import math
import threading
import warnings
from collections import Counter
from multiprocessing import Pool, Value, cpu_count

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# [멀티프로세싱 워커] 모듈 최상위 정의 필수 (Windows pickle)
# ══════════════════════════════════════════════════════════
_shared_counter = None

def _init_worker(shared_val):
    """Pool initializer: 공유 카운터를 워커 전역 변수에 주입."""
    global _shared_counter
    _shared_counter = shared_val


def _walk_worker(args: tuple) -> Counter:
    """
    단일 워커 RWR 시뮬레이션 (ver1과 동일 구조).
    전이 확률은 호출 전 build_attention_neighbor_cache()로 결정됨.
    """
    chunk_nodes, neighbor_cache, n_walks, walk_length, restart_prob, worker_seed = args
    rng   = np.random.default_rng(worker_seed)
    local = Counter()

    for start_node in chunk_nodes:
        nbrs, probs = neighbor_cache[start_node]
        for _ in range(n_walks):
            current   = start_node
            cur_nbrs  = nbrs
            cur_probs = probs
            for _ in range(walk_length):
                local[current] += 1
                if rng.random() < restart_prob:
                    current, cur_nbrs, cur_probs = start_node, nbrs, probs
                    continue
                if cur_nbrs is None:
                    current, cur_nbrs, cur_probs = start_node, nbrs, probs
                    continue
                idx       = rng.choice(len(cur_nbrs), p=cur_probs)
                current   = cur_nbrs[idx]
                cur_nbrs, cur_probs = neighbor_cache[current]
        if _shared_counter is not None:
            with _shared_counter.get_lock():
                _shared_counter.value += n_walks
    return local


# ══════════════════════════════════════════════════════════
# 0. 파일 탐색 유틸
# ══════════════════════════════════════════════════════════
def find_file(base_path: str, prefix: str) -> str:
    """prefix로 시작하는 첫 번째 TSV 반환."""
    matches = glob.glob(os.path.join(base_path, f"{prefix}*.tsv"))
    if not matches:
        raise FileNotFoundError(f"'{prefix}*.tsv' 없음: {base_path}")
    return sorted(matches)[0]


# ══════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════
def load_data(base_path: str):
    """_A_ / _B_ / _C_ TSV 로드 및 gene_set 기준 필터링."""
    path_a = find_file(base_path, "_A_")
    path_b = find_file(base_path, "_B_")
    path_c = find_file(base_path, "_C_")

    gga      = pd.read_csv(path_a, sep="\t")
    gene_set = set(gga["gene"])
    print(f"[Load] GGA: {len(gga):,} rows ({len(gene_set):,} genes)  ← {os.path.basename(path_a)}")

    ppi_raw = pd.read_csv(path_b, sep="\t")
    ppi     = ppi_raw[ppi_raw["gene1"].isin(gene_set) | ppi_raw["gene2"].isin(gene_set)].reset_index(drop=True)
    print(f"[Load] PPI: {len(ppi_raw):,} → {len(ppi):,} rows  ← {os.path.basename(path_b)}")

    dgt_raw          = pd.read_csv(path_c, sep="\t")
    dgt_raw["use_gene"] = dgt_raw["final_target_gene"].fillna(dgt_raw["target_gene"])
    dgt              = dgt_raw[dgt_raw["use_gene"].isin(gene_set)].reset_index(drop=True)
    print(f"[Load] DGT: {len(dgt_raw):,} → {len(dgt):,} rows  ← {os.path.basename(path_c)}")

    return gga, ppi, dgt


# ══════════════════════════════════════════════════════════
# 2. Z-score → Sigmoid 정규화
# ══════════════════════════════════════════════════════════
def zscore_sigmoid_normalize(series: pd.Series) -> pd.Series:
    """Z-score 표준화 후 Sigmoid로 0~1 확률값 변환."""
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    z = (series - mu) / sigma
    return 1.0 / (1.0 + np.exp(-z))


# ══════════════════════════════════════════════════════════
# 3. 이종 네트워크 구성 (ver1 동일)
# ══════════════════════════════════════════════════════════
def build_heterogeneous_network(gga, ppi, dgt) -> nx.Graph:
    """GGA/PPI/DGT 통합 이종 그래프 구성 (Z-score→Sigmoid 정규화)."""
    G = nx.Graph()
    GGA_W, PPI_W = 0.4, 0.6

    gga = gga.copy(); gga["norm_w"] = zscore_sigmoid_normalize(gga["absPearsonR"].astype(float))
    for _, row in gga.iterrows():
        gene, w = str(row["gene"]), float(row["norm_w"])
        G.add_node("ERG", node_type="gene"); G.add_node(gene, node_type="gene")
        if gene == "ERG": continue
        if G.has_edge("ERG", gene):
            ew = G["ERG"][gene]["weight"]
            G["ERG"][gene].update({"weight": GGA_W*w + PPI_W*ew, "edge_type": "gga+ppi"})
        else:
            G.add_edge("ERG", gene, weight=w, edge_type="gga")
    print(f"[Network] GGA 완료 | 엣지: {G.number_of_edges():,}")

    ppi = ppi.copy(); ppi["norm_w"] = zscore_sigmoid_normalize(ppi["score"].astype(float))
    for _, row in ppi.iterrows():
        g1, g2, w = str(row["gene1"]), str(row["gene2"]), float(row["norm_w"])
        G.add_node(g1, node_type="gene"); G.add_node(g2, node_type="gene")
        if G.has_edge(g1, g2):
            ew = G[g1][g2]["weight"]
            G[g1][g2].update({"weight": GGA_W*ew + PPI_W*w, "edge_type": "gga+ppi"})
        else:
            G.add_edge(g1, g2, weight=w, edge_type="ppi")
    print(f"[Network] PPI 완료 | 엣지: {G.number_of_edges():,}")

    dgt = dgt.copy(); dgt["norm_w"] = zscore_sigmoid_normalize(dgt["score"].astype(float))
    for _, row in dgt.iterrows():
        drug, gene, w = str(row["drug"]), str(row["use_gene"]), float(row["norm_w"])
        if not drug or not gene: continue
        G.add_node(drug, node_type="drug"); G.add_node(gene, node_type="gene")
        if G.has_edge(drug, gene):
            if w > G[drug][gene]["weight"]: G[drug][gene]["weight"] = w
        else:
            G.add_edge(drug, gene, weight=w, edge_type="dgt")
    print(f"[Network] DGT 완료 | 엣지: {G.number_of_edges():,}")

    gene_n = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "gene")
    drug_n = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "drug")
    ws     = [d["weight"] for _, _, d in G.edges(data=True)]
    etypes = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    print(f"\n[Network] 노드: {G.number_of_nodes():,} (Gene:{gene_n:,} Drug:{drug_n:,})")
    print(f"  엣지: {G.number_of_edges():,}  타입: { {k:f'{v:,}' for k,v in etypes.items()} }")
    print(f"  Weight: min={min(ws):.4f} mean={np.mean(ws):.4f} max={max(ws):.4f}")
    return G


# ══════════════════════════════════════════════════════════
# 4. 그래프 → PyTorch Tensor 변환
# ══════════════════════════════════════════════════════════
EDGE_TYPE_MAP = {"gga": 0, "ppi": 1, "gga+ppi": 2, "dgt": 3}

def graph_to_tensors(G: nx.Graph):
    """
    nx.Graph → PyTorch Tensor 변환.

    반환:
      X          : (N, 8)   노드 특징 행렬
      edge_index : (2, 2E)  양방향 엣지 인덱스
      edge_attr  : (2E, 5)  엣지 특징 (원-핫 타입 4 + weight 1)
      node_list  : [str]    노드 이름 순서
      node_types : (N,)     0=gene, 1=drug
    """
    node_list = list(G.nodes())
    n2i       = {n: i for i, n in enumerate(node_list)}
    N         = len(node_list)

    # ── 노드 특징 (N × 8)
    X = np.zeros((N, 8), dtype=np.float32)
    for i, node in enumerate(node_list):
        ntype  = G.nodes[node].get("node_type", "gene")
        nbrs   = list(G.neighbors(node))
        degree = len(nbrs)
        ws     = [G[node][nb]["weight"] for nb in nbrs] if nbrs else [0.0]
        ec     = Counter(G[node][nb].get("edge_type","gga") for nb in nbrs)
        X[i, 0] = 1.0 if ntype == "gene" else 0.0         # gene 여부
        X[i, 1] = 1.0 if ntype == "drug" else 0.0         # drug 여부
        X[i, 2] = math.log(1 + degree)                     # log-degree
        X[i, 3] = ec.get("gga", 0) / max(degree, 1)       # gga 비율
        X[i, 4] = ec.get("ppi", 0) / max(degree, 1)       # ppi 비율
        X[i, 5] = ec.get("dgt", 0) / max(degree, 1)       # dgt 비율
        X[i, 6] = float(np.mean(ws))                       # 평균 weight
        X[i, 7] = float(np.max(ws))                        # 최대 weight

    # ── 양방향 엣지 인덱스 + 특징 (2E × 5)
    srcs, dsts, attrs = [], [], []
    for u, v, d in G.edges(data=True):
        i, j  = n2i[u], n2i[v]
        etype = d.get("edge_type", "gga")
        w     = float(d.get("weight", 0.5))
        oh    = [0.0] * 4
        oh[EDGE_TYPE_MAP.get(etype, 0)] = 1.0
        feat  = oh + [w]
        # u→v
        srcs.append(i); dsts.append(j); attrs.append(feat)
        # v→u (양방향)
        srcs.append(j); dsts.append(i); attrs.append(feat)

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_attr  = torch.tensor(attrs, dtype=torch.float32)
    X_t        = torch.tensor(X, dtype=torch.float32)
    node_types = torch.tensor(
        [0 if G.nodes[n].get("node_type","gene")=="gene" else 1 for n in node_list],
        dtype=torch.long
    )

    print(f"[Tensor] X:{tuple(X_t.shape)}  edge_index:{tuple(edge_index.shape)}  edge_attr:{tuple(edge_attr.shape)}")
    return X_t, edge_index, edge_attr, node_list, node_types


# ══════════════════════════════════════════════════════════
# 5. Edge-Aware GAT 레이어 (Pure PyTorch)
# ══════════════════════════════════════════════════════════
class EdgeAwareGATLayer(nn.Module):
    """
    단일 GAT 레이어.

    Attention:
      e_ij = LeakyReLU(a^T · [W·h_i || W·h_j || W_e·edge_feat_ij])
      alpha_ij = softmax over src-neighbors j

    엣지 특징을 attention에 통합해 이종 그래프의 엣지 타입 정보 보존.
    degree-aware scaling: e_ij *= 1/sqrt(degree_src) (허브 attention 희석 방지)
    """
    def __init__(self, F_in: int, F_out: int, n_heads: int,
                 F_edge: int = 5, dropout: float = 0.3,
                 concat: bool = True, leaky_slope: float = 0.2):
        super().__init__()
        self.F_out   = F_out
        self.n_heads = n_heads
        self.concat  = concat

        # 노드 변환 파라미터 (head별)
        self.W = nn.Parameter(torch.empty(n_heads, F_in, F_out))
        # Attention 벡터: [Wh_i || Wh_j || edge_feat]
        self.a = nn.Parameter(torch.empty(n_heads, 2 * F_out + F_edge))
        # 엣지 특징 변환
        self.W_edge = nn.Linear(F_edge, F_edge, bias=False)

        self.leaky   = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)
        self.elu     = nn.ELU()

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr, degree_src=None):
        """
        x          : (N, F_in)
        edge_index : (2, E)
        edge_attr  : (E, F_edge)
        degree_src : (E,) 선택적 degree 스케일링
        반환: (out, alpha)  out:(N, F_out*n_heads or F_out)  alpha:(E, n_heads)
        """
        N   = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        # 1. 노드 변환: (N, n_heads, F_out)
        Wh = torch.einsum("nf,hfo->nho", x, self.W)

        # 2. src/dst 특징 수집
        Wh_src = Wh[src]    # (E, n_heads, F_out)
        Wh_dst = Wh[dst]    # (E, n_heads, F_out)

        # 3. 엣지 특징 변환 → head 차원 확장
        ef = self.W_edge(edge_attr)                          # (E, F_edge)
        ef = ef.unsqueeze(1).expand(-1, self.n_heads, -1)   # (E, n_heads, F_edge)

        # 4. Attention 입력 concatenation
        cat = torch.cat([Wh_src, Wh_dst, ef], dim=2)        # (E, n_heads, 2F_out+F_edge)

        # 5. Attention score
        e = (cat * self.a.unsqueeze(0)).sum(dim=2)           # (E, n_heads)
        e = self.leaky(e)

        # 6. degree-aware scaling (허브 attention 희석 방지)
        if degree_src is not None:
            scale = 1.0 / torch.sqrt(degree_src.float().clamp(min=1)).unsqueeze(1)
            e = e * scale

        # 7. src 기준 softmax (수치 안정화)
        e_max = torch.zeros(N, self.n_heads, device=x.device)
        e_max.scatter_reduce_(0, src.unsqueeze(1).expand_as(e), e,
                              reduce="amax", include_self=True)
        e_exp = torch.exp(e - e_max[src])
        e_sum = torch.zeros(N, self.n_heads, device=x.device)
        e_sum.scatter_add_(0, src.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha = e_exp / (e_sum[src] + 1e-16)                # (E, n_heads)
        alpha = self.dropout(alpha)

        # 8. 메시지 집계 (weighted sum)
        msg = Wh_dst * alpha.unsqueeze(2)                   # (E, n_heads, F_out)
        out = torch.zeros(N, self.n_heads, self.F_out, device=x.device)
        out.scatter_add_(0,
                         src.view(-1,1,1).expand_as(msg),
                         msg)                               # (N, n_heads, F_out)

        # 9. concat or mean + 활성화
        if self.concat:
            out = out.view(N, self.n_heads * self.F_out)
        else:
            out = out.mean(dim=1)
        out = self.elu(out)

        return out, alpha


# ══════════════════════════════════════════════════════════
# 6. 이종 GAT 모델 (2-layer)
# ══════════════════════════════════════════════════════════
class HeterogeneousGAT(nn.Module):
    """
    2-레이어 Edge-Aware GAT.

    노드 타입별 별도 선형 변환(type-specific projection) 적용:
      gene → W_gene, drug → W_drug
    → 이종 노드 간 의미론적 차이 보존
    """
    def __init__(self, F_in: int = 8, hidden: int = 16, out: int = 16,
                 n_heads: int = 8, F_edge: int = 5, dropout: float = 0.3):
        super().__init__()
        # 타입별 입력 투영
        self.W_gene = nn.Linear(F_in, F_in, bias=False)
        self.W_drug = nn.Linear(F_in, F_in, bias=False)

        # Layer 1: concat → hidden × n_heads
        self.gat1 = EdgeAwareGATLayer(F_in, hidden, n_heads, F_edge,
                                      dropout=dropout, concat=True)
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: mean → out
        self.gat2 = EdgeAwareGATLayer(hidden * n_heads, out, n_heads, F_edge,
                                      dropout=dropout, concat=False)

    def forward(self, x, edge_index, edge_attr, node_types, degree_src=None):
        """
        반환: Z (N, out), alpha (E, n_heads) ← Layer2 attention
        """
        # 타입별 투영
        h = torch.where(
            node_types.unsqueeze(1) == 0,
            self.W_gene(x),
            self.W_drug(x)
        )

        h, _     = self.gat1(h, edge_index, edge_attr, degree_src)
        h        = self.drop1(h)
        h, alpha = self.gat2(h, edge_index, edge_attr, degree_src)
        return h, alpha


# ══════════════════════════════════════════════════════════
# 7. Link Prediction 손실 함수
# ══════════════════════════════════════════════════════════
class LinkPredLoss(nn.Module):
    """
    Negative Sampling 기반 Link Prediction BCE Loss.
    drug 노드 가중치 ×10 (소수 노드 학습 보완).
    """
    def __init__(self, drug_weight: float = 10.0, reg_lambda: float = 0.01):
        super().__init__()
        self.drug_w   = drug_weight
        self.reg_lam  = reg_lambda

    def forward(self, Z, pos_edge, neg_edge, node_types):
        """
        Z         : (N, out) 임베딩
        pos_edge  : (2, E_pos) 실제 엣지
        neg_edge  : (2, E_neg) 가짜 엣지
        """
        # Positive score
        pos_score = (Z[pos_edge[0]] * Z[pos_edge[1]]).sum(dim=1)
        # Negative score
        neg_score = (Z[neg_edge[0]] * Z[neg_edge[1]]).sum(dim=1)

        # drug 노드 관련 엣지 가중치 적용
        pos_w = torch.where(
            (node_types[pos_edge[0]] == 1) | (node_types[pos_edge[1]] == 1),
            torch.full_like(pos_score, self.drug_w),
            torch.ones_like(pos_score)
        )
        neg_w = torch.where(
            (node_types[neg_edge[0]] == 1) | (node_types[neg_edge[1]] == 1),
            torch.full_like(neg_score, self.drug_w),
            torch.ones_like(neg_score)
        )

        loss = -(pos_w * F.logsigmoid(pos_score)).mean() \
               -(neg_w * F.logsigmoid(-neg_score)).mean()

        # degree 정규화 항 (허브 임베딩 크기 억제)
        reg = self.reg_lam * Z.pow(2).mean()
        return loss + reg


def _sample_negative_edges(pos_edge, N, n_sample, existing_set):
    """실제 엣지를 제외한 음성 엣지 랜덤 샘플링."""
    neg = []
    while len(neg) < n_sample:
        u = np.random.randint(0, N)
        v = np.random.randint(0, N)
        if u != v and (u, v) not in existing_set and (v, u) not in existing_set:
            neg.append((u, v))
    return torch.tensor(neg, dtype=torch.long).t()


# ══════════════════════════════════════════════════════════
# 8. GAT 학습
# ══════════════════════════════════════════════════════════
def train_gat(G: nx.Graph,
              X: torch.Tensor,
              edge_index: torch.Tensor,
              edge_attr: torch.Tensor,
              node_types: torch.Tensor,
              n_epochs: int = 200,
              lr: float = 0.001,
              patience: int = 20,
              batch_size: int = 4096,
              seed: int = 42) -> nn.Module:
    """
    GAT 학습.

    학습 전략:
      - 엣지를 Train/Val/Test = 80/10/10% 분할
      - 각 배치마다 negative sampling (1:1 비율)
      - gradient clipping (max_norm=1.0)
      - Val loss 기준 조기 종료 (patience=20)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N  = X.size(0)
    E  = edge_index.size(1) // 2   # 단방향 엣지 수

    # degree 계산 (degree-aware scaling용)
    degree = torch.zeros(N, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0],
                        torch.ones(edge_index.size(1), dtype=torch.long))
    degree_src = degree[edge_index[0]]    # (2E,) 각 엣지의 src degree

    # 단방향 엣지만 추출 (짝수 인덱스 = 원래 방향)
    uni_idx    = torch.arange(0, edge_index.size(1), 2)
    uni_edges  = edge_index[:, uni_idx]   # (2, E)

    # Train / Val / Test 분할
    perm    = torch.randperm(E)
    n_train = int(E * 0.8)
    n_val   = int(E * 0.1)
    train_e = uni_edges[:, perm[:n_train]]
    val_e   = uni_edges[:, perm[n_train:n_train+n_val]]

    existing_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    model     = HeterogeneousGAT(F_in=X.size(1), hidden=16, out=16,
                                  n_heads=8, F_edge=edge_attr.size(1))
    criterion = LinkPredLoss(drug_weight=10.0, reg_lambda=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val   = float("inf")
    no_improve = 0
    best_state = None

    print(f"\n[GAT] 학습 시작 | 노드:{N:,} 엣지:{E:,} | epochs:{n_epochs} patience:{patience}")
    print(f"  full-graph forward, link-pred batch_size:{batch_size} lr:{lr} n_heads:8 hidden:16 out:16")

    for epoch in range(1, n_epochs + 1):
        # ── 에폭당 1회 전체 그래프 forward (미니배치 반복 X)
        model.train()
        optimizer.zero_grad()
        Z, _ = model(X, edge_index, edge_attr, node_types, degree_src)

        # 전체 train 엣지에서 배치 크기만큼 샘플링해 loss 계산
        idx       = torch.randperm(n_train)[:batch_size]
        pos_batch = train_e[:, idx]
        neg_batch = _sample_negative_edges(pos_batch, N, idx.size(0), existing_set)
        loss      = criterion(Z, pos_batch, neg_batch, node_types)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss = loss.item()

        # Validation (no_grad)
        model.eval()
        with torch.no_grad():
            Z, _     = model(X, edge_index, edge_attr, node_types, degree_src)
            neg_val  = _sample_negative_edges(val_e, N, min(batch_size, val_e.size(1)), existing_set)
            val_loss = criterion(Z, val_e[:, :batch_size], neg_val, node_types).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        # 조기 종료
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [EarlyStopping] Epoch {epoch} | best_val={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[GAT] 학습 완료 | best_val_loss={best_val:.4f}")
    return model


# ══════════════════════════════════════════════════════════
# 9. Attention 추출 → 방향별 dict
# ══════════════════════════════════════════════════════════
def extract_directed_attention(model: nn.Module,
                                X, edge_index, edge_attr,
                                node_types, node_list) -> dict:
    """
    학습된 GAT에서 방향별 엣지 attention 추출.
    alpha(i→j) ≠ alpha(j→i) 를 별도로 저장.

    반환: {(src_name, dst_name): float}  (8 head 평균)
    """
    N = X.size(0)
    degree = torch.zeros(N, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0],
                        torch.ones(edge_index.size(1), dtype=torch.long))
    degree_src = degree[edge_index[0]]

    model.eval()
    with torch.no_grad():
        _, alpha = model(X, edge_index, edge_attr, node_types, degree_src)
        # alpha: (2E, n_heads) → head 평균 → (2E,)
        alpha_mean = alpha.mean(dim=1).cpu().numpy()

    directed = {}
    for e_idx in range(edge_index.size(1)):
        src_name = node_list[edge_index[0, e_idx].item()]
        dst_name = node_list[edge_index[1, e_idx].item()]
        directed[(src_name, dst_name)] = float(alpha_mean[e_idx])

    print(f"[GAT] attention 추출 완료 | {len(directed):,}개 방향별 엣지")
    return directed


# ══════════════════════════════════════════════════════════
# 10. Attention → 전이 확률 캐시 (ver1 대체)
# ══════════════════════════════════════════════════════════
def build_attention_neighbor_cache(G: nx.Graph,
                                    directed_attention: dict,
                                    alpha_mix: float = 0.7) -> dict:
    """
    GAT attention + 원본 weight 혼합으로 전이 확률 캐시 구성.

    combined = alpha_mix × GAT_att + (1-alpha_mix) × 원본_weight_normalized
    → alpha_mix=0.7: GAT 구조 반영 + 원본 weight fallback 보장
    """
    cache = {}
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if not nbrs:
            cache[node] = ([], None)
            continue

        # GAT attention (방향별, 없으면 epsilon)
        gat_w = np.array([
            directed_attention.get((node, nb), 1e-8)
            for nb in nbrs
        ], dtype=float)
        gat_w = np.clip(gat_w, 1e-8, None)
        gat_w /= gat_w.sum()

        # 원본 weight 정규화
        orig_w = np.array([G[node][nb]["weight"] for nb in nbrs], dtype=float)
        orig_w = np.clip(orig_w + 1e-8, 1e-8, None)
        orig_w /= orig_w.sum()

        # 혼합
        combined  = alpha_mix * gat_w + (1 - alpha_mix) * orig_w
        combined /= combined.sum()

        cache[node] = (nbrs, combined)
    return cache


# ══════════════════════════════════════════════════════════
# 11. 진행률 모니터 (ver1 동일)
# ══════════════════════════════════════════════════════════
def _progress_monitor(shared_done, total_walks, stop_event):
    """0.5초마다 ASCII 진행바 출력."""
    t0 = time.time()
    while not stop_event.is_set():
        done    = shared_done.value
        pct     = done / total_walks * 100
        elapsed = time.time() - t0
        speed   = done / elapsed if elapsed > 0 else 0
        eta     = (total_walks - done) / speed if speed > 0 else 0
        filled  = int(30 * done / total_walks)
        bar     = "#" * filled + "-" * (30 - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {done:,}/{total_walks:,}  "
              f"{speed:,.0f} walks/s  ETA {eta:.0f}s   ", end="", flush=True)
        if done >= total_walks:
            break
        stop_event.wait(timeout=0.5)
    print()


# ══════════════════════════════════════════════════════════
# 12. RWR 시뮬레이션 (attention 기반 캐시 사용)
# ══════════════════════════════════════════════════════════
def simulate_random_walk_v2(G: nx.Graph,
                              neighbor_cache: dict,
                              n_walks:      int   = 1000,
                              walk_length:  int   = 100,
                              restart_prob: float = 0.15,
                              seed:         int   = 42,
                              n_workers:    int   = 12) -> Counter:
    """
    GAT attention 기반 전이 확률을 사용하는 RWR.
    멀티프로세싱 + 실시간 진행률 (ver1 구조 재사용).
    """
    nodes       = list(G.nodes())
    n_nodes     = len(nodes)
    total_walks = n_nodes * n_walks
    n_workers   = min(n_workers, n_nodes, cpu_count())

    print(f"\n[RWR-v2] 노드 {n_nodes:,} × {n_walks}회 = {total_walks:,}번 워크")
    print(f"  walk_length={walk_length} restart_prob={restart_prob} n_workers={n_workers}")

    chunks      = [nodes[i::n_workers] for i in range(n_workers)]
    shared_done = Value("l", 0)
    worker_args = [
        (chunk, neighbor_cache, n_walks, walk_length, restart_prob, seed + i)
        for i, chunk in enumerate(chunks)
    ]

    stop_event = threading.Event()
    monitor    = threading.Thread(
        target=_progress_monitor,
        args=(shared_done, total_walks, stop_event),
        daemon=True
    )
    monitor.start()

    t0          = time.time()
    visit_count = Counter()
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(shared_done,)) as pool:
        for r in pool.map(_walk_worker, worker_args):
            visit_count += r

    stop_event.set()
    monitor.join()

    elapsed = time.time() - t0
    print(f"[RWR-v2] 완료 | {elapsed:.1f}s | {total_walks/elapsed:,.0f} walks/s | "
          f"총 방문: {sum(visit_count.values()):,}")
    return visit_count


# ══════════════════════════════════════════════════════════
# 13. 결과 출력
# ══════════════════════════════════════════════════════════
def print_results(G, visit_count, top_k=20):
    """전체/Gene/Drug Top-K 출력."""
    total        = sum(visit_count.values())
    visit_prob   = {n: c / total for n, c in visit_count.items()}
    sorted_nodes = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 75)
    print(f"    [GAT+RWR 노드 중요도 Top {top_k}]")
    print("=" * 75)
    print(f"{'Rank':<5} {'Node':<22} {'Type':<6} {'방문횟수':>12} {'방문확률':>10} {'Degree':>8}")
    print("-" * 75)
    for rank, (node, cnt) in enumerate(sorted_nodes[:top_k], 1):
        ntype  = G.nodes[node].get("node_type", "?")
        prob   = visit_prob[node]
        degree = G.degree(node)
        print(f"{rank:<5} {node:<22} {ntype:<6} {cnt:>12,} {prob:>10.6f} {degree:>8}")
    print("=" * 75)

    gene_top = [(n,c) for n,c in sorted_nodes if G.nodes[n].get("node_type")=="gene"]
    drug_top = [(n,c) for n,c in sorted_nodes if G.nodes[n].get("node_type")=="drug"]

    print(f"\n[Gene Top 10]")
    for rank, (n, c) in enumerate(gene_top[:10], 1):
        print(f"  {rank}. {n:<20} {c:>12,}  ({visit_prob[n]:.6f})  degree={G.degree(n)}")

    print(f"\n[Drug Top 10]")
    for rank, (n, c) in enumerate(drug_top[:10], 1):
        print(f"  {rank}. {n:<22} {c:>12,}  ({visit_prob[n]:.6f})  degree={G.degree(n)}")

    top_node, top_cnt = sorted_nodes[0]
    print(f"\n★ 최고 중요도 노드: [{top_node}]  ({visit_prob[top_node]*100:.3f}%)")
    return visit_prob, sorted_nodes


# ══════════════════════════════════════════════════════════
# 14. 결과 저장
# ══════════════════════════════════════════════════════════
def save_results_v2(G, visit_count, visit_prob, directed_attention, base_path):
    """
    node_importance_v2.tsv 저장.
    gat_hub_score: 해당 노드로 향하는 평균 attention (허브 지표).
    """
    # 노드별 평균 incoming attention 계산
    incoming = {}
    for (src, dst), att in directed_attention.items():
        incoming.setdefault(dst, []).append(att)
    gat_hub = {n: float(np.mean(v)) if v else 0.0 for n, v in incoming.items()}

    rows = [
        {
            "node":          node,
            "node_type":     G.nodes[node].get("node_type", "?"),
            "visit_count":   visit_count.get(node, 0),
            "visit_prob":    round(visit_prob.get(node, 0.0), 8),
            "degree":        G.degree(node),
            "gat_hub_score": round(gat_hub.get(node, 0.0), 8),
        }
        for node in G.nodes()
    ]
    df = (pd.DataFrame(rows)
            .sort_values("visit_count", ascending=False)
            .reset_index(drop=True))
    df.index += 1

    out = os.path.join(base_path, "node_importance_v2.tsv")
    df.to_csv(out, sep="\t", index_label="rank")
    print(f"\n[Saved] {out}  ({len(df):,} 노드)")


# ══════════════════════════════════════════════════════════
# 15. Main
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    BASE_PATH = "C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/"

    # 1. 데이터 로드 + 네트워크 구성
    gga, ppi, dgt = load_data(BASE_PATH)
    G             = build_heterogeneous_network(gga, ppi, dgt)

    # 2. 그래프 → Tensor 변환
    X, edge_index, edge_attr, node_list, node_types = graph_to_tensors(G)

    # 3. GAT 학습
    model = train_gat(
        G, X, edge_index, edge_attr, node_types,
        n_epochs  = 200,
        lr        = 0.001,
        patience  = 20,
        batch_size= 4096,
        seed      = 42,
    )

    # 4. Attention 추출 → 방향별 dict
    directed_attention = extract_directed_attention(
        model, X, edge_index, edge_attr, node_types, node_list
    )

    # 5. Attention → 전이 확률 캐시 (GAT 0.7 + 원본 0.3)
    neighbor_cache = build_attention_neighbor_cache(
        G, directed_attention, alpha_mix=0.7
    )

    # 6. GAT attention 기반 RWR
    visit_count = simulate_random_walk_v2(
        G, neighbor_cache,
        n_walks      = 1000,
        walk_length  = 100,
        restart_prob = 0.15,
        seed         = 42,
        n_workers    = 12,
    )

    # 7. 결과 출력 및 저장
    visit_prob, sorted_nodes = print_results(G, visit_count, top_k=20)
    save_results_v2(G, visit_count, visit_prob, directed_attention, BASE_PATH)
