"""
Network RWR ver3: GAT + RWR (새 A/B/C 데이터 기반)
=====================================================
ver2 대비 변경점:
  - _A_ (GGA): ERG-gene 단일 벡터 → gene-gene pairwise absPearsonR 행렬
    (TMPRSS2-ERG PRAD TCGA 발현 기반, 3,267,720 쌍, threshold=0.3)
  - _B_ (PPI): 동일 (9606.ppi.physical, 1,477,610 rows)
  - _C_ (DGT): 동일 (CIViC drug-gene target, 1,628 rows)
  - 결과 저장: node_importance_v3.tsv (rank / node / node_type /
                visit_count / visit_prob / degree / gat_hub_score)

[GGA 로드 방식 변경]
  기존: gene | absPearsonR  (ERG 기준 단방향)
  신규: gene_A | gene_B | pearsonR | absPearsonR  (pairwise 양방향)
  → build_heterogeneous_network()의 GGA 추가 로직을 pairwise 방식으로 전면 교체

[파이프라인]
  데이터 로드 + 이종 네트워크 구성
    → 그래프 → Tensor 변환 (X, edge_index, edge_attr)
    → GAT 학습 (200 epochs, patience=20, Early Stopping)
    → Attention 추출 → 전이 확률 혼합 캐시
    → RWR 시뮬레이션 (멀티프로세싱, n_workers=12)
    → 결과 저장

[GAT 아키텍처] (ver2 동일)
  Layer1: EdgeAwareGATLayer(8→16, 8-head, concat) → (N, 128)
  Layer2: EdgeAwareGATLayer(128→16, 8-head, mean) → (N, 16)
  Attention: e_ij = LeakyReLU(a^T·[Wh_i || Wh_j || W_e·edge_feat])
             alpha_ij = softmax over src-neighbors
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
    """단일 워커 RWR 시뮬레이션."""
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
#    __A__: gene-gene pairwise absPearsonR (신규 방식)
#    __B__: PPI (ver2 동일)
#    __C__: DGT (ver2 동일)
# ══════════════════════════════════════════════════════════
def load_data(base_path: str, gga_path: str):
    """
    GGA: gene_A / gene_B / pearsonR / absPearsonR 형식 파일
    PPI, DGT: base_path 내 _B_ / _C_ 파일
    """
    # ── GGA (새 pairwise 형식)
    gga = pd.read_csv(gga_path, sep="\t")
    # GGA에 등장하는 모든 유전자 집합
    gene_set = set(gga["gene_A"]) | set(gga["gene_B"])
    print(f"[Load] GGA: {len(gga):,} 쌍 / {len(gene_set):,} genes  <- {os.path.basename(gga_path)}")

    # ── PPI
    path_b  = find_file(base_path, "_B_")
    ppi_raw = pd.read_csv(path_b, sep="\t")
    ppi     = ppi_raw[
        ppi_raw["gene1"].isin(gene_set) | ppi_raw["gene2"].isin(gene_set)
    ].reset_index(drop=True)
    print(f"[Load] PPI: {len(ppi_raw):,} -> {len(ppi):,} rows  <- {os.path.basename(path_b)}")

    # ── DGT
    path_c          = find_file(base_path, "_C_")
    dgt_raw         = pd.read_csv(path_c, sep="\t")
    dgt_raw["use_gene"] = dgt_raw["final_target_gene"].fillna(dgt_raw["target_gene"])
    dgt             = dgt_raw[dgt_raw["use_gene"].isin(gene_set)].reset_index(drop=True)
    print(f"[Load] DGT: {len(dgt_raw):,} -> {len(dgt):,} rows  <- {os.path.basename(path_c)}")

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
# 3. 이종 네트워크 구성 (GGA pairwise 방식으로 전면 교체)
# ══════════════════════════════════════════════════════════
def build_heterogeneous_network(gga: pd.DataFrame,
                                 ppi: pd.DataFrame,
                                 dgt: pd.DataFrame) -> nx.Graph:
    """
    GGA/PPI/DGT 통합 이종 그래프 구성.

    GGA (신규 pairwise):
      - gene_A <-> gene_B 엣지, 가중치 = zscore_sigmoid(absPearsonR)
      - GGA+PPI 중복 시 weighted average (GGA=0.4, PPI=0.6)
    PPI, DGT: ver2 동일
    """
    G = nx.Graph()
    GGA_W, PPI_W = 0.4, 0.6

    # ── GGA: pairwise gene-gene 엣지 추가
    gga = gga.copy()
    gga["norm_w"] = zscore_sigmoid_normalize(gga["absPearsonR"].astype(float))
    for _, row in gga.iterrows():
        g1, g2, w = str(row["gene_A"]), str(row["gene_B"]), float(row["norm_w"])
        G.add_node(g1, node_type="gene")
        G.add_node(g2, node_type="gene")
        if G.has_edge(g1, g2):
            ew = G[g1][g2]["weight"]
            G[g1][g2].update({"weight": GGA_W * w + PPI_W * ew,
                               "edge_type": "gga+ppi"})
        else:
            G.add_edge(g1, g2, weight=w, edge_type="gga")
    print(f"[Network] GGA 완료 | 노드: {G.number_of_nodes():,} 엣지: {G.number_of_edges():,}")

    # ── PPI
    ppi = ppi.copy()
    ppi["norm_w"] = zscore_sigmoid_normalize(ppi["score"].astype(float))
    for _, row in ppi.iterrows():
        g1, g2, w = str(row["gene1"]), str(row["gene2"]), float(row["norm_w"])
        G.add_node(g1, node_type="gene")
        G.add_node(g2, node_type="gene")
        if G.has_edge(g1, g2):
            ew = G[g1][g2]["weight"]
            G[g1][g2].update({"weight": GGA_W * ew + PPI_W * w,
                               "edge_type": "gga+ppi"})
        else:
            G.add_edge(g1, g2, weight=w, edge_type="ppi")
    print(f"[Network] PPI 완료 | 노드: {G.number_of_nodes():,} 엣지: {G.number_of_edges():,}")

    # ── DGT
    dgt = dgt.copy()
    dgt["norm_w"] = zscore_sigmoid_normalize(dgt["score"].astype(float))
    for _, row in dgt.iterrows():
        drug, gene, w = str(row["drug"]), str(row["use_gene"]), float(row["norm_w"])
        if not drug or not gene:
            continue
        G.add_node(drug, node_type="drug")
        G.add_node(gene, node_type="gene")
        if G.has_edge(drug, gene):
            if w > G[drug][gene]["weight"]:
                G[drug][gene]["weight"] = w
        else:
            G.add_edge(drug, gene, weight=w, edge_type="dgt")
    print(f"[Network] DGT 완료 | 노드: {G.number_of_nodes():,} 엣지: {G.number_of_edges():,}")

    gene_n = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "gene")
    drug_n = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "drug")
    ws     = [d["weight"] for _, _, d in G.edges(data=True)]
    etypes = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    print(f"\n[Network] 총 노드: {G.number_of_nodes():,} (Gene:{gene_n:,} Drug:{drug_n:,})")
    print(f"  총 엣지: {G.number_of_edges():,}  타입: { {k: f'{v:,}' for k, v in etypes.items()} }")
    print(f"  Weight: min={min(ws):.4f} mean={np.mean(ws):.4f} max={max(ws):.4f}")
    return G


# ══════════════════════════════════════════════════════════
# 4. 그래프 → PyTorch Tensor 변환 (ver2 동일)
# ══════════════════════════════════════════════════════════
EDGE_TYPE_MAP = {"gga": 0, "ppi": 1, "gga+ppi": 2, "dgt": 3}

def graph_to_tensors(G: nx.Graph):
    """
    nx.Graph → PyTorch Tensor 변환.
    반환: X(N,8), edge_index(2,2E), edge_attr(2E,5),
          node_list[str], node_types(N,)
    """
    node_list = list(G.nodes())
    n2i       = {n: i for i, n in enumerate(node_list)}
    N         = len(node_list)

    X = np.zeros((N, 8), dtype=np.float32)
    for i, node in enumerate(node_list):
        ntype  = G.nodes[node].get("node_type", "gene")
        nbrs   = list(G.neighbors(node))
        degree = len(nbrs)
        ws     = [G[node][nb]["weight"] for nb in nbrs] if nbrs else [0.0]
        ec     = Counter(G[node][nb].get("edge_type", "gga") for nb in nbrs)
        X[i, 0] = 1.0 if ntype == "gene" else 0.0
        X[i, 1] = 1.0 if ntype == "drug" else 0.0
        X[i, 2] = math.log(1 + degree)
        X[i, 3] = ec.get("gga", 0) / max(degree, 1)
        X[i, 4] = ec.get("ppi", 0) / max(degree, 1)
        X[i, 5] = ec.get("dgt", 0) / max(degree, 1)
        X[i, 6] = float(np.mean(ws))
        X[i, 7] = float(np.max(ws))

    srcs, dsts, attrs = [], [], []
    for u, v, d in G.edges(data=True):
        i, j  = n2i[u], n2i[v]
        etype = d.get("edge_type", "gga")
        w     = float(d.get("weight", 0.5))
        oh    = [0.0] * 4
        oh[EDGE_TYPE_MAP.get(etype, 0)] = 1.0
        feat  = oh + [w]
        srcs.append(i); dsts.append(j); attrs.append(feat)
        srcs.append(j); dsts.append(i); attrs.append(feat)

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_attr  = torch.tensor(attrs, dtype=torch.float32)
    X_t        = torch.tensor(X, dtype=torch.float32)
    node_types = torch.tensor(
        [0 if G.nodes[n].get("node_type", "gene") == "gene" else 1
         for n in node_list],
        dtype=torch.long
    )

    print(f"[Tensor] X:{tuple(X_t.shape)}  "
          f"edge_index:{tuple(edge_index.shape)}  "
          f"edge_attr:{tuple(edge_attr.shape)}")
    return X_t, edge_index, edge_attr, node_list, node_types


# ══════════════════════════════════════════════════════════
# 5. Edge-Aware GAT 레이어 (ver2 동일)
# ══════════════════════════════════════════════════════════
class EdgeAwareGATLayer(nn.Module):
    """
    단일 GAT 레이어 (Edge-Aware).
    e_ij = LeakyReLU(a^T · [Wh_i || Wh_j || W_e·edge_feat])
    alpha_ij = softmax over src-neighbors
    degree-aware scaling: e_ij *= 1/sqrt(degree_src)
    """
    def __init__(self, F_in, F_out, n_heads, F_edge=5,
                 dropout=0.3, concat=True, leaky_slope=0.2):
        super().__init__()
        self.F_out   = F_out
        self.n_heads = n_heads
        self.concat  = concat

        self.W      = nn.Parameter(torch.empty(n_heads, F_in, F_out))
        self.a      = nn.Parameter(torch.empty(n_heads, 2 * F_out + F_edge))
        self.W_edge = nn.Linear(F_edge, F_edge, bias=False)
        self.leaky  = nn.LeakyReLU(leaky_slope)
        self.dropout= nn.Dropout(dropout)
        self.elu    = nn.ELU()

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr, degree_src=None):
        N   = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        Wh     = torch.einsum("nf,hfo->nho", x, self.W)
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]
        ef     = self.W_edge(edge_attr).unsqueeze(1).expand(-1, self.n_heads, -1)
        cat    = torch.cat([Wh_src, Wh_dst, ef], dim=2)
        e      = (cat * self.a.unsqueeze(0)).sum(dim=2)
        e      = self.leaky(e)

        if degree_src is not None:
            scale = 1.0 / torch.sqrt(degree_src.float().clamp(min=1)).unsqueeze(1)
            e     = e * scale

        e_max = torch.zeros(N, self.n_heads, device=x.device)
        e_max.scatter_reduce_(0, src.unsqueeze(1).expand_as(e), e,
                              reduce="amax", include_self=True)
        e_exp = torch.exp(e - e_max[src])
        e_sum = torch.zeros(N, self.n_heads, device=x.device)
        e_sum.scatter_add_(0, src.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha = e_exp / (e_sum[src] + 1e-16)
        alpha = self.dropout(alpha)

        msg = Wh_dst * alpha.unsqueeze(2)
        out = torch.zeros(N, self.n_heads, self.F_out, device=x.device)
        out.scatter_add_(0, src.view(-1, 1, 1).expand_as(msg), msg)

        if self.concat:
            out = out.view(N, self.n_heads * self.F_out)
        else:
            out = out.mean(dim=1)
        return self.elu(out), alpha


# ══════════════════════════════════════════════════════════
# 6. 이종 GAT 모델 2-layer (ver2 동일)
# ══════════════════════════════════════════════════════════
class HeterogeneousGAT(nn.Module):
    """
    2-레이어 Edge-Aware GAT (type-specific projection 포함).
    gene → W_gene, drug → W_drug 별도 선형 변환.
    """
    def __init__(self, F_in=8, hidden=16, out=16,
                 n_heads=8, F_edge=5, dropout=0.3):
        super().__init__()
        self.W_gene = nn.Linear(F_in, F_in, bias=False)
        self.W_drug = nn.Linear(F_in, F_in, bias=False)
        self.gat1   = EdgeAwareGATLayer(F_in, hidden, n_heads, F_edge,
                                        dropout=dropout, concat=True)
        self.drop1  = nn.Dropout(dropout)
        self.gat2   = EdgeAwareGATLayer(hidden * n_heads, out, n_heads, F_edge,
                                        dropout=dropout, concat=False)

    def forward(self, x, edge_index, edge_attr, node_types, degree_src=None):
        h = torch.where(node_types.unsqueeze(1) == 0,
                        self.W_gene(x), self.W_drug(x))
        h, _     = self.gat1(h, edge_index, edge_attr, degree_src)
        h        = self.drop1(h)
        h, alpha = self.gat2(h, edge_index, edge_attr, degree_src)
        return h, alpha


# ══════════════════════════════════════════════════════════
# 7. Link Prediction 손실 (ver2 동일)
# ══════════════════════════════════════════════════════════
class LinkPredLoss(nn.Module):
    """BCE Link Prediction Loss + degree 정규화. drug×10 가중치."""
    def __init__(self, drug_weight=10.0, reg_lambda=0.01):
        super().__init__()
        self.drug_w  = drug_weight
        self.reg_lam = reg_lambda

    def forward(self, Z, pos_edge, neg_edge, node_types):
        pos_score = (Z[pos_edge[0]] * Z[pos_edge[1]]).sum(dim=1)
        neg_score = (Z[neg_edge[0]] * Z[neg_edge[1]]).sum(dim=1)
        pos_w = torch.where(
            (node_types[pos_edge[0]] == 1) | (node_types[pos_edge[1]] == 1),
            torch.full_like(pos_score, self.drug_w), torch.ones_like(pos_score))
        neg_w = torch.where(
            (node_types[neg_edge[0]] == 1) | (node_types[neg_edge[1]] == 1),
            torch.full_like(neg_score, self.drug_w), torch.ones_like(neg_score))
        loss = -(pos_w * F.logsigmoid(pos_score)).mean() \
               -(neg_w * F.logsigmoid(-neg_score)).mean()
        return loss + self.reg_lam * Z.pow(2).mean()


def _sample_negative_edges(pos_edge, N, n_sample, existing_set):
    neg = []
    while len(neg) < n_sample:
        u, v = np.random.randint(0, N), np.random.randint(0, N)
        if u != v and (u, v) not in existing_set and (v, u) not in existing_set:
            neg.append((u, v))
    return torch.tensor(neg, dtype=torch.long).t()


# ══════════════════════════════════════════════════════════
# 8. GAT 학습 (ver2 동일: epoch당 1회 full-graph forward)
# ══════════════════════════════════════════════════════════
def train_gat(G, X, edge_index, edge_attr, node_types,
              n_epochs=200, lr=0.001, patience=20,
              batch_size=4096, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    N  = X.size(0)
    E  = edge_index.size(1) // 2

    degree = torch.zeros(N, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0],
                        torch.ones(edge_index.size(1), dtype=torch.long))
    degree_src = degree[edge_index[0]]

    uni_idx   = torch.arange(0, edge_index.size(1), 2)
    uni_edges = edge_index[:, uni_idx]

    perm    = torch.randperm(E)
    n_train = int(E * 0.8)
    n_val   = int(E * 0.1)
    train_e = uni_edges[:, perm[:n_train]]
    val_e   = uni_edges[:, perm[n_train:n_train + n_val]]

    existing_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    model     = HeterogeneousGAT(F_in=X.size(1), hidden=16, out=16,
                                  n_heads=8, F_edge=edge_attr.size(1))
    criterion = LinkPredLoss(drug_weight=10.0, reg_lambda=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, no_improve, best_state = float("inf"), 0, None

    print(f"\n[GAT] 학습 시작 | 노드:{N:,} 엣지:{E:,} | epochs:{n_epochs} patience:{patience}")
    print(f"  full-graph forward / link-pred batch:{batch_size} lr:{lr}")

    for epoch in range(1, n_epochs + 1):
        # epoch당 1회 full-graph forward
        model.train()
        optimizer.zero_grad()
        Z, _ = model(X, edge_index, edge_attr, node_types, degree_src)

        idx       = torch.randperm(n_train)[:batch_size]
        pos_batch = train_e[:, idx]
        neg_batch = _sample_negative_edges(pos_batch, N, idx.size(0), existing_set)
        loss      = criterion(Z, pos_batch, neg_batch, node_types)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            Z, _     = model(X, edge_index, edge_attr, node_types, degree_src)
            neg_val  = _sample_negative_edges(
                val_e, N, min(batch_size, val_e.size(1)), existing_set)
            val_loss = criterion(
                Z, val_e[:, :batch_size], neg_val, node_types).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"train={loss.item():.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [EarlyStopping] Epoch {epoch} | best_val={best_val:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"[GAT] 완료 | best_val={best_val:.4f}")
    return model


# ══════════════════════════════════════════════════════════
# 9. Attention 추출 (ver2 동일)
# ══════════════════════════════════════════════════════════
def extract_directed_attention(model, X, edge_index, edge_attr,
                                node_types, node_list):
    N = X.size(0)
    degree = torch.zeros(N, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0],
                        torch.ones(edge_index.size(1), dtype=torch.long))
    degree_src = degree[edge_index[0]]
    model.eval()
    with torch.no_grad():
        _, alpha = model(X, edge_index, edge_attr, node_types, degree_src)
        alpha_mean = alpha.mean(dim=1).cpu().numpy()
    directed = {
        (node_list[edge_index[0, e].item()],
         node_list[edge_index[1, e].item()]): float(alpha_mean[e])
        for e in range(edge_index.size(1))
    }
    print(f"[GAT] attention 추출 완료 | {len(directed):,}개 방향별 엣지")
    return directed


# ══════════════════════════════════════════════════════════
# 10. Attention → 전이 확률 캐시 (ver2 동일)
# ══════════════════════════════════════════════════════════
def build_attention_neighbor_cache(G, directed_attention, alpha_mix=0.7):
    """GAT attention(0.7) + 원본 weight(0.3) 혼합 전이 확률 캐시."""
    cache = {}
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if not nbrs:
            cache[node] = ([], None)
            continue
        gat_w = np.array([
            directed_attention.get((node, nb), 1e-8) for nb in nbrs
        ], dtype=float)
        gat_w = np.clip(gat_w, 1e-8, None)
        gat_w /= gat_w.sum()

        orig_w = np.array([G[node][nb]["weight"] for nb in nbrs], dtype=float)
        orig_w = np.clip(orig_w + 1e-8, 1e-8, None)
        orig_w /= orig_w.sum()

        combined  = alpha_mix * gat_w + (1 - alpha_mix) * orig_w
        combined /= combined.sum()
        cache[node] = (nbrs, combined)
    return cache


# ══════════════════════════════════════════════════════════
# 11. 진행률 모니터
# ══════════════════════════════════════════════════════════
def _progress_monitor(shared_done, total_walks, stop_event):
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
# 12. RWR 시뮬레이션 (ver2 동일)
# ══════════════════════════════════════════════════════════
def simulate_random_walk_v3(G, neighbor_cache,
                             n_walks=1000, walk_length=100,
                             restart_prob=0.15, seed=42, n_workers=12):
    nodes       = list(G.nodes())
    n_nodes     = len(nodes)
    total_walks = n_nodes * n_walks
    n_workers   = min(n_workers, n_nodes, cpu_count())

    print(f"\n[RWR-v3] 노드 {n_nodes:,} x {n_walks}회 = {total_walks:,} walks")
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
    print(f"[RWR-v3] 완료 | {elapsed:.1f}s | {total_walks/elapsed:,.0f} walks/s | "
          f"총 방문: {sum(visit_count.values()):,}")
    return visit_count


# ══════════════════════════════════════════════════════════
# 13. 결과 출력
# ══════════════════════════════════════════════════════════
def print_results(G, visit_count, top_k=20):
    total        = sum(visit_count.values())
    visit_prob   = {n: c / total for n, c in visit_count.items()}
    sorted_nodes = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 78)
    print(f"    [GAT+RWR v3 노드 중요도 Top {top_k}]")
    print("=" * 78)
    print(f"{'Rank':<5} {'Node':<22} {'Type':<6} {'방문횟수':>12} {'방문확률':>10} {'Degree':>8}")
    print("-" * 78)
    for rank, (node, cnt) in enumerate(sorted_nodes[:top_k], 1):
        ntype  = G.nodes[node].get("node_type", "?")
        prob   = visit_prob[node]
        degree = G.degree(node)
        print(f"{rank:<5} {node:<22} {ntype:<6} {cnt:>12,} {prob:>10.6f} {degree:>8}")
    print("=" * 78)

    gene_top = [(n, c) for n, c in sorted_nodes if G.nodes[n].get("node_type") == "gene"]
    drug_top = [(n, c) for n, c in sorted_nodes if G.nodes[n].get("node_type") == "drug"]

    print(f"\n[Gene Top 10]")
    for rank, (n, c) in enumerate(gene_top[:10], 1):
        print(f"  {rank}. {n:<22} {c:>12,}  ({visit_prob[n]:.6f})  degree={G.degree(n)}")

    print(f"\n[Drug Top 10]")
    for rank, (n, c) in enumerate(drug_top[:10], 1):
        print(f"  {rank}. {n:<22} {c:>12,}  ({visit_prob[n]:.6f})  degree={G.degree(n)}")

    top_node, top_cnt = sorted_nodes[0]
    print(f"\n* 최고 중요도 노드: [{top_node}]  ({visit_prob[top_node]*100:.3f}%)")
    return visit_prob, sorted_nodes


# ══════════════════════════════════════════════════════════
# 14. 결과 저장
# ══════════════════════════════════════════════════════════
def save_results_v3(G, visit_count, visit_prob, directed_attention, base_path):
    """node_importance_v3.tsv 저장."""
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

    out = os.path.join(base_path, "node_importance_v3.tsv")
    df.to_csv(out, sep="\t", index_label="rank")
    print(f"\n[Saved] {out}  ({len(df):,} 노드)")
    return out


# ══════════════════════════════════════════════════════════
# 15. Main
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    BASE_PATH = "C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/"
    GGA_PATH  = (r"C:\Users\user\Desktop\CNN_CVPR\__DATAAn__\random_work_test"
                 r"\_3_TMPRSS2ERG\__A__gene_gene_asso.tsv")

    # 1. 데이터 로드
    gga, ppi, dgt = load_data(BASE_PATH, GGA_PATH)

    # 2. 이종 네트워크 구성
    G = build_heterogeneous_network(gga, ppi, dgt)

    # 3. 그래프 → Tensor 변환
    X, edge_index, edge_attr, node_list, node_types = graph_to_tensors(G)

    # 4. GAT 학습
    model = train_gat(
        G, X, edge_index, edge_attr, node_types,
        n_epochs   = 200,
        lr         = 0.001,
        patience   = 20,
        batch_size = 4096,
        seed       = 42,
    )

    # 5. Attention 추출
    directed_attention = extract_directed_attention(
        model, X, edge_index, edge_attr, node_types, node_list
    )

    # 6. 전이 확률 캐시 (GAT 0.7 + 원본 0.3)
    neighbor_cache = build_attention_neighbor_cache(
        G, directed_attention, alpha_mix=0.7
    )

    # 7. GAT+RWR 시뮬레이션
    visit_count = simulate_random_walk_v3(
        G, neighbor_cache,
        n_walks      = 1000,
        walk_length  = 100,
        restart_prob = 0.15,
        seed         = 42,
        n_workers    = 12,
    )

    # 8. 결과 출력 및 저장
    visit_prob, sorted_nodes = print_results(G, visit_count, top_k=20)
    save_results_v3(G, visit_count, visit_prob, directed_attention, BASE_PATH)
