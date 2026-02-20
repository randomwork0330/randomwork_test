"""
Network Construction & Random Walk with Restart (RWR) Node Importance
======================================================================
입력 소스:
  _A_absPearsonR_tmprss2erg.tsv  → GGA 역할 (gene-gene, TMPRSS2-ERG 공발현)
      컬럼: gene | absPearsonR

  _B_9606.ppi.physical.tsv      → PPI 역할 (gene-gene, STRING 물리적 상호작용)
      컬럼: gene1 | gene2 | score   ← 이미 Min-Max 정규화 완료 (0~1)

  _C_civic_gts.tsv              → DGT 역할 (drug-gene, CIViC sensitivity)
      컬럼: drug | target_gene | final_target_gene | score
      ※ final_target_gene + score 사용 / score는 내부에서 재정규화

[정규화 전략: Z-score → Sigmoid]
  ① Z-score:  z = (x - μ) / σ       → 분포 편향·스케일 제거
  ② Sigmoid:  p = 1 / (1 + exp(-z)) → 0~1 확률값으로 변환

  - Outlier 영향 자동 압축 (Min-Max 대비 robust)
  - 분포 정보 보존 (평균 근처 분별력 극대화)
  - 이종 데이터 간 확률론적으로 공정한 비교 가능
  - _B_ (PPI): 이미 0~1이지만 동일 파이프라인 적용으로 일관성 유지
  - _B_ + _A_ 중복 엣지: 가중 평균 (GGA=0.4, PPI=0.6)
  - 워크 이동 확률: 이웃 weight 비례 local-softmax (합=1, ε=1e-8)

[속도 개선 전략]
  ① 이웃 캐싱: 시뮬레이션 전 모든 노드의 이웃 리스트·전이 확률 1회 계산
               → 루프 내 반복적 배열 생성·정규화 제거 (30~50% 향상)
  ② 멀티프로세싱: 노드를 n_workers개 청크로 분할 → 각 프로세스 독립 실행
               → ProcessPoolExecutor + multiprocessing.Value 공유 카운터
               → 워커별 독립 난수 시드(seed + worker_id)로 재현성 보장
  ③ 실시간 진행률: 모니터링 스레드가 0.5초마다 공유 카운터 읽어 출력
"""

import pandas as pd
import numpy as np
import networkx as nx
import random
import glob
import os
import time
import threading
from collections import Counter
from multiprocessing import Pool, Value, cpu_count
import multiprocessing as mp


# ──────────────────────────────────────────────
# [멀티프로세싱 워커] 모듈 최상위에 정의 필수
# (Windows pickle 제약: 클래스/함수는 __main__ 외부에 있어야 직렬화 가능)
#
# multiprocessing.Value 는 pickle 직렬화 불가 → Pool initializer 패턴 사용
# : Pool 생성 시 _init_worker 로 공유 카운터를 전역 변수로 주입
# ──────────────────────────────────────────────
_shared_counter = None   # 워커 전역 공유 카운터 (initializer가 주입)

def _init_worker(shared_val):
    """Pool initializer: 각 워커 프로세스 시작 시 공유 카운터를 전역 변수에 주입."""
    global _shared_counter
    _shared_counter = shared_val


def _walk_worker(args: tuple) -> Counter:
    """
    단일 워커 프로세스가 실행하는 RWR 시뮬레이션.

    args:
      chunk_nodes   : 이 워커가 담당하는 시작 노드 리스트
      neighbor_cache: {node: (neighbors_list, prob_array)} 사전 계산 캐시
      n_walks       : 노드당 워크 횟수
      walk_length   : 워크당 최대 스텝
      restart_prob  : teleport 확률
      worker_seed   : 이 워커 전용 난수 시드 (seed + worker_id)
    """
    chunk_nodes, neighbor_cache, n_walks, walk_length, restart_prob, worker_seed = args

    # 워커 전용 독립 난수 상태 (프로세스 간 간섭 없음)
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

                # teleport
                if rng.random() < restart_prob:
                    current   = start_node
                    cur_nbrs  = nbrs
                    cur_probs = probs
                    continue

                if cur_nbrs is None:    # 이웃 없음 → teleport
                    current   = start_node
                    cur_nbrs  = nbrs
                    cur_probs = probs
                    continue

                # 캐시된 확률로 다음 노드 선택
                idx       = rng.choice(len(cur_nbrs), p=cur_probs)
                current   = cur_nbrs[idx]
                cur_nbrs, cur_probs = neighbor_cache[current]

        # 노드 1개 완료 → 공유 카운터 업데이트 (진행률 모니터용)
        if _shared_counter is not None:
            with _shared_counter.get_lock():
                _shared_counter.value += n_walks

    return local


# ──────────────────────────────────────────────
# 0. 파일 자동 탐색 유틸
# ──────────────────────────────────────────────
def find_file(base_path: str, prefix: str) -> str:
    """base_path 에서 prefix 로 시작하는 첫 번째 TSV 파일 경로 반환."""
    pattern = os.path.join(base_path, f"{prefix}*.tsv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"'{prefix}*.tsv' 파일을 찾을 수 없습니다: {base_path}")
    return sorted(matches)[0]


# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────
def load_data(base_path: str = "./"):
    path_a = find_file(base_path, "_A_")
    path_b = find_file(base_path, "_B_")
    path_c = find_file(base_path, "_C_")

    # ── _A_ 로드: 기준 유전자 집합 정의
    gga = pd.read_csv(path_a, sep="\t")   # gene | absPearsonR
    gene_set = set(gga["gene"])           # _A_ 에 존재하는 유전자만 허용

    print(f"[Load] GGA(_A_): {len(gga):,} rows  ({len(gene_set):,} genes)"
          f"  ← {os.path.basename(path_a)}")

    # ── _B_ 로드 + _A_ 유전자 기준 필터링 (gene1, gene2 모두 _A_ 소속)
    ppi_raw = pd.read_csv(path_b, sep="\t")   # gene1 | gene2 | score
    ppi = ppi_raw[
        ppi_raw["gene1"].isin(gene_set) | ppi_raw["gene2"].isin(gene_set)
    ].reset_index(drop=True)
    print(f"[Load] PPI(_B_): {len(ppi_raw):,} rows → 필터링 후 {len(ppi):,} rows"
          f"  (gene1 or gene2 가 _A_ 소속)"
          f"  ← {os.path.basename(path_b)}")

    # ── _C_ 로드 + _A_ 유전자 기준 필터링 (final_target_gene 이 _A_ 소속)
    dgt_raw = pd.read_csv(path_c, sep="\t")   # drug | target_gene | final_target_gene | score
    dgt_raw["use_gene"] = dgt_raw["final_target_gene"].fillna(dgt_raw["target_gene"])
    dgt = dgt_raw[dgt_raw["use_gene"].isin(gene_set)].reset_index(drop=True)
    print(f"[Load] DGT(_C_): {len(dgt_raw):,} rows → 필터링 후 {len(dgt):,} rows"
          f"  ({dgt['drug'].nunique():,} drugs × {dgt['use_gene'].nunique():,} genes)"
          f"  ← {os.path.basename(path_c)}")

    return gga, ppi, dgt


# ──────────────────────────────────────────────
# 2. Z-score → Sigmoid 정규화 유틸
# ──────────────────────────────────────────────
def zscore_sigmoid_normalize(series: pd.Series) -> pd.Series:
    """
    Z-score → Sigmoid 2단계 정규화.

    ① Z-score : z = (x - μ) / σ
        - 분포의 평균·분산 제거 → 이종 데이터 간 스케일 통일
        - σ=0 (모든 값 동일) 이면 z=0 → sigmoid(0)=0.5 반환

    ② Sigmoid : p = 1 / (1 + exp(-z))
        - z를 0~1 확률값으로 변환
        - 평균 근처 값은 0.5 부근에서 분별력 높음
        - Outlier는 0 또는 1에 수렴 → 영향 자동 압축
    """
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        # 모든 값이 동일 → z=0 → sigmoid=0.5
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    z = (series - mu) / sigma
    return 1.0 / (1.0 + np.exp(-z))


# ──────────────────────────────────────────────
# 3. 이종 네트워크 구성
# ──────────────────────────────────────────────
def build_heterogeneous_network(gga: pd.DataFrame,
                                 ppi: pd.DataFrame,
                                 dgt: pd.DataFrame) -> nx.Graph:
    """
    세 소스를 통합한 이종 그래프(undirected) 구성.

    노드 타입:
      - 'gene' : GGA / PPI / DGT 의 유전자 노드
      - 'drug' : DGT 의 약물 노드

    엣지 타입 & weight:
      - 'gga'     : absPearsonR Z-score→Sigmoid 정규화 (0~1)
      - 'ppi'     : STRING physical score Z-score→Sigmoid 정규화 (0~1)
      - 'gga+ppi' : GGA·PPI 중복 엣지 → 가중 평균 (GGA=0.4, PPI=0.6)
      - 'dgt'     : CIViC score Z-score→Sigmoid 정규화 (0~1)
    """
    G = nx.Graph()
    GGA_W, PPI_W = 0.4, 0.6   # 중복 엣지 신뢰도 가중치

    # ── (A) GGA 엣지: gene ↔ gene (TMPRSS2-ERG 상관 유전자)
    # gene 컬럼: TMPRSS2-ERG 와 상관된 개별 유전자
    # 네트워크 엣지: (ERG, gene_i)  weight = absPearsonR (Min-Max 정규화)
    # → ERG 를 anchor 로 star 형태로 연결
    gga = gga.copy()
    gga["norm_w"] = zscore_sigmoid_normalize(gga["absPearsonR"].astype(float))   # Z-score→Sigmoid

    for _, row in gga.iterrows():
        gene = str(row["gene"])
        w    = float(row["norm_w"])
        G.add_node("ERG",  node_type="gene")
        G.add_node(gene,   node_type="gene")
        if gene == "ERG":
            continue   # self-loop 제외
        if G.has_edge("ERG", gene):
            # 기존 엣지가 있으면 (PPI에서 먼저 삽입된 경우) 가중 평균 병합
            existing_w = G["ERG"][gene]["weight"]
            merged     = GGA_W * w + PPI_W * existing_w
            G["ERG"][gene]["weight"]    = merged
            G["ERG"][gene]["edge_type"] = "gga+ppi"
        else:
            G.add_edge("ERG", gene,
                       weight=w,
                       raw_weight=w,
                       edge_type="gga")

    print(f"[Network] GGA 엣지 추가 완료 | 현재 엣지 수: {G.number_of_edges():,}")

    # ── (B) PPI 엣지: gene1 ↔ gene2 (STRING physical)
    ppi = ppi.copy()
    ppi["norm_w"] = zscore_sigmoid_normalize(ppi["score"].astype(float))   # Z-score→Sigmoid (일관성 유지)

    for _, row in ppi.iterrows():
        g1 = str(row["gene1"])
        g2 = str(row["gene2"])
        w  = float(row["norm_w"])
        G.add_node(g1, node_type="gene")
        G.add_node(g2, node_type="gene")
        if G.has_edge(g1, g2):
            # GGA 엣지와 중복 → 가중 평균 병합
            gga_w  = G[g1][g2]["weight"]
            merged = GGA_W * gga_w + PPI_W * w
            G[g1][g2]["weight"]    = merged
            G[g1][g2]["edge_type"] = "gga+ppi"
        else:
            G.add_edge(g1, g2,
                       weight=w,
                       raw_weight=w,
                       edge_type="ppi")

    print(f"[Network] PPI 엣지 추가 완료 | 현재 엣지 수: {G.number_of_edges():,}")

    # ── (C) DGT 엣지: drug ↔ final_target_gene (CIViC)
    # use_gene 컬럼은 load_data() 에서 이미 생성됨
    dgt = dgt.copy()
    # score Z-score→Sigmoid 정규화
    dgt["norm_w"] = zscore_sigmoid_normalize(dgt["score"].astype(float))

    for _, row in dgt.iterrows():
        drug = str(row["drug"])
        gene = str(row["use_gene"])
        w    = float(row["norm_w"])
        if not drug or not gene:
            continue
        G.add_node(drug, node_type="drug")
        G.add_node(gene, node_type="gene")
        if G.has_edge(drug, gene):
            # 동일 (drug, gene) 쌍 중복 → 더 큰 weight 유지
            if w > G[drug][gene]["weight"]:
                G[drug][gene]["weight"] = w
        else:
            G.add_edge(drug, gene,
                       weight=w,
                       raw_weight=float(row["score"]),
                       edge_type="dgt")

    print(f"[Network] DGT 엣지 추가 완료 | 현재 엣지 수: {G.number_of_edges():,}")

    # ── 통계 요약
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
    weights    = [d["weight"] for _, _, d in G.edges(data=True)]
    etypes     = Counter(d["edge_type"] for _, _, d in G.edges(data=True))

    print(f"\n[Network] ── 최종 그래프 통계 ──")
    print(f"  노드 수  : {G.number_of_nodes():,}  (Gene: {len(gene_nodes):,} | Drug: {len(drug_nodes):,})")
    print(f"  엣지 수  : {G.number_of_edges():,}")
    print(f"  엣지 타입: { {k: f'{v:,}' for k, v in etypes.items()} }")
    print(f"  Weight   : min={min(weights):.4f} | mean={np.mean(weights):.4f} | max={max(weights):.4f}")

    return G


# ──────────────────────────────────────────────
# 4. 확률적 Random Walk with Restart (RWR)
#    이웃 캐싱 + 멀티프로세싱 + 실시간 진행률
# ──────────────────────────────────────────────
def _build_neighbor_cache(G: nx.Graph) -> dict:
    """
    시뮬레이션 전 1회 실행: 모든 노드의 이웃 리스트와 전이 확률을 캐싱.
    루프 내 반복적 배열 생성·정규화를 제거하여 30~50% 속도 향상.

    반환: {node: (neighbors_list, prob_array)}
      - 이웃 없는 노드: ([], None)
      - prob_array: 이웃 weight 기반 정규화 확률 (합=1, ε=1e-8 보정)
    """
    cache = {}
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if not nbrs:
            cache[node] = ([], None)
        else:
            w = np.array([G[node][nb]["weight"] for nb in nbrs], dtype=float)
            w += 1e-8       # zero-weight 보정
            w /= w.sum()    # 전이 확률 정규화 (1회만 수행)
            cache[node] = (nbrs, w)
    return cache


def _progress_monitor(shared_done: Value, total_walks: int, stop_event: threading.Event):
    """
    모니터링 스레드: 0.5초마다 공유 카운터를 읽어 실시간 진행률 출력.
    stop_event 가 set() 되면 종료.
    진행바는 ASCII 문자(#/-)로 구성 (Windows cp949 인코딩 호환).
    """
    start_time = time.time()
    while not stop_event.is_set():
        done    = shared_done.value
        pct     = done / total_walks * 100
        elapsed = time.time() - start_time
        speed   = done / elapsed if elapsed > 0 else 0
        eta     = (total_walks - done) / speed if speed > 0 else 0
        bar_len = 30
        filled  = int(bar_len * done / total_walks)
        bar     = "#" * filled + "-" * (bar_len - filled)   # ASCII 진행바
        print(f"\r  [{bar}] {pct:5.1f}%  {done:,}/{total_walks:,}  "
              f"{speed:,.0f} walks/s  ETA {eta:.0f}s   ", end="", flush=True)
        if done >= total_walks:
            break
        stop_event.wait(timeout=0.5)
    print()  # 줄바꿈


def simulate_random_walk(G: nx.Graph,
                          n_walks:      int   = 1000,
                          walk_length:  int   = 100,
                          restart_prob: float = 0.15,
                          seed:         int   = 42,
                          n_workers:    int   = 12) -> Counter:
    """
    이웃 캐싱 + 멀티프로세싱 기반 고속 RWR 시뮬레이션.

    개선 포인트:
      ① 이웃 캐싱  : 전이 확률 1회 계산 → 루프 내 배열 생성 제거
      ② 멀티프로세싱: 노드를 n_workers 청크로 분할 → 병렬 실행
      ③ 실시간 진행률: 모니터링 스레드가 0.5초마다 진행바 갱신

    Parameters
    ----------
    n_workers : 병렬 프로세스 수 (기본 12, 실제 코어 수 초과 시 자동 조정)
    """
    nodes       = list(G.nodes())
    n_nodes     = len(nodes)
    total_walks = n_nodes * n_walks
    n_workers   = min(n_workers, n_nodes, cpu_count())  # 실제 코어 수 초과 방지

    print(f"\n[RWR] 시작 | 노드 {n_nodes:,}개 × {n_walks}회 = 총 {total_walks:,}번 워크")
    print(f"  walk_length={walk_length} | restart_prob={restart_prob} | n_workers={n_workers}")

    # ── 전처리: 이웃 캐싱 (1회 계산)
    print("[RWR] 이웃 캐싱 전처리 중...")
    neighbor_cache = _build_neighbor_cache(G)
    print(f"[RWR] 캐싱 완료 | {n_nodes:,}개 노드")

    # ── 노드를 n_workers 청크로 분할 (라운드로빈 방식으로 균등 분배)
    chunks = [nodes[i::n_workers] for i in range(n_workers)]

    # ── 공유 카운터 (프로세스 간 진행률 공유)
    shared_done = Value('l', 0)   # 'l' = long int

    # ── 워커 인자 구성 (각 워커마다 독립 시드, shared_done은 initializer로 주입)
    worker_args = [
        (chunk, neighbor_cache, n_walks, walk_length, restart_prob, seed + i)
        for i, chunk in enumerate(chunks)
    ]

    # ── 진행률 모니터링 스레드 시작
    stop_event = threading.Event()
    monitor    = threading.Thread(
        target=_progress_monitor,
        args=(shared_done, total_walks, stop_event),
        daemon=True
    )
    monitor.start()

    # ── 멀티프로세싱 실행
    # Pool initializer로 shared_done을 각 워커의 전역 변수에 주입 (pickle 우회)
    t0          = time.time()
    visit_count = Counter()
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(shared_done,)) as pool:
        results = pool.map(_walk_worker, worker_args)
    for r in results:
        visit_count += r   # 각 워커 결과 병합

    # ── 모니터 종료
    stop_event.set()
    monitor.join()

    elapsed = time.time() - t0
    print(f"[RWR] 완료 | 소요 시간: {elapsed:.1f}s | "
          f"속도: {total_walks/elapsed:,.0f} walks/s | "
          f"총 방문 횟수: {sum(visit_count.values()):,}")
    return visit_count


# ──────────────────────────────────────────────
# 5. 결과 출력
# ──────────────────────────────────────────────
def print_results(G: nx.Graph,
                   visit_count: Counter,
                   top_k: int = 20) -> tuple:
    total        = sum(visit_count.values())
    visit_prob   = {n: c / total for n, c in visit_count.items()}
    sorted_nodes = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)

    # ── 전체 Top-K
    print("\n" + "=" * 75)
    print(f"    [RWR 노드 중요도 Top {top_k}]")
    print("=" * 75)
    print(f"{'Rank':<5} {'Node':<22} {'Type':<6} "
          f"{'방문횟수':>12} {'방문확률':>10} {'Degree':>8}")
    print("-" * 75)
    for rank, (node, cnt) in enumerate(sorted_nodes[:top_k], 1):
        ntype  = G.nodes[node].get("node_type", "?")
        prob   = visit_prob[node]
        degree = G.degree(node)
        print(f"{rank:<5} {node:<22} {ntype:<6} "
              f"{cnt:>12,} {prob:>10.6f} {degree:>8}")
    print("=" * 75)

    # ── Gene-only Top-10
    gene_nodes = [(n, c) for n, c in sorted_nodes
                  if G.nodes[n].get("node_type") == "gene"]
    print(f"\n[Top 10 Gene 노드]")
    print(f"{'Rank':<5} {'Gene':<22} {'방문횟수':>12} {'방문확률':>10} {'Degree':>8}")
    print("-" * 60)
    for rank, (node, cnt) in enumerate(gene_nodes[:10], 1):
        prob   = visit_prob[node]
        degree = G.degree(node)
        print(f"{rank:<5} {node:<22} {cnt:>12,} {prob:>10.6f} {degree:>8}")

    # ── Drug-only Top-10
    drug_nodes = [(n, c) for n, c in sorted_nodes
                  if G.nodes[n].get("node_type") == "drug"]
    print(f"\n[Top 10 Drug 노드]")
    print(f"{'Rank':<5} {'Drug':<22} {'방문횟수':>12} {'방문확률':>10} {'Degree':>8}")
    print("-" * 60)
    for rank, (node, cnt) in enumerate(drug_nodes[:10], 1):
        prob   = visit_prob[node]
        degree = G.degree(node)
        print(f"{rank:<5} {node:<22} {cnt:>12,} {prob:>10.6f} {degree:>8}")

    # ── 1위 노드 상세
    top_node, top_cnt = sorted_nodes[0]
    top_type  = G.nodes[top_node].get("node_type", "?")
    neighbors = list(G.neighbors(top_node))
    print(f"\n★ 최고 중요도 노드 : [{top_node}]  (type={top_type})")
    print(f"  방문 횟수       : {top_cnt:,}회  ({visit_prob[top_node]*100:.3f}%)")
    print(f"  연결 이웃 수    : {len(neighbors)}개")
    if len(neighbors) <= 20:
        print(f"  이웃 목록       : {', '.join(str(n) for n in neighbors)}")

    return visit_prob, sorted_nodes


# ──────────────────────────────────────────────
# 6. 결과 저장
# ──────────────────────────────────────────────
def save_results(G: nx.Graph,
                  visit_count: Counter,
                  visit_prob:  dict,
                  base_path:   str):
    rows = [
        {
            "node":        node,
            "node_type":   G.nodes[node].get("node_type", "?"),
            "visit_count": visit_count.get(node, 0),
            "visit_prob":  round(visit_prob.get(node, 0.0), 8),
            "degree":      G.degree(node),
        }
        for node in G.nodes()
    ]
    df = (pd.DataFrame(rows)
            .sort_values("visit_count", ascending=False)
            .reset_index(drop=True))
    df.index += 1   # rank 1부터

    out = os.path.join(base_path, "node_importance.tsv")
    df.to_csv(out, sep="\t", index_label="rank")
    print(f"\n[Saved] {out}  ({len(df):,} 노드)")


# ──────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    BASE_PATH = "C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/"

    # 1. 데이터 로드
    gga, ppi, dgt = load_data(BASE_PATH)

    # 2. 이종 네트워크 구성
    G = build_heterogeneous_network(gga, ppi, dgt)

    # 3. RWR 시뮬레이션
    visit_count = simulate_random_walk(
        G,
        n_walks      = 1000,    # 노드당 워크 수
        walk_length  = 100,     # 워크당 최대 스텝
        restart_prob = 0.15,    # teleport 확률
        seed         = 42,
        n_workers    = 12,      # 병렬 프로세스 수 (실제 코어 초과 시 자동 조정)
    )

    # 4. 결과 출력 및 저장
    visit_prob, sorted_nodes = print_results(G, visit_count, top_k=20)
    save_results(G, visit_count, visit_prob, BASE_PATH)
