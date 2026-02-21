"""
v3 후처리 통합 실행 스크립트
1. annotate_prad_v3.py  → node_importance_v3_annotated.tsv 생성
2. stats_enrichment_v3.py → 통계 분석 및 결과 저장
"""
import subprocess
import sys
import os

BASE = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test"
V3_TSV = os.path.join(BASE, "node_importance_v3.tsv")

# v3 결과 파일 존재 확인
if not os.path.exists(V3_TSV):
    print(f"[ERROR] {V3_TSV} 파일이 아직 없습니다.")
    print("  RWR 완료 후 재실행하세요.")
    sys.exit(1)

print("=" * 60)
print("  v3 후처리 시작")
print("=" * 60)

python = sys.executable

# Step 1: Annotation
print("\n[Step 1] PRAD annotation 추가...")
r1 = subprocess.run([python, os.path.join(BASE, "annotate_prad_v3.py")], check=True)

# Step 2: 통계 분석
print("\n[Step 2] 통계 농축 분석...")
r2 = subprocess.run([python, os.path.join(BASE, "stats_enrichment_v3.py")], check=True)

print("\n" + "=" * 60)
print("  v3 후처리 완료!")
print(f"  - node_importance_v3_annotated.tsv")
print(f"  - stats_enrichment_v3_result.tsv")
print(f"  - stats_enrichment_v3_result.txt")
print("=" * 60)
