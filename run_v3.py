"""v3 실행 래퍼 - 로그 파일에 stdout/stderr 모두 저장"""
import sys, os, subprocess

BASE = r"C:\Users\user\Desktop\CNN_CVPR\__DATAAn__\random_work_test"
LOG  = os.path.join(BASE, "v3_run.log")
PY   = r"C:\Anaconda3\envs\cosmic_ai\python.exe"
SCRIPT = os.path.join(BASE, "network_random_walk_v3_GAT.py")

print(f"실행: {PY} -u {SCRIPT}", flush=True)
print(f"로그: {LOG}", flush=True)

with open(LOG, "w", encoding="utf-8") as f:
    proc = subprocess.Popen(
        [PY, "-u", SCRIPT],
        cwd=BASE,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(f"PID: {proc.pid}", flush=True)
    ret = proc.wait()
    print(f"완료. exit={ret}", flush=True)
