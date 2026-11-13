"""points.npz -> scene.ply (INRIA 3DGS 형식, 알갱이당 17 float = 68바이트)

사용: uv run --with numpy python make_ply.py [K] [N_KEEP] [출력경로]
  K       알갱이 크기 계수 (기본 1.4)
  N_KEEP  사용할 점 개수 (기본 0 = 전체, 지정하면 고르게 추출)
"""
import sys, time, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "recon", "points.npz")
K = float(sys.argv[1]) if len(sys.argv) > 1 else 1.4
N_KEEP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "scene.ply")
OPACITY = 0.9
SH_C0 = 0.28209479177387814

t0 = time.time()
d = np.load(SRC)
xyz, rgb = d["xyz"], d["rgb"]

if N_KEEP and N_KEEP < len(xyz):
    idx = np.linspace(0, len(xyz) - 1, N_KEEP).astype(np.int64)  # 고르게 추출
    xyz, rgb = xyz[idx], rgb[idx]

n = len(xyz)
extent = float((xyz.max(0) - xyz.min(0)).max())      # 장면의 가장 긴 변
r = K * extent / np.sqrt(n)                          # 알갱이 반지름

p = xyz.astype(np.float32).copy()
p[:, 1] *= -1.0                                      # 모델은 Y가 아래, three.js는 위
p[:, 2] *= -1.0

out = np.zeros((n, 17), dtype=np.float32)
out[:, 0:3] = p                                      # 위치
# out[:, 3:6] 법선 = 0
out[:, 6:9] = (rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0   # 0차 구면 조화 (f_dc)
out[:, 9] = np.log(OPACITY / (1.0 - OPACITY))        # 불투명도 (시그모이드 이전 값)
out[:, 10:13] = np.log(r)                            # 크기 (로그)
out[:, 13] = 1.0                                     # 회전 사원수 (w,x,y,z) = 안 돌림

header = (
    "ply\nformat binary_little_endian 1.0\n"
    f"element vertex {n}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property float nx\nproperty float ny\nproperty float nz\n"
    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    "property float opacity\n"
    "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    "end_header\n"
).encode("ascii")

with open(OUT, "wb") as f:
    f.write(header)
    f.write(out.tobytes())

size = os.path.getsize(OUT)
print(f"알갱이 개수      : {n:,}")
print(f"반지름 r         : {r:.5f}  (K={K}, 가장 긴 변={extent:.3f})")
print(f"파일 크기        : {size:,} bytes ({size/1e6:.1f}MB / {size/1048576:.1f}MiB)")
print(f"헤더 크기        : {len(header)} bytes")
print(f"알갱이당 바이트  : {(size-len(header))/n:.1f} (본문 기준), {size/n:.2f} (헤더 포함)")
print(f"f_rest 포함 여부 : {'f_rest' in header.decode()}")
print(f"저장 시간        : {time.time()-t0:.1f}초")
