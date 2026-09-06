"""기획서 §3.5의 Attach 시나리오를 그대로 - 기존 학습 스크립트에 두 줄."""
import torch, torch.nn.functional as F
from torch import nn
import torchflow as tf


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x)


class Net(nn.Module):
    def __init__(self, depth=4):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(32) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))


model = Net()
x = torch.randn(8, 3, 32, 32)
y = torch.randint(0, 10, (8,))

# 여기 두 줄이 전부다
sess = tf.watch(model, x, objective=lambda out, batch: F.cross_entropy(out, y),
                port=8799, open_browser=False,
                state_dir=".torchflow/attach")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
tf.hook_step(model, optimizer, every=2)
# ─

import time; time.sleep(4)   # hub 기동 대기

result = sess.probe()
print(f"\nprobe: loss={result['loss']:.4f}  {result['elapsed_ms']:.0f} ms  obj={result['objective']}")
graded = [n for n in result["nodes"] if n["grad_norm"] is not None]
print(f"grad 수집 노드 {len(graded)}개, 경고 {sum(1 for n in graded if n['warn'])}개")
for n in graded[:4] + graded[-2:]:
    print(f"  {n['node']:10} ‖g‖={n['grad_norm']:.3e}  ratio={n['grad_ratio']:.2e}  warn={n['warn']}")

print("\n학습 루프 (probe가 학습을 바꾸지 않는지 확인):")
before = [p.detach().clone() for p in model.parameters()]
losses = []
for step in range(6):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    tf.log(step, loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    losses.append(loss.item())
print("  loss:", " → ".join(f"{v:.4f}" for v in losses))
print("  가중치가 실제로 갱신됨:", not all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

# probe 이후 모델 상태 불변 확인
snapshot = [p.detach().clone() for p in model.parameters()]
grads_before = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
sess.probe()
print("  probe 후 가중치 불변:", all(torch.equal(a, b) for a, b in zip(snapshot, model.parameters())))
print("  probe 후 grad 복원:", all((a is None and b.grad is None) or torch.equal(a, b.grad)
                                   for a, b in zip(grads_before, model.parameters())))
print("  probe 후 train 모드 유지:", model.training)

import urllib.request, json
req = urllib.request.Request(f"http://127.0.0.1:8799/api/health", headers={"Authorization": f"token {sess.token}"})
print("\nhub:", json.loads(urllib.request.urlopen(req).read()))
req = urllib.request.Request(f"http://127.0.0.1:8799/api/runs/curve?key=loss", headers={"Authorization": f"token {sess.token}"})
print("기록된 loss 점:", sum(len(s["points"]) for s in json.loads(urllib.request.urlopen(req).read())["series"]), "개")
print("URL:", sess.url)
sess.close()
