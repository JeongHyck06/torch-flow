"""프로브 예산과 캐시 쿼터 (기획서 §5.1.4, §5.2)."""

import pytest

from torchflow.hub.cache import DiskCAS
from torchflow.kernel.budget import ProbeBudget


# 프로브 예산


def test_budget_degrades_in_a_fixed_order():
    """B 축소 -> forward-only -> 비활성. 순서가 흔들리면 사용자가 예측할 수 없다."""
    budget = ProbeBudget(budget_ms=100, batch=4)

    budget.record(500, now=0)
    assert budget.decide(now=0).batch == 2 and budget.decide(now=0).backward

    budget.record(500, now=1)
    assert budget.decide(now=1).batch == 1 and budget.decide(now=1).backward

    budget.record(500, now=2)
    assert not budget.decide(now=2).backward, "forward-only로 내려가지 않았다"

    budget.record(500, now=3)
    assert not budget.decide(now=3).enabled


def test_budget_leaves_fast_probes_alone():
    budget = ProbeBudget(budget_ms=2000, batch=4)
    for step in range(10):
        budget.record(50, now=step)
    decision = budget.decide(now=10)
    assert decision.enabled and decision.backward and decision.batch == 4
    assert not decision.degraded


def test_gpu_seconds_per_minute_disables_l1():
    """반응형 GPU 예산: 분당 상한을 넘으면 L1이 스스로 물러난다(§3.4)."""
    budget = ProbeBudget(budget_ms=10_000, gpu_sec_per_min=5)
    for step in range(6):
        budget.record(1000, now=step)     # 6초 소비
    assert not budget.decide(now=6).enabled


def test_l1_comes_back_after_the_cooldown():
    budget = ProbeBudget(budget_ms=10_000, gpu_sec_per_min=5)
    for step in range(6):
        budget.record(1000, now=step)
    assert not budget.decide(now=6).enabled
    assert budget.decide(now=6 + 61).enabled, "1분 유휴 후에도 돌아오지 않았다"


def test_spent_seconds_only_counts_the_last_minute():
    budget = ProbeBudget()
    budget.record(1000, now=0)
    budget.record(1000, now=100)
    assert budget.spent_gpu_seconds(now=100) == pytest.approx(1.0)


# 디스크 CAS


def test_cas_deduplicates_identical_payloads(tmp_path):
    cas = DiskCAS(tmp_path / "cache")
    first = cas.put(b"same bytes")
    second = cas.put(b"same bytes")
    assert first == second
    assert len(cas.entries()) == 1
    assert cas.get(first) == b"same bytes"


def test_cas_shards_by_digest_prefix(tmp_path):
    cas = DiskCAS(tmp_path / "cache")
    digest = cas.put(b"payload")
    assert (cas.root / digest[:2] / digest).exists()


def test_action_cache_maps_keys_to_content(tmp_path):
    cas = DiskCAS(tmp_path / "cache")
    cas.remember("node_key:abc", {"shape": [4, 8]})
    assert cas.recall("node_key:abc") == {"shape": [4, 8]}
    assert cas.recall("missing") is None


def test_action_index_survives_a_restart(tmp_path):
    cas = DiskCAS(tmp_path / "cache")
    cas.remember("k", {"v": 1})
    cas.flush()
    assert DiskCAS(tmp_path / "cache").recall("k") == {"v": 1}


def test_corrupt_index_does_not_break_the_cache(tmp_path):
    """캐시는 재구축 가능한 파생 상태다 - 인덱스가 깨져도 죽지 않는다."""
    cas = DiskCAS(tmp_path / "cache")
    cas.remember("k", {"v": 1})
    cas.flush()
    cas.index_path.write_text("{not json", encoding="utf-8")
    assert DiskCAS(tmp_path / "cache").actions == {}


def test_gc_evicts_until_the_quota_is_met(tmp_path):
    cas = DiskCAS(tmp_path / "cache", quota_gb=4096 / 1024**3)   # 4 KB
    digests = [cas.put(bytes(1024) + str(index).encode()) for index in range(10)]
    assert cas.size_bytes() > cas.quota_bytes

    report = cas.gc()
    assert report.removed > 0
    assert cas.size_bytes() <= cas.quota_bytes
    assert sum(1 for digest in digests if cas.get(digest) is not None) < len(digests)


def test_gc_keeps_referenced_digests(tmp_path):
    cas = DiskCAS(tmp_path / "cache", quota_gb=2048 / 1024**3)
    digests = [cas.put(bytes(1024) + str(index).encode()) for index in range(8)]
    pinned = {digests[0], digests[1]}

    cas.gc(keep=pinned)
    assert all(cas.get(digest) is not None for digest in pinned)


def test_gc_dry_run_changes_nothing(tmp_path):
    cas = DiskCAS(tmp_path / "cache", quota_gb=1024 / 1024**3)
    for index in range(6):
        cas.put(bytes(1024) + str(index).encode())
    before = cas.size_bytes()

    report = cas.gc(dry_run=True)
    assert report.dry_run and report.removed > 0
    assert cas.size_bytes() == before
