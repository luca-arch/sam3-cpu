"""Tests for IntraChunkMonitor, AdaptiveChunkManager, and AsyncIOWorker."""

import json
import math
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sam3.async_io import AsyncIOWorker
from sam3.memory_optimizer import (
    AdaptiveChunkManager,
    FrameMemorySample,
    GrowthCalibration,
    IntraChunkMonitor,
    MonitorResult,
    _linear_regression,
    get_memory_tier,
)

# ---------------------------------------------------------------------------
# _linear_regression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    """Tests for the pure-Python linear regression helper."""

    def test_perfect_fit(self):
        slope, intercept, r_sq = _linear_regression([0, 1, 2, 3], [10, 20, 30, 40])
        assert abs(slope - 10.0) < 0.01
        assert abs(intercept - 10.0) < 0.01
        assert r_sq > 0.999

    def test_zero_slope(self):
        slope, intercept, r_sq = _linear_regression([0, 1, 2, 3], [5, 5, 5, 5])
        assert abs(slope) < 0.01
        assert abs(intercept - 5.0) < 0.01

    def test_single_point(self):
        slope, intercept, r_sq = _linear_regression([1], [42])
        assert intercept == 42

    def test_two_points(self):
        slope, intercept, r_sq = _linear_regression([0, 10], [100, 200])
        assert abs(slope - 10.0) < 0.01
        assert abs(intercept - 100.0) < 0.01

    def test_noisy_data(self):
        xs = [0, 1, 2, 3, 4, 5]
        ys = [10, 22, 28, 42, 48, 62]  # roughly y = 10x + 10
        slope, intercept, r_sq = _linear_regression(xs, ys)
        assert 8 < slope < 12
        assert r_sq > 0.95


# ---------------------------------------------------------------------------
# IntraChunkMonitor
# ---------------------------------------------------------------------------


class TestIntraChunkMonitor:
    """Tests for the proactive intra-chunk memory monitor."""

    def test_init_defaults(self):
        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cpu",
        )
        assert monitor.expected_iterations == 100
        assert monitor.device == "cpu"
        assert monitor.vram_limit == 0  # CPU has no VRAM limit
        assert monitor.effective_limit == 0

    def test_init_custom_thresholds(self):
        """Custom soft/hard limits override __globals defaults."""
        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cpu",
            soft_limit_pct=0.70,
            hard_limit_pct=0.85,
            ram_soft_limit_pct=0.60,
            ram_hard_limit_pct=0.80,
        )
        assert monitor._soft_pct == 0.70
        assert monitor._hard_pct == 0.85
        assert monitor._ram_soft_pct == 0.60
        assert monitor._ram_hard_pct == 0.80

    def test_per_frame_evaluation(self):
        """Every frame should be evaluated (no checkpoint skipping)."""
        monitor = IntraChunkMonitor(
            expected_iterations=20,
            device="cpu",
        )
        monitor.start()
        for i in range(20):
            monitor.check(frame_idx=i)

        result = monitor.finalize()
        # Every frame should be evaluated
        assert result.checkpoints_evaluated == 20
        assert result.iterations_completed == 20

    def test_check_always_continues_on_cpu(self):
        """CPU device has no VRAM limit; RAM limit is high so check() passes."""
        monitor = IntraChunkMonitor(
            expected_iterations=10,
            device="cpu",
            ram_hard_limit_pct=0.99,  # effectively disable RAM stop for this test
        )
        monitor.start()
        for i in range(10):
            assert monitor.check(frame_idx=i) is True

    def test_check_increments_iteration(self):
        monitor = IntraChunkMonitor(
            expected_iterations=10,
            device="cpu",
        )
        monitor.start()
        for i in range(5):
            monitor.check(frame_idx=i)
        assert monitor._iteration == 5

    @patch("sam3.memory_optimizer.get_gpu_memory")
    def test_hard_stop_threshold(self, mock_gpu):
        """Monitor should signal stop when VRAM exceeds hard threshold (92%)."""
        mock_gpu.return_value = MagicMock(total=10 * 1024**3)

        monitor = IntraChunkMonitor(
            expected_iterations=20,
            device="cuda",
            vram_limit_bytes=10 * 1024**3,
            hard_limit_pct=0.92,
        )

        # Mock torch.cuda to simulate high memory usage
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # 95% of 9.5GB effective limit → well above 92% hard limit
        mock_torch.cuda.memory_allocated.return_value = int(9.5 * 1024**3 * 0.95)
        mock_torch.cuda.reset_peak_memory_stats.return_value = None

        with patch.dict("sys.modules", {"torch": mock_torch}):
            monitor.start()
            result = monitor.check(frame_idx=0)
            assert result is False
            assert monitor._stop_reason == "hard_limit"

    @patch("sam3.memory_optimizer.get_gpu_memory")
    def test_soft_limit_warning(self, mock_gpu):
        """Monitor should issue warning when VRAM exceeds soft threshold."""
        mock_gpu.return_value = MagicMock(total=10 * 1024**3)

        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cuda",
            vram_limit_bytes=10 * 1024**3,
            soft_limit_pct=0.80,
            hard_limit_pct=0.95,  # high hard limit to not trigger it
        )

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # 85% of 9.5GB effective → above 80% soft, below 95% hard
        mock_torch.cuda.memory_allocated.return_value = int(9.5 * 1024**3 * 0.85)
        mock_torch.cuda.reset_peak_memory_stats.return_value = None

        with patch.dict("sys.modules", {"torch": mock_torch}):
            monitor.start()
            # First few checks during calibration — should continue but set warning
            for i in range(6):
                monitor.check(frame_idx=i)
            assert monitor._soft_warning_issued is True

    @patch("sam3.memory_optimizer.get_gpu_memory")
    def test_below_soft_limit_no_warning(self, mock_gpu):
        """No warning when VRAM usage is below soft threshold."""
        mock_gpu.return_value = MagicMock(total=10 * 1024**3)

        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cuda",
            vram_limit_bytes=10 * 1024**3,
            soft_limit_pct=0.80,
            hard_limit_pct=0.92,
        )

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # 50% of effective limit → well below soft limit
        mock_torch.cuda.memory_allocated.return_value = int(9.5 * 1024**3 * 0.50)
        mock_torch.cuda.reset_peak_memory_stats.return_value = None

        with patch.dict("sys.modules", {"torch": mock_torch}):
            monitor.start()
            for i in range(10):
                assert monitor.check(frame_idx=i) is True
            assert monitor._soft_warning_issued is False

    def test_finalize_returns_correct_result(self):
        monitor = IntraChunkMonitor(
            expected_iterations=50,
            device="cpu",
        )
        monitor.start()
        for i in range(10):
            monitor.check(frame_idx=i)

        result = monitor.finalize()
        assert isinstance(result, MonitorResult)
        assert result.iterations_planned == 50
        assert result.iterations_completed == 10
        assert result.early_stopped is False
        assert result.stop_reason == "completed"

    def test_to_dict_serialization(self):
        monitor = IntraChunkMonitor(
            expected_iterations=20,
            device="cpu",
        )
        monitor.start()
        for i in range(5):
            monitor.check(frame_idx=i)

        d = monitor.to_dict()
        assert isinstance(d, dict)
        assert d["iterations_planned"] == 20
        assert d["iterations_completed"] == 5
        assert d["early_stopped"] is False
        assert d["stop_reason"] == "completed"
        assert "soft_limit_pct" in d
        assert "hard_limit_pct" in d
        assert "soft_warning_issued" in d
        assert "ram_soft_limit_pct" in d
        assert "ram_hard_limit_pct" in d
        assert "ram_soft_warning_issued" in d
        # Should be JSON-serializable
        json.dumps(d)

    def test_calibration_with_simulated_growth(self):
        """Test calibration with mock CUDA that has growing memory."""
        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cpu",  # use CPU to avoid actual CUDA calls
        )
        monitor.start()

        # Manually inject samples to simulate growth
        base = 1_000_000_000  # 1GB
        growth_per_iter = 10_000_000  # 10MB per iter
        for i in range(5):
            sample = FrameMemorySample(
                iteration=i,
                frame_idx=i,
                vram_allocated=base + growth_per_iter * i,
                timestamp=time.time(),
            )
            monitor._samples.append(sample)

        monitor._calibrate()
        assert monitor._calibration is not None
        assert monitor._calibration.n_samples == 5
        assert monitor._calibration.growth_rate_per_iter > 0
        # R² should be very high for perfectly linear data
        assert monitor._calibration.r_squared > 0.99

    def test_predict_at_uses_calibration(self):
        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cpu",
        )
        monitor._calibration = GrowthCalibration(
            growth_rate_per_iter=10_000_000,
            baseline_bytes=1_000_000_000,
            safe_iterations=80,
            confidence=0.8,
            n_samples=5,
            r_squared=0.99,
        )
        predicted = monitor._predict_at(50)
        expected = 1_000_000_000 + 10_000_000 * 50
        assert predicted == expected

    @patch("sam3.memory_optimizer.get_gpu_memory")
    def test_no_predicted_oom_from_warmup(self, mock_gpu):
        """Steep warmup slope must NOT cause a premature predicted_oom stop.

        This was the original bug: 5 calibration frames during model init
        had steep memory growth; extrapolating that across 1800 iterations
        falsely predicted OOM at frame 6.
        """
        mock_gpu.return_value = MagicMock(total=80 * 1024**3)  # 80GB A100

        monitor = IntraChunkMonitor(
            expected_iterations=1800,  # 900 frames × 2 "both"
            device="cuda",
            vram_limit_bytes=80 * 1024**3,
        )

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.reset_peak_memory_stats.return_value = None
        mock_torch.cuda.max_memory_allocated.return_value = 0

        # Simulate warmup: steep growth from 3GB → 19GB in 5 frames
        warmup_vram = [3, 7, 12, 16, 19]  # GB — steep model init
        steady_vram = 19.5  # GB — after init, growth is near-zero

        call_count = [0]

        def memory_allocated():
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(warmup_vram):
                return int(warmup_vram[idx] * 1024**3)
            return int(steady_vram * 1024**3)

        mock_torch.cuda.memory_allocated.side_effect = memory_allocated

        with patch.dict("sys.modules", {"torch": mock_torch}):
            monitor.start()
            # Run 20 frames — should NOT stop (19.5GB on 80GB = 24%)
            for i in range(20):
                result = monitor.check(frame_idx=i)
                assert result is True, f"False stop at frame {i}: {monitor._stop_reason}"

    def test_recalibration_updates_slope(self):
        """Calibration should be re-fitted periodically."""
        monitor = IntraChunkMonitor(
            expected_iterations=200,
            device="cpu",
        )
        monitor.start()

        # Inject steep warmup samples
        base = 1_000_000_000
        for i in range(5):
            monitor._samples.append(
                FrameMemorySample(
                    iteration=i,
                    frame_idx=i,
                    vram_allocated=base + 500_000_000 * i,  # 500MB/iter startup
                    timestamp=time.time(),
                )
            )
        monitor._calibrate()
        initial_slope = monitor._calibration.growth_rate_per_iter

        # Inject steady-state samples (much flatter)
        for i in range(10, 60, 10):
            monitor._samples.append(
                FrameMemorySample(
                    iteration=i,
                    frame_idx=i,
                    vram_allocated=base + 3_000_000_000 + 5_000_000 * (i - 10),
                    timestamp=time.time(),
                )
            )
        monitor._calibrate()
        recal_slope = monitor._calibration.growth_rate_per_iter

        # Recalibrated slope should be much lower than initial
        assert recal_slope < initial_slope * 0.5

    def test_ram_hard_limit_stops(self):
        """Monitor should stop when RAM exceeds hard threshold."""
        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cpu",
            ram_limit_bytes=32 * 1024**3,  # 32GB total
            ram_hard_limit_pct=0.85,
        )
        # effective_ram_limit = 32GB * (1 - 0.30) = 22.4GB
        # 85% of 22.4GB = 19.04GB
        # Simulate RSS = 20GB → 89% of effective → above 85%
        with patch("sam3.memory_optimizer.psutil") as mock_psutil:
            mock_process = MagicMock()
            mock_process.memory_info.return_value = MagicMock(rss=int(20 * 1024**3))
            mock_psutil.Process.return_value = mock_process
            mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024**3)

            monitor.start()
            result = monitor.check(frame_idx=0)
            assert result is False
            assert monitor._stop_reason == "ram_hard_limit"

    def test_ram_below_threshold_continues(self):
        """Monitor should continue when RAM usage is normal."""
        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cpu",
            ram_limit_bytes=32 * 1024**3,
            ram_hard_limit_pct=0.85,
            ram_soft_limit_pct=0.70,
        )
        # effective_ram_limit = 32GB * 0.70 = 22.4GB
        # 4GB RSS = ~18% → well below limits
        with patch("sam3.memory_optimizer.psutil") as mock_psutil:
            mock_process = MagicMock()
            mock_process.memory_info.return_value = MagicMock(rss=int(4 * 1024**3))
            mock_psutil.Process.return_value = mock_process
            mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024**3)

            monitor.start()
            result = monitor.check(frame_idx=0)
            assert result is True
            assert monitor._ram_soft_warning_issued is False


# ---------------------------------------------------------------------------
# AdaptiveChunkManager — object-aware growth
# ---------------------------------------------------------------------------


class TestObjectAwareGrowth:
    """Tests for damped growth when many objects are tracked.

    The exact GROW_FACTOR comes from the auto-detected memory tier,
    so tests derive expected values from ``mgr.GROW_FACTOR`` rather
    than hardcoding 1.25.
    """

    def test_single_object_full_growth(self):
        """With 1 object, growth factor should be full (undamped)."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        gf = mgr.GROW_FACTOR
        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=int(80 * 1024**3 * 0.95 * 0.3),  # 30% usage
            n_objects=1,
        )
        assert rec.action == "GROW"
        # growth = 1 + (gf-1) / (1 + log2(1)) = gf
        assert rec.adjusted_chunk_size == int(500 * gf)

    def test_many_objects_dampened_growth(self):
        """With 29 objects (like the 1080p scenario), growth is minimal."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        gf = mgr.GROW_FACTOR
        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=int(80 * 1024**3 * 0.95 * 0.3),
            n_objects=29,
        )
        assert rec.action == "GROW"
        dampened = 1.0 + (gf - 1.0) / (1.0 + math.log2(29))
        expected = int(500 * dampened)
        assert rec.adjusted_chunk_size == expected
        # Should be much less than the undamped 500*gf
        assert rec.adjusted_chunk_size < int(500 * gf) - 20

    def test_two_objects_moderate_dampening(self):
        """With 2 objects, growth is about half of full."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        gf = mgr.GROW_FACTOR
        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=int(80 * 1024**3 * 0.95 * 0.3),
            n_objects=2,
        )
        assert rec.action == "GROW"
        dampened = 1.0 + (gf - 1.0) / (1.0 + 1.0)
        assert rec.adjusted_chunk_size == int(500 * dampened)

    def test_zero_objects_treated_as_one(self):
        """Edge case: 0 objects should use max(n, 1) = 1."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        gf = mgr.GROW_FACTOR
        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=int(80 * 1024**3 * 0.95 * 0.3),
            n_objects=0,
        )
        assert rec.action == "GROW"
        assert rec.adjusted_chunk_size == int(500 * gf)

    def test_soft_warning_blocks_grow(self):
        """When soft_warning_seen=True, action should NOT be GROW even
        if the aggregated peak_vram is low enough to qualify."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        # Low peak_vram (30%) would normally trigger GROW
        low_peak = int(80 * 1024**3 * 0.95 * 0.3)
        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=low_peak,
            n_objects=1,
            soft_warning_seen=True,
        )
        assert rec.action == "CONTINUE", f"Expected CONTINUE when soft_warning_seen=True, got {rec.action}"
        assert rec.adjusted_chunk_size == 500  # unchanged

    def test_soft_warning_allows_shrink(self):
        """soft_warning_seen should NOT prevent SHRINK — only growth."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        # On CPU, vram_limit defaults to 0.  Override for pressure eval.
        mgr.vram_limit = 80 * 1024**3
        # High peak_vram (87%) → WARNING → SHRINK regardless of flag
        # (use 0.87, safely above WARNING_THRESHOLD=0.85 to avoid
        # floating-point truncation with int())
        high_peak = int(80 * 1024**3 * 0.95 * 0.87)
        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=high_peak,
            n_objects=1,
            soft_warning_seen=True,
        )
        assert rec.action == "SHRINK"


# ---------------------------------------------------------------------------
# AsyncIOWorker
# ---------------------------------------------------------------------------


class TestAsyncIOWorker:
    """Tests for the background I/O worker."""

    def test_lifecycle(self):
        worker = AsyncIOWorker()
        worker.start()
        assert worker._started is True
        worker.shutdown()
        assert worker._started is False

    def test_submit_and_drain(self):
        worker = AsyncIOWorker()
        worker.start()

        results = []
        worker.submit(lambda: results.append(42))
        worker.drain()
        assert results == [42]
        worker.shutdown()

    def test_submit_raises_if_not_started(self):
        worker = AsyncIOWorker()
        with pytest.raises(RuntimeError, match="not started"):
            worker.submit(lambda: None)

    def test_drain_catches_errors(self):
        worker = AsyncIOWorker()
        worker.start()

        def fail():
            raise ValueError("test error")

        worker.submit(fail)
        errors = worker.drain()
        assert errors == 1
        assert worker._errors == 1
        worker.shutdown()

    def test_file_writing(self):
        """Test actual file writing through the worker."""
        worker = AsyncIOWorker()
        worker.start()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}

            def write_json():
                with open(path, "w") as f:
                    json.dump(data, f)

            worker.submit(write_json)
            worker.drain()

            assert path.exists()
            with open(path) as f:
                assert json.load(f) == data

        worker.shutdown()

    def test_to_dict(self):
        worker = AsyncIOWorker()
        worker.start()
        worker.submit(lambda: None)
        worker.drain()

        d = worker.to_dict()
        assert d["tasks_submitted"] == 1
        assert d["errors"] == 0
        assert d["pending"] == 0
        assert d["wall_time_s"] >= 0
        worker.shutdown()

    def test_multiple_tasks_ordering(self):
        """Tasks should execute in submission order (single thread)."""
        worker = AsyncIOWorker()
        worker.start()

        results = []
        for i in range(10):
            worker.submit(lambda x=i: results.append(x))
        worker.drain()
        assert results == list(range(10))
        worker.shutdown()

    def test_pending_count(self):
        worker = AsyncIOWorker()
        worker.start()

        import threading

        event = threading.Event()

        def blocking():
            event.wait(timeout=5)

        worker.submit(blocking)
        time.sleep(0.1)  # let task start
        assert worker.pending >= 0  # may be 0 or 1 depending on timing

        event.set()
        worker.drain()
        assert worker.pending == 0
        worker.shutdown()

    def test_double_start_idempotent(self):
        worker = AsyncIOWorker()
        worker.start()
        worker.start()  # should not raise
        assert worker._started is True
        worker.shutdown()

    def test_double_shutdown_idempotent(self):
        worker = AsyncIOWorker()
        worker.start()
        worker.shutdown()
        worker.shutdown()  # should not raise
        assert worker._started is False


# ---------------------------------------------------------------------------
# Memory tier detection
# ---------------------------------------------------------------------------


class TestMemoryTier:
    """Tests for get_memory_tier auto-detection."""

    def test_tier_s_small_gpu(self):
        """8 GB GPU → tier S."""
        tier = get_memory_tier(vram_bytes=8 * 1024**3)
        assert tier["tier"] == "S"
        assert tier["grow_factor"] == 1.3
        assert tier["min_chunk_frames"] == 15

    def test_tier_m_mid_gpu(self):
        """16 GB GPU → tier M."""
        tier = get_memory_tier(vram_bytes=16 * 1024**3)
        assert tier["tier"] == "M"
        assert tier["grow_factor"] == 1.4

    def test_tier_l_large_gpu(self):
        """40 GB GPU → tier L."""
        tier = get_memory_tier(vram_bytes=40 * 1024**3)
        assert tier["tier"] == "L"
        assert tier["grow_factor"] == 1.5

    def test_tier_xl_datacenter_gpu(self):
        """80 GB GPU → tier XL."""
        tier = get_memory_tier(vram_bytes=80 * 1024**3)
        assert tier["tier"] == "XL"
        assert tier["grow_factor"] == 1.5
        assert tier["min_chunk_frames"] == 100

    def test_tier_cpu_small(self):
        """8 GB RAM, no GPU → CPU_S."""
        tier = get_memory_tier(vram_bytes=0, ram_bytes=8 * 1024**3)
        assert tier["tier"] == "CPU_S"
        assert tier["grow_factor"] == 1.2

    def test_tier_cpu_large(self):
        """64 GB RAM, no GPU → CPU_L."""
        tier = get_memory_tier(vram_bytes=0, ram_bytes=64 * 1024**3)
        assert tier["tier"] == "CPU_L"
        assert tier["grow_factor"] == 1.4

    def test_tier_applied_to_manager(self):
        """AdaptiveChunkManager should pick up tier parameters."""
        tier = get_memory_tier(vram_bytes=80 * 1024**3)
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cuda",
            vram_limit_bytes=80 * 1024**3,
            tier=tier,
        )
        assert mgr.GROW_FACTOR == tier["grow_factor"]
        assert mgr.GROW_THRESHOLD == tier["grow_threshold"]
        assert mgr.min_chunk_frames == tier["min_chunk_frames"]
        assert mgr.max_growth_factor == tier["max_growth_factor"]


# ---------------------------------------------------------------------------
# Object count trend in growth
# ---------------------------------------------------------------------------


class TestObjectCountTrend:
    """Tests for chunk-to-chunk object count trend adjustment."""

    def test_increasing_objects_dampens_growth(self):
        """When objects increase >20% between chunks, growth is further dampened."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        low_peak = int(80 * 1024**3 * 0.95 * 0.2)
        # Chunk 0: 10 objects
        rec0 = mgr.record_chunk(chunk_id=0, chunk_size=500, peak_vram_bytes=low_peak, n_objects=10)
        # Chunk 1: 20 objects (100% increase)
        rec1 = mgr.record_chunk(chunk_id=1, chunk_size=rec0.adjusted_chunk_size, peak_vram_bytes=low_peak, n_objects=20)
        assert rec1.action == "GROW"
        # The dampened growth for 20 objects with increasing trend should be
        # less than the dampened growth for 20 objects without trend
        mgr2 = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        ref = mgr2.record_chunk(chunk_id=0, chunk_size=rec0.adjusted_chunk_size, peak_vram_bytes=low_peak, n_objects=20)
        assert rec1.adjusted_chunk_size <= ref.adjusted_chunk_size

    def test_decreasing_objects_boosts_growth(self):
        """When objects decrease >20% between chunks, growth is boosted."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        low_peak = int(80 * 1024**3 * 0.95 * 0.2)
        # Chunk 0: 20 objects
        rec0 = mgr.record_chunk(chunk_id=0, chunk_size=500, peak_vram_bytes=low_peak, n_objects=20)
        # Chunk 1: 5 objects (75% decrease)
        rec1 = mgr.record_chunk(chunk_id=1, chunk_size=rec0.adjusted_chunk_size, peak_vram_bytes=low_peak, n_objects=5)
        assert rec1.action == "GROW"
        # With decreasing objects, growth should be higher than baseline
        mgr2 = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        ref = mgr2.record_chunk(chunk_id=0, chunk_size=rec0.adjusted_chunk_size, peak_vram_bytes=low_peak, n_objects=5)
        assert rec1.adjusted_chunk_size >= ref.adjusted_chunk_size

    def test_stable_objects_no_trend_effect(self):
        """When objects are stable (±20%), no trend adjustment."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        low_peak = int(80 * 1024**3 * 0.95 * 0.2)
        # Chunk 0: 10 objects
        rec0 = mgr.record_chunk(chunk_id=0, chunk_size=500, peak_vram_bytes=low_peak, n_objects=10)
        # Chunk 1: 11 objects (10% increase, within ±20%)
        rec1 = mgr.record_chunk(chunk_id=1, chunk_size=rec0.adjusted_chunk_size, peak_vram_bytes=low_peak, n_objects=11)
        assert rec1.action == "GROW"
        # Compare with fresh manager (no history)
        mgr2 = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        ref = mgr2.record_chunk(chunk_id=0, chunk_size=rec0.adjusted_chunk_size, peak_vram_bytes=low_peak, n_objects=11)
        # Same result — no trend penalty/bonus
        assert rec1.adjusted_chunk_size == ref.adjusted_chunk_size


# ---------------------------------------------------------------------------
# Baseline-aware shrink targeting
# ---------------------------------------------------------------------------


class TestBaselineAwareShrink:
    """Tests for proportional, baseline-aware shrink in record_chunk."""

    VRAM_LIMIT = 80 * 1024**3  # 80 GB

    def _make_mgr(self, initial=1000):
        mgr = AdaptiveChunkManager(
            initial_chunk_size=initial,
            device="cpu",
            vram_limit_bytes=self.VRAM_LIMIT,
        )
        # On CPU device, constructor sets vram_limit=0. Override for
        # pressure evaluation to work with our simulated VRAM.
        mgr.vram_limit = self.VRAM_LIMIT
        return mgr

    @property
    def eff(self):
        """effective_vram_limit for 80 GB with 5% reserve → 76 GB."""
        return int(self.VRAM_LIMIT * 0.95)

    def test_critical_with_baseline_targets_shrink_pct(self):
        """CRITICAL (>=95%) with baseline should target SHRINK_TARGET_PCT."""
        mgr = self._make_mgr(975)
        baseline = int(self.eff * 0.78)  # 78% baseline (model weights)
        # Use 0.96 for peak — safely above CRITICAL_THRESHOLD (0.95)
        peak = int(self.eff * 0.96)  # 96% peak → CRITICAL

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=975,
            peak_vram_bytes=peak,
            n_objects=5,
            baseline_vram_bytes=baseline,
        )

        assert rec.action == "SHRINK"
        assert rec.pressure == "CRITICAL"
        assert rec.target_utilization_pct == round(mgr.SHRINK_TARGET_PCT * 100, 1)

        # Baseline-aware targeting SHRINK_TARGET_PCT (0.80)
        expected_raw = int(975 * ((self.eff * 0.80 - baseline) / (peak - baseline)) * 0.95)
        floor_size = int(975 * mgr.SHRINK_CRITICAL_FACTOR)
        expected = min(expected_raw, floor_size)
        expected = max(expected, mgr.min_chunk_frames)
        assert rec.adjusted_chunk_size == expected

    def test_critical_without_baseline_proportional(self):
        """CRITICAL without baseline uses proportional fallback."""
        mgr = self._make_mgr(975)
        # Use 0.96 peak — safely above CRITICAL_THRESHOLD (0.95)
        peak = int(self.eff * 0.96)
        usage_pct = peak / mgr.effective_vram_limit

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=975,
            peak_vram_bytes=peak,
            n_objects=5,
            baseline_vram_bytes=0,  # no baseline
        )

        assert rec.action == "SHRINK"
        assert rec.pressure == "CRITICAL"
        # Proportional: 975 * (0.80 / usage_pct) * 0.95
        # Floor: 975 * 0.50 = 487.5 → 487
        proportional = int(975 * (mgr.SHRINK_TARGET_PCT / max(usage_pct, 0.01)) * 0.95)
        floor_size = int(975 * mgr.SHRINK_CRITICAL_FACTOR)
        expected = min(proportional, floor_size)
        assert rec.adjusted_chunk_size == expected

    def test_warning_with_baseline(self):
        """WARNING (85-95%) with baseline → milder than CRITICAL."""
        mgr = self._make_mgr(500)
        baseline = int(self.eff * 0.50)  # 50% baseline
        # Use 0.88 — safely above WARNING_THRESHOLD (0.85), below CRITICAL (0.95)
        peak = int(self.eff * 0.88)  # 88% peak → WARNING

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=peak,
            n_objects=3,
            baseline_vram_bytes=baseline,
        )

        assert rec.action == "SHRINK"
        assert rec.pressure == "WARNING"

        # Baseline-aware computation
        target_growth = self.eff * 0.80 - baseline
        actual_growth = peak - baseline
        raw = int(500 * (target_growth / actual_growth) * 0.95)
        floor_size = int(500 * mgr.SHRINK_WARNING_FACTOR)
        expected = max(min(raw, floor_size), mgr.min_chunk_frames)
        assert rec.adjusted_chunk_size == expected

    def test_baseline_exceeds_target_uses_minimum(self):
        """When baseline alone exceeds target_pct, use minimum chunk."""
        mgr = self._make_mgr(500)
        baseline = int(self.eff * 0.85)  # 85% baseline > 80% target
        peak = int(self.eff * 0.93)  # CRITICAL

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=peak,
            n_objects=1,
            baseline_vram_bytes=baseline,
        )

        assert rec.action == "SHRINK"
        assert rec.adjusted_chunk_size == mgr.min_chunk_frames

    def test_low_baseline_proportional_wins(self):
        """With low baseline, baseline-aware gives larger chunk than floor."""
        mgr = self._make_mgr(1000)
        baseline = int(self.eff * 0.10)  # 10% baseline
        # Use 0.96 — safely above CRITICAL_THRESHOLD (0.95)
        peak = int(self.eff * 0.96)  # CRITICAL

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=1000,
            peak_vram_bytes=peak,
            n_objects=2,
            baseline_vram_bytes=baseline,
        )

        assert rec.action == "SHRINK"
        assert rec.pressure == "CRITICAL"
        # Floor: 1000 * 0.50 = 500
        floor_size = int(1000 * mgr.SHRINK_CRITICAL_FACTOR)
        assert rec.adjusted_chunk_size <= floor_size

    def test_floor_prevents_overly_generous_shrink(self):
        """Floor factor ensures a meaningful reduction even with baseline."""
        mgr = self._make_mgr(800)
        # WARNING pressure (87%) with low baseline
        # (use 0.87, safely above WARNING_THRESHOLD=0.85)
        baseline = int(self.eff * 0.05)
        peak = int(self.eff * 0.87)

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=800,
            peak_vram_bytes=peak,
            n_objects=1,
            baseline_vram_bytes=baseline,
        )

        assert rec.action == "SHRINK"
        assert rec.pressure == "WARNING"
        # Floor is 800 * 0.75 = 600
        assert rec.adjusted_chunk_size <= int(800 * mgr.SHRINK_WARNING_FACTOR)

    def test_target_utilization_pct_set_on_shrink(self):
        """ChunkMemoryRecord should have target_utilization_pct on SHRINK."""
        mgr = self._make_mgr(500)
        peak = int(self.eff * 0.92)

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=peak,
            n_objects=1,
        )

        assert rec.action == "SHRINK"
        assert rec.target_utilization_pct == round(mgr.SHRINK_TARGET_PCT * 100, 1)

    def test_target_utilization_pct_on_grow(self):
        """ChunkMemoryRecord should have GROW_TARGET_PCT on GROW."""
        mgr = self._make_mgr(500)
        peak = int(self.eff * 0.20)

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=peak,
            n_objects=1,
        )

        assert rec.action == "GROW"
        assert rec.target_utilization_pct == round(mgr.GROW_TARGET_PCT * 100, 1)


# ---------------------------------------------------------------------------
# Calibration-based GROW
# ---------------------------------------------------------------------------


class TestCalibrationBasedGrow:
    """Tests for aggressive, calibration-based growth in record_chunk."""

    VRAM_LIMIT = 80 * 1024**3  # 80 GB

    def _make_mgr(self, initial=500):
        mgr = AdaptiveChunkManager(
            initial_chunk_size=initial,
            device="cpu",
            vram_limit_bytes=self.VRAM_LIMIT,
        )
        mgr.vram_limit = self.VRAM_LIMIT
        return mgr

    @property
    def eff(self):
        return int(self.VRAM_LIMIT * 0.95)

    def test_calibrated_grow_targets_soft_limit(self):
        """With calibration data, GROW should compute frames to reach GROW_TARGET_PCT."""
        mgr = self._make_mgr(260)
        # Simulate: baseline 20% of eff, peak 27% (260 frames, "both"=520 iters)
        baseline = int(self.eff * 0.20)
        peak = int(self.eff * 0.27)
        # growth_rate = (peak - baseline) / 520 iters
        growth_rate = (peak - baseline) / 520

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=260,
            peak_vram_bytes=peak,
            n_objects=2,
            baseline_vram_bytes=baseline,
            growth_rate_per_iter=growth_rate,
        )

        assert rec.action == "GROW"

        # Expected: target_vram = eff * 0.85
        # target_growth = eff * 0.85 - eff * 0.20 = eff * 0.65
        # target_iters = target_growth / growth_rate
        # target_frames = target_iters / 2
        target_vram = self.eff * mgr.GROW_TARGET_PCT
        target_growth = target_vram - baseline
        target_iters = target_growth / growth_rate
        expected_frames = int(target_iters / 2)

        # Should be much larger than the old 260 * 1.25 = 325
        assert rec.adjusted_chunk_size > 260 * 1.25
        # But capped at max_growth_factor × initial
        cap = int(260 * mgr.max_growth_factor)
        expected_capped = min(expected_frames, cap)
        assert rec.adjusted_chunk_size == expected_capped

    def test_calibrated_grow_exceeds_dampened_floor(self):
        """Calibrated size should always be >= the old dampened multiplier."""
        mgr = self._make_mgr(200)
        baseline = int(self.eff * 0.10)
        peak = int(self.eff * 0.15)
        growth_rate = (peak - baseline) / 400  # 200 frames * 2

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=200,
            peak_vram_bytes=peak,
            n_objects=1,
            baseline_vram_bytes=baseline,
            growth_rate_per_iter=growth_rate,
        )

        assert rec.action == "GROW"
        dampened_floor = int(200 * mgr.GROW_FACTOR)  # 1 object = full factor
        # Calibrated should be >= floor (though capped by max_growth_factor)
        assert rec.adjusted_chunk_size >= dampened_floor

    def test_no_calibration_uses_dampened_fallback(self):
        """Without calibration data, GROW falls back to dampened multiplier."""
        mgr = self._make_mgr(260)
        peak = int(self.eff * 0.27)

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=260,
            peak_vram_bytes=peak,
            n_objects=2,
            # No baseline or growth_rate
        )

        assert rec.action == "GROW"
        # With 2 objects: dampened = 1 + (1.25-1) / (1+1) = 1.125
        dampened = 1.0 + (mgr.GROW_FACTOR - 1.0) / (1.0 + 1.0)
        expected = min(int(260 * dampened), int(260 * mgr.max_growth_factor))
        assert rec.adjusted_chunk_size == expected

    def test_calibrated_grow_capped_by_max_growth(self):
        """Even with calibration, growth can't exceed max_growth_factor × initial."""
        mgr = self._make_mgr(500)
        # Very low baseline → calibration wants huge chunk
        baseline = int(self.eff * 0.01)
        peak = int(self.eff * 0.02)
        growth_rate = (peak - baseline) / 1000  # tiny growth = wants huge chunk

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=peak,
            n_objects=1,
            baseline_vram_bytes=baseline,
            growth_rate_per_iter=growth_rate,
        )

        assert rec.action == "GROW"
        cap = int(500 * mgr.max_growth_factor)
        assert rec.adjusted_chunk_size == cap

    def test_calibrated_grow_with_many_objects(self):
        """Calibration still works correctly with many objects."""
        mgr = self._make_mgr(300)
        baseline = int(self.eff * 0.30)
        peak = int(self.eff * 0.40)
        growth_rate = (peak - baseline) / 600

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=300,
            peak_vram_bytes=peak,
            n_objects=20,
            baseline_vram_bytes=baseline,
            growth_rate_per_iter=growth_rate,
        )

        assert rec.action == "GROW"
        # Calibrated: target = eff * 0.85, target_growth = eff * 0.55
        target_growth = self.eff * mgr.GROW_TARGET_PCT - baseline
        target_iters = target_growth / growth_rate
        calibrated = int(target_iters / 2)
        # Floor: dampened growth with 20 objects is minimal
        import math

        dampened_floor = int(300 * (1.0 + (mgr.GROW_FACTOR - 1.0) / (1.0 + math.log2(20))))
        expected = max(calibrated, dampened_floor)
        expected = min(expected, int(300 * mgr.max_growth_factor))
        assert rec.adjusted_chunk_size == expected

    def test_soft_warning_blocks_calibrated_grow(self):
        """soft_warning_seen should prevent GROW even with calibration."""
        mgr = self._make_mgr(260)
        baseline = int(self.eff * 0.20)
        peak = int(self.eff * 0.27)
        growth_rate = (peak - baseline) / 520

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=260,
            peak_vram_bytes=peak,
            n_objects=2,
            baseline_vram_bytes=baseline,
            growth_rate_per_iter=growth_rate,
            soft_warning_seen=True,
        )

        assert rec.action == "CONTINUE"
        assert rec.adjusted_chunk_size == 260

    def test_grow_target_pct_in_record(self):
        """GROW record should show GROW_TARGET_PCT in target_utilization_pct."""
        mgr = self._make_mgr(300)
        peak = int(self.eff * 0.20)

        rec = mgr.record_chunk(
            chunk_id=0,
            chunk_size=300,
            peak_vram_bytes=peak,
            n_objects=1,
        )

        assert rec.action == "GROW"
        assert rec.target_utilization_pct == round(mgr.GROW_TARGET_PCT * 100, 1)


# ---------------------------------------------------------------------------
# AdaptiveMultiplier
# ---------------------------------------------------------------------------


class TestAdaptiveMultiplier:
    """Tests for the AdaptiveMultiplier that learns per-frame cost."""

    def _make(self, w=1920, h=1080):
        from sam3.memory_optimizer import AdaptiveMultiplier

        return AdaptiveMultiplier(width=w, height=h, device="cpu")

    def test_not_calibrated_initially(self):
        am = self._make()
        assert not am.is_calibrated
        # Falls back to static estimate
        est = am.estimate_per_frame_bytes()
        assert est > 0

    def test_update_sets_calibrated(self):
        am = self._make()
        am.update(growth_rate_per_iter=50_000_000, baseline_bytes=1_000_000_000, n_objects=1, confidence=0.9)
        assert am.is_calibrated

    def test_low_confidence_rejected(self):
        am = self._make()
        am.update(growth_rate_per_iter=50_000_000, baseline_bytes=1_000_000_000, n_objects=1, confidence=0.2)
        assert not am.is_calibrated  # confidence < 0.3 → rejected

    def test_negative_growth_rejected(self):
        am = self._make()
        am.update(growth_rate_per_iter=-1, baseline_bytes=1_000_000_000, n_objects=1, confidence=0.9)
        assert not am.is_calibrated

    def test_calibrated_estimate_uses_growth_rate(self):
        am = self._make()
        growth = 25_000_000  # 25 MB/iter
        am.update(growth_rate_per_iter=growth, baseline_bytes=2_000_000_000, n_objects=2, confidence=0.95)

        est = am.estimate_per_frame_bytes()
        # growth × 2 (for "both" direction) = 50 MB
        expected = growth * 2
        assert est == expected

    def test_window_limits_samples(self):
        am = self._make()
        for i in range(10):
            am.update(
                growth_rate_per_iter=10_000_000 * (i + 1), baseline_bytes=1_000_000_000, n_objects=1, confidence=0.8
            )
        assert len(am._samples) == am._WINDOW

    def test_compute_safe_frames(self):
        am = self._make()
        growth = 50_000_000  # 50 MB/iter → 100 MB/frame
        am.update(growth_rate_per_iter=growth, baseline_bytes=2_000_000_000, n_objects=1, confidence=0.9)

        avail = 10_000_000_000  # 10 GB
        baseline = 2_000_000_000  # 2 GB
        frames = am.compute_safe_frames(avail, baseline)
        # (10GB - 2GB) / 100MB = 80 frames
        assert frames == 80

    def test_compute_safe_frames_minimum(self):
        am = self._make()
        growth = 500_000_000  # 500 MB/iter → 1 GB/frame
        am.update(growth_rate_per_iter=growth, baseline_bytes=9_000_000_000, n_objects=1, confidence=0.9)

        # Barely any available → should return minimum 25
        frames = am.compute_safe_frames(9_500_000_000, 9_500_000_000)
        assert frames == 25

    def test_to_dict(self):
        am = self._make()
        am.update(growth_rate_per_iter=30_000_000, baseline_bytes=2_000_000_000, n_objects=3, confidence=0.85)

        d = am.to_dict()
        assert d["calibrated"] is True
        assert d["n_samples"] == 1
        assert d["adaptive_per_frame_mb"] is not None
        assert d["static_per_frame_mb"] > 0
        assert len(d["samples"]) == 1

    def test_manager_init_and_feed(self):
        """AdaptiveChunkManager.init_adaptive_multiplier + feed_calibration."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        mgr.vram_limit = 80 * 1024**3
        mgr.init_adaptive_multiplier(1920, 1080)
        assert mgr.adaptive_multiplier is not None
        assert not mgr.adaptive_multiplier.is_calibrated

        mgr.feed_calibration(
            growth_rate_per_iter=40_000_000,
            baseline_bytes=5_000_000_000,
            n_objects=2,
            confidence=0.9,
        )
        assert mgr.adaptive_multiplier.is_calibrated
        est = mgr.get_adaptive_per_frame_bytes()
        assert est is not None
        assert est == 40_000_000 * 2

    def test_record_chunk_feeds_multiplier(self):
        """record_chunk should automatically feed calibration data."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        mgr.vram_limit = 80 * 1024**3
        mgr.init_adaptive_multiplier(1920, 1080)

        eff = mgr.effective_vram_limit
        peak = int(eff * 0.40)  # low usage → GROW
        baseline = int(eff * 0.20)
        growth_rate = 35_000_000

        mgr.record_chunk(
            chunk_id=0,
            chunk_size=500,
            peak_vram_bytes=peak,
            n_objects=2,
            baseline_vram_bytes=baseline,
            growth_rate_per_iter=growth_rate,
            calibration_confidence=0.88,
        )

        assert mgr.adaptive_multiplier.is_calibrated
        assert mgr.get_adaptive_per_frame_bytes() == growth_rate * 2

    def test_to_dict_includes_multiplier(self):
        """to_dict should include adaptive_multiplier when initialised."""
        mgr = AdaptiveChunkManager(
            initial_chunk_size=500,
            device="cpu",
            vram_limit_bytes=80 * 1024**3,
        )
        mgr.vram_limit = 80 * 1024**3
        mgr.init_adaptive_multiplier(1920, 1080)
        mgr.feed_calibration(20_000_000, 3_000_000_000, 1, 0.85)

        d = mgr.to_dict()
        assert "adaptive_multiplier" in d
        assert d["adaptive_multiplier"]["calibrated"] is True
