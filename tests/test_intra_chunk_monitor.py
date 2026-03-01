"""Tests for IntraChunkMonitor and AsyncIOWorker."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sam3.memory_optimizer import (
    IntraChunkMonitor,
    FrameMemorySample,
    GrowthCalibration,
    MonitorResult,
    _linear_regression,
)
from sam3.async_io import AsyncIOWorker


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

    def test_build_checkpoints_includes_calibration_frames(self):
        monitor = IntraChunkMonitor(
            expected_iterations=100,
            device="cpu",
        )
        # First CALIBRATION_FRAMES should be included
        for i in range(IntraChunkMonitor.CALIBRATION_FRAMES):
            assert i in monitor._checkpoints

    def test_build_checkpoints_exponential_decay(self):
        monitor = IntraChunkMonitor(
            expected_iterations=1000,
            device="cpu",
        )
        # Should include 500 (N/2), 750 (3N/4), 875 (7N/8), etc.
        assert 500 in monitor._checkpoints
        assert 750 in monitor._checkpoints
        assert 999 in monitor._checkpoints  # last iteration

    def test_build_checkpoints_small_chunk(self):
        """Very small chunk should still have valid checkpoints."""
        monitor = IntraChunkMonitor(
            expected_iterations=3,
            device="cpu",
        )
        assert 0 in monitor._checkpoints
        assert 1 in monitor._checkpoints
        assert 2 in monitor._checkpoints

    def test_check_always_continues_on_cpu(self):
        """CPU device has no VRAM limit, so check() should always return True."""
        monitor = IntraChunkMonitor(
            expected_iterations=10,
            device="cpu",
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
        """Monitor should signal stop when VRAM exceeds hard threshold."""
        mock_gpu.return_value = MagicMock(total=10 * 1024**3)

        monitor = IntraChunkMonitor(
            expected_iterations=20,
            device="cuda",
            vram_limit_bytes=10 * 1024**3,
        )

        # Mock torch.cuda to simulate high memory usage
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # 96% of 9.5GB effective limit (10GB * 0.95 reserve)
        mock_torch.cuda.memory_allocated.return_value = int(9.5 * 1024**3 * 0.96)
        mock_torch.cuda.reset_peak_memory_stats.return_value = None

        with patch.dict("sys.modules", {"torch": mock_torch}):
            monitor.start()
            # First few frames are calibration — will sample and trigger hard stop
            result = monitor.check(frame_idx=0)
            # At 96% of effective limit, should trigger hard stop
            assert result is False
            assert monitor._stop_reason == "hard_limit"

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
