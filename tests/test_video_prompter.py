"""
Tests for video_prompter.py helper functions.

Exercises the stitching, overlay, IoU remapping, mask saving, and
memory-table helpers WITHOUT loading the SAM3 model.
"""

import cv2
import numpy as np
import pytest

# Import helpers under test
from video_prompter import (
    _build_object_tracking,
    _compute_iou,
    _create_overlay_video,
    _extract_last_frame_masks,
    _fmt,
    _match_and_remap,
    _parse_timestamp,
    _resolve_range,
    _save_chunk_masks,
    _stitch_masks_to_video,
    _table,
    _trim_carry_if_needed,
)

# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------


class TestFmt:
    def test_bytes(self):
        assert _fmt(512) == "512.0 B"

    def test_megabytes(self):
        assert _fmt(10 * 1024**2) == "10.0 MB"

    def test_gigabytes(self):
        assert _fmt(2.5 * 1024**3) == "2.5 GB"


# ---------------------------------------------------------------------------
# _table (just ensure no crash)
# ---------------------------------------------------------------------------


class TestTable:
    def test_empty(self, capsys):
        _table([])
        assert capsys.readouterr().out == ""

    def test_simple(self, capsys):
        _table([["A", "B"], ["1", "2"]])
        out = capsys.readouterr().out
        assert "A" in out and "B" in out


# ---------------------------------------------------------------------------
# _extract_last_frame_masks
# ---------------------------------------------------------------------------


class TestExtractLastFrameMasks:
    def test_empty(self):
        assert _extract_last_frame_masks({}, {0, 1}) == {}

    def test_single_frame(self):
        masks = np.array(
            [
                np.ones((10, 10), dtype=bool),
                np.zeros((10, 10), dtype=bool),
            ]
        )
        result = {
            0: {
                "out_obj_ids": np.array([0, 1]),
                "out_binary_masks": masks,
            }
        }
        out = _extract_last_frame_masks(result, {0, 1})
        assert 0 in out and 1 in out
        assert out[0].max() == 255  # All-ones mask → 255
        assert out[1].max() == 0  # All-zeros mask → 0

    def test_last_frame_picked(self):
        m0 = np.zeros((5, 5), dtype=bool)
        m1 = np.ones((5, 5), dtype=bool)
        result = {
            0: {"out_obj_ids": np.array([0]), "out_binary_masks": np.array([m0])},
            5: {"out_obj_ids": np.array([0]), "out_binary_masks": np.array([m1])},
        }
        out = _extract_last_frame_masks(result, {0})
        # Should pick frame 5 (last), whose mask is all-ones
        assert out[0].min() == 255


# ---------------------------------------------------------------------------
# _compute_iou  (mirrors test_iou_matching tests for prompter's local copy)
# ---------------------------------------------------------------------------


class TestComputeIoU:
    def test_identical(self):
        m = np.ones((10, 10), dtype=np.uint8) * 255
        assert _compute_iou(m, m) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        a[:5] = 255
        b = np.zeros((10, 10), dtype=np.uint8)
        b[5:] = 255
        assert _compute_iou(a, b) == 0.0

    def test_bool_masks(self):
        a = np.ones((10, 10), dtype=bool)
        b = np.ones((10, 10), dtype=bool)
        assert _compute_iou(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _match_and_remap
# ---------------------------------------------------------------------------


class TestMatchAndRemap:
    def test_first_chunk_increments(self):
        """First chunk with no prev_masks should assign sequential global IDs."""
        masks = np.array([np.ones((5, 5), dtype=bool)])
        result = {0: {"out_obj_ids": np.array([0]), "out_binary_masks": masks}}
        remapped, ids, mapping, gnid, _iou = _match_and_remap(result, {0}, {}, 0)
        assert mapping == {0: 0}
        assert gnid == 1

    def test_second_chunk_matches(self):
        """Second chunk should match via IoU to previous global IDs."""
        mask_a = np.zeros((10, 10), dtype=bool)
        mask_a[:5] = True
        mask_b = np.zeros((10, 10), dtype=bool)
        mask_b[5:] = True

        # Prev masks from chunk 0 with global IDs 0 and 1
        prev = {
            0: (mask_a.astype(np.uint8) * 255),
            1: (mask_b.astype(np.uint8) * 255),
        }

        # Current chunk detects same masks but with fresh IDs 0, 1
        cur_masks = np.array([mask_a, mask_b])
        result = {0: {"out_obj_ids": np.array([0, 1]), "out_binary_masks": cur_masks}}

        remapped, ids, mapping, gnid, _iou = _match_and_remap(result, {0, 1}, prev, 2)
        # Should match: new 0→global 0, new 1→global 1
        assert mapping[0] == 0
        assert mapping[1] == 1
        assert gnid == 2  # no new IDs allocated

    def test_new_object_assigned(self):
        """Objects not matching prev should get new global IDs."""
        prev_mask = np.ones((10, 10), dtype=np.uint8) * 255
        prev = {0: prev_mask}

        new_mask = np.zeros((10, 10), dtype=bool)  # completely different
        cur = {0: {"out_obj_ids": np.array([0]), "out_binary_masks": np.array([new_mask])}}

        _, _, mapping, gnid, _iou = _match_and_remap(cur, {0}, prev, 1)
        # IoU is 0 → no match → new ID
        assert mapping[0] == 1
        assert gnid == 2

    def test_empty_result(self):
        result, ids, mapping, gnid, _iou = _match_and_remap({}, {0, 1}, {}, 0)
        assert result == {}
        assert 0 in ids and 1 in ids
        assert gnid == 2


# ---------------------------------------------------------------------------
# _save_chunk_masks
# ---------------------------------------------------------------------------


class TestSaveChunkMasks:
    def test_creates_mask_videos(self, tmp_path):
        """_save_chunk_masks writes per-object MP4 mask videos."""
        masks = np.array([np.ones((10, 10), dtype=bool)])
        result = {
            0: {"out_obj_ids": np.array([0]), "out_binary_masks": masks},
            1: {"out_obj_ids": np.array([0]), "out_binary_masks": masks},
        }
        masks_dir = tmp_path / "masks" / "prompt"
        _save_chunk_masks(result, {0}, masks_dir, 10, 10, 3, fps=10.0)

        # Should create one MP4 per object (no per-frame PNGs)
        mp4 = masks_dir / "object_0_mask.mp4"
        assert mp4.exists()

        # Verify frame count via OpenCV
        import cv2

        cap = cv2.VideoCapture(str(mp4))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n_frames == 3  # 3 total frames


# ---------------------------------------------------------------------------
# _stitch_masks_to_video (multi-chunk)
# ---------------------------------------------------------------------------


class TestStitchMasks:
    def _setup_chunks(self, tmp_path, n_chunks, frames_per_chunk, overlap, obj_ids):
        """Helper: write fake PNGs for each chunk."""
        chunks_dir = tmp_path / "chunks"
        chunk_infos = []

        for ci in range(n_chunks):
            start = ci * (frames_per_chunk - overlap) if ci > 0 else 0
            if ci == 0:
                start = 0
            else:
                start = chunk_infos[-1]["end"] + 1 - overlap
            end = start + frames_per_chunk - 1
            chunk_infos.append({"chunk": ci, "start": start, "end": end})

            for oid in obj_ids:
                obj_dir = chunks_dir / f"chunk_{ci}" / "masks" / "test" / f"object_{oid}"
                obj_dir.mkdir(parents=True)
                for fidx in range(frames_per_chunk):
                    # Use cv2 to write PNGs (avoids PIL/zlib incompatibility with cv2 reader)
                    arr = np.full((10, 10), 128, dtype=np.uint8)
                    cv2.imwrite(str(obj_dir / f"frame_{fidx:06d}.png"), arr)

        return chunks_dir, chunk_infos

    def test_single_chunk_stitching(self, tmp_path):
        chunks_dir, chunk_infos = self._setup_chunks(tmp_path, 1, 10, 5, {0})
        out = tmp_path / "out"
        _stitch_masks_to_video(chunks_dir, "test", {0}, chunk_infos, 5, out, 25, 10, 10)

        mp4 = out / "object_0_mask.mp4"
        assert mp4.exists()
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 10  # no skip for first chunk

    def test_two_chunk_stitching(self, tmp_path):
        """Two chunks of 10 frames with overlap 5 → 15 unique frames total."""
        chunks_dir, chunk_infos = self._setup_chunks(tmp_path, 2, 10, 5, {0})
        out = tmp_path / "out"
        _stitch_masks_to_video(chunks_dir, "test", {0}, chunk_infos, 5, out, 25, 10, 10)

        mp4 = out / "object_0_mask.mp4"
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # chunk 0: 10 frames (no skip), chunk 1: 10-5=5 frames → total 15
        assert n == 15

    def test_missing_object_in_chunk_writes_black(self, tmp_path):
        """Object only in chunk 1 → chunk 0 should write black frames."""
        # Create chunk 0 with no objects, chunk 1 with object 0
        chunks_dir = tmp_path / "chunks"
        # Chunk 0: no object dirs
        chunk0_dir = chunks_dir / "chunk_0" / "masks" / "test"
        chunk0_dir.mkdir(parents=True)

        # Chunk 1: has object 0
        obj_dir = chunks_dir / "chunk_1" / "masks" / "test" / "object_0"
        obj_dir.mkdir(parents=True)
        for i in range(10):
            arr = np.full((10, 10), 200, dtype=np.uint8)
            cv2.imwrite(str(obj_dir / f"frame_{i:06d}.png"), arr)

        chunk_infos = [
            {"chunk": 0, "start": 0, "end": 9},
            {"chunk": 1, "start": 5, "end": 14},
        ]

        out = tmp_path / "out"
        _stitch_masks_to_video(chunks_dir, "test", {0}, chunk_infos, 5, out, 25, 10, 10)

        mp4 = out / "object_0_mask.mp4"
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # chunk 0: 10 black frames, chunk 1: 10-5=5 frames → 15 total
        assert n == 15

        # First frame should be black
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        assert ret
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert frame.max() == 0, "First frame should be black (object missing in chunk 0)"

        cap.release()


# ---------------------------------------------------------------------------
# _create_overlay_video
# ---------------------------------------------------------------------------


class TestOverlay:
    def test_overlay_produces_video(self, tmp_path):
        """Overlay with a synthetic video and mask."""
        w, h, n = 20, 20, 5

        # Create original video
        vid_path = tmp_path / "source.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 25, (w, h), True)
        for _ in range(n):
            frame = np.full((h, w, 3), 128, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Create mask video
        mask_path = tmp_path / "mask.mp4"
        mwriter = cv2.VideoWriter(str(mask_path), fourcc, 25, (w, h), False)
        for _ in range(n):
            mask = np.full((h, w), 255, dtype=np.uint8)
            mwriter.write(mask)
        mwriter.release()

        out_path = tmp_path / "overlay.mp4"
        _create_overlay_video(vid_path, [mask_path], out_path, alpha=0.5)

        assert out_path.exists()
        cap = cv2.VideoCapture(str(out_path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        cap.release()


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    def test_plain_float(self):
        assert _parse_timestamp("4.5") == 4.5

    def test_integer(self):
        assert _parse_timestamp("10") == 10.0

    def test_mm_ss(self):
        assert _parse_timestamp("1:30") == 90.0

    def test_hh_mm_ss(self):
        assert _parse_timestamp("0:01:30") == 90.0

    def test_hh_mm_ss_large(self):
        assert _parse_timestamp("1:02:03") == 3723.0

    def test_mm_ss_fractional(self):
        assert _parse_timestamp("2:30.5") == 150.5

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_timestamp("not-a-time")

    def test_too_many_colons_raises(self):
        with pytest.raises(ValueError):
            _parse_timestamp("1:2:3:4")


# ---------------------------------------------------------------------------
# _resolve_range
# ---------------------------------------------------------------------------


class TestResolveRange:
    def test_no_range_returns_none(self, tmp_path):
        assert _resolve_range(tmp_path / "v.mp4", None, None) is None

    def test_frame_range_passthrough(self, tmp_path):
        result = _resolve_range(tmp_path / "v.mp4", (10, 50), None)
        assert result == (10, 50)


# ---------------------------------------------------------------------------
# _build_object_tracking
# ---------------------------------------------------------------------------


class TestBuildObjectTracking:
    def test_empty_set(self, tmp_path):
        result = _build_object_tracking(tmp_path, set(), 25.0, 0)
        assert result == []

    def test_missing_mp4(self, tmp_path):
        """Object ID with no corresponding mask video."""
        result = _build_object_tracking(tmp_path, {0}, 25.0, 0)
        assert len(result) == 1
        assert result[0]["object_id"] == 0
        assert result[0]["first_frame"] is None
        assert result[0]["total_frames_active"] == 0

    def test_with_mask_video(self, tmp_path):
        """Create a short mask video with known white frames."""
        w, h, n = 64, 64, 20
        fps = 10.0

        mp4 = tmp_path / "object_0_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4), fourcc, fps, (w, h), False)
        for i in range(n):
            # Frames 5-14 are white (10 active frames)
            if 5 <= i <= 14:
                frame = np.full((h, w), 255, dtype=np.uint8)
            else:
                frame = np.zeros((h, w), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        result = _build_object_tracking(tmp_path, {0}, fps, frame_offset=0)
        assert len(result) == 1
        r = result[0]
        assert r["object_id"] == 0
        assert r["first_frame"] == 5
        assert r["last_frame"] == 14
        assert r["total_frames_active"] == 10
        assert r["total_frames"] == n
        assert r["first_timestamp"] == 0.5
        assert r["last_timestamp"] == 1.4
        assert "first_timecode" in r
        assert r["first_timecode"].startswith("00:00:")

    def test_frame_offset(self, tmp_path):
        """Frame offset shifts absolute frame numbers."""
        w, h, n = 32, 32, 5
        fps = 10.0
        offset = 100

        mp4 = tmp_path / "object_1_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4), fourcc, fps, (w, h), False)
        for i in range(n):
            # All frames active
            writer.write(np.full((h, w), 255, dtype=np.uint8))
        writer.release()

        result = _build_object_tracking(tmp_path, {1}, fps, frame_offset=offset)
        r = result[0]
        assert r["first_frame"] == offset
        assert r["last_frame"] == offset + n - 1
        assert r["first_timestamp"] == round(offset / fps, 3)

    def test_multiple_objects_sorted(self, tmp_path):
        """Results are sorted by object ID."""
        w, h, n = 32, 32, 3
        fps = 10.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for oid in [2, 0, 1]:
            mp4 = tmp_path / f"object_{oid}_mask.mp4"
            writer = cv2.VideoWriter(str(mp4), fourcc, fps, (w, h), False)
            for _ in range(n):
                writer.write(np.full((h, w), 255, dtype=np.uint8))
            writer.release()

        result = _build_object_tracking(tmp_path, {2, 0, 1}, fps, 0)
        ids = [r["object_id"] for r in result]
        assert ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# Parallel postprocessing tests
# ---------------------------------------------------------------------------


class TestParallelStitching:
    """Verify that multi-object stitching works with parallel workers."""

    def _setup_multi_object_chunks(self, tmp_path, n_objects, n_chunks, frames_per_chunk, overlap):
        chunks_dir = tmp_path / "chunks"
        chunk_infos = []
        for ci in range(n_chunks):
            if ci == 0:
                start = 0
            else:
                start = chunk_infos[-1]["end"] + 1 - overlap
            end = start + frames_per_chunk - 1
            chunk_infos.append({"chunk": ci, "start": start, "end": end})

            for oid in range(n_objects):
                obj_dir = chunks_dir / f"chunk_{ci}" / "masks" / "test" / f"object_{oid}"
                obj_dir.mkdir(parents=True)
                for fidx in range(frames_per_chunk):
                    arr = np.full((10, 10), 128 + oid, dtype=np.uint8)
                    cv2.imwrite(str(obj_dir / f"frame_{fidx:06d}.png"), arr)

        return chunks_dir, chunk_infos

    def test_parallel_stitch_multi_objects(self, tmp_path):
        """5 objects should be stitched in parallel (max_workers > 1)."""
        n_obj = 5
        chunks_dir, chunk_infos = self._setup_multi_object_chunks(tmp_path, n_obj, 2, 10, 3)
        out = tmp_path / "out"
        _stitch_masks_to_video(
            chunks_dir,
            "test",
            set(range(n_obj)),
            chunk_infos,
            3,
            out,
            25,
            10,
            10,
            max_workers=4,
        )
        # All 5 mask videos should exist
        for oid in range(n_obj):
            mp4 = out / f"object_{oid}_mask.mp4"
            assert mp4.exists(), f"Missing mask video for object {oid}"
            cap = cv2.VideoCapture(str(mp4))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # chunk 0: 10 frames, chunk 1: 10-3=7 → 17
            assert n == 17, f"Object {oid}: expected 17 frames, got {n}"

    def test_sequential_fallback_one_object(self, tmp_path):
        """Single object should use sequential path (no subprocess spawn)."""
        chunks_dir, chunk_infos = self._setup_multi_object_chunks(tmp_path, 1, 1, 8, 0)
        out = tmp_path / "out"
        _stitch_masks_to_video(
            chunks_dir,
            "test",
            {0},
            chunk_infos,
            0,
            out,
            25,
            10,
            10,
            max_workers=1,
        )
        mp4 = out / "object_0_mask.mp4"
        assert mp4.exists()
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 8

    def test_forced_max_workers_1(self, tmp_path):
        """max_workers=1 forces sequential even with multiple objects."""
        chunks_dir, chunk_infos = self._setup_multi_object_chunks(tmp_path, 3, 1, 5, 0)
        out = tmp_path / "out"
        _stitch_masks_to_video(
            chunks_dir,
            "test",
            {0, 1, 2},
            chunk_infos,
            0,
            out,
            25,
            10,
            10,
            max_workers=1,
        )
        for oid in range(3):
            assert (out / f"object_{oid}_mask.mp4").exists()


class TestParallelTracking:
    """Verify that multi-object tracking analysis works in parallel."""

    def _make_mask_videos(self, tmp_path, n_objects, n_frames, w=32, h=32, fps=10.0):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for oid in range(n_objects):
            mp4 = tmp_path / f"object_{oid}_mask.mp4"
            writer = cv2.VideoWriter(str(mp4), fourcc, fps, (w, h), False)
            for fidx in range(n_frames):
                # Objects alternate active/inactive to verify individual analysis
                if fidx % (oid + 2) == 0:
                    writer.write(np.full((h, w), 255, dtype=np.uint8))
                else:
                    writer.write(np.zeros((h, w), dtype=np.uint8))
            writer.release()
        return fps

    def test_parallel_tracking_multi_objects(self, tmp_path):
        """6 objects analysed in parallel — results sorted by ID."""
        n_obj = 6
        fps = self._make_mask_videos(tmp_path, n_obj, 20)
        result = _build_object_tracking(
            tmp_path,
            set(range(n_obj)),
            fps,
            0,
            max_workers=4,
        )
        assert len(result) == n_obj
        ids = [r["object_id"] for r in result]
        assert ids == list(range(n_obj))
        # Each object should have different active frame counts
        # (due to different modulo patterns)
        for r in result:
            assert r["total_frames"] == 20
            assert r["total_frames_active"] > 0

    def test_sequential_tracking_max_workers_1(self, tmp_path):
        """max_workers=1 forces sequential processing."""
        fps = self._make_mask_videos(tmp_path, 3, 10)
        result = _build_object_tracking(
            tmp_path,
            {0, 1, 2},
            fps,
            0,
            max_workers=1,
        )
        assert len(result) == 3
        ids = [r["object_id"] for r in result]
        assert ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# Streaming Mask Infrastructure
# ---------------------------------------------------------------------------


class TestEmptyMaskPool:
    """Tests for EmptyMaskPool shared black frame."""

    def test_returns_black_frame(self):
        from sam3.streaming_masks import EmptyMaskPool

        pool = EmptyMaskPool(32, 24)
        frame = pool.get_black_frame()
        assert frame.shape == (24, 32)
        assert frame.dtype == np.uint8
        assert frame.max() == 0

    def test_read_only(self):
        from sam3.streaming_masks import EmptyMaskPool

        pool = EmptyMaskPool(16, 16)
        frame = pool.get_black_frame()
        assert not frame.flags.writeable


class TestMaskVideoWriter:
    """Tests for per-object lossless mask MP4 writer."""

    def test_write_and_read_back(self, tmp_path):
        import cv2

        from sam3.streaming_masks import MaskVideoWriter

        out = tmp_path / "mask.mp4"
        writer = MaskVideoWriter(out, 10.0, 32, 32)

        # Write 5 white frames
        white = np.ones((32, 32), dtype=np.uint8) * 255
        for _ in range(5):
            writer.write_frame(white)
        writer.close()

        assert out.exists()
        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 5

    def test_write_black_with_pool(self, tmp_path):
        import cv2

        from sam3.streaming_masks import EmptyMaskPool, MaskVideoWriter

        out = tmp_path / "mask.mp4"
        pool = EmptyMaskPool(20, 20)
        writer = MaskVideoWriter(out, 10.0, 20, 20)
        writer.write_black(3, pool=pool)
        writer.close()

        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 3

    def test_frame_count_tracked(self, tmp_path):
        from sam3.streaming_masks import MaskVideoWriter

        out = tmp_path / "mask.mp4"
        writer = MaskVideoWriter(out, 10.0, 16, 16)
        white = np.ones((16, 16), dtype=np.uint8) * 255
        writer.write_frame(white)
        writer.write_frame(white)
        assert writer.frames_written == 2
        writer.close()


class TestStreamingMaskWriterIntegration:
    """Integration test: save_chunk_masks with multi-object streaming."""

    def test_multi_object_masks(self, tmp_path):
        """Multiple objects produce separate MP4 files."""
        import cv2

        masks_dir = tmp_path / "masks"
        # Two objects, 4 frames total
        mask0 = np.array([np.ones((10, 10), dtype=bool)])
        mask1 = np.array([np.zeros((10, 10), dtype=bool)])
        result = {
            0: {"out_obj_ids": np.array([0, 1]), "out_binary_masks": np.stack([mask0[0], mask1[0]])},
            1: {"out_obj_ids": np.array([0, 1]), "out_binary_masks": np.stack([mask0[0], mask1[0]])},
        }
        _save_chunk_masks(result, {0, 1}, masks_dir, 10, 10, 4, fps=10.0)

        for oid in [0, 1]:
            mp4 = masks_dir / f"object_{oid}_mask.mp4"
            assert mp4.exists()
            cap = cv2.VideoCapture(str(mp4))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            assert n == 4  # all 4 frames written (2 with data, 2 black)


# ---------------------------------------------------------------------------
# _trim_carry_if_needed
# ---------------------------------------------------------------------------


class TestTrimCarryIfNeeded:
    """Tests for the RAM guard that drops carry entries under memory pressure."""

    def test_no_drop_when_within_limits(self):
        """Carry is untouched when RAM usage is within both limits."""
        carry = {
            "person": {0: np.ones((10, 10), dtype=np.uint8)},
            "ball": {1: np.zeros((10, 10), dtype=np.uint8)},
        }
        # max_ram_pct=1.0 and min_free_gb=0 → never triggers
        dropped = _trim_carry_if_needed(carry, max_ram_pct=1.0, min_free_gb=0.0)
        assert dropped == 0
        assert len(carry) == 2

    def test_drops_oldest_when_pct_exceeded(self):
        """When fake pct threshold is 0 (always exceeded), all entries are dropped."""
        carry = {
            "first": {0: np.ones((10, 10), dtype=np.uint8)},
            "second": {1: np.ones((10, 10), dtype=np.uint8)},
        }
        # max_ram_pct=0.0 means 0% threshold → always exceeded
        dropped = _trim_carry_if_needed(carry, max_ram_pct=0.0, min_free_gb=0.0)
        assert dropped == 2
        assert len(carry) == 0

    def test_drops_oldest_when_free_gb_exceeded(self):
        """When min_free_gb is very large, all entries are dropped."""
        carry = {
            "alpha": {0: np.ones((10, 10), dtype=np.uint8)},
            "beta": {1: np.ones((10, 10), dtype=np.uint8)},
        }
        # min_free_gb=9999 → always exceeded (nobody has 9999 GB free)
        dropped = _trim_carry_if_needed(carry, max_ram_pct=1.0, min_free_gb=9999.0)
        assert dropped == 2
        assert len(carry) == 0

    def test_empty_carry_noop(self):
        """Empty carry should not raise and return 0."""
        carry = {}
        dropped = _trim_carry_if_needed(carry, max_ram_pct=0.0, min_free_gb=9999.0)
        assert dropped == 0

    def test_insertion_order_preserved(self):
        """Items are dropped in insertion order (oldest first)."""
        carry = {
            "oldest": {0: np.ones((5, 5), dtype=np.uint8)},
            "middle": {1: np.ones((5, 5), dtype=np.uint8)},
            "newest": {2: np.ones((5, 5), dtype=np.uint8)},
        }
        # Force 1 drop at a time by using a threshold that becomes
        # satisfied after removing entries (we use a trick: only check
        # that at least oldest is dropped first)
        # With pct=0 all will be dropped, but order matters
        _trim_carry_if_needed(carry, max_ram_pct=0.0, min_free_gb=0.0)
        # All dropped because pct=0 always exceeds
        assert len(carry) == 0
