from __future__ import annotations

from copy import deepcopy

import pytest

from src.evaluation import runtime_benchmark as benchmark


def test_stopped_camera_is_not_a_measurement(monkeypatch):
    monkeypatch.setattr(benchmark, '_hardware', lambda: {})
    responses = {'/api/status': {'inspection_status': 'stopped', 'camera_status': 'disconnected'},
                 '/api/debug/runtime': {'enabled': False, 'paused': False}, '/api/camera/settings': {}}
    result = benchmark.collect('http://127.0.0.1:8000', 'empty', fetch=lambda _base, path: responses[path],
                               sleep=lambda _seconds: pytest.fail('should not wait for stopped camera'))
    assert result['status'] == 'NOT_MEASURED'
    assert 'not running' in result['reason']
    assert 'rates' not in result


def test_debug_capture_is_not_a_clean_baseline(monkeypatch):
    monkeypatch.setattr(benchmark, '_hardware', lambda: {})
    responses = {'/api/status': {'inspection_status': 'running', 'camera_status': 'connected'},
                 '/api/debug/runtime': {'enabled': True}, '/api/camera/settings': {}}
    result = benchmark.collect('http://localhost:8000', 'one_fish', fetch=lambda _base, path: responses[path])
    assert result['status'] == 'NOT_MEASURED'
    assert 'Debug' in result['reason']


def test_rates_use_counter_deltas_and_preserve_rolling_timings(monkeypatch):
    monkeypatch.setattr(benchmark, '_hardware', lambda: {})
    elapsed = [0.0]
    def sleep(value): elapsed[0] += value
    def fetch(_base, path):
        if path == '/api/status':
            return {'inspection_status': 'running', 'camera_status': 'connected'}
        if path == '/api/camera/settings':
            return {}
        count = int(elapsed[0])
        return deepcopy({'enabled': False, 'paused': False,
                         'frame_queue': {'captured_frames': count * 10, 'processed_frames': count * 5, 'dropped_frames': count * 5},
                         'performance': {'counters': {'processed_frame': count * 4, 'model1': count * 5, 'model2': count},
                                         'timings': {'model1_inference': {'count': 100, 'p50': 100, 'p95': 200}}}})
    result = benchmark.collect('http://localhost:8000', 'overlap', duration=60, warmup=0, interval=10,
                               fetch=fetch, sleep=sleep, clock=lambda: elapsed[0])
    assert result['status'] == 'MEASURED'
    assert result['rates']['captured_frames_per_second'] == 10
    assert result['rates']['processed_frame_per_second'] == 4
    assert result['rates']['processed_frames_per_second'] == 5
    assert result['final_rolling_timings']['model1_inference']['p95'] == 200


def test_remote_url_or_invalid_duration_is_rejected():
    with pytest.raises(ValueError):
        benchmark.collect('http://example.com', 'empty')
    with pytest.raises(ValueError):
        benchmark.collect('http://localhost', 'empty', duration=2)
