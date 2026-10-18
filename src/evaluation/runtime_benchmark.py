"""Collect clean, read-only live runtime telemetry from an existing local server.

This tool never starts/stops inspection, opens a camera, or changes debug state.
Rolling percentile windows are retained as supplied by the server, not averaged.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def _get(base: str, route: str) -> dict:
    with urlopen(base.rstrip('/') + route, timeout=5) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f'{route}: expected a JSON object')
    return result


def _hardware() -> dict:
    result = {'os': platform.platform(), 'cpu': platform.processor() or None, 'python': platform.python_version()}
    try:
        import torch
        available = bool(torch.cuda.is_available())
        result.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                      torch_cuda_available=available,
                      cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else [])
    except Exception as exc:
        result['torch_probe_error'] = str(exc)
    return result


def _reason(status: dict, runtime: dict) -> str | None:
    if runtime.get('enabled') or runtime.get('paused'):
        return 'Debug capture or pause is active; a clean baseline cannot be measured.'
    if status.get('inspection_status') != 'running':
        return 'Inspection is not running; camera and processing FPS are not measured.'
    if status.get('camera_status') != 'connected':
        return 'Camera is not connected; live FPS is not measured.'
    return None


def _safe_metadata(status: dict, camera: dict) -> dict:
    # Do not persist event histories, debug artifacts, frames, or review data.
    model_info = status.get('model_info') or {}
    return {key: status.get(key) for key in ('inspection_status', 'camera_status', 'runtime_mode', 'resolution',
                                               'active_model2_id', 'device_report', 'durable_session')} | {
        'model_info': {name: model_info.get(name) for name in ('detector', 'quality')}, 'camera': camera}


def _counter_delta(start: dict, end: dict, key: str) -> int | None:
    before, after = start.get(key), end.get(key)
    if not isinstance(before, int) or isinstance(before, bool) or not isinstance(after, int) or after < before:
        return None
    return after - before


def collect(base: str, scenario: str, duration: int = 60, warmup: int = 15, interval: float = 2.0,
            *, fetch=_get, sleep=time.sleep, clock=time.monotonic) -> dict:
    parsed = urlparse(base)
    if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or parsed.username or parsed.password:
        raise ValueError('Benchmark URL must be a local HTTP server.')
    if scenario not in {'empty', 'one_fish', 'overlap'} or not 60 <= duration <= 120 or warmup < 0 or interval <= 0:
        raise ValueError('Choose a supported scenario, duration 60-120 s, nonnegative warmup, and positive interval.')
    result = {'schema': 'runtime-benchmark-v1', 'recorded_at': datetime.now(timezone.utc).isoformat(),
              'scenario_operator_declared': scenario, 'duration_requested_seconds': duration, 'warmup_seconds': warmup,
              'hardware_collector': _hardware(), 'status': 'NOT_MEASURED', 'samples': [],
              'limitations': ['Rolling server percentile windows can contain frames from before this scenario.',
                              'Timing stages may overlap; do not add them or assume the largest is exclusive.',
                              'MJPEG dispatch is server pacing, not browser-visible preview FPS.',
                              'Frame-rate evidence is not fish accuracy or throughput ground truth.']}
    try:
        status, runtime = fetch(base, '/api/status'), fetch(base, '/api/debug/runtime')
        camera = fetch(base, '/api/camera/settings')
        result['initial_metadata'] = _safe_metadata(status, camera)
        reason = _reason(status, runtime)
        if reason:
            result['reason'] = reason
            return result
        sleep(warmup)
        status, runtime = fetch(base, '/api/status'), fetch(base, '/api/debug/runtime')
        reason = _reason(status, runtime)
        if reason:
            result['reason'] = 'After warmup: ' + reason
            return result
        started = clock()
        while True:
            runtime = fetch(base, '/api/debug/runtime')
            if runtime.get('enabled') or runtime.get('paused'):
                result['reason'] = 'Debug capture or pause changed during the run.'
                return result
            elapsed = max(0.0, clock() - started)
            result['samples'].append({'elapsed_seconds': round(elapsed, 3), 'active_tracks': runtime.get('active_tracks'),
                                      'frame_queue': runtime.get('frame_queue', {}), 'performance': runtime.get('performance', {})})
            if elapsed >= duration:
                break
            sleep(min(interval, duration - elapsed))
        final_status = fetch(base, '/api/status')
        final_camera = fetch(base, '/api/camera/settings')
        result['final_metadata'] = _safe_metadata(final_status, final_camera)
        reason = _reason(final_status, runtime)
        if reason:
            result['reason'] = 'At end of run: ' + reason
            return result
        first, last = result['samples'][0], result['samples'][-1]
        elapsed = last['elapsed_seconds'] - first['elapsed_seconds']
        initial_queue, final_queue = first['frame_queue'], last['frame_queue']
        rates = {}
        for key in ('captured_frames', 'processed_frames', 'dropped_frames'):
            change = _counter_delta(initial_queue, final_queue, key)
            rates[key + '_delta'] = change
            rates[key + '_per_second'] = round(change / elapsed, 3) if change is not None and elapsed > 0 else None
        rates['processed_queue_note'] = 'Queue takes, not successful published frames; use processed_frame counter for publication.'
        calls_before = first['performance'].get('counters', {})
        calls_after = last['performance'].get('counters', {})
        for key in ('processed_frame', 'model1', 'model2', 'model2_roi'):
            change = _counter_delta(calls_before, calls_after, key)
            rates[key + '_calls'] = change
            rates[key + '_per_second'] = round(change / elapsed, 3) if change is not None and elapsed > 0 else None
        result.update(status='MEASURED', duration_observed_seconds=round(elapsed, 3), rates=rates,
                      final_rolling_timings=last['performance'].get('timings', {}))
    except (OSError, URLError, ValueError, TimeoutError) as exc:
        result['reason'] = f'Server or measurement unavailable: {exc}'
    return result


def markdown(report: dict) -> str:
    lines = ['# Runtime baseline', '', f"Status: **{report['status']}**", f"Scenario (operator declared): {report['scenario_operator_declared']}",
             f"Reason: {report.get('reason', 'Measurements collected; inspect conditions and limitations below.')}", '']
    if report['status'] == 'MEASURED':
        lines += ['## Rates', '', '```json', json.dumps(report.get('rates', {}), indent=2), '```', '',
                  '## Final rolling timing window (milliseconds)', '', '| Stage | Samples | p50 | p95 |', '|---|---:|---:|---:|']
        for name, values in sorted(report.get('final_rolling_timings', {}).items()):
            lines.append(f"| {name} | {values.get('count')} | {values.get('p50')} | {values.get('p95')} |")
    lines += ['', '## Limitations', ''] + [f'- {item}' for item in report['limitations']]
    lines += ['', 'Processed FPS, camera capture FPS, and browser preview FPS are different quantities.',
              'Do not enable bounded debug capture for this procedure. The tool never controls the camera.', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--scenario', choices=['empty', 'one_fish', 'overlap'], required=True)
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--output-dir', type=Path, default=Path('results/performance'))
    args = parser.parse_args()
    report = collect(args.url, args.scenario, args.duration, args.warmup, args.interval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime('baseline_%Y%m%d_%H%M%S_') + args.scenario
    json_path, md_path = args.output_dir / (name + '.json'), args.output_dir / (name + '.md')
    with json_path.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    with md_path.open('x', encoding='utf-8') as file:
        file.write(markdown(report))
    print(f"{report['status']}: {json_path} and {md_path}")


if __name__ == '__main__':
    main()
