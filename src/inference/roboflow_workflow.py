"""Roboflow workflow client for dried fish grading."""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_WORKSPACE_NAME = "shah-rukh-biao"
DEFAULT_WORKFLOW_ID = "dried-fish-quality-grading-vdried-fish-quality-grading-6-yolo26n-seg-t1-logic"
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3


class RoboflowWorkflowError(RuntimeError):
    """Base error for workflow invocation failures."""


class RoboflowAuthenticationError(RoboflowWorkflowError):
    """Raised when the API key is missing or rejected."""


class RoboflowWorkflowResponseError(RoboflowWorkflowError):
    """Raised when the response cannot be parsed safely."""


@dataclass(frozen=True)
class WorkflowImageInput:
    """Image input accepted by the Roboflow workflow endpoint."""

    type: str
    value: str

    @classmethod
    def from_path(cls, image_path: Path) -> "WorkflowImageInput":
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return cls(type="base64", value=encoded)

    @classmethod
    def from_url(cls, image_url: str) -> "WorkflowImageInput":
        return cls(type="url", value=image_url)


def _api_key() -> str:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RoboflowAuthenticationError(
            "ROBOFLOW_API_KEY is not set. Create a key in the Roboflow dashboard and export it before running inference."
        )
    return api_key


def _workflow_url(api_url: str, workspace_name: str, workflow_id: str) -> str:
    return f"{api_url.rstrip('/')}/{workspace_name}/workflows/{workflow_id}"


def _json_request(payload: dict[str, Any], api_key: str, api_url: str, workspace_name: str, workflow_id: str) -> Request:
    body = json.dumps(payload).encode("utf-8")
    return Request(
        _workflow_url(api_url, workspace_name, workflow_id),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )


def _read_json_response(response: Any) -> Any:
    raw = response.read()
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RoboflowWorkflowResponseError("Roboflow returned a non-JSON response.") from exc


def _extract_workflow_outputs(response_payload: Any) -> list[dict[str, Any]]:
    if isinstance(response_payload, list):
        outputs = response_payload
    elif isinstance(response_payload, dict):
        outputs = []
        for key in ("outputs", "result", "results", "data"):
            candidate = response_payload.get(key)
            if isinstance(candidate, list):
                outputs = candidate
                break
        if not outputs:
            outputs = [response_payload]
    else:
        raise RoboflowWorkflowResponseError("Roboflow response has an unexpected shape.")

    parsed: list[dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, dict):
            raise RoboflowWorkflowResponseError("Roboflow response entries must be dictionaries.")
        parsed.append(_normalize_output_item(item))
    return parsed


def _normalize_output_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in item.items():
        normalized[key] = _maybe_persist_image_output(key, value)
    return normalized


def _maybe_persist_image_output(key: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    output_type = value.get("type")
    output_value = value.get("value")
    if output_type not in {"base64", "image"} or not isinstance(output_value, str):
        return value

    if "image" not in key.lower() and not key.lower().endswith(("_png", "_jpg", "_jpeg", "_webp")):
        return value

    decoded = base64.b64decode(output_value)
    temp_dir = Path(tempfile.mkdtemp(prefix="roboflow_workflow_"))
    suffix = ".png" if output_type == "base64" else ".bin"
    output_path = temp_dir / f"{key}{suffix}"
    output_path.write_bytes(decoded)
    return {"type": output_type, "path": str(output_path)}


def run_workflow(
    *,
    image: str | Path,
    parameters: dict[str, Any] | None = None,
    workspace_name: str = DEFAULT_WORKSPACE_NAME,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    api_url: str = DEFAULT_API_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict[str, Any]]:
    """Run a Roboflow workflow against one image or image URL."""

    api_key = _api_key()
    image_input: WorkflowImageInput
    image_path = Path(image)

    if image_path.exists():
        image_input = WorkflowImageInput.from_path(image_path)
    else:
        image_value = str(image)
        if not image_value.startswith("https://"):
            raise ValueError("image must be a local file path or an https URL")
        image_input = WorkflowImageInput.from_url(image_value)

    payload: dict[str, Any] = {"inputs": {"image": image_input.__dict__}}
    if parameters:
        payload["parameters"] = parameters

    request = _json_request(payload, api_key, api_url, workspace_name, workflow_id)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                parsed_payload = _read_json_response(response)
            return _extract_workflow_outputs(parsed_payload)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise RoboflowAuthenticationError(
                    f"Roboflow rejected the API key while calling workflow '{workflow_id}' ({exc.code})."
                ) from exc
            last_error = exc
        except (URLError, TimeoutError, RoboflowWorkflowResponseError) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(0.5 * (2**attempt))

    raise RoboflowWorkflowError(
        f"Failed to run Roboflow workflow '{workflow_id}' after {max_retries + 1} attempts."
    ) from last_error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Roboflow dried-fish-quality workflow.")
    parser.add_argument("image", help="Local image path or https URL")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the raw workflow response")
    parser.add_argument("--parameter", action="append", default=[], help="Workflow parameter in key=value form")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    return parser


def _parse_parameters(raw_parameters: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_parameter in raw_parameters:
        key, separator, value = raw_parameter.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid parameter format: {raw_parameter!r}. Expected key=value.")
        parsed[key] = value
    return parsed


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    parameters = _parse_parameters(args.parameter)
    outputs = run_workflow(
        image=args.image,
        parameters=parameters or None,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())