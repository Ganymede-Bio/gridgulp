"""Command-line interface for GridGulp."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from gridgulp.ai import GemmaClient
from gridgulp.config import Config, DetectionStrategy, VisionMode
from gridgulp.gridgulp import GridGulp
from gridgulp.models import WorkbookSchemaResult


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gridgulp",
        description="Gemma-first workbook schematization",
    )
    parser.add_argument(
        "--config",
        help="Path to gridgulp.toml. Defaults to ./gridgulp.toml when present.",
    )
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in DetectionStrategy],
        help="Override the detection strategy.",
    )
    parser.add_argument("--gemma-base-url", help="Override the Gemma API base URL.")
    parser.add_argument("--gemma-api-key", help="Override the Gemma API key.")
    parser.add_argument("--gemma-model-id", help="Override the exact Gemma model id.")
    parser.add_argument(
        "--vision-enabled",
        action=argparse.BooleanOptionalAction,
        help="Enable or disable multimodal vision fallback.",
    )
    parser.add_argument(
        "--vision-mode",
        choices=[mode.value for mode in VisionMode],
        help="Override the multimodal vision mode.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Override the minimum detection confidence.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect tables in a file or directory.")
    detect_parser.add_argument("path", help="File or directory to inspect.")
    detect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    detect_parser.add_argument(
        "--save-debug-images",
        help="Write rendered overview/crop images to this directory.",
    )
    detect_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into directories.",
    )

    schematize_parser = subparsers.add_parser(
        "schematize",
        help="Build canonical workbook schema for a file or directory.",
    )
    schematize_parser.add_argument("path", help="File or directory to schematize.")
    schematize_parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    schematize_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into directories.",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Validate Gemma configuration.")
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = Config.from_sources(
            config_path=args.config,
            overrides=_collect_overrides(args),
        )
        if args.command == "detect":
            return _run_detect(config, args)
        if args.command == "schematize":
            return _run_schematize(config, args)
        if args.command == "doctor":
            return _run_doctor(config, args)
    except Exception as exc:
        parser.exit(status=1, message=f"gridgulp: error: {exc}\n")

    parser.exit(status=2, message="gridgulp: error: unknown command\n")


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "detection_strategy": args.strategy,
        "gemma_base_url": args.gemma_base_url,
        "gemma_api_key": args.gemma_api_key,
        "gemma_model_id": args.gemma_model_id,
        "vision_enabled": getattr(args, "vision_enabled", None),
        "vision_mode": getattr(args, "vision_mode", None),
        "vision_debug_image_dir": getattr(args, "save_debug_images", None),
        "confidence_threshold": args.confidence_threshold,
    }


def _run_detect(config: Config, args: argparse.Namespace) -> int:
    porter = GridGulp(config=config)
    target = Path(args.path).expanduser()

    if target.is_dir():
        results = porter.detect_tables_in_directory_sync(target, recursive=args.recursive)
        payload = {str(path): result.model_dump(mode="json") for path, result in results.items()}
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0

        for path, result in results.items():
            print(f"{path}: {result.total_tables} tables ({result.detection_time:.2f}s)")
        return 0

    result = porter.detect_tables_sync(target)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0

    print(f"File: {result.file_info.path}")
    print(f"Strategy: {result.metadata.get('detection_strategy')}")
    print(f"Tables: {result.total_tables}")
    print(f"Methods: {', '.join(result.methods_used)}")
    for sheet in result.sheets:
        print(f"- {sheet.name}: {len(sheet.tables)} tables")
        for table in sheet.tables:
            name_suffix = f" ({table.suggested_name})" if table.suggested_name else ""
            print(
                f"  - {table.range.excel_range} | {table.confidence:.0%} | "
                f"{table.detection_method}{name_suffix}"
            )
    return 0


def _run_schematize(config: Config, args: argparse.Namespace) -> int:
    porter = GridGulp(config=config)
    target = Path(args.path).expanduser()

    if target.is_dir():
        results = porter.schematize_in_directory_sync(target, recursive=args.recursive)
        payload = {str(path): result.model_dump(mode="json") for path, result in results.items()}
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0

        for path, result in results.items():
            print(
                f"{path}: {result.total_entities} entities, "
                f"{len(result.relationships)} relationships ({result.schematization_time:.2f}s)"
            )
        return 0

    result = porter.schematize_sync(target)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0

    _print_schema_summary(result)
    return 0


def _run_doctor(config: Config, args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "detection_strategy": config.detection_strategy.value,
        "gemma": {
            "enabled": config.gemma.enabled,
            "base_url": config.gemma.base_url,
            "model_id": config.gemma.model_id,
        },
        "vision": {
            "enabled": config.vision.enabled,
            "mode": config.vision.mode.value,
        },
        "checks": [],
    }

    config.validate_runtime()
    report["checks"].append({"name": "config", "status": "ok"})

    if config.detection_strategy == DetectionStrategy.GEMMA_FIRST:
        client = GemmaClient(config.gemma)
        models = asyncio.run(client.list_models())
        report["checks"].append({"name": "endpoint", "status": "ok", "models": models})
        if config.gemma.model_id not in models:
            raise ValueError(
                f"Configured Gemma model_id '{config.gemma.model_id}' was not advertised by /models."
            )
        report["checks"].append({"name": "model", "status": "ok"})
        if config.vision.is_active():
            asyncio.run(client.probe_multimodal_support())
            report["checks"].append({"name": "multimodal", "status": "ok"})

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("GridGulp doctor")
        print(f"Strategy: {report['detection_strategy']}")
        print(f"Gemma model: {report['gemma']['model_id']}")
        print(f"Vision mode: {report['vision']['mode']}")
        for check in report["checks"]:
            print(f"- {check['name']}: {check['status']}")

    return 0


def _print_schema_summary(result: WorkbookSchemaResult) -> None:
    print(f"File: {result.file_info.path}")
    print(f"Entities: {result.total_entities}")
    print(f"Relationships: {len(result.relationships)}")
    print(f"Regions: {result.total_regions}")
    for entity in result.entities:
        print(
            f"- {entity.suggested_name} | fields={len(entity.fields)} | "
            f"records={entity.record_count_estimate} | sheets={', '.join(entity.sheet_names)}"
        )
    if result.relationships:
        print("Relationships:")
        for relationship in result.relationships:
            print(
                f"  - {relationship.from_entity_id} -> {relationship.to_entity_id} "
                f"({relationship.relationship_type.value})"
            )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
