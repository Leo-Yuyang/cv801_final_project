#!/usr/bin/env python3
"""Build audited GitHub and Hugging Face release trees from explicit allowlists."""

from __future__ import annotations

import argparse
import errno
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("release_config.json")
REPO_MAP_TEMPLATE = Path(__file__).with_name("repo-map.env.example")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "wkdrs" / "cv801-release",
        help="Parent directory for a new timestamped release package.",
    )
    parser.add_argument(
        "--timestamp",
        help="Package name override; defaults to UTC YYYYMMDDTHHMMSSZ.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip SHA-256 generation for a quick structural check.",
    )
    return parser.parse_args()


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported release_config.json schema_version")
    return config


def pattern_matches(relative: PurePosixPath, pattern: str) -> bool:
    value = relative.as_posix()
    return fnmatch.fnmatch(value, pattern) or fnmatch.fnmatch(relative.name, pattern)


def selected(relative: PurePosixPath, entry: dict, default_excludes: list[str]) -> bool:
    excludes = [*default_excludes, *entry.get("exclude", [])]
    if any(pattern_matches(relative, pattern) for pattern in excludes):
        return False
    includes = entry.get("include")
    return not includes or any(pattern_matches(relative, pattern) for pattern in includes)


def expand_entry(entry: dict, default_excludes: list[str]) -> list[tuple[Path, PurePosixPath]]:
    source_spec = entry["src"]
    kind = entry.get("kind", "file")
    destination = PurePosixPath(entry["dest"])
    pairs: list[tuple[Path, PurePosixPath]] = []

    if kind == "glob":
        sources = sorted(PROJECT_ROOT.glob(source_spec))
        for source in sources:
            if source.is_file() and selected(PurePosixPath(source.name), entry, default_excludes):
                pairs.append((source, destination / source.name))
        return pairs

    source = PROJECT_ROOT / source_spec
    complete_if = entry.get("complete_if")
    if complete_if and not (source / complete_if).is_file():
        return pairs
    if kind == "file":
        if source.is_file():
            pairs.append((source, destination))
        return pairs

    if kind != "tree":
        raise ValueError(f"Unsupported entry kind {kind!r}: {source_spec}")
    if not source.is_dir():
        return pairs

    max_depth = entry.get("max_depth")
    for candidate in sorted(source.rglob("*")):
        if not candidate.is_file():
            continue
        relative = PurePosixPath(candidate.relative_to(source).as_posix())
        depth = len(relative.parts) - 1
        if max_depth is not None and depth > int(max_depth):
            continue
        if selected(relative, entry, default_excludes):
            pairs.append((candidate, destination / relative))
    return pairs


def normalized_destination(path: PurePosixPath) -> PurePosixPath:
    parts = [part for part in path.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"Destination escapes release tree: {path}")
    return PurePosixPath(*parts)


def replacement_bytes(config: dict) -> list[tuple[bytes, bytes]]:
    replacements: list[tuple[bytes, bytes]] = []
    for old, new in config["replacements"].items():
        old = old.replace("{project_root}", str(PROJECT_ROOT))
        replacements.append((old.encode(), new.encode()))
    return sorted(replacements, key=lambda pair: len(pair[0]), reverse=True)


def is_text(path: Path, config: dict) -> bool:
    return path.suffix.lower() in set(config["text_suffixes"]) or path.name in {
        ".gitignore",
        "Dockerfile",
        "Makefile",
    }


def copy_sanitized(source: Path, destination: Path, replacements: list[tuple[bytes, bytes]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    for old, new in replacements:
        data = data.replace(old, new)
    destination.write_bytes(data)
    shutil.copymode(source, destination)


def materialize(
    pairs: Iterable[tuple[Path, PurePosixPath]],
    destination_root: Path,
    config: dict,
    *,
    hardlink_binary: bool,
) -> list[dict]:
    replacements = replacement_bytes(config)
    records: list[dict] = []
    occupied: dict[PurePosixPath, Path] = {}
    for source, raw_destination in pairs:
        relative = normalized_destination(raw_destination)
        if relative in occupied:
            raise ValueError(
                f"Release destination collision: {relative} from {source} and {occupied[relative]}"
            )
        occupied[relative] = source
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if is_text(source, config):
            copy_sanitized(source, destination, replacements)
            mode = "sanitized-copy"
        elif hardlink_binary:
            try:
                os.link(source, destination)
                mode = "hardlink"
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, destination)
                mode = "copy-cross-device"
        else:
            shutil.copy2(source, destination)
            mode = "copy"
        records.append(
            {
                "source": str(source.relative_to(PROJECT_ROOT)),
                "destination": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "mode": mode,
            }
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(repo_root: Path, skip_checksums: bool) -> list[dict]:
    records: list[dict] = []
    files = sorted(
        path
        for path in repo_root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    manifest_lines: list[str] = []
    for path in files:
        relative = path.relative_to(repo_root).as_posix()
        digest = None if skip_checksums else sha256_file(path)
        if digest is not None:
            manifest_lines.append(f"{digest}  {relative}\n")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    if not skip_checksums:
        (repo_root / "MANIFEST.sha256").write_text("".join(manifest_lines), encoding="utf-8")
    return records


SECRET_PATTERNS = {
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(rb"(?:github_pat_|ghp_)[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # Split the literal so the scanner can safely ship inside the GitHub package.
    "local /vast path": re.compile(rb"/vast" + rb"/users/"),
}


def scan_release(release_root: Path, config: dict) -> None:
    findings: list[str] = []
    for path in sorted(release_root.rglob("*")):
        if not path.is_file() or not is_text(path, config):
            continue
        data = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                findings.append(f"{label}: {path.relative_to(release_root)}")
    if findings:
        raise RuntimeError("Sensitive/local material found:\n" + "\n".join(findings))


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_pending(path: Path, optional_missing: list[str], grid_complete: bool) -> None:
    optional_lines = "\n".join(f"- `{item}`" for item in optional_missing) or "- None"
    if grid_complete:
        grid_status = """- The complete paired `3 x 3` Writer-by-reviewer analysis is included.
- All nine cells passed the identical 1,000-ID audit before table generation.
- The paired-analysis JSON, Markdown tables, LaTeX tables, and repair lineage are included additively; prior results remain present."""
    else:
        grid_status = """- The frozen 2B/4B Writer-only checkpoints, predictions, and metrics are included.
- The `3 x 3` Writer-by-reviewer grid is deliberately excluded until all nine cells pass the identical 1,000-ID audit and paired bootstrap analysis.
- The original PDF remains unchanged; the Writer-scale LaTeX file is labelled as an additive draft."""
    text = f"""# Pre-release checklist

## Optional experiment artifacts not found

{optional_lines}

Missing optional files belong to additive experiment branches. Rerun the builder after a branch completes to include it; existing results are retained.

## Writer-scale publication boundary

{grid_status}

## External targets and authorization

- Set the Hugging Face namespace and run `hf auth login` / `hf auth whoami` on the upload host.
- Set `GITHUB_REPO_URL` to the intended public release repository.
- Confirm the project code licence, data permissions, authorship, and citation details; no DOI or blanket open licence is asserted by this package.
- Preserve Golden Test reference isolation until every evaluated configuration is frozen.
- Recheck the upstream non-commercial research and redistribution terms before publishing the full LocateAnything checkpoint.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config()
    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root.resolve()
    package_root = output_root / stamp
    package_root.mkdir(parents=True, exist_ok=False)

    default_excludes = config["default_excludes"]
    inventory: dict = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_revision": git_revision(),
        "package": stamp,
        "public_base_weights_included": False,
        "github": {},
        "hf": {},
        "optional_missing": [],
        "writer_reviewer_grid_complete": False,
    }

    github_pairs: list[tuple[Path, PurePosixPath]] = []
    for entry in config["github"]:
        pairs = expand_entry(entry, default_excludes)
        if not pairs:
            if entry.get("optional"):
                inventory["optional_missing"].append(entry["src"])
                continue
            raise FileNotFoundError(f"Required GitHub release entry is empty: {entry['src']}")
        github_pairs.extend(pairs)
    github_root = package_root / "github"
    github_records = materialize(
        github_pairs,
        github_root,
        config,
        hardlink_binary=False,
    )
    github_files = write_manifest(github_root, args.skip_checksums)
    oversized = [record for record in github_files if record["bytes"] >= 100_000_000]
    if oversized:
        names = ", ".join(record["path"] for record in oversized)
        raise RuntimeError(f"GitHub package contains files >=100 MB: {names}")
    inventory["github"] = {
        "files": len(github_files),
        "bytes": sum(record["bytes"] for record in github_files),
        "materialization": github_records,
    }

    for repo_key, repo_config in config["hf_repositories"].items():
        repo_root = package_root / "hf" / repo_key
        repo_root.mkdir(parents=True)
        card_source = PROJECT_ROOT / repo_config["card"]
        if not card_source.is_file():
            raise FileNotFoundError(card_source)
        copy_sanitized(card_source, repo_root / "README.md", replacement_bytes(config))

        repo_pairs: list[tuple[Path, PurePosixPath]] = []
        for entry in repo_config["entries"]:
            source_text = entry["src"]
            if source_text.startswith("inits/"):
                raise RuntimeError(f"Public/base inits material is forbidden: {source_text}")
            pairs = expand_entry(entry, default_excludes)
            if not pairs:
                if entry.get("optional"):
                    inventory["optional_missing"].append(source_text)
                    continue
                raise FileNotFoundError(f"Required HF release entry is empty: {source_text}")
            repo_pairs.extend(pairs)

        materialized = materialize(
            repo_pairs,
            repo_root,
            config,
            hardlink_binary=True,
        )
        files = write_manifest(repo_root, args.skip_checksums)
        inventory["hf"][repo_key] = {
            "repo_name": repo_config["repo_name"],
            "repo_type": repo_config["repo_type"],
            "restricted": bool(repo_config.get("restricted")),
            "files": len(files),
            "bytes": sum(record["bytes"] for record in files),
            "materialization": materialized,
        }

    grid_complete = all(
        (PROJECT_ROOT / "wkdrs" / "cv801-writer-reviewer-statistics" / name).is_file()
        for name in (
            "grid_samples.csv",
            "paired_analysis.json",
            "paired_analysis.md",
            "report_tables.tex",
            "report_tables.md",
            "repair_lineage.json",
            "repair_lineage.md",
        )
    )
    inventory["writer_reviewer_grid_complete"] = grid_complete
    scan_release(package_root, config)
    write_pending(package_root / "PENDING.md", inventory["optional_missing"], grid_complete)
    shutil.copy2(REPO_MAP_TEMPLATE, package_root / "repo-map.env.example")
    inventory_path = package_root / "release_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"release_package={package_root}")
    print(f"github_files={inventory['github']['files']} github_bytes={inventory['github']['bytes']}")
    for repo_key, details in inventory["hf"].items():
        print(f"hf_repo={repo_key} files={details['files']} bytes={details['bytes']}")
    print(f"optional_missing={len(inventory['optional_missing'])}")
    print("release_preparation_status=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"release_preparation_status=FAIL error={exc}", file=sys.stderr)
        raise
