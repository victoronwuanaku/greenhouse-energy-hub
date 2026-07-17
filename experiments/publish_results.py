"""Thin command-line caller for publication evidence regeneration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from greenhouse_energy_hub.evaluation import (
    publication_bundle_ids,
    regenerate_publication_artifacts,
)


ROOT = Path(__file__).resolve().parent.parent


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate publication artifacts from verified pinned Run Bundles."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "publication-candidates.json",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    manifest = regenerate_publication_artifacts(
        candidate_index=args.candidates,
        manifest_path=args.manifest,
        repository_root=ROOT,
        runs_root=ROOT / "results" / "runs",
        figures_root=ROOT / "results" / "figures",
        readme_path=ROOT / "README.md",
    )
    print(
        "Regenerated publication artifacts from "
        f"{len(publication_bundle_ids(manifest))} verified pinned Run Bundles."
    )


if __name__ == "__main__":
    main()
