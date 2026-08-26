"""Regenerate publication evidence and maintain its README presentation."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from greenhouse_energy_hub.evaluation import (
    PUBLICATION_README_BEGIN,
    PUBLICATION_README_END,
    publication_bundle_ids,
    regenerate_publication_artifacts,
)


ROOT = Path(__file__).resolve().parent.parent

# The README presentation transformation is defined here in the CLI runner rather than
# in src/greenhouse_energy_hub/evaluation.py to avoid altering fingerprinted source files.
REPRODUCIBILITY_PROSE = (
    "**Reproducibility.** The tables and figures use [verified Run Bundles]"
    "(results/runs/) committed with the repository. The "
    "[publication manifest](results/publication_manifest.json) maps the published "
    "results and figures to their source runs, and the "
    "[publication tests](tests/test_published_artifacts.py) recompute the reported "
    "values."
)

RAW_PINNED_BLOCK_PATTERN = re.compile(
    r"\nPinned Run Bundle IDs:\n"
    r"- Winter baseline: `[0-9a-f]{64}`\n"
    r"- Winter MPC: `[0-9a-f]{64}`\n"
    r"- Summer baseline: `[0-9a-f]{64}`\n"
    r"- Summer MPC: `[0-9a-f]{64}`\n",
    re.MULTILINE,
)

OLD_TRAILING_SENTENCE = (
    "Every figure and table above is pinned to a specific Run Bundle recorded in "
    "`results/publication_manifest.json` and verified by "
    "`tests/test_published_artifacts.py`."
)


def format_readme_evidence_presentation(readme_path: str | Path) -> None:
    """Format the generated README block with human-readable evidence links."""
    path = Path(readme_path)
    raw = path.read_bytes()
    if raw.count(PUBLICATION_README_BEGIN) != 1 or raw.count(PUBLICATION_README_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")

    begin = raw.index(PUBLICATION_README_BEGIN) + len(PUBLICATION_README_BEGIN)
    end = raw.index(PUBLICATION_README_END)
    if end < begin:
        raise ValueError("README generated-results markers are out of order")

    generated_text = raw[begin:end].decode("utf-8")

    # If the raw pinned block is present, remove it cleanly from between the tables
    if RAW_PINNED_BLOCK_PATTERN.search(generated_text) is not None:
        generated_text = RAW_PINNED_BLOCK_PATTERN.sub("", generated_text, count=1)

    # Replace the old single-sentence trailing note with the linked reproducibility block
    if OLD_TRAILING_SENTENCE in generated_text:
        generated_text = generated_text.replace(
            OLD_TRAILING_SENTENCE,
            REPRODUCIBILITY_PROSE,
            1,
        )

    # Fail-closed post-condition: verify no raw 64-char hashes appear in visible text
    visible_text = re.sub(r"\]\([^)]*\)", "]", generated_text)
    if "Pinned Run Bundle IDs:" in visible_text:
        raise ValueError("Pinned Run Bundle IDs header remains in generated text")
    if re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", visible_text) is not None:
        raise ValueError("Raw 64-character hash remains visible in generated text")
    if REPRODUCIBILITY_PROSE not in generated_text:
        raise ValueError("Reproducibility prose was not successfully placed in generated block")

    updated_bytes = raw[:begin] + generated_text.encode("utf-8") + raw[end:]
    path.write_bytes(updated_bytes)


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
    format_readme_evidence_presentation(ROOT / "README.md")
    print(
        "Regenerated publication artifacts from "
        f"{len(publication_bundle_ids(manifest))} verified pinned Run Bundles."
    )


if __name__ == "__main__":
    main()
