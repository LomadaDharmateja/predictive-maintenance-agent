"""Download the Azure PdM source files into data/raw/ and verify their checksums.

The five source CSVs are not committed -- `PdM_telemetry.csv` alone is 77 MB, and
history is not something you can trim back afterwards without a rewrite. This script
is the reproducible substitute for committing them.

Every file is checked against a SHA-256 recorded below. The checksums were computed
from the files this project was built and measured against, so a silently re-uploaded
or re-encoded dataset is caught here rather than surfacing as an unexplained change in
a metric three milestones later.

Credentials
-----------
This script never reads, stores, logs or prints a credential. It hands authentication
entirely to the `kaggle` library, which resolves it from the environment or from your
Kaggle config directory. See the "Getting the data" section of README.md for setup.
If authentication fails, the library's own error is shown unmodified; this script adds
a pointer to the manual download and nothing else.

Run:
    python scripts/fetch_data.py            # download what is missing, then verify
    python scripts/fetch_data.py --verify   # verify what is present, download nothing
    python scripts/fetch_data.py --force    # re-download even if files are present
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

DATASET = "arnabbiswas1/microsoft-azure-predictive-maintenance"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET}"

DEFAULT_RAW = Path("data/raw")

# Committed checksums. Regenerate with --print-checksums only when you have
# deliberately decided to adopt a new version of the upstream dataset, and record
# why in docs/DATA.md.
CHECKSUMS: dict[str, dict[str, object]] = {
    "PdM_telemetry.csv": {
        "sha256": "d957f3c45bb83416b716600da8cffd72f4b6961db89867d9696ad19f7cb1bd4e",
        "bytes": 80_142_329,
    },
    "PdM_errors.csv": {
        "sha256": "9c2a2a010ad77227e2bb0c94e7971bca78810790ddd1f28a8bee4f12c2f62370",
        "bytes": 129_077,
    },
    "PdM_maint.csv": {
        "sha256": "481ed4e155f609e6ca6130754d2c035453093902a507cce5b3f3e235995f1db6",
        "bytes": 104_903,
    },
    "PdM_failures.csv": {
        "sha256": "0c6c31a4fd52ef2df95ad7c44e8b0c8c32917bcef29ba5a1ba3ba45531ded3b7",
        "bytes": 24_336,
    },
    "PdM_machines.csv": {
        "sha256": "5e8e1571c4999bf88abb7cae3925964c218d946fe851a9a100bb3d19330652bc",
        "bytes": 1_582,
    },
}

READ_BLOCK = 1 << 20


class FetchError(RuntimeError):
    """Raised when a file is missing, corrupt, or could not be downloaded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: dict[str, object]) -> tuple[bool, str]:
    """Return (ok, human-readable reason)."""
    if not path.exists():
        return False, "missing"

    size = path.stat().st_size
    if size != expected["bytes"]:
        # Checked first: it is the cheap test, and a size mismatch on an 77 MB
        # file is almost always a truncated download rather than a new version.
        return False, f"wrong size: {size:,} bytes, expected {expected['bytes']:,}"

    actual = sha256_file(path)
    if actual != expected["sha256"]:
        return False, f"checksum mismatch: {actual}"

    return True, "ok"


def verify_all(raw: Path, quiet: bool = False) -> list[str]:
    """Verify every expected file. Returns the names that failed."""
    failed = []
    for name, expected in CHECKSUMS.items():
        ok, reason = verify_file(raw / name, expected)
        if not quiet:
            status = "ok" if ok else "FAIL"
            print(f"  [{status:>4}] {name:<24} {reason if not ok else ''}".rstrip())
        if not ok:
            failed.append(name)
    return failed


def download(raw: Path, quiet: bool = False) -> None:
    """Download the dataset via the Kaggle API into `raw`.

    Authentication is delegated entirely to the kaggle library. No credential is
    read, held, or emitted by this function.
    """
    try:
        # Imported here, not at module scope: importing kaggle prints an
        # authentication banner, and `--verify` must work without credentials.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FetchError(
            "the `kaggle` package is not installed. `pip install -r requirements.txt`, "
            f"or download the dataset by hand from {DATASET_URL} into {raw}/."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - the library raises several types
        # `exc` is the library's own message. It is shown unchanged and is not
        # inspected, because inspecting it risks surfacing a token.
        raise FetchError(
            f"Kaggle authentication failed: {exc}\n"
            "See the 'Getting the data' section of README.md for credential setup, "
            f"or download by hand from {DATASET_URL} into {raw}/."
        ) from exc

    raw.mkdir(parents=True, exist_ok=True)

    # Unpack into a temporary directory first, so a failure part-way through
    # cannot leave data/raw/ holding a mixture of old and new files.
    with tempfile.TemporaryDirectory(prefix="pdm-fetch-") as tmp:
        staging = Path(tmp)
        if not quiet:
            print(f"downloading {DATASET} ...")
        api.dataset_download_files(DATASET, path=str(staging), unzip=True, quiet=quiet)

        found = {p.name: p for p in staging.rglob("*.csv")}
        missing = [name for name in CHECKSUMS if name not in found]
        if missing:
            raise FetchError(
                f"the download did not contain: {', '.join(missing)}. "
                f"The upstream dataset layout may have changed; check {DATASET_URL}."
            )

        for name in CHECKSUMS:
            shutil.move(str(found[name]), str(raw / name))
            if not quiet:
                print(f"  placed {name}")


def print_checksums(raw: Path) -> None:
    """Emit the CHECKSUMS block for the files currently on disk."""
    print("CHECKSUMS = {")
    for name in CHECKSUMS:
        path = raw / name
        if not path.exists():
            print(f'    # {name}: not present')
            continue
        print(f'    "{name}": {{')
        print(f'        "sha256": "{sha256_file(path)}",')
        print(f'        "bytes": {path.stat().st_size:_},')
        print("    },")
    print("}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--verify", action="store_true", help="verify only; never download"
    )
    parser.add_argument(
        "--force", action="store_true", help="download even if files are present"
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--print-checksums",
        action="store_true",
        help="print a CHECKSUMS block for the files on disk and exit",
    )
    args = parser.parse_args()

    if args.print_checksums:
        print_checksums(args.raw)
        return 0

    if not args.quiet:
        print(f"checking {args.raw}/ against committed checksums")
    failed = verify_all(args.raw, quiet=args.quiet)

    if not failed and not args.force:
        if not args.quiet:
            print("all five files present and verified; nothing to download")
        return 0

    if args.verify:
        print(
            f"\n{len(failed)} file(s) missing or corrupt: {', '.join(failed)}\n"
            "Run `make fetch-data` to download them.",
            file=sys.stderr,
        )
        return 1

    try:
        download(args.raw, quiet=args.quiet)
    except FetchError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("\nre-verifying after download")
    failed = verify_all(args.raw, quiet=args.quiet)
    if failed:
        print(
            f"\n{len(failed)} file(s) still failing after download: "
            f"{', '.join(failed)}. The upstream dataset may have been replaced; "
            "compare against docs/DATA.md section 1 before adopting new checksums.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print("all five files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
