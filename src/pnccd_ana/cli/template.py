"""
pnccd_ana.cli.template
=======================
Generate a template analysis.yaml configuration file.

Usage
-----
  pnccd-template                        # writes analysis.yaml in current dir
  pnccd-template my_experiment.yaml     # writes to specified path
  pnccd-template --print                # print to stdout instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a template pnccd_ana configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="analysis.yaml",
        help="Path to write the template (default: analysis.yaml)")
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the template to stdout instead of writing a file")
    args = parser.parse_args(argv)

    from ..config import _TEMPLATE

    if args.print_only:
        sys.stdout.write(_TEMPLATE)
        return

    out = Path(args.output)
    if out.exists():
        answer = input(f"  {out} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("  Aborted.")
            return

        out.write_text(_TEMPLATE)
    print(f"  Template written to: {out}")
    print(f"  Edit it, then run:")
    print(f"    pnccd-offset          {out}")
    print(f"    pnccd-event-rec       {out}")
    print(f"    pnccd-energy-cal      {out}")
    print(f"    pnccd-time-dependency {out}")


if __name__ == "__main__":
    main()
