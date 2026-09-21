#!/usr/bin/env python3
"""Convenience wrapper; shares arguments with the installed open-jev CLI."""

import sys

from open_jev.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-synthetic", *sys.argv[1:]]))
