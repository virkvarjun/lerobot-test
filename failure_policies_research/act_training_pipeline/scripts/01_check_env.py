#!/usr/bin/env python3
"""
Sanity-check script: verifies that the environment is set up correctly.
Run:  python scripts/01_check_env.py
"""

import sys


def main():
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)

    # Python
    print(f"Python version : {sys.version}")
    print(f"Python path    : {sys.executable}")

    # PyTorch
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
        print(f"MPS available  : {torch.backends.mps.is_available()}")
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        sys.exit(1)

    # LeRobot
    try:
        import lerobot

        version = getattr(lerobot, "__version__", "unknown")
        print(f"LeRobot version: {version}")
    except ImportError:
        print("ERROR: lerobot is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Feetech SDK (needed for SO-101)
    try:
        import scservo_sdk  # noqa: F401

        print("Feetech SDK    : installed")
    except ImportError:
        print("WARNING: feetech-servo-sdk not installed (needed for SO-101 motors).")
        print("         Run: pip install feetech-servo-sdk")

    # USB ports
    import glob

    ports = glob.glob("/dev/tty.usbmodem*")
    if ports:
        print(f"USB ports      : {', '.join(ports)}")
    else:
        print("USB ports      : none detected (robot not connected?)")

    print("=" * 50)
    print("All checks passed." if "lerobot" in sys.modules else "Some checks failed.")


if __name__ == "__main__":
    main()
