from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--run-weights",
        action="store_true",
        default=False,
        help="run tests that need downloaded Kronos weights and torch",
    )
