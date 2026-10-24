"""Retirement notice for the former three-stage long-form probe.

Historical implementation and its environment are available at commit d4104f0.
This entry point intentionally performs no model loading or filesystem writes.
"""


def main() -> None:
    raise SystemExit(
        "qwen3-sortformer is retired; the gateway no longer runs its three-stage "
        "pipeline. See MODELS.md (Retired Profiles) and SPEC-016. "
        "Historical source: git show d4104f0:scripts/probe_qwen3_sortformer_longform.py. "
        "That snapshot also needs its historical pipeline profile restored for reproduction."
    )


if __name__ == "__main__":
    main()
