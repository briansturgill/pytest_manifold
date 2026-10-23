"""Run manifold3d extrude/cylinder in N fresh processes and report inconsistent results.

Usage: python check_determinism.py [runs]   (default 20)
Exit code 0 if every run matches the expected values, 1 otherwise.
"""
import collections
import math
import subprocess
import sys

CHILD = """
import json
import manifold3d as m
sq = m.CrossSection.square((2, 3))
print(json.dumps({
    "extrude 2x3x5": m.Manifold.extrude(sq, 5.0).volume(),
    "cylinder r=-1": m.Manifold.cylinder(1, 1, -1, 32).volume(),
    "cylinder r=1": m.Manifold.cylinder(1, 1, 1, 32).volume(),
}))
"""

EXPECTED = {
    "extrude 2x3x5": 30.0,
    "cylinder r=-1": 0.5 * 32 * math.sin(2 * math.pi / 32),
    "cylinder r=1": 0.5 * 32 * math.sin(2 * math.pi / 32),
}


def main():
    import json

    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seen = {name: collections.Counter() for name in EXPECTED}
    for _ in range(runs):
        out = subprocess.run(
            [sys.executable, "-c", CHILD], capture_output=True, text=True, check=True
        ).stdout
        for name, value in json.loads(out).items():
            seen[name][round(value, 4)] += 1

    failed = False
    for name, counts in seen.items():
        expected = round(EXPECTED[name], 4)
        bad = {v: c for v, c in counts.items() if not math.isclose(v, expected, abs_tol=1e-3)}
        failed |= bool(bad)
        status = "FAIL" if bad else "ok"
        print(f"{status:4} {name}: expected {expected}, observed {dict(sorted(counts.items()))}")

    print(f"\n{runs} fresh processes:", "INCONSISTENT" if failed else "all correct")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
