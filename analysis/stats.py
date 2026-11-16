"""Statistics used for the result tables.

    python analysis/stats.py rate K N                 success rate with Wilson 95% interval
    python analysis/stats.py compare K1 N1 K2 N2      difference of two rates: Newcombe 95% interval, chi-square and Fisher tests
    python analysis/stats.py cv R1 R2 R3 ...          mean and coefficient of variation of per-partner success rates
"""
import math
import statistics as st
import sys

from scipy import stats

Z = 1.959963984540054


def wilson(k, n, z=Z):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """Interval for p1 - p2 (Newcombe 1998, method 10: square-and-add of the two Wilson intervals)."""
    p1, p2 = k1 / n1, k2 / n2; (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    return (p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def compare(k1, n1, k2, n2):
    chi2, p_chi, _, _ = stats.chi2_contingency([[k1, n1 - k1], [k2, n2 - k2]], correction=False)
    return dict(diff=k1 / n1 - k2 / n2, ci=newcombe(k1, n1, k2, n2), chi2=chi2, p_chi2=p_chi,
                p_fisher=stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def cv(rates):
    m = st.mean(rates)
    return m, (st.pstdev(rates) / m if m > 0 else float('nan'))


if __name__ == '__main__':
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ''), sys.argv[2:]
    if cmd == 'rate' and len(args) == 2:
        k, n = map(int, args); lo, hi = wilson(k, n); print(f"{k}/{n} = {k / n:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
    elif cmd == 'compare' and len(args) == 4:
        r = compare(*map(int, args))
        print(f"difference {r['diff']:+.3f}  Newcombe 95% [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  chi2={r['chi2']:.2f} p={r['p_chi2']:.3g}  Fisher p={r['p_fisher']:.3g}")
    elif cmd == 'cv' and args:
        m, c = cv([float(x) for x in args]); print(f"mean {m:.3f}  CV {c:.3f}  (n={len(args)})")
    else:
        print(__doc__)
