"""Does the ego improve across the episodes of a round?

Episodes are grouped into rounds (see events.rounds) and statistics are reported by position in the round
(1 to 5): success rate, and how often the ego picks before its partner. In the heterogeneous setting (FI),
give each file the type of the partners it contains as TYPE:path (e.g. R1R2:heldout_R1R2.jsonl); the script
then also reports how often the ego's first picked ingredient belongs to the partner's type, which measures
identification of the partner across episodes, and compares position 1 against positions 2-5 (Fisher).

Usage: python analysis/round_position.py <FC|PC|FI> <[TYPE:]rollout.jsonl> [...] [--max-position 5]
"""
import argparse
import collections

from scipy import stats

from events import FI_TYPE, load_episodes, parse_episode, rounds


def by_position(files, domain, max_position=5):
    paths, types = [], []
    for f in files:
        t, _, p = f.rpartition(':')
        paths.append(p); types.append(t or None)
    pos = collections.defaultdict(lambda: collections.defaultdict(list)); n_rounds = 0
    for (i, _env), eps in rounds(load_episodes(paths)):
        n_rounds += 1
        for k, steps in enumerate(eps[:max_position], 1):
            ep = parse_episode(steps, domain); e, p = ep['ego_picks'], ep['partner_picks']
            pos[k]['success'].append(ep['success'])
            if e:
                pos[k]['ego picks first'].append(not p or e[0][0] < p[0][0])
                if domain == 'FI' and types[i]:
                    pos[k]['first pick in partner type'].append(FI_TYPE[e[0][1]] == types[i])
    return pos, n_rounds


def report(pos, n_rounds):
    keys = [k for k in ('success', 'first pick in partner type', 'ego picks first') if any(k in pos[p] for p in pos)]
    print(f"rounds={n_rounds}")
    print("position  " + "  ".join(f"{k:>28s}" for k in keys))
    for p in sorted(pos):
        print(f"{p:8d}  " + "  ".join(f"{sum(pos[p][k]) / max(1, len(pos[p][k])):22.3f} (n={len(pos[p][k]):3d})" for k in keys))
    for k in keys[:2]:
        a = pos[1][k]; b = [x for p in pos if p > 1 for x in pos[p][k]]
        if a and b:
            pval = stats.fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]])[1]
            print(f"{k}: position 1 {sum(a) / len(a):.3f} vs positions 2+ {sum(b) / len(b):.3f}  Fisher p={pval:.2g}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('domain', choices=['FC', 'PC', 'FI']); ap.add_argument('files', nargs='+'); ap.add_argument('--max-position', type=int, default=5)
    a = ap.parse_args()
    report(*by_position(a.files, a.domain, a.max_position))
