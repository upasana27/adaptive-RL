"""Which recipes an agent goes for, and whether its partner follows.

Per rollout file: success rate, the distribution of the ego's and the partner's first picked ingredient
(by recipe), the distribution of delivered recipes, the fraction of episodes in which the ego picks first,
and, for those episodes, how often the partner's first pick is of the same recipe (the partner adopts the
ego's recipe). A delivered-recipe distribution concentrated on one recipe together with a high lead and
adoption rate is the signature of an ego that imposes a fixed recipe instead of adapting.
`--pool` adds the same statistics over all files together; the entropy of the delivered recipes is in bits.

Usage: python analysis/recipe_distribution.py <FC|PC|FI> <rollout.jsonl> [more.jsonl ...] [--pool]
"""
import argparse
import collections
import math
import os

from events import load_episodes, parse_episode


def distribution(episodes, domain):
    ego_first, par_first, delivered = collections.Counter(), collections.Counter(), collections.Counter()
    n = lead = 0; adopted, success = [], []
    for steps in episodes.values():
        ep = parse_episode(steps, domain); n += 1; success.append(ep['success'])
        e, p = ep['ego_picks'], ep['partner_picks']
        if e: ego_first[e[0][1]] += 1
        if p: par_first[p[0][1]] += 1
        if ep['success']: delivered[ep['target'] or '?'] += 1
        if e and p and e[0][0] < p[0][0]:
            lead += 1; adopted.append(p[0][1] == e[0][1])
    return dict(n=n, success=success, ego_first=ego_first, par_first=par_first, delivered=delivered, lead=lead, adopted=adopted)


def entropy(counter):
    tot = sum(counter.values())
    return abs(-sum(c / tot * math.log2(c / tot) for c in counter.values() if c)) if tot else float('nan')


def report(res, label):
    fmt = lambda c: ", ".join(f"{k}:{v / max(1, sum(c.values())):.2f}" for k, v in sorted(c.items())) or "-"
    n = res['n']
    print(f"{label}: episodes={n} success={sum(res['success']) / max(1, n):.2f}")
    print(f"    ego first pick     : {fmt(res['ego_first'])}")
    print(f"    partner first pick : {fmt(res['par_first'])}")
    print(f"    delivered recipe   : {fmt(res['delivered'])}   (n={sum(res['delivered'].values())}, entropy {entropy(res['delivered']):.2f} bits)")
    print(f"    ego picked first in {res['lead'] / max(1, n):.2f} of episodes; then the partner adopted the ego's recipe {sum(res['adopted']) / max(1, len(res['adopted'])):.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('domain', choices=['FC', 'PC', 'FI']); ap.add_argument('files', nargs='+'); ap.add_argument('--pool', action='store_true')
    a = ap.parse_args()
    for f in a.files:
        eps = load_episodes([f])
        seat = next(iter(eps.values()))[0]['ego'] if eps else '?'
        report(distribution(eps, a.domain), f"{os.path.basename(f)} (ego seat {seat})")
    if a.pool and len(a.files) > 1:
        report(distribution(load_episodes(a.files), a.domain), "pooled")
