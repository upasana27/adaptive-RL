"""Who commits first: waiting versus leading.

An agent's commitment time in an episode is its first pick-up of an ingredient of the recipe the pair ends
up preparing (the recipe of the last ingredient put on the plate). First pick-ups alone are not used because
agents often touch other ingredients before settling. Episodes in which nothing is plated, or in which the
ego never picks up an ingredient of the prepared recipe, are left out.

Reported: the fraction of episodes in which the ego commits before its partner (the ego also counts as first
when the partner never picks up an ingredient of that recipe), the median difference of the commitment times
(ego minus partner; negative = ego leads) with a one-sided Wilcoxon signed-rank test and effect size
d = mean / sd, and the success rate when the ego led versus followed (Mann-Whitney U).

Usage: python analysis/commitment.py <FC|PC|FI> <rollout.jsonl> [more.jsonl ...] [--label NAME]
"""
import argparse
import statistics as st

from scipy import stats

from events import load_episodes, parse_episode


def commitment(episodes, domain):
    diffs, led, followed, n, ego_first, partner_never, explored = [], [], [], 0, 0, 0, []
    for steps in episodes.values():
        ep = parse_episode(steps, domain)
        if ep['target'] is None or not ep['ego_picks']:
            continue
        t_ego = next((t for t, r in ep['ego_picks'] if r == ep['target']), None)
        t_par = next((t for t, r in ep['partner_picks'] if r == ep['target']), None)
        if t_ego is None:
            continue
        n += 1
        explored.append(len({r for _, r in ep['ego_picks']}) > 1)
        if t_par is None:
            partner_never += 1; ego_first += 1; led.append(ep['success']); continue
        diffs.append(t_ego - t_par)
        if t_ego < t_par:
            ego_first += 1; led.append(ep['success'])
        elif t_par < t_ego:
            followed.append(ep['success'])
    return dict(n=n, ego_first=ego_first, partner_never=partner_never, diffs=diffs, led=led, followed=followed, explored=explored)


def report(res, label):
    n, diffs = res['n'], res['diffs']
    if n == 0:
        print(f"{label}: no episode with a plated ingredient"); return
    line = f"{label}: N={n}  ego commits first {res['ego_first'] / n:.3f}  (partner never committed: {res['partner_never']})"
    if len(diffs) > 5:
        sd = st.pstdev(diffs)
        w = stats.wilcoxon(diffs, alternative='less')
        line += f"  median(t_ego - t_partner)={st.median(diffs):.0f}  d={st.mean(diffs) / sd if sd > 0 else float('nan'):.2f}  Wilcoxon W={w.statistic:.0f} p={w.pvalue:.2g}"
    print(line)
    led, fol = res['led'], res['followed']
    line = f"    success when ego led {st.mean(led) if led else float('nan'):.3f} (n={len(led)})  when ego followed {st.mean(fol) if fol else float('nan'):.3f} (n={len(fol)})"
    if led and fol:
        u = stats.mannwhitneyu(led, fol, alternative='greater')
        line += f"  Mann-Whitney U={u.statistic:.0f} p={u.pvalue:.3g}"
    print(line + f"  |  ego touched more than one recipe {st.mean(res['explored']):.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('domain', choices=['FC', 'PC', 'FI']); ap.add_argument('files', nargs='+'); ap.add_argument('--label', default='all files')
    a = ap.parse_args()
    report(commitment(load_episodes(a.files), a.domain), a.label)
