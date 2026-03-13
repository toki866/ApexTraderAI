from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd


def compute_cluster_stats(labels: Iterable[int]) -> pd.DataFrame:
    s = pd.Series(list(labels), dtype=int)
    if s.empty:
        return pd.DataFrame(columns=["cluster_id", "count", "share", "mean_run"])
    run_id = (s != s.shift(1)).cumsum()
    runs = (
        pd.DataFrame({"cluster_id": s, "run_id": run_id})
        .groupby(["cluster_id", "run_id"], as_index=False)
        .size()
        .rename(columns={"size": "run_len"})
    )
    mean_run = runs.groupby("cluster_id")["run_len"].mean()
    counts = s.value_counts().sort_index()
    share = (counts / float(len(s))).sort_index()
    rows = []
    for cid in counts.index.tolist():
        rows.append(
            {
                "cluster_id": int(cid),
                "count": int(counts.loc[cid]),
                "share": float(share.loc[cid]),
                "mean_run": float(mean_run.get(cid, 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def derive_valid_and_rare_clusters(
    stats: pd.DataFrame,
    *,
    share_min: float,
    mean_run_min: float,
) -> Tuple[List[int], List[int]]:
    if stats.empty:
        return [], []
    rare = []
    valid = []
    for r in stats.itertuples(index=False):
        cid = int(r.cluster_id)
        is_rare = float(r.share) < float(share_min) and float(r.mean_run) < float(mean_run_min)
        if is_rare:
            rare.append(cid)
        else:
            valid.append(cid)
    return valid, rare
