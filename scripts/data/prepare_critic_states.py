"""Select fixed observations from disjoint D4RL episodes before critic evaluation."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for task in config["tasks"]:
        output = Path(task["output"])
        output.mkdir(parents=True, exist_ok=False)
        rng = np.random.default_rng(config["seed"])
        with h5py.File(task["dataset"], "r") as data:
            ends = np.flatnonzero(data["terminals"][:] | data["timeouts"][:])
            starts = np.r_[0, ends[:-1] + 1]
            eligible = np.flatnonzero(ends - starts >= 2)
            excluded = []
            if "exclude_selection" in task:
                previous = json.loads(Path(task["exclude_selection"]).read_text())
                excluded = [row["episode"] for row in previous["rows"]]
                eligible = eligible[~np.isin(eligible, excluded)]
            count = sum(config["split_sizes"].values())
            episodes = rng.permutation(eligible)[:count]
            assert len(episodes) == count
            rows, observations, actions = [], [], []
            cursor = 0
            for split, size in config["split_sizes"].items():
                for position in range(size):
                    episode = int(episodes[cursor])
                    # Exclude the terminal boundary; each split uses distinct episodes.
                    index = int(rng.integers(starts[episode], ends[episode]))
                    observations.append(data["observations"][index])
                    actions.append(data["actions"][index])
                    rows.append({"array_index": cursor, "split": split, "split_index": position,
                                 "episode": episode, "dataset_index": index,
                                 "episode_start": int(starts[episode]), "episode_end": int(ends[episode])})
                    cursor += 1
        np.save(output / "observations.npy", np.stack(observations))
        np.save(output / "actions.npy", np.stack(actions))
        metadata = {"task": task["name"], "dataset": task["dataset"],
                    "dataset_url": task["dataset_url"], "seed": config["seed"],
                    "normalization": "none", "selection": "one nonterminal observation per distinct complete episode",
                    "split_sizes": config["split_sizes"], "rows": rows}
        if "exclude_selection" in task:
            metadata.update(exclude_selection=task["exclude_selection"],
                            excluded_episodes=excluded)
        (output / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"task": task["name"], "selected": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
