from __future__ import annotations

import argparse
import pandas as pd

from llm_rewards.runner import run_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4, help="Default 4 (true/proxy/misleading/delayed)")
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--plan_len", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--no_llm_eval", action="store_true", help="Disable post-episode LLM failure-mode labeling")
    args = ap.parse_args()

    ep_df, step_df, breakdown_df = run_experiment(
        episodes=args.episodes,
        max_steps=args.max_steps,
        plan_len=args.plan_len,
        temperature=args.temperature,
        use_llm_eval=(not args.no_llm_eval),
        out_dir="outputs",
    )

    # Print summary table (performance per episode)
    cols = [
        "episode", "reward_mode", "steps", "reached_goal",
        "return_observed", "return_true", "return_proxy", "return_misleading",
        "proxy_steps", "misleading_steps",
        "revisit_ratio", "osc_rate", "no_move_rate",
        "failure_modes_auto", "failure_modes_llm",
    ]
    print("\n=== Episode Summary ===")
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(ep_df[cols].to_string(index=False))

    print("\n=== Failure Mode Breakdown (AUTO) ===")
    print(breakdown_df.to_string(index=False))

    print("\nSaved logs under: outputs/run_<timestamp>/ (see episode_summary.csv, step_log.csv)")

if __name__ == "__main__":
    main()