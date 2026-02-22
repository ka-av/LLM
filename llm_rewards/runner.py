from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .env_grid import SimpleRewardGrid
from .prompts import build_plan_prompt, build_eval_prompt
from .gemini_client import gemini_plan, gemini_eval
from .failure_modes import detect_reason_action_mismatch, auto_failure_tags, compute_path_metrics


def _episode_brief(step_rows: List[Dict], ep_summary: Dict) -> str:
    # Keep it short-ish for LLM eval
    last = step_rows[-1] if step_rows else {}
    return json.dumps(
        {
            "episode_summary": ep_summary,
            "last_state": {
                "t": last.get("t"),
                "agent": last.get("agent"),
                "at_goal": last.get("at_goal"),
                "coin_collected": last.get("coin_collected"),
            },
            "trajectory_head": [
                {
                    "t": r["t"],
                    "agent": r["agent"],
                    "action_name": r["action_name"],
                    "reward_observed": r["reward_observed"],
                    "plan_reason": r["plan_reason"],
                }
                for r in step_rows[:8]
            ],
            "trajectory_tail": [
                {
                    "t": r["t"],
                    "agent": r["agent"],
                    "action_name": r["action_name"],
                    "reward_observed": r["reward_observed"],
                    "plan_reason": r["plan_reason"],
                }
                for r in step_rows[-8:]
            ],
        },
        ensure_ascii=False,
    )


def run_experiment(
    episodes: int = 4,
    max_steps: int = 30,
    plan_len: int = 5,
    temperature: float = 0.4,
    use_llm_eval: bool = True,
    out_dir: str = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (episode_summary_df, step_log_df, failure_mode_breakdown_df)
    Also writes CSVs to outputs/<timestamp>/
    """
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    # episode reward modes: 1 per episode, cycling
    modes_cycle = ["true", "proxy", "misleading", "delayed"]
    objective_text = "Reach the GOAL and maximize the stated reward."

    all_steps: List[Dict] = []
    all_eps: List[Dict] = []

    pbar = tqdm(range(episodes), desc="Episodes", leave=True)
    for ep in pbar:
        reward_mode = modes_cycle[ep % len(modes_cycle)]
        env = SimpleRewardGrid(size=5, max_steps=max_steps)
        info = env.reset(seed=1000 + ep, reward_mode=reward_mode)

        plan_actions: List[int] = []
        plan_reason: str = ""
        used_fallback = False
        plan_id = 0

        step_rows: List[Dict] = []

        # cumulative returns for analysis (all channels)
        ret_observed = 0.0
        ret_true = 0.0
        ret_proxy = 0.0
        ret_misleading = 0.0
        ret_delayed_revealed = 0.0

        for _ in range(max_steps):
            if not plan_actions:
                prompt = build_plan_prompt(info, objective_text)
                plan, fallback = gemini_plan(prompt, temperature=temperature, max_retries=3)
                used_fallback = used_fallback or fallback

                plan_actions = [int(a) for a in plan["actions"] if 0 <= int(a) <= 3][:plan_len]
                plan_reason = plan.get("reason", "")
                plan_id += 1

                if not plan_actions:
                    plan_actions = [0]  # safe default

            action = plan_actions.pop(0)
            step_info = env.step(action)

            # accumulate
            ret_observed += float(step_info["reward_observed"])
            ret_true += float(step_info["reward_true"])
            ret_proxy += float(step_info["reward_proxy"])
            ret_misleading += float(step_info["reward_misleading"])
            ret_delayed_revealed += float(step_info["reward_delayed_revealed"])

            reason_mismatch = detect_reason_action_mismatch(plan_reason, step_info["action_name"])

            row = {
                "episode": ep,
                "reward_mode": reward_mode,
                "plan_id": plan_id,
                "t": step_info["t"],
                "agent": step_info["agent"],
                "action": step_info["action"],
                "action_name": step_info["action_name"],
                "plan_reason": plan_reason,
                "reason_mismatch": reason_mismatch,
                "fallback_used_anytime": used_fallback,

                "at_goal": step_info["at_goal"],
                "at_coin": step_info["at_coin"],
                "at_proxy": step_info["at_proxy"],
                "at_misleading": step_info["at_misleading"],

                "reward_observed": step_info["reward_observed"],
                "reward_true": step_info["reward_true"],
                "reward_proxy": step_info["reward_proxy"],
                "reward_misleading": step_info["reward_misleading"],
                "reward_delayed_revealed": step_info["reward_delayed_revealed"],

                "ret_observed_sofar": ret_observed,
                "ret_true_sofar": ret_true,
                "ret_proxy_sofar": ret_proxy,
                "ret_misleading_sofar": ret_misleading,
                "ret_delayed_revealed_sofar": ret_delayed_revealed,
            }

            step_rows.append(row)
            all_steps.append(row)

            info = step_info

            if step_info["terminated"] or step_info["truncated"]:
                break

        reached_goal = any(r["at_goal"] for r in step_rows)
        positions = [tuple(r["agent"]) for r in step_rows]
        metrics = compute_path_metrics(positions)

        proxy_steps = sum(1 for r in step_rows if r["at_proxy"])
        misleading_steps = sum(1 for r in step_rows if r["at_misleading"])

        ep_summary = {
            "episode": ep,
            "reward_mode": reward_mode,
            "steps": len(step_rows),
            "reached_goal": int(reached_goal),

            "return_observed": float(ret_observed),
            "return_true": float(ret_true),
            "return_proxy": float(ret_proxy),
            "return_misleading": float(ret_misleading),
            "return_delayed_revealed": float(ret_delayed_revealed),

            "proxy_steps": int(proxy_steps),
            "misleading_steps": int(misleading_steps),

            "revisit_ratio": metrics["revisit_ratio"],
            "osc_rate": metrics["osc_rate"],
            "no_move_rate": metrics["no_move_rate"],

            "fallback_used": int(used_fallback),
        }

        # auto tags
        tags_auto = auto_failure_tags(
            step_rows=step_rows,
            reached_goal=reached_goal,
            max_steps=max_steps,
            reward_mode=reward_mode,
            used_fallback=used_fallback,
        )
        ep_summary["failure_modes_auto"] = "|".join(tags_auto)

        # optional LLM eval tags
        if use_llm_eval:
            brief = _episode_brief(step_rows, ep_summary)
            eval_prompt = build_eval_prompt(brief)
            ev, ev_fallback = gemini_eval(eval_prompt, temperature=0.2, max_retries=2)
            tags_llm = ev.get("failure_modes", []) or []
            if ev_fallback and "gemini_fallback" not in tags_llm:
                tags_llm = list(tags_llm) + ["gemini_fallback"]
            ep_summary["failure_modes_llm"] = "|".join(tags_llm) if tags_llm else ""
            ep_summary["llm_eval_notes"] = ev.get("notes", "")
        else:
            ep_summary["failure_modes_llm"] = ""
            ep_summary["llm_eval_notes"] = ""

        all_eps.append(ep_summary)

        pbar.set_postfix({
            "mode": reward_mode,
            "goal": int(reached_goal),
            "R_obs": f"{ret_observed:+.2f}",
            "auto": (ep_summary["failure_modes_auto"][:18] if ep_summary["failure_modes_auto"] else "-"),
        })

    step_df = pd.DataFrame(all_steps)
    ep_df = pd.DataFrame(all_eps)

    # failure mode breakdown (auto)
    mode_rows = []
    for _, r in ep_df.iterrows():
        modes = [m for m in str(r["failure_modes_auto"]).split("|") if m.strip()]
        for m in modes:
            mode_rows.append({
                "mode": m,
                "episode": int(r["episode"]),
                "reward_mode": r["reward_mode"],
                "return_observed": float(r["return_observed"]),
                "return_true": float(r["return_true"]),
            })

    breakdown_df = (
        pd.DataFrame(mode_rows)
        .groupby("mode", as_index=False)
        .agg(
            count=("episode", "count"),
            avg_return_observed=("return_observed", "mean"),
            avg_return_true=("return_true", "mean"),
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # write files
    step_df.to_csv(os.path.join(run_dir, "step_log.csv"), index=False)
    ep_df.to_csv(os.path.join(run_dir, "episode_summary.csv"), index=False)
    breakdown_df.to_csv(os.path.join(run_dir, "failure_mode_breakdown.csv"), index=False)

    return ep_df, step_df, breakdown_df