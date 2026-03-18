"""LLM-Powered Conjunction Event Summarizer.

Generates human-readable risk assessments from CDM data + model predictions.
Two modes:
  1. Template-based (always works, no API needed): rule-based NLP from structured data
  2. Claude-enhanced (optional): uses Anthropic API for richer, contextual summaries

For Demo Day: pre-generate summaries in the pipeline, display in webapp.
For production: could run on-demand via API endpoint.
"""

import math
import os
from typing import Optional


def summarize_pair(pair: dict, model_predictions: dict = None) -> str:
    """Generate a human-readable risk summary for a conjunction pair.

    Args:
        pair: dict with CDM pair data (from cdm_forecast.json)
        model_predictions: optional dict with LSTM/ensemble/GNN predictions

    Returns:
        Natural language summary string
    """
    # Extract key data
    sat1 = pair.get("sat1_name", "Object 1")
    sat2 = pair.get("sat2_name", "Object 2")
    current_pc = pair.get("current_pc", 0)
    forecast_pc = pair.get("forecast_pc", 0)
    miss_km = pair.get("current_miss_km", 0)
    trend = pair.get("risk_direction", "unknown")
    n_updates = pair.get("n_updates", 0)
    tca = pair.get("tca", "")
    exceedance = pair.get("exceedance_probability", 0)
    ensemble_prob = pair.get("ensemble_probability")
    lstm_prob = pair.get("lstm_escalation_prob")
    pred_max_pc = pair.get("predicted_max_log10_pc")
    pred_min_miss = pair.get("predicted_min_miss_km")
    attention = pair.get("attention_weights")
    conf_lower = pair.get("exceedance_lower")
    conf_upper = pair.get("exceedance_upper")

    # Merge additional model predictions
    if model_predictions:
        ensemble_prob = model_predictions.get("ensemble_probability", ensemble_prob)
        lstm_prob = model_predictions.get("lstm_escalation_prob", lstm_prob)
        pred_max_pc = model_predictions.get("predicted_max_log10_pc", pred_max_pc)

    parts = []

    # Opening: identify the pair and headline risk level
    risk_word = _risk_level_word(current_pc, ensemble_prob or exceedance)
    parts.append(f"{sat1} vs {sat2}: {risk_word}.")

    # Current state
    pc_str = _format_pc(current_pc)
    parts.append(f"Current collision probability is {pc_str} with a miss distance of {miss_km:.1f} km.")

    # Trend analysis
    if trend == "escalating":
        parts.append("Risk is escalating across recent CDM updates.")
        if pair.get("pc_trend", 0) > 0:
            pct_change = abs(pair["pc_trend"]) * 100
            parts.append(f"Pc is increasing at approximately {pct_change:.0f}% per update.")
    elif trend == "de-escalating":
        parts.append("Risk appears to be decreasing based on recent CDM updates.")
    elif trend == "stable":
        parts.append("Risk has been stable across recent updates.")

    # TCA timing
    if tca:
        try:
            from datetime import datetime, timezone
            tca_dt = datetime.fromisoformat(tca.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_left = (tca_dt - now).total_seconds() / 3600
            if hours_left > 0:
                if hours_left < 24:
                    parts.append(f"Time of closest approach is in {hours_left:.0f} hours.")
                else:
                    parts.append(f"Closest approach is in {hours_left / 24:.1f} days.")
            else:
                parts.append("This conjunction has already reached closest approach (resolved).")
        except (ValueError, AttributeError):
            pass

    # Model predictions
    if ensemble_prob is not None:
        prob_pct = ensemble_prob * 100
        if ensemble_prob >= 0.7:
            parts.append(f"The ensemble model estimates a {prob_pct:.0f}% probability of exceeding the maneuver planning threshold (Pc > 5e-4).")
        elif ensemble_prob >= 0.3:
            parts.append(f"The ensemble model gives a {prob_pct:.0f}% chance of exceeding the maneuver threshold — borderline case requiring monitoring.")
        else:
            parts.append(f"The ensemble model estimates only a {prob_pct:.0f}% probability of threshold exceedance.")

    # Uncertainty
    if conf_lower is not None and conf_upper is not None:
        width = conf_upper - conf_lower
        if width > 0.5:
            parts.append(f"Uncertainty is high (range: {conf_lower*100:.0f}%-{conf_upper*100:.0f}%). More CDM updates needed for confident assessment.")
        elif width > 0.2:
            parts.append(f"Moderate uncertainty in prediction ({conf_lower*100:.0f}%-{conf_upper*100:.0f}%).")

    # LSTM deep learning insights
    if pred_max_pc is not None:
        pred_pc_val = 10 ** pred_max_pc
        parts.append(f"The deep learning model predicts a maximum Pc of {_format_pc(pred_pc_val)} (log\u2081\u2080 = {pred_max_pc:.2f}).")
        if pred_max_pc > -3.3:
            parts.append("This exceeds the 5e-4 maneuver planning threshold.")

    if pred_min_miss is not None and pred_min_miss > 0:
        parts.append(f"Predicted minimum miss distance: {pred_min_miss:.1f} km.")

    # Attention interpretation
    if attention and len(attention) >= 2:
        max_idx = max(range(len(attention)), key=lambda i: attention[i])
        max_weight = attention[max_idx]
        if max_weight > 0.5:
            if max_idx == len(attention) - 1:
                parts.append("The model focuses most attention on the latest CDM update, suggesting recent orbital data is most informative.")
            elif max_idx == 0:
                parts.append("The model weights the earliest CDM update heavily, indicating the initial conjunction geometry is a strong predictor.")
            else:
                parts.append(f"CDM update #{max_idx + 1} (of {len(attention)}) receives the most attention ({max_weight*100:.0f}%), suggesting a significant change occurred at that point.")

    # Data quality caveat
    if n_updates <= 2:
        parts.append("Note: Only {0} CDM update{1} available. Predictions will improve as more data arrives.".format(
            n_updates, "s" if n_updates != 1 else ""))

    # Recommendation
    if (ensemble_prob or exceedance or 0) >= 0.5 or current_pc >= 5e-4:
        parts.append("Recommendation: Active monitoring at 6-hour intervals. Consider maneuver planning if Pc continues to rise.")
    elif (ensemble_prob or exceedance or 0) >= 0.2:
        parts.append("Recommendation: Continue routine monitoring. Reassess if trend becomes escalating.")
    else:
        parts.append("Recommendation: Low priority. Standard monitoring schedule.")

    return " ".join(parts)


def summarize_pair_llm(pair: dict, model_predictions: dict = None,
                       api_key: str = None) -> str:
    """Generate an LLM-enhanced summary using Claude API.

    Falls back to template-based summary if API is unavailable.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return summarize_pair(pair, model_predictions)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build structured context for the LLM
        context = _build_llm_context(pair, model_predictions)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""You are a satellite conjunction analyst. Generate a concise,
professional risk assessment (3-5 sentences) from this conjunction data.
Be specific about numbers. Do NOT use jargon without explanation.
Mention what operators should do next.

{context}"""
            }],
        )
        return response.content[0].text

    except Exception as e:
        # Fallback to template
        return summarize_pair(pair, model_predictions)


def batch_summarize(pairs: list[dict], use_llm: bool = False,
                    api_key: str = None) -> list[dict]:
    """Generate summaries for a batch of conjunction pairs.

    Returns list of {pair_key, summary} dicts.
    """
    results = []
    for pair in pairs:
        key = f"{pair.get('sat1_norad', 0)}-{pair.get('sat2_norad', 0)}"
        if use_llm:
            summary = summarize_pair_llm(pair, api_key=api_key)
        else:
            summary = summarize_pair(pair)

        results.append({
            "sat1_norad": pair.get("sat1_norad", 0),
            "sat2_norad": pair.get("sat2_norad", 0),
            "summary": summary,
        })
    return results


def _risk_level_word(current_pc: float, exceedance_prob: float) -> str:
    """Generate headline risk description."""
    if current_pc >= 5e-3:
        return "CRITICAL RISK — collision probability extremely high"
    if current_pc >= 5e-4:
        return "HIGH RISK — above maneuver planning threshold"
    if exceedance_prob >= 0.7:
        return "ELEVATED RISK — likely to exceed maneuver threshold"
    if exceedance_prob >= 0.3:
        return "MODERATE RISK — possible threshold exceedance"
    return "LOW RISK — below monitoring threshold"


def _format_pc(pc: float) -> str:
    """Format collision probability for human readability."""
    if pc >= 0.01:
        return f"{pc*100:.1f}%"
    if pc >= 1e-4:
        return f"{pc:.1e}"
    return f"{pc:.1e}"


def _build_llm_context(pair: dict, model_predictions: dict = None) -> str:
    """Build structured text context for LLM prompt."""
    lines = []
    lines.append(f"Conjunction: {pair.get('sat1_name', '?')} (NORAD {pair.get('sat1_norad', '?')}) "
                  f"vs {pair.get('sat2_name', '?')} (NORAD {pair.get('sat2_norad', '?')})")
    lines.append(f"Current Pc: {pair.get('current_pc', 0):.2e}")
    lines.append(f"Miss distance: {pair.get('current_miss_km', 0):.1f} km")
    lines.append(f"Trend: {pair.get('risk_direction', 'unknown')}")
    lines.append(f"CDM updates: {pair.get('n_updates', 0)}")
    lines.append(f"TCA: {pair.get('tca', 'unknown')}")

    ep = pair.get("ensemble_probability") or pair.get("exceedance_probability", 0)
    lines.append(f"Ensemble P(exceed 5e-4): {ep*100:.0f}%")

    if pair.get("predicted_max_log10_pc") is not None:
        lines.append(f"LSTM predicted max log10(Pc): {pair['predicted_max_log10_pc']:.2f}")
    if pair.get("predicted_min_miss_km") is not None:
        lines.append(f"LSTM predicted min miss: {pair['predicted_min_miss_km']:.1f} km")

    lines.append(f"Maneuver threshold: Pc > 5e-4 (log10 = -3.3)")
    return "\n".join(lines)
