"""
Rule-based lifestyle & nutrition guidance.  DESIGN RULES (please review clinically):
  * Tips depend on cycle PHASE, the user's GOAL and PATTERN FLAGS, never on a specific LH number:
    no diet has been shown to change LH levels or the LH surge in a reliable way.
  * Only general, guideline-style advice. No calorie/weight targets, no supplement doses, no named diets.
  * Every tip carries an evidence label and a source.
"""
PHASES = ("baseline", "approaching_surge", "surge_window", "post_surge")
GOALS = ("tracking", "trying_to_conceive", "avoiding_pregnancy")

SOURCES = {
    "pcos2023": {"title": "International evidence-based guideline for PCOS 2023 (Teede et al.), summary",
                 "url": "https://assets.monashivf.com/assets/Summary-International-evidence-based-guideline-for-the-assessment-and-management-of-polycystic-ovary-syndrome-2023.pdf"},
    "nordic": {"title": "Forslund et al. 2024, PCOS guideline: a Nordic perspective (Acta Obstet Gynecol Scand)",
               "url": "https://pubmed.ncbi.nlm.nih.gov/37983875/"},
    "bcmj": {"title": "Optimizing fertility, Part 1: evidence-based lifestyle changes (BCMJ)",
             "url": "https://bcmj.org/articles/optimizing-fertility-part-1-evidence-based-lifestyle-changes"},
    "cowan": {"title": "Cowan et al. 2023, Lifestyle management in PCOS: beyond diet and physical activity",
              "url": "https://link.springer.com/content/pdf/10.1186/s12902-022-01208-y.pdf"},
}
EVIDENCE = {"guideline": "Guideline-based", "general": "General health advice",
            "limited": "Limited evidence", "safety": "Safety information"}

BASIS = ("These tips depend on your cycle phase, your goal and your pattern flags, not on a specific LH number, "
         "because no diet has been shown to change LH levels or the LH surge reliably.")
DISCLAIMER = ("General wellness information, not personal medical, dietary or weight-loss advice. "
              "Prototype built on simulated data.")

HEADLINES = {
    "baseline": "No surge expected right now. Keep your routine steady and keep logging.",
    "approaching_surge": "A surge may start soon. Keep measuring at the same time each day.",
    "surge_window": "A surge may be under way. Ovulation usually follows in about 1 to 1.5 days.",
    "post_surge": "The surge has probably passed. You are likely in the second half of the cycle.",
}

WHEN_TO_SEEK_CARE = [
    "No period for 3 months or more, or cycles that are very irregular.",
    "Very heavy bleeding, or severe pelvic pain.",
    "Trying to conceive for 12 months without success (6 months if you are 35 or older).",
    "Rapid weight change, very restrictive eating, or you stop having periods while training hard.",
]


def _item(category, title, text, evidence, source=None):
    return {"category": category, "title": title, "text": text,
            "evidence": evidence, "evidence_label": EVIDENCE[evidence], "source": source}


def get_guidance(phase, goal="tracking", risk_band=None, cycle_irregular=False, no_surge_detected=False):
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if goal not in GOALS:
        raise ValueError(f"goal must be one of {GOALS}")
    items = []

    # ---- phase-specific (cycle information) ----
    if phase == "surge_window":
        items.append(_item("Your cycle", "Surge may be under way",
                           "Ovulation typically follows the start of an LH surge by roughly 1 to 1.5 days, though timing varies "
                           "between people and cycles.", "general", "bcmj"))
        if goal == "trying_to_conceive":
            items.append(_item("Your cycle", "Most fertile days",
                               "The fertile window is about 6 days and ends on the day of ovulation, so the days around a surge "
                               "are usually the most fertile.", "general", "bcmj"))
        items.append(_item("Nutrition", "No special diet needed for the surge",
                           "There is no good evidence that any food, drink or supplement brings on, strengthens or times an LH surge. "
                           "Keep eating regular meals and stay hydrated.", "general", "bcmj"))
    elif phase == "approaching_surge":
        items.append(_item("Your cycle", "Keep daily readings consistent",
                           "Measure at about the same time each day so the forecast can follow the rise. Missed days make the "
                           "forecast less certain.", "general"))
    elif phase == "post_surge":
        items.append(_item("Your cycle", "Second half of the cycle",
                           "After a surge, ovulation usually happens within about a day. If your period is late, a pregnancy "
                           "test is the next step.", "general", "bcmj"))
        if goal == "trying_to_conceive":
            items.append(_item("Nutrition", "Ask about folic acid before conception",
                               "Folic acid is recommended before conception and in early pregnancy. Ask a clinician or pharmacist "
                               "what is right for you.", "guideline", "bcmj"))
    else:
        items.append(_item("Your cycle", "Keep logging",
                           "A steady daily log is what lets the app recognise your own pattern over several cycles.", "general"))

    if goal == "avoiding_pregnancy":
        items.append(_item("Safety", "This app is not contraception",
                           "LH readings can miss or mistime ovulation, especially with irregular cycles. Do not rely on this "
                           "app to avoid pregnancy.", "safety"))

    # ---- foundation habits (always shown) ----
    items.append(_item("Nutrition", "Regular, balanced eating",
                       "Regular meals built around vegetables, fruit, whole grains, protein and healthy fats. Guidelines for PCOS "
                       "say no single diet is better than the others, so choose a pattern you can keep up.", "guideline", "pcos2023"))
    items.append(_item("Movement", "Stay regularly active",
                       "Regular activity you enjoy is recommended for everyone, including people with PCOS or irregular cycles. "
                       "A common adult target is about 150 minutes of moderate activity a week plus some strength work. "
                       "Very intense training with too little food can stop ovulation, so balance effort with fuel and rest.",
                       "general", "pcos2023"))
    items.append(_item("Sleep & stress", "Protect sleep, note disruptions",
                       "Poor sleep, high stress, illness and travel can shift ovulation timing in a given cycle. Note them "
                       "when you log a reading, as it helps explain unusual cycles.", "limited"))

    # ---- pattern-specific ----
    flagged = risk_band in ("moderate", "elevated") or cycle_irregular or no_surge_detected
    if flagged:
        items.append(_item("Talk to a clinician", "Your pattern is worth discussing",
                           "Cycles without a clear surge, irregular cycles or higher baseline LH have many possible causes, "
                           "including PCOS, thyroid problems, stress, weight change and others. This app cannot tell which, "
                           "and a diagnosis needs a clinical assessment.", "safety"))
        items.append(_item("Nutrition", "Fuel your body adequately",
                           "Crash diets and heavy restriction can suppress ovulation and make cycles less regular. If you are "
                           "eating much less than usual, losing weight quickly or training very hard, tell a clinician.",
                           "limited", "bcmj"))
    if risk_band == "elevated":
        items.append(_item("Nutrition", "If PCOS is confirmed",
                           "International guidelines make healthy lifestyle first-line care in PCOS, personalised to your "
                           "preferences, with no single diet or exercise plan proven best and without weight stigma. Evidence for "
                           "supplements such as inositol is limited, so discuss them with a clinician before starting.",
                           "guideline", "pcos2023"))

    used = [i["source"] for i in items if i["source"]]
    sources = [dict(key=k, **SOURCES[k]) for k in dict.fromkeys(used)]
    return {"phase": phase, "goal": goal, "headline": HEADLINES[phase], "basis": BASIS, "items": items,
            "when_to_seek_care": WHEN_TO_SEEK_CARE, "sources": sources, "disclaimer": DISCLAIMER}
