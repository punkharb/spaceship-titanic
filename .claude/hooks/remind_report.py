"""PostToolUse hook: after a pipeline file is edited, remind Claude to
(1) run it from the project folder and (2) update the owner's role report.

It never blocks anything; it only adds context for Claude.
"""
import json
import re
import sys

REPORTS = {
    1: "reports/role1_lead.md",
    2: "reports/role2_data_baseline.md",
    3: "reports/role3_random_forest.md",
    4: "reports/role4_lightgbm.md",
    5: "reports/role5_presentation.md",
}
# Which role owns which file (mirrors CLAUDE.md > Project structure).
# Matched against the END of the path, so Windows and Git-Bash paths both work.
OWNERS = [
    (r"(^|/)01_explore\.py$", 2), (r"(^|/)02_prepare\.py$", 2), (r"(^|/)03_baseline\.py$", 2),
    (r"(^|/)04_random_forest\.py$", 3), (r"(^|/)05_lightgbm\.py$", 4),
    (r"(^|/)06_compare\.py$", 5), (r"(^|/)07_final_model\.py$", 1),
    (r"(^|/)src/features\.py$", 2), (r"(^|/)src/(config|evaluation|__init__)\.py$", 1),
]

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "").replace("\\", "/")

match = next(((re.search(p, path), r) for p, r in OWNERS if re.search(p, path)), None)
if match is None:
    sys.exit(0)
rel = match[0].group(0).lstrip("/")
role = match[1]
shared = rel.startswith("src/")
msg = (
    f"You edited {rel} (owned by Role {role}). Before finishing: "
    f"run the affected script from the project folder to confirm it works, "
    f"then update {REPORTS[role]} using the template in CLAUDE.md > Role reports "
    f"(real numbers from the output only, chart paths with captions, decisions and why)."
)
if shared:
    msg += " This is a SHARED file: tell the user Role 1 must review it, and check the contract in CLAUDE.md did not change."

print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
sys.exit(0)
