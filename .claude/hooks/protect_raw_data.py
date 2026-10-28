"""PreToolUse hook: never let Claude change the raw data files.

train.csv and test.csv are the source of truth for the whole team. Reading and
copying them is fine; editing, overwriting, moving or deleting them is blocked.
Exit code 2 blocks the tool call and shows the message to Claude.
"""
import json
import re
import sys

PROTECTED = {"train.csv", "test.csv"}
MESSAGE = (
    "Blocked: train.csv and test.csv are the team's raw data and must never be "
    "changed (see CLAUDE.md > Shared rules). Load them with pandas and write any "
    "cleaned or derived data to a NEW file, or do the change inside the pipeline."
)

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(0)  # not our business if the payload is unreadable

tool = data.get("tool_name", "")
tool_input = data.get("tool_input") or {}

if tool in {"Edit", "MultiEdit", "Write", "NotebookEdit"}:
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()  # works for Windows and / paths
    if name in PROTECTED:
        print(MESSAGE, file=sys.stderr)
        sys.exit(2)

elif tool == "Bash":
    cmd = tool_input.get("command", "")
    names = r"(?:train|test)\.csv"
    if re.search(names, cmd, re.IGNORECASE):
        destructive = [
            rf">>?\s*[\"']?[^\s\"']*{names}",                 # redirect into the file
            r"\bsed\s+(-\w*\s+)*-i",                           # in-place edit
            r"\b(rm|del|erase|rmdir|unlink|mv|move|ren|rename|truncate|shred)\b",
            r"\b(Remove-Item|Move-Item|Rename-Item|Set-Content|Out-File|Clear-Content)\b",
        ]
        if any(re.search(p, cmd, re.IGNORECASE) for p in destructive):
            print(MESSAGE, file=sys.stderr)
            sys.exit(2)

sys.exit(0)
