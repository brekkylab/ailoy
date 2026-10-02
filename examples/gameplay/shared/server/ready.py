"""Open the agent's company, hand it to the bridge, and pause the game.

The company is an AI's, since an AI is what opens a company on a dedicated server; on a
saved game it is there already. The bridge acts for it from then on.
"""

import os
import re
import sys
import time

sys.path.insert(0, os.environ.get("OPENTTD_SKILL", "/skills/openttd"))
from admin import Admin, AdminError  # noqa: E402
from ttd import call  # noqa: E402


def ai_company(admin):
    for line in admin.rcon("companies").splitlines():
        m = re.match(r"#:(\d+)\(.*\bAI$", line.strip())
        if m:
            return int(m.group(1)) - 1
    return None


with Admin(timeout=120) as admin:
    company = ai_company(admin)
    if company is None:
        print(admin.rcon("start_ai AiloyCompany"))
        for _ in range(50):
            company = ai_company(admin)
            if company is not None:
                break
            time.sleep(0.2)
        else:
            raise AdminError("the company did not open")
    # The bridge may still be starting: ask until it answers.
    for attempt in range(10):
        try:
            print(call(admin, "setup", {"company": company}, timeout=15))
            break
        except AdminError as e:
            print(f"setup: {e}")
            if attempt == 9:
                raise
    admin.rcon("pause")
