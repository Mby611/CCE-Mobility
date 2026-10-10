# Mobility Recovery Skill

Use the latest observable task state as the source of truth.

- If a previously selected resource, route, or tool result is no longer usable, do not keep recommending it.
- Verify replacement resources against the current observation or tool result before presenting them as available.
- If a tool result is empty or incomplete, explain the uncertainty and ask a targeted clarification or suggest a safe next check.
- Keep unaffected user constraints stable while adapting only the parts touched by the latest environment update.
- Separate confirmed alternatives from unverified possibilities in the final response.
