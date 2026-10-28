---
title: Password Reset
category: accounts
reliability_score: 0.95
workflow_id: password_reset
---

# Password Reset

1. Verify the customer using the approved account-verification process.
2. Confirm the account exists and is not suspended; a locked account should follow the account-locked workflow.
3. Use the password-reset action to initiate the reset process.
4. Tell the customer only that the reset action was initiated if the tool confirms it. Never request, generate, or disclose their password or reset token in chat.
5. If the account is missing, suspended, or the action fails, explain the safe next step and escalate when needed.