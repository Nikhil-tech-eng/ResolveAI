---
title: Account Locked
category: accounts
reliability_score: 0.96
workflow_id: account_locked
---

# Account Locked

1. Verify the customer through the approved account-verification process.
2. Read the account status and failed-login-attempt count. A locked status is the source of truth; do not infer it from the customer's description.
3. If the account is locked and identity verification succeeds, use the unlock action.
4. Confirm that the account status is active and the failed-attempt count is cleared before telling the customer access is restored.
5. If the account is suspended, already active, or cannot be verified, do not force an unlock; explain the result and escalate when appropriate.