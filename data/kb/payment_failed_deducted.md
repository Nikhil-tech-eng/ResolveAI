---
title: Payment Failed but Amount Was Deducted
category: payments
reliability_score: 0.98
workflow_id: payment_deducted
---

# Payment Failed but Amount Was Deducted

1. Verify the customer and order, then read the order payment status and payment logs.
2. Confirm whether a payment attempt is marked failed and whether `amount_deducted` is true. Never infer a deduction from the customer's bank notification alone.
3. If the records confirm a deduction, use the refund action; its policy check allows a refund when an amount was actually deducted.
4. If records show no deduction or disagree with the customer-provided evidence, explain the recorded facts and escalate for payment review.
5. Confirm the action result without inventing a bank posting or settlement time.