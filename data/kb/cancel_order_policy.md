---
title: Cancel Order Policy
category: orders
reliability_score: 0.98
workflow_id: cancel_order
---

# Cancel Order Policy

1. Verify the customer and retrieve the requested order.
2. Check the order status. Cancellation is allowed only while the order status is `placed`.
3. If the order is shipped, delivered, cancelled, or returned, do not cancel it. Explain the current recorded status and available next step.
4. If it is still placed, use the cancellation action and confirm only the status returned by the tool.
5. Escalate if the customer disputes a blocked cancellation or the order status is missing or inconsistent.