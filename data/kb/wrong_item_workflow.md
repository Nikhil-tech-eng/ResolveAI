---
title: Wrong Item Workflow
category: fulfillment
reliability_score: 0.97
workflow_id: wrong_item
---

# Wrong Item Workflow

1. Confirm the customer and order, then retrieve the ordered item SKUs and shipment details.
2. Compare the delivered-item SKU with the SKU on the order. Treat a mismatch as verified evidence; do not infer a mismatch from the customer's description alone.
3. If the records match or shipment evidence is missing, ask for the needed details or escalate for review rather than scheduling a replacement.
4. For a verified wrong item, offer the available return or replacement path. Check return eligibility before creating a return request.
5. Schedule a replacement only after verification, and communicate the action result and any remaining steps.