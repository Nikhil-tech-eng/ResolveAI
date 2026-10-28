---
title: Order Tracking
category: shipping
reliability_score: 0.96
workflow_id: order_status
---

# Order Tracking

1. Confirm the customer and retrieve the requested order from the database.
2. Read the shipment carrier, tracking number, shipment status, and latest recorded delivery details.
3. Share the recorded status and tracking reference. Do not invent scans, locations, or estimated delivery dates.
4. If there is no shipment record, explain that tracking is not available in the support records and check the order status.
5. For a delayed or disputed shipment, follow the shipping-delays workflow and escalate when the available records cannot resolve the issue.