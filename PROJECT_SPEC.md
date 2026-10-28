# PROJECT_SPEC.md: PS-04 Agentic Customer Support System

> AI coding assistant: READ THIS FILE FULLY before every task. Follow it exactly. If a request conflicts with this spec, ask before deviating.

## 1. Goal

Build a multi-agent AI customer support platform for an **e-commerce** company that:
- understands customer issues (intent, priority, sentiment)
- fetches customer context (account, orders, past tickets, subscription)
- retrieves knowledge and selects the most reliable support workflow
- troubleshoots step by step, like a support engineer
- takes real actions (return, refund, cancel, reset password), not just answers
- maintains multi-turn memory and tracks the issue lifecycle
- decides intelligently when to auto-resolve vs escalate
- generates structured tickets with full context, so humans never reread the chat
- tracks resolution status and collects CSAT

**Success = most queries resolved with zero human involvement, with a visible, explainable agent trace.**

## 2. Tech Stack (fixed, do not change without asking)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph (state graph) |
| LLM | Google geminin, JSON-structured outputs. Wrap in `llm/client.py` so provider is swappable |
| Validation | Pydantic v2 for ALL LLM inputs/outputs |
| DB | SQLite via SQLAlchemy |
| Vector store | ChromaDB (persistent, local) + sentence-transformers `all-MiniLM-L6-v2` |
| UI | Streamlit (chat page + supervisor dashboard page) |
| Tests | pytest |
| Config | `.env` (Gemini_API_KEY, DEMO_MODE) |

## 3. Folder Structure

```
support-agent/
├── PROJECT_SPEC.md
├── README.md
├── .env.example
├── requirements.txt
├── data/
│   ├── seed_data.py
│   ├── support.db            (generated)
│   └── kb/                   (markdown knowledge articles)
├── db/
│   ├── models.py
│   └── session.py
├── tools/
│   ├── read_tools.py
│   ├── action_tools.py
│   └── policy.py
├── kb/
│   ├── ingest.py
│   └── search.py
├── llm/
│   └── client.py
├── agents/
│   ├── schemas.py
│   ├── intent_agent.py
│   ├── context_agent.py
│   ├── knowledge_agent.py
│   ├── troubleshooting_agent.py
│   ├── action_agent.py
│   ├── memory_agent.py
│   ├── escalation_agent.py
│   ├── ticket_agent.py
│   ├── tracking_agent.py
│   ├── csat_agent.py
│   └── orchestrator.py
├── api/
│   └── main.py
├── ui/
│   ├── chat_app.py
│   └── pages/dashboard.py
├── evals/
│   ├── test_conversations.json
│   ├── run_eval.py
│   └── baseline_rag.py
└── tests/
```

## 4. Database Schema (SQLite)

- **customers**: id, name, email, phone, tier (`standard`|`premium`), account_status (`active`|`locked`|`suspended`), failed_login_attempts, subscription_status, created_at
- **orders**: id (e.g. `ORD12345`), customer_id, status (`placed`|`shipped`|`delivered`|`cancelled`|`returned`), total, payment_status (`paid`|`failed`|`refunded`|`pending`), placed_at, delivered_at
- **order_items**: id, order_id, sku, name, qty, price
- **shipments**: id, order_id, carrier, tracking_no, status, delivered_item_sku (what was actually delivered), delivered_at
- **payments**: id, order_id, amount, status, gateway_ref, amount_deducted (bool), created_at
- **tickets**: id, customer_id, order_id, intent, category, priority, sentiment, summary, investigation, actions_taken (JSON), status (`opened`|`assigned`|`pending_customer`|`resolved`|`closed`), assigned_to, escalation_reason, sla_deadline, created_at, updated_at
- **messages**: id, conversation_id, role (`customer`|`agent`|`system`), content, channel, timestamp
- **conversations**: id, customer_id, channel, state (JSON), status, started_at
- **action_log**: id, conversation_id, tool_name, args (JSON), result (JSON), success, timestamp
- **csat**: id, ticket_id, conversation_id, rating (1-5), sentiment, resolution_quality, comment, created_at

## 5. Tools (deterministic Python, NO LLM inside)

**Read tools:** `get_customer`, `get_order`, `get_order_items`, `get_shipment`, `get_payment_logs`, `get_ticket_history`, `get_subscription`.

**Action tools:** `create_return_request`, `generate_return_label`, `initiate_refund`, `cancel_order`, `reset_password`, `unlock_account`, `schedule_replacement`, `send_notification`.

Rules:
- Every action tool writes a row to `action_log`.
- Every action tool calls `tools/policy.py` first (e.g. returns allowed only within 30 days of delivery, refunds only for paid orders, cancel only before shipped).
- Tools return `{success: bool, data: {...}, error: str|None}`.
- **The LLM must never invent order/customer data. All facts come from tools.**

## 6. Agents (each = one file, one function, Pydantic in/out)

| # | Agent | Input | Output |
|---|---|---|---|
| 1 | Intent | conversation history | `IntentResult{intent, category, priority, sentiment, extracted_entities}` |
| 2 | Context | customer_id | `CustomerContext{profile, tier, recent_orders, past_tickets_count, subscription, summary}` |
| 3 | Knowledge | intent + context | `KBResult{articles[], selected_workflow, confidence}`; picks the best **workflow**, not just a paragraph |
| 4 | Troubleshooting | intent + entities + workflow | `TroubleshootResult{steps_run[{step, finding}], diagnosis, proposed_action}` |
| 5 | Action | proposed_action | `ActionResult{actions_executed[], success, customer_message}` |
| 6 | Memory | messages + state | `ConversationState{entities, missing_fields, lifecycle_stage, turn_count}` |
| 7 | Escalation | everything so far | `EscalationDecision{decision: auto_resolve|escalate|ask_more_info, reason, confidence}` |
| 8 | Ticket | full state | `Ticket{...}`, complete, human-ready |
| 9 | Tracking | ticket events | status transitions + customer notification |
| 10 | CSAT | resolved conversation | `CSATResult{rating, sentiment, resolution_quality}` |
| 11 | Orchestrator | user message | LangGraph graph coordinating all above |

### Intent list (minimum)
`wrong_item`, `damaged_item`, `missing_item`, `order_status`, `payment_failure`, `refund_request`, `cancel_order`, `login_issue`, `subscription_issue`, `fraud_complaint`, `legal_request`, `data_breach`, `general_query`.

### Escalation rules (hard-coded + LLM confidence)
Escalate when ANY is true:
- intent in {`fraud_complaint`, `legal_request`, `data_breach`}
- confidence < 0.70
- sentiment strongly negative AND issue unresolved after 2 attempts
- requested action blocked by policy and customer disputes it
- SLA at risk (premium: 15 min, standard: 60 min)
- 3+ failed troubleshooting steps

Otherwise: auto-resolve.

### Ticket format
```
Ticket ID, Customer (name, tier), Order ID
Issue (1-2 lines)
Investigation (what was checked and found)
Actions Taken (list from action_log)
Status, Priority, Assigned To
Escalation Reason
Conversation Summary
Suggested Next Step for human agent
```

### Ticket state machine
`opened -> assigned -> pending_customer -> resolved -> closed` (also `pending_customer -> assigned` and `resolved -> opened` on reopen). Every transition triggers `send_notification`.

## 7. Orchestrator Flow (LangGraph)

```
START -> memory_load -> intent -> context -> [missing info? -> ask_customer -> END(turn)]
      -> knowledge -> troubleshoot -> escalation_check
            -> auto_resolve: action -> verify -> reply -> csat_prompt -> END
            -> escalate: ticket -> tracking -> reply(handoff message) -> END
            -> ask_more_info: ask_customer -> END(turn)
```

- Shared `AgentState` TypedDict: messages, conversation_id, customer_id, intent, context, kb, troubleshoot, actions, escalation, ticket, trace[], confidence.
- Every node appends `{agent, input_summary, output_summary, tools_called, duration_ms, confidence}` to `trace`.
- State persists per conversation_id in the DB (multi-turn).
- Sentiment-aware: if sentiment is negative, empathetic tone is prioritized and escalation threshold is stricter.

## 8. API Endpoints

- `POST /chat` {conversation_id?, customer_id, message, channel} -> {reply, trace, state, ticket?}
- `GET /tickets`, `GET /tickets/{id}`, `PATCH /tickets/{id}` (status change)
- `POST /csat`
- `GET /metrics` -> open tickets, resolution rate, avg resolution time, escalation rate, avg CSAT, intent distribution
- `GET /alerts` -> root-cause spike alerts

## 9. UI

- **Chat page:** customer selector (dropdown of seed customers), chat window, right panel "Agent Trace" (live agent steps, tools called, confidence).
- **Dashboard page:** KPIs above, ticket table with status filter, charts (intents, escalation rate over time), root-cause alerts, SLA-at-risk list.

## 10. Advanced Features (build after core works)

1. Supervisor dashboard
2. Sentiment-aware tone
3. SLA monitoring (background check, auto-escalate)
4. Root-cause discovery (spike in same intent within time window)
5. Auto KB expansion (draft FAQ from resolved ticket)
6. Omnichannel (Telegram bot reusing `/chat`)

## 11. Coding Rules

- Type hints everywhere. Docstring per function.
- All LLM outputs validated with Pydantic; on validation failure retry once, then fall back safely (escalate).
- LLM prompts live in constants at the top of each agent file.
- `DEMO_MODE=true` returns cached/deterministic responses for scripted demo scenarios if the API fails.
- No secrets in code. Use `.env`.
- Each agent has unit tests in `/tests` (min. 5 cases).
- Keep functions small. Log with the `logging` module.
- After each phase: run tests, update README, confirm what works.

## 12. Seed Data Requirements (must include these scenarios)

| Customer | Scenario |
|---|---|
| C001 (premium, 3 past tickets) | Order ORD10001 delivered, wrong item (shipment.delivered_item_sku != order_items.sku) |
| C002 | Payment failed but amount_deducted = true (ORD10002) |
| C003 | Account locked after 5 failed attempts |
| C004 (premium) | Subscription paid but not activated (sync failed) |
| C005 | Order ORD10005 shipped, wants to cancel (policy: blocked) |
| C006 | Order ORD10006 placed, wants to cancel (allowed) |
| C007 | Delivered 45 days ago, wants return (policy: outside window -> escalate) |
| C008 | Fraud complaint: unknown charges |
| C009 | Order delayed, angry customer |
| C010 | Simple order status query |
Plus 10+ filler customers and ~50 orders total. Also seed 30+ extra "payment_failure" conversations clustered in one hour to enable root-cause detection.

## 13. Definition of Done

- All 3 demo scenarios work end to end (auto-resolve, multi-turn, escalation).
- Trace visible in UI. Dashboard shows live metrics.
- Eval script reports metrics for agentic vs. baseline RAG.
- README has setup and run instructions.
