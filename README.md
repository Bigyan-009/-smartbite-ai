# SmartBite AI 🍽️

**Design and Development of an Agent-Based Restaurant Food Recommendation Chatbot Using LangGraph, MongoDB, ChromaDB, and Hybrid Recommendation Techniques**

SmartBite AI is an AI-powered restaurant food recommendation and ordering chatbot,
built as my final-year project. Users chat in natural language ("Recommend spicy
non-veg food under Rs. 500"), and a LangGraph agent workflow understands the request,
retrieves the user's long-term memory and semantically similar foods, filters and
ranks candidates with a hybrid scoring algorithm, and replies with clear
recommendations and reasons. Orders can be placed and tracked straight from the chat.

**Highlights**

- LangGraph multi-node agent with intent routing (chat / recommend / order / order status / cancel)
- Hybrid recommendation ranker: vector similarity + structured filters + user memory + nutrition goals + the user's own ratings
- Two databases, each used for what it does best: MongoDB (source of truth, validators, TTL indexes, aggregation pipelines) and ChromaDB (semantic search, per-user memories)
- Context-aware: live weather, season and time of day shape the recommendations
- Runs fully offline with rule-based fallbacks when no LLM key is configured
- Secure auth: bcrypt, hashed single-use reset codes, server-side sessions
- 15 pytest modules; integration tests skip themselves when MongoDB is absent

---

## Features

### User
- Register / login / logout (bcrypt-hashed passwords, role-based access)
- **Forgot password** — a six-digit code is emailed over Gmail SMTP; codes are
  stored hashed, expire in 15 minutes, are single-use, and are discarded after
  5 wrong attempts. Changing the password logs out every existing session.
- Change your password from the Profile page
- **Persistent login & chat:** refreshing the page keeps you logged in and
  restores your conversation where you left off (session token + saved history)
- Chat with the AI food recommendation bot — it also handles greetings, questions
  and small talk naturally (not just food dumps)
- **Conversation memory:** the bot remembers the current chat, so follow-ups like
  "make it cheaper" or "order the first one" work
- **Weather / season / time-aware:** recommendations adapt to the moment — warm,
  spicy soups on a cold rainy evening; light, cool dishes when it's hot; breakfast
  items in the morning
- Personalised recommendations with human-readable reasons and scores
- **Food photos on every card** — 32 dish photos are bundled in `assets/foods/`,
  so the menu, chat recommendations and favourites all show real pictures with
  no internet connection needed (sources listed in `assets/foods/CREDITS.md`)
- **Nutrition-aware recommendations:** ask for "something light", "not too oily",
  "high protein for the gym" or "something filling" and the ranker uses each
  dish's real calorie and protein figures (shown on every food card)
- **Recommendations learn from your ratings:** dishes you rate 1-2 ⭐ stop being
  suggested, dishes you rate 4-5 ⭐ come back more often
- **Place orders** — by chat ("order 2 Chicken Momo", "I'll take the first one")
  or with the 🛒 Order button, and review them on the *My Orders* page
- Save favourites and rate foods (1–5 ⭐)
- View and edit profile preferences
- The bot remembers durable preferences ("I don't like mushroom. Remember that.")

### Admin
- Dashboard with metrics and charts
- Add / edit / deactivate / delete foods (including a photo upload, calories
  and protein per dish)
- Automatic ChromaDB indexing on every food change + manual reindex
- Recommendation logs and chat-history summary
- Analytics & result-analysis page (success rate, average score, top foods)

### AI Chatbot (LangGraph)
- **Intent routing** — a router node classifies each turn as *chat*, *recommend*,
  or *order* and takes the right branch (so greetings don't trigger food ranking)
- **Ambient-context agent** — a `load_context` node adds live weather (free,
  keyless Open-Meteo API), season, and time of day, which bias semantic search,
  the ranking score, and the bot's wording
- Natural-language understanding → structured preferences
- Conversation history is passed to the agents for context-aware follow-ups
- ChromaDB semantic search + per-user memory retrieval
- MongoDB structured filtering with progressive relaxation
- Hybrid ranking algorithm (vector similarity + structured signals + memory +
  nutrition goals + the user's own star ratings)
- Order-taking agent that resolves "the first one" / quantities and saves the order
- Long-term preference memory stored in ChromaDB

> **Works offline too:** if no LLM API key is configured, the system falls back to
> rule-based understanding and the hybrid ranking engine, so the demo always runs.

---

## Project structure

```text
app.py              Streamlit entry point and page routing
agents/             LangGraph workflow: state, nodes, prompts, graph wiring
recommendation/     Hybrid ranking algorithm
services/           Business logic: food, order, cart, chat, memory, context, email, sessions
database/           MongoDB client, schema/migrations/indexes, ChromaDB client, seed data
auth/               Login, registration, session restore
ui/                 Streamlit pages (user and admin) and shared components
utils/              Validators, date and id helpers
config/             Settings loaded from .env
assets/foods/       Bundled dish photos (credits in CREDITS.md)
tests/              pytest suite
```

---

## Tech Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| Language         | Python 3.11+                        |
| UI               | Streamlit                           |
| Agent workflow   | LangGraph                           |
| LLM              | Gemini (default) or OpenAI          |
| Main database    | MongoDB                             |
| Vector database  | ChromaDB (persistent, local)        |
| Passwords        | bcrypt                              |
| Charts           | Plotly                              |
| Testing          | pytest                              |

---

## Architecture

```text
Streamlit UI
├── Login / Register
├── User: Chatbot · My Orders · Favourites · Profile
└── Admin: Dashboard · Manage Foods · Logs · Analytics
        ↓
Python Application Layer
├── Authentication Service      (auth/)
├── MongoDB Service             (database/mongo.py)
├── ChromaDB Service            (database/chroma.py)
├── Food / User / Analytics / Memory services  (services/)
├── Hybrid Recommendation Ranker (recommendation/)
└── LangGraph Agent Workflow    (agents/)
        ↓
Data Layer
├── MongoDB: users, foods, chat_history, recommendation_logs, ratings, favourites, orders, sessions
└── ChromaDB: food_items, user_memories
```

### LangGraph workflow

```text
START → load_user_profile → load_context → retrieve_user_memory → waiter_agent ─┐
   (load_context = live weather + season + time of day, Nepal time)             │
   ┌────────────────────────────────────────────────────────────────────────────┤
   │ intent = order:         place_order_node   (writes the order to MongoDB)   │
   │ intent = order_status:  show_orders_node   (status / tracking / history)   │
   │ intent = cancel_order:  cancel_order_node  (cancels via chat)              │
   │ otherwise:              present_foods_node (hybrid-ranks recommendations)  │
   └────────────────────────────────────────────────────────────────────────────┘
                     → save_chat_history → update_user_memory_agent → END
```

The `waiter_agent` is the single reasoning brain: it receives the full menu, the
user's profile, memories, recent orders, and live context, then decides the intent,
writes the reply, and picks dishes. Deterministic Python handles everything
involving data and money. Without an LLM API key, rule-based fallbacks keep the
whole flow working offline.

---

## Database design

Two databases, each doing what it is good at. **MongoDB** is the source of
truth for structured data; **ChromaDB** is a searchable copy of the food
descriptions plus per-user memories, kept in sync on every write.

### Schema management (`database/schema.py`)

The shape of the data is owned by one module, applied automatically the first
time the database is opened, and safe to run again:

| Job | What it does |
| --- | --- |
| **Migrations** | Numbered, forward-only repairs. The applied version is stored in the `meta` collection, so each runs exactly once. |
| **Indexes** | Every index exists because a real query needs it. |
| **TTL indexes** | MongoDB deletes expired login sessions and password-reset codes by itself. |
| **Validators** | JSON Schema rules held *in the database*, so bad data is refused even if something writes to MongoDB directly. |

Migrations applied so far:

1. **Normalise legacy food documents** — some dishes had been inserted straight
   into MongoDB with a numeric `spice_level`, an old `is_veg` flag instead of
   `diet_type`, and no list fields. This repairs them.
2. **Enforce one rating per user per dish** — removes duplicate ratings, then
   replaces the old non-unique index with a unique one.

### Validation happens twice

`utils/validators.py` checks input in the application, *and* MongoDB enforces
its own rules underneath. The database will refuse:

- a negative price, or a dish with no name
- a `diet_type` outside `veg / non_veg / vegan / egg`
- a numeric `spice_level` (it must be one of the four named levels)
- a rating outside 1–5, or a second rating by the same user for the same dish
- a `role` other than `user` / `admin`
- a `password_hash` short enough to be a plain-text password

### Analytics use aggregation pipelines

Dashboard and result-analysis figures are computed **inside MongoDB** with
`$group`, `$avg` and `$unwind`, so only the finished numbers travel back to
Python. Plain `find().sort().limit()` is still used where nothing is being
aggregated — "the five highest-rated dishes" is a sort, not a calculation.

### Indexes

| Collection | Index | Why |
| --- | --- | --- |
| `users` | `email` (unique) | Login lookup; no two accounts may share an email |
| `foods` | `name`, `cuisine`, `available + price` | Menu browsing and filtering |
| `chat_history` | `user_id + session_id + created_at` | Replaying one conversation in order |
| `recommendation_logs` | `created_at`, `user_id + created_at` | Admin log page and result analysis |
| `ratings` | `user_id + food_id` (unique), `food_id` | One rating per dish per user; recomputing averages |
| `favourites` | `user_id + food_id` (unique) | No duplicate favourites |
| `orders` | `user_id + created_at`, `status + created_at` | "My orders" and the admin queue |
| `sessions` | `token` (unique), `expires_at` (**TTL**) | Session lookup; expired rows self-delete |
| `password_resets` | `email` (unique), `expires_at` (**TTL**) | One pending reset per address; codes self-delete |

### The vector database (ChromaDB)

- Two collections: `food_items` and `user_memories`, both using **cosine**
  distance, which is the right measure for sentence embeddings.
- Embeddings are produced locally (all-MiniLM-L6-v2) — no API key needed.
- Memory searches are **always** filtered by `user_id`, so one user can never
  see another's memories.
- Food searches filter on `available` inside ChromaDB, so a dish the kitchen
  switched off never takes up a candidate slot.
- A rebuild embeds every dish in **one batch** rather than one call per dish.
- *Manage Foods* shows a live "MongoDB vs ChromaDB" count so you can see at a
  glance whether the index is in sync.

---

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment setup

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` if needed. Key variables:

- `MONGODB_URI` — default `mongodb://localhost:27017`
- `LLM_PROVIDER` — `gemini` (default) or `openai`
- `GEMINI_API_KEY` / `OPENAI_API_KEY` — optional; leave the placeholder to run offline
- `CHROMA_PATH` — default `./chroma_db`
- `ADMIN_EMAIL` / `ADMIN_PASSWORD` — seeded admin account
- `APP_CITY` / `APP_LAT` / `APP_LON` — location for weather (default Kathmandu);
  `WEATHER_ENABLED=false` disables the weather lookup (season + time still apply)
- `SESSION_TTL_DAYS` — how long a browser stays logged in across refreshes

No API keys are hardcoded anywhere in the source.

### Run MongoDB

Local install: start the `mongod` service (default `mongodb://localhost:27017`).
Or use MongoDB Atlas and put the connection string in `MONGODB_URI`.

### Seed demo data

```bash
python database/seed_data.py
```

This creates the admin user and 25 demo foods, and indexes them in ChromaDB.

### Run the app

```bash
streamlit run app.py
```

---

## Email setup (forgot-password codes)

Reset codes are sent with Gmail SMTP. Gmail refuses normal account passwords,
so you need an **App Password**:

1. Turn on 2-Step Verification: <https://myaccount.google.com/security>
2. Create an App Password: <https://myaccount.google.com/apppasswords>
3. Put it in `.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your.address@gmail.com
SMTP_PASSWORD=abcdefghijklmnop      # the 16-character App Password
SMTP_FROM_NAME=SmartBite AI
```

4. Check it works:

```bash
python -m services.email_service your.address@gmail.com
```

**If you leave `SMTP_USER` / `SMTP_PASSWORD` blank** and `APP_ENV=development`,
the reset code is shown on screen instead of emailed, so the demo still runs
with no email account at all.

---

## Login details

- **Admin:** `admin@smartbite.ai` / `admin12345` (from `.env`)
- **User:** register a new account from the UI.

---

## Demo flow

1. Register a user and log in.
2. Say: *"Hi!"* → the bot chats back (it doesn't force a recommendation).
3. Ask: *"What should I eat right now?"* → notice it adapts to the weather/time
   shown in the banner (e.g. warm spicy dishes on a cold rainy evening).
4. Ask: *"Recommend spicy non-veg food under Rs. 500."*
5. Review recommendations, reasons and scores.
6. Say: *"Order the first one"* → order is placed (conversation memory in action).
7. Open **My Orders**, view the total, and **cancel** an order.
8. **Refresh the page** → you stay logged in and the chat is restored.
9. Save a favourite and rate a recommendation.
10. Say: *"I do not like mushroom. Remember that."*
11. Ask: *"Recommend something for dinner."* → memory affects results.
12. Log out, log in as admin.
13. Add a food → stored in MongoDB and indexed in ChromaDB.
14. Explore the dashboard, logs, and analytics.

---

## Deployment

The app runs anywhere Python 3.11+ runs. Two things must exist wherever you
deploy it: a **MongoDB** the app can reach, and a **writable folder** for the
ChromaDB index (`CHROMA_PATH`).

### 1. Prepare configuration

Never deploy your `.env` file. Set the same keys as environment variables on
the host instead (`.env.example` lists every one of them). The values that must
be set for a real deployment:

| Variable | Why |
| --- | --- |
| `MONGODB_URI` | A MongoDB Atlas connection string, not `localhost` |
| `OPENAI_API_KEY` or `GEMINI_API_KEY` | The chatbot's LLM (the app still runs without one, in rule-based offline mode) |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD` | **Change these** before going live |
| `SMTP_USER`, `SMTP_PASSWORD` | Gmail App Password, for forgot-password emails |
| `APP_ENV` | Set to `production` so reset codes are never shown on screen |

### 2. Seed the database once

```bash
python -m database.seed_data
```

This creates the admin account, inserts the demo menu, and builds the ChromaDB
index. It is safe to run again; it refreshes rather than duplicating.

### 3. Run it

```bash
streamlit run app.py --server.port 8501
```

### Streamlit Community Cloud (free, easiest)

1. Push this repository to GitHub. `.gitignore` already keeps `.env` out.
2. On <https://share.streamlit.io>, point a new app at `app.py`.
3. Paste your configuration into **Settings → Secrets** in `.env` format.
4. Use **MongoDB Atlas** for the database and allow access from anywhere
   (`0.0.0.0/0`) so the Streamlit host can connect.
5. Deploy, then run the seed script once from the app's terminal.

### Deployment checklist

- [ ] `ADMIN_PASSWORD` changed from the default
- [ ] `APP_ENV=production`
- [ ] `.env` is not in the repository (`git status` should never list it)
- [ ] `MONGODB_URI` points at Atlas, not localhost
- [ ] Seed script has been run once
- [ ] `pytest` passes

> **Known limitation:** the login session token is kept in the page URL
> (`?sid=`), so treat a copied link as a credential. Moving it to an
> `HttpOnly` cookie is listed under *Future improvements*.

---

## Testing

```bash
pytest
```

- Unit tests (auth, ranker, schema, fallbacks) run anywhere with no database.
- Integration tests (services, graph, orders, sessions) need MongoDB and are
  **skipped automatically** when it is unavailable.

---

## Known limitations

**Persistent login uses a URL token.** Staying logged in across a page refresh
is implemented by putting a session token in the query string
(`?sid=<token>`), which `restore_session()` reads back on the next page load.
Streamlit has no way to set a cookie without a third-party component, so the
URL is the only place a plain Streamlit app can keep it.

Two consequences:

1. *The URL is part of browser history.* Pressing Back lands on the pre-login
   URL, which carries no token, so refreshing there shows the login page even
   though the session is still valid. `auth_service.sync_session_url()` writes
   the token back on the next rerun to keep this working.
2. *The token is only as private as the address bar.* It is visible in browser
   history and in any copied link, so anyone with that URL can resume the
   session until it expires (`SESSION_TTL_DAYS`, default 7 days).

The proper fix is to move the token into an `HttpOnly` cookie — see *Future
improvements*.

---

## Future improvements

Mobile app, real restaurant integration, online payment, delivery tracking,
voice chatbot, multi-language (Nepali–English), collaborative filtering, and
cloud deployment.

Security-wise, the clearest next step is moving the session token out of the
URL and into an `HttpOnly`, `Secure` cookie (via a small Streamlit component or
a reverse proxy), which removes both limitations noted above.

---

## License

MIT. See [LICENSE](LICENSE). Food photo sources and licences are listed in
`assets/foods/CREDITS.md`.
