FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Every local module the app imports, listed explicitly rather than `COPY . .`:
# there is no .dockerignore here, so a blanket copy would bake .env and
# tokens.json into the image.
#
# `auth.py` was missing until 2026-08-19 and the container had been crash-looping
# on `ModuleNotFoundError: No module named 'auth'` -- 27 restarts, exit 0, nothing
# reaching port 8890. Both main.py and chatgpt_client.py import it. Adding a module
# to the codebase does NOT add it to the image: this line is the whole manifest,
# and it has to be updated by hand whenever a new local module appears.
#
# It happened again on 2026-08-20, the same way: `capabilities.py` (the
# capability contract) and `tool_calls.py` (emulated function calling) both
# landed on main, both are imported by main.py at module scope, and neither was
# on this line -- so the container crash-looped on
# `ModuleNotFoundError: No module named 'capabilities'` and port 8890 went dark.
# If you add a .py file that main.py or chatgpt_client.py imports, add it HERE
# in the same commit.
COPY chatgpt_client.py main.py auth.py dpop.py session_web.py \
     capabilities.py tool_calls.py tool_detect.py conv_store.py ./

ENV PYTHONUNBUFFERED=1
# The anonymous conversation index. It holds the device ids that are the only
# way to reopen an anonymous conversation, so it belongs on a MOUNTED VOLUME:
# left inside the image layer it is wiped by the next deploy, which is the exact
# durability the index exists to provide.
#
# /app/data is not an arbitrary choice: it is where docker-compose.yml mounts the
# named volume `chatgpt-data`, and that mount is what production actually runs on.
#
# An earlier version of this note said the opposite -- that Coolify builds this app
# with build_pack "dockerfile", never reads docker-compose.yml, and backs /app/data
# with a separately registered volume (rs3okqn9jehjs7k6mj43haxm-data). All three are
# wrong, and believing them leads to declaring persistence twice. Checked against the
# live deployment on 2026-09-02: the app's build_pack is "dockercompose", and the
# running container's only mount is volume
# `xok2kjvrhtbrurzircv21b6d_chatgpt-data` -> /app/data.
#
# The practical consequence, and the reason this matters beyond tidiness: the compose
# file IS the production contract. A service-level key added there (a volume, an
# environment entry) reaches production; one added only in the Coolify UI does not
# reach the container unless this file also names it.
#
# There is deliberately no `VOLUME /app/data` instruction. Without a NAMED
# volume it creates an anonymous one, which survives a container restart but not
# the recreation a deploy performs -- the appearance of durability without it.
ENV CONV_DB_PATH=/app/data/conversations.db

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
