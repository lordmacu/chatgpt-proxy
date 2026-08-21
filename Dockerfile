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
     capabilities.py tool_calls.py tool_detect.py ./

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
