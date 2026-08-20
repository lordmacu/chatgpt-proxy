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
COPY chatgpt_client.py main.py auth.py dpop.py session_web.py ./

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
