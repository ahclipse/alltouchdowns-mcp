# Two stages: build the virtual environment with uv, then copy only that into a clean image.
#
# Base images are pinned by digest, so a rebuild cannot silently pick up a different image
# under the same tag. Dependabot proposes the new digest when a tag moves.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 /uv /bin/uv
WORKDIR /app
# Use the image's own Python. A downloaded one would live outside the folder that is copied
# into the final image, and the virtual environment would point at nothing.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Dependencies first, so this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
# A fixed, unprivileged user. /data is where the database volume is mounted.
RUN useradd --uid 10001 --no-create-home app && mkdir /data && chown 10001:10001 /data
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ATD_DB_PATH=/data/mcp.db \
    ATD_HOST=0.0.0.0 \
    ATD_PORT=8000
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "alltouchdowns_mcp"]
