FROM agrigorev/zoomcamp-model:2025

LABEL authors="AFMac"

RUN pip install uv

WORKDIR /code

COPY "uv.lock" "pyproject.toml" ./

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
RUN uv sync --locked --no-dev

RUN pip install uvicorn fastapi

COPY "webserve.py" ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "webserve:app", "--host", "0.0.0.0", "--port", "9696"]