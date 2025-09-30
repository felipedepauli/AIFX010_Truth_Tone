PY=python
PKG=dspfusion

uv-setup:
uv venv -p 3.11
. .venv/bin/activate && uv pip install -e ".[dev]"

lint:
ruff check src tests
black --check src tests

fmt:
ruff check --fix src tests
black src tests

test:
pytest -q

run-batch:
$(PY) -m dspfusion.cli --input data/processed --out metrics/report.csv --config configs/base.yaml

serve:
uvicorn dspfusion.service.api:app --host 0.0.0.0 --port 8000 --reload

dk-build:
docker compose -f docker/compose.yml build

dk-up:
docker compose -f docker/compose.yml up

dk-down:
docker compose -f docker/compose.yml down
