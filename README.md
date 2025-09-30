# Audio Splicing (DSP/ML Clássico)

Framework leve, explicável e reprodutível para detecção/localização de *splicing* de voz.
- Modos: (1) avaliação em lote (EER, F1@±100ms) • (2) microsserviço web com frontend simples
- Tech: Python, MLflow, Docker, `uv`, FastAPI, scikit-learn (sem deep learning)

## Estrutura
- `src/dspfusion/`: código-fonte (pré-processamento, features, detectores, fusão, serviço web)
- `configs/`: YAMLs de configuração
- `scripts/`: CLIs utilitários (gerar forgeries, preparar dados, avaliar)
- `deploy/` e `docker/`: Dockerfiles e Compose
- `models/`: artefatos exportados (registrados no MLflow)
- `mlruns/`: tracking local (não versionado)
- `frontend/`: página estática para visualização dos resultados do serviço
- `tests/`: testes unitários
- `data/`: dados (não versionado)

## Uso rápido
Ver `MAKE.md` ou `Makefile` para comandos comuns.
