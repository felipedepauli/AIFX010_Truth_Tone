# Audio Splicing (DSP/ML Clássico)

Framework leve, explicável e reprodutível para detecção/localização de *splicing* de voz.
- Modos: (1) avaliação em lote (EER, F1@±100ms) • (2) microsserviço web com frontend simples
- Tech: Python, MLflow, Docker, `uv`, FastAPI, scikit-learn (sem deep learning)

## Estrutura
- `src/hoaxhertz/`: código-fonte (pré-processamento, features, detectores, fusão, serviço web)
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




Pré-processamento:
  - LoadAndStandardize
Features:
  - MFCC (librosa): prontos para framewise e vetor de clipe
  - Ambient/AES   : (log-mel + estatísticas robustas). Pronto para assinatura de ambiente /AEI (Assinatura acústica do local)
  - ENF           : (zc ou stft). Pront para encontrar a frequência da rede elétrica

Detectores:
  - Splicing por Sliding-Window + Ruptures/PELT (modelo = rbf)
  - Alinhamento ENF contra log externo
  - VAD (webrtcvad)
  - SAD (GMM)
  - CUSUM
  - Detecção de eventos acústicos (AED): baseado em CNN (Keras/TensorFlow)
  - Detecção de eventos acústicos (AED): baseado em CRNN (Keras/TensorFlow)


Datasets baixados:
  - MUSAN (ruído, fala, música)
  - FSD50K (eventos acústicos)
  - AudioSet (eventos acústicos)
  - LibriSpeech (fala)
  - CommonVoice (fala)
  - UrbanSound8K (eventos acústicos urbanos)
  - ESC-50 (eventos acústicos ambientais)
  - TIMIT (fala)
  - VCTK (fala)
  - DEMAND (ruído ambiental)
  - QUT-NOISE (ruído ambiental)
  - TUT Acoustic Scenes 2017 (assinatura acústica do local)
  - TUT Acoustic Scenes 2016 (assinatura acústica do local)
  - TUT Sound Events 2017 (eventos acústicos)
  - TUT Sound Events 2016 (eventos acústicos)   

O que falta p/ a entrega final

  1. Criar splits/CSV de treino/val para o classificador de clipes.
  2. Treinar (LogReg/SVM) com MFCC (+Ambient opcional).
  3. Validar (relatório/accuracy) e rodar inferência no predict_clip_classifier.py.
  4. Rodar splicing em material com cortes sintéticos e medir (precision/recall de fronteiras).
  5. (Opcional) Aprimorar ENF: expor --stft-nperseg/--stft-noverlap no enf_align.py e oferecer harmonic folding quando o áudio tiver 120 Hz dominando.