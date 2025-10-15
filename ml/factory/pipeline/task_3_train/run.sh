python ./train_clip_classifier.py \
  --csv ../../../../ml/factory/pipeline/task_0_data/data_processed/train_clips.csv \
  --suffix .feat.npz \
  --clf svm \
  --out-model ml/factory/experiments/run0/clipclf.joblib \
  --out-report ml/factory/experiments/run0/clipclf_report.json
