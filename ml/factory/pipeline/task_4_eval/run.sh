python ./predict.py \
  --audio /home/fpauli/aif/git/AIFX010-Hoax-Hertz/ml/data/raw/ADD/ADD_train_dev/train/ADD_T_00000000.wav \
  --model ../task_3_train/ml/factory/experiments/run0/clipclf.joblib \
  --suffix .feat.npz \
  --out-json artifacts/pred_one.json
