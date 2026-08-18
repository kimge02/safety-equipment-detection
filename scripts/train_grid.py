"""
YOLOv11 기반 안전장비(안전모/반사조끼) 탐지 모델 - 하이퍼파라미터 그리드 학습

논문 재현: YOLOv11-n / YOLOv11-s 모델을
batch size(16/32/64), learning rate(0.001/0.01) 조합별로 학습하고
mAP, F1 score, 학습시간을 CSV로 기록한다.

실행 전 확인:
- data.yaml 경로가 맞는지 DATA_YAML 변수에서 확인
- 회사 노트북 GPU(RTX 4070) 기준, 조합 하나당 수십 분~수 시간 걸릴 수 있음
- 오래 걸리므로 nohup이나 tmux로 백그라운드 실행 권장
"""

import csv
import time
import os
from ultralytics import YOLO

DATA_YAML = os.path.expanduser(
    "~/projects/safety-equipment-detection/safety_dataset/safety-Helmet-Reflective-Jacket/data.yaml"
)
RESULT_CSV = "grid_results.csv"
EPOCHS = 100
IMGSZ = 640
PATIENCE = 20  # early stopping patience (논문엔 명시 안 됐지만, 그리드 10개 다 도니까 시간 절약용)

# 논문 표 1, 2 기준 조합
# YOLOv11-n: batch 16/32/64 x lr 0.001/0.01
# YOLOv11-s: batch 16/32 x lr 0.001/0.01 (논문에서 64는 생략)
COMBOS = []
for batch in [16, 32, 64]:
    for lr in [0.001, 0.01]:
        COMBOS.append(("yolo11n.pt", "n", batch, lr))
for batch in [16, 32]:
    for lr in [0.001, 0.01]:
        COMBOS.append(("yolo11s.pt", "s", batch, lr))


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    # CSV 헤더 준비 (파일 없으면 새로 생성)
    write_header = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "model", "batch", "lr", "epochs_trained", "train_time_sec",
                "mAP50", "mAP50_95", "precision", "recall", "f1"
            ])

    for weights, model_tag, batch, lr in COMBOS:
        run_name = f"{model_tag}_b{batch}_lr{lr}"
        print(f"\n{'='*60}\n[시작] {run_name}\n{'='*60}")

        model = YOLO(weights)
        start = time.time()

        train_results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=batch,
            lr0=lr,
            imgsz=IMGSZ,
            patience=PATIENCE,
            project="runs_grid",
            name=run_name,
            exist_ok=True,
            verbose=False,
        )

        elapsed = time.time() - start
        epochs_trained = train_results.epoch + 1 if hasattr(train_results, "epoch") else EPOCHS

        # test set으로 최종 검증
        val_results = model.val(data=DATA_YAML, split="test")
        metrics = val_results.results_dict

        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)
        f1 = compute_f1(precision, recall)

        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                model_tag, batch, lr, epochs_trained, round(elapsed, 1),
                round(map50, 4), round(map50_95, 4),
                round(precision, 4), round(recall, 4), round(f1, 4)
            ])

        print(f"[완료] {run_name} - mAP50-95: {map50_95:.4f}, F1: {f1:.4f}, 소요시간: {elapsed/60:.1f}분")

    print(f"\n전체 그리드 학습 완료. 결과는 {RESULT_CSV} 확인")


if __name__ == "__main__":
    main()
