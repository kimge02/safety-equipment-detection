"""
YOLOv11 안전장비 탐지 - 조합 하나씩 실행하는 학습 스크립트

사용법:
    python3 train_one.py n 16 0.001
    python3 train_one.py n 16 0.01
    python3 train_one.py n 32 0.001
    python3 train_one.py n 32 0.01
    python3 train_one.py n 64 0.001
    python3 train_one.py n 64 0.01
    python3 train_one.py s 16 0.001
    python3 train_one.py s 16 0.01
    python3 train_one.py s 32 0.001
    python3 train_one.py s 32 0.01

- 실행할 때마다 딱 1개 조합만 학습하고 끝남
- 결과는 grid_results.csv에 한 줄씩 계속 쌓임 (이미 있으면 이어서 추가)
- 이미 끝난 조합인지 자동으로 체크해서, 중복 실행 방지
"""

import csv
import sys
import time
import os
from ultralytics import YOLO

DATA_YAML = os.path.expanduser(
    "~/projects/safety-equipment-detection/safety_dataset/safety-Helmet-Reflective-Jacket/data.yaml"
)
RESULT_CSV = "grid_results.csv"
EPOCHS = 100
IMGSZ = 640
PATIENCE = 20


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def already_done(model_tag, batch, lr):
    if not os.path.exists(RESULT_CSV):
        return False
    with open(RESULT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == model_tag and int(row["batch"]) == batch and float(row["lr"]) == lr:
                return True
    return False


def main():
    if len(sys.argv) != 4:
        print("사용법: python3 train_one.py [n|s] [batch] [lr]")
        print("예시:  python3 train_one.py n 16 0.001")
        sys.exit(1)

    model_tag = sys.argv[1]
    batch = int(sys.argv[2])
    lr = float(sys.argv[3])
    weights = f"yolo11{model_tag}.pt"

    if already_done(model_tag, batch, lr):
        print(f"이미 완료된 조합입니다: {model_tag}, batch={batch}, lr={lr}. grid_results.csv 확인.")
        sys.exit(0)

    write_header = not os.path.exists(RESULT_CSV)
    if write_header:
        with open(RESULT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "batch", "lr", "epochs_trained", "train_time_sec",
                "mAP50", "mAP50_95", "precision", "recall", "f1"
            ])

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
	workers=2,
        project="runs_grid",
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    elapsed = time.time() - start
    epochs_trained = train_results.epoch + 1 if hasattr(train_results, "epoch") else EPOCHS

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

    print(f"\n[완료] {run_name} - mAP50-95: {map50_95:.4f}, F1: {f1:.4f}, 소요시간: {elapsed/60:.1f}분")
    print(f"결과가 {RESULT_CSV}에 저장되었습니다.")


if __name__ == "__main__":
    main()
