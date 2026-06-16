# GPU 小规模验证

## 测试目的
验证 GPU 下训练 Pipeline 正常工作（CUDA 张量、显存、吞吐量）。

## 测试条件
```
python3 train.py --num_labels 40 --device cuda --total_steps 50 \
    --log_interval 10 --eval_interval 50 --batch_size 16 \
    --uratio 3 --num_workers 2
```

## 原始输出
```
[Config] num_labels=40, batch_size=16, uratio=3, lr=0.03, total_steps=50
[Config] confidence_threshold=0.95, ema_decay=0.999
[Train] Using device: cuda
[Data] num_labels=40, labeled=40, unlabeled=49960, test=10000
[Data] labeled batches=2, unlabeled batches=1040
[Train] Starting training: 50 steps
[Step       1/50] lr=0.029970 sup=2.2923 unsup=0.0000 total=2.2923 mask=0.000 time=1.0s
[Eval  Step       1] Test Accuracy: 10.03%
[Step      10/50] lr=0.027135 sup=2.2065 unsup=0.0000 total=2.2065 mask=0.000 time=3.1s
[Step      20/50] lr=0.019635 sup=1.8681 unsup=0.0000 total=1.8681 mask=0.000 time=3.9s
[Step      30/50] lr=0.010365 sup=1.9718 unsup=0.0000 total=1.9718 mask=0.000 time=4.6s
[Step      40/50] lr=0.002865 sup=1.4499 unsup=0.0000 total=1.4499 mask=0.000 time=5.3s
[Step      50/50] lr=0.000000 sup=1.2994 unsup=0.0000 total=1.2994 mask=0.000 time=6.0s
[Eval  Step      50] Test Accuracy: 10.09%

[Final] Test Accuracy: 9.81% (best: 10.09%)
[Final] Total time: 8.8s
[Result] num_labels=40, best_acc=10.09%, final_acc=9.81%
```

## 每步耗时分析
从日志计算各区间每步耗时（累加时间为 `time` 列，含日志开销）:

| 区间 | 步数 | 累计时间差 | 每步 (ms) |
|------|------|-----------|-----------|
| step 1    | 1  | 1.0s (含初始化) | ~1000 |
| step 1-10 | 9  | 2.1s | 233 |
| step 10-20| 10 | 0.8s | 80 |
| step 20-30| 10 | 0.7s | 70 |
| step 30-40| 10 | 0.7s | 70 |
| step 40-50| 10 | 0.7s | 70 |
| **稳态**  | —  | — | **~70 ms** |

## 分析
- 首步含 JIT 编译/CUDA 初始化开销，后续稳定
- 稳态 70ms/步 (batch=16+48=64 张图)
- 50 步 8.8s 总耗时，Pipeline 正常
- 无监督损失持续为 0（正常，模型尚未收敛）
