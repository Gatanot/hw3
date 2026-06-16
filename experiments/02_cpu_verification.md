# CPU 小规模验证

## 测试目的
验证训练 Pipeline 正确性（数据流、前向传播、反向传播、梯度流动）。

## 测试条件
```
python3 train.py --num_labels 40 --device cpu --total_steps 10 \
    --log_interval 5 --eval_interval 100 --batch_size 4 \
    --uratio 2 --num_workers 0
```

## 原始输出
```
[Config] num_labels=40, batch_size=4, uratio=2, lr=0.03, total_steps=10
[Config] confidence_threshold=0.95, ema_decay=0.999
[Train] Using device: cpu
[Data] num_labels=40, labeled=40, unlabeled=49960, test=10000
[Data] labeled batches=10, unlabeled batches=6245
[Train] Starting training: 10 steps
[Step       1/10] lr=0.029266 sup=2.3067 unsup=0.0000 total=2.3067 mask=0.000 time=0.4s
[Eval  Step       1] Test Accuracy: 10.06%
[Step       5/10] lr=0.015000 sup=2.3872 unsup=0.0000 total=2.3872 mask=0.000 time=29.7s
[Step      10/10] lr=0.000000 sup=2.3852 unsup=0.0000 total=2.3852 mask=0.000 time=30.5s

[Final] Test Accuracy: 8.77% (best: 10.06%)
[Final] Total time: 60.0s
[Result] num_labels=40, best_acc=10.06%, final_acc=8.77%
```

## 分析
- 前向/反向传播正常（loss ≈ 2.3，符合随机初始化预期）
- 无监督损失恒为 0（随机模型无法产生 >0.95 置信度的伪标签，符合预期）
- 测试准确率 ≈ 10%（随机猜测水平，符合预期）
- CPU 每步约 6 秒（batch=4+8=12张图），确认全量不可行
