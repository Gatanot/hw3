# RTX 4090 云端训练指南

## 0. 环境准备

```bash
# 克隆代码 (或 scp/rsync 上传)
git clone <repo-url> hw3 && cd hw3
# 或: rsync -avz hw3/ user@server:/path/to/hw3/

# 创建虚拟环境
python3 -m venv venv && source venv/bin/activate

# 安装依赖 (PyTorch 按 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 验证
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: True, NVIDIA GeForce RTX 4090
```

## 1. 推荐训练命令

```bash
# === 40 labeled ===
python3 -u train.py \
    --num_labels 40 --batch_size 64 --uratio 7 \
    --lr 0.03 --total_steps 1048576 \
    --confidence_threshold 0.95 --ema_decay 0.999 \
    --use_amp --cudnn_benchmark \
    --checkpoint_dir ./checkpoints/fixmatch_40 \
    --log_dir ./logs/fixmatch_40 \
    --seed 0 --eval_interval 1024 --log_interval 128 \
    2>&1 | tee logs/fixmatch_40_train.log

# === 250 labeled ===
python3 -u train.py \
    --num_labels 250 --batch_size 64 --uratio 7 \
    --lr 0.03 --total_steps 1048576 \
    --confidence_threshold 0.95 --ema_decay 0.999 \
    --use_amp --cudnn_benchmark \
    --checkpoint_dir ./checkpoints/fixmatch_250 \
    --log_dir ./logs/fixmatch_250 \
    --seed 0 --eval_interval 1024 --log_interval 128 \
    2>&1 | tee logs/fixmatch_250_train.log

# === 4000 labeled ===
python3 -u train.py \
    --num_labels 4000 --batch_size 64 --uratio 7 \
    --lr 0.03 --total_steps 1048576 \
    --confidence_threshold 0.95 --ema_decay 0.999 \
    --use_amp --cudnn_benchmark \
    --checkpoint_dir ./checkpoints/fixmatch_4000 \
    --log_dir ./logs/fixmatch_4000 \
    --seed 0 --eval_interval 1024 --log_interval 128 \
    2>&1 | tee logs/fixmatch_4000_train.log
```

如果需要多轮求平均 (论文做法)，修改 `--seed` 为 1, 2。

## 2. 后台运行 (screen/tmux)

```bash
# screen
screen -S fixmatch_40
python3 -u train.py --num_labels 40 --use_amp --cudnn_benchmark \
    --checkpoint_dir ./checkpoints/fixmatch_40 --log_dir ./logs/fixmatch_40 \
    2>&1 | tee logs/fixmatch_40_train.log
# Ctrl+A D 断开

# 或 nohup
nohup python3 -u train.py --num_labels 40 --use_amp --cudnn_benchmark \
    --checkpoint_dir ./checkpoints/fixmatch_40 --log_dir ./logs/fixmatch_40 \
    > logs/fixmatch_40_train.log 2>&1 &
```

## 3. 训练产物 (必须保留)

训练完成后保留以下文件:

```
checkpoints/fixmatch_40/
├── best_model.pth              # 最优模型 (EMA权重)
├── checkpoint_65536.pth        # 中间检查点 (间隔 65536 步)
├── checkpoint_131072.pth
├── ...
└── checkpoint_1048576.pth      # 最终检查点

logs/fixmatch_40/
├── train.log                   # 完整训练日志 (含每一步 loss/mask/acc)
└── events.out.tfevents.*       # TensorBoard 事件文件

logs/fixmatch_250/
└── (同上)

logs/fixmatch_4000/
└── (同上)
```

## 4. 训练后提取关键指标

```bash
# 最佳准确率
grep "best_acc" logs/fixmatch_*/train.log

# 最终准确率
grep "Final" logs/fixmatch_*/train.log

# 提取所有评估记录
grep "Eval" logs/fixmatch_40_train.log > results_fixmatch_40_eval.txt
```

## 5. TensorBoard 可视化

```bash
tensorboard --logdir ./logs --port 6006 --bind_all
# 浏览器访问 http://<server-ip>:6006
```

观察指标:
- `train/sup_loss` — 有监督损失 (应持续下降)
- `train/unsup_loss` — 无监督损失 (mask 比例上升后出现)
- `train/mask_ratio` — 伪标签保留比例 (从 0 逐渐上升)
- `test/accuracy` — 测试准确率

## 6. 预计时间 (RTX 4090)

| 标注量 | 单轮 | seed=0/1/2 三轮 |
|--------|------|-----------------|
| 40     | ~5 h | ~15 h |
| 250    | ~5 h | ~15 h |
| 4000   | ~5 h | ~15 h |
| **合计** | **~15 h** | **~45 h** |

## 7. 常见问题

### Q: 显存不足 (OOM)
减少 `--uratio 3` 并相应减小 batch_size:
```bash
python3 train.py --num_labels 40 --batch_size 32 --uratio 3 --use_amp ...
```

### Q: num_workers 报错
设为 0 避免多进程问题:
```bash
python3 train.py --num_labels 40 --num_workers 0 --use_amp ...
```

### Q: CIFAR-10 下载慢
手动下载并放置:
```bash
mkdir -p data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
# 代码会自动提取
```

### Q: 如何从断点恢复
当前版本不支持断点恢复。建议确保训练中途不被中断。

## 8. 代码修复记录

- `src/dataset.py:109` — labeled_loader 改为 `drop_last=False`
  (修复: num_labels=40 时之前会丢弃唯一 batch)
