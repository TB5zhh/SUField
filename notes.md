# 2022-05-13

Frameworks:

- Dataloaders: load data + preprocess
- Network: 
  - train: from data of dataloader to loss
  - eval: from data of dataloader to score


# 2022-05-25

scene split:
train: https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_train.txt
val: https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt
test: https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_test.txt

# 2022-05-26

checkpoint v1:
```python
state_dict = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'step': step,
    'metrics': metrics,
    'args': args,
}
```

checkpoint v2:
```python
state_dict = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'step': step,
    'metrics': metrics,
    'args': args,
    'scaler': scaler.state_dict(),
}
```