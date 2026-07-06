<#
run_all_experiments.ps1

功能：
1. 一键运行 DML-Bench Core 版所有主要实验；
2. 包括 Centralized SGD、Sync-SGD、Local SGD、Async-SGD；
3. 包括 Local SGD 不同 local_steps 对比；
4. 包括 Sync/Async 在 straggler 场景下的对比；
5. 自动创建 results/raw、results/tables、results/figures 目录；
6. 每个实验运行前打印清晰标题；
7. 若某个实验失败，脚本会立即停止，避免后续汇总使用错误结果；
8. 最后自动运行 straggler_analysis.py 和 summary_all_experiments.py。

运行方式：
powershell -ExecutionPolicy Bypass -File scripts/run_all_experiments.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "================================================================================"
Write-Host "DML-Bench: Run All Experiments"
Write-Host "================================================================================"

# -------------------------------
# 0. Basic settings
# -------------------------------

$MODEL = "mlp"
$EPOCHS = 10
$BATCH_SIZE = 64
$LR = 0.01
$NUM_WORKERS = 4
$SEED = 42

$RAW_DIR = "results/raw"
$TABLES_DIR = "results/tables"
$FIGURES_DIR = "results/figures"

# Optional: reduce PyTorch CUDA deterministic warning.
# If you still see warning, this is not fatal.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# -------------------------------
# 1. Create directories
# -------------------------------

Write-Host ""
Write-Host "[0/9] Creating output directories..."
New-Item -ItemType Directory -Force $RAW_DIR | Out-Null
New-Item -ItemType Directory -Force $TABLES_DIR | Out-Null
New-Item -ItemType Directory -Force $FIGURES_DIR | Out-Null
New-Item -ItemType Directory -Force "scripts" | Out-Null
New-Item -ItemType Directory -Force "report" | Out-Null

# -------------------------------
# 2. Check scripts
# -------------------------------

Write-Host ""
Write-Host "[Check] Checking required files..."

$requiredFiles = @(
    "dmlbench/algorithms/centralized_sgd.py",
    "dmlbench/algorithms/sync_sgd.py",
    "dmlbench/algorithms/local_sgd.py",
    "dmlbench/algorithms/async_sgd.py",
    "scripts/straggler_analysis.py",
    "scripts/summary_all_experiments.py"
)

foreach ($file in $requiredFiles) {
    if (-Not (Test-Path $file)) {
        Write-Host "Missing required file: $file" -ForegroundColor Red
        throw "Required file not found: $file"
    }
}

Write-Host "All required files exist." -ForegroundColor Green

# -------------------------------
# 3. Centralized SGD
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[1/9] Running Centralized SGD"
Write-Host "================================================================================"

python -m dmlbench.algorithms.centralized_sgd `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --seed $SEED

# -------------------------------
# 4. Sync-SGD, equal delay
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[2/9] Running Sync-SGD, equal delay = 1,1,1,1"
Write-Host "================================================================================"

python -m dmlbench.algorithms.sync_sgd `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --num-workers $NUM_WORKERS `
    --worker-delays "1,1,1,1" `
    --seed $SEED

# -------------------------------
# 5. Sync-SGD, straggler delay
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[3/9] Running Sync-SGD, straggler delay = 1,1,1,5"
Write-Host "================================================================================"

python -m dmlbench.algorithms.sync_sgd `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --num-workers $NUM_WORKERS `
    --worker-delays "1,1,1,5" `
    --seed $SEED

# -------------------------------
# 6. Local SGD, local_steps = 1, 5, 10, 20
# -------------------------------

$LOCAL_STEPS_LIST = @(1, 5, 10, 20)

foreach ($LOCAL_STEPS in $LOCAL_STEPS_LIST) {
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host "[4/9] Running Local SGD, local_steps = $LOCAL_STEPS"
    Write-Host "================================================================================"

    python -m dmlbench.algorithms.local_sgd `
        --model $MODEL `
        --epochs $EPOCHS `
        --batch-size $BATCH_SIZE `
        --lr $LR `
        --num-workers $NUM_WORKERS `
        --local-steps $LOCAL_STEPS `
        --seed $SEED
}

# -------------------------------
# 7. Async-SGD, equal delay
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[5/9] Running Async-SGD, equal delay = 1,1,1,1"
Write-Host "================================================================================"

python -m dmlbench.algorithms.async_sgd `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --num-workers $NUM_WORKERS `
    --worker-delays "1,1,1,1" `
    --seed $SEED

# -------------------------------
# 8. Async-SGD, straggler delay
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[6/9] Running Async-SGD, straggler delay = 1,1,1,5"
Write-Host "================================================================================"

python -m dmlbench.algorithms.async_sgd `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --num-workers $NUM_WORKERS `
    --worker-delays "1,1,1,5" `
    --seed $SEED

# -------------------------------
# 9. Optional severe straggler
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[7/9] Running Optional Async-SGD, severe straggler delay = 1,1,1,10"
Write-Host "================================================================================"

python -m dmlbench.algorithms.async_sgd `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --num-workers $NUM_WORKERS `
    --worker-delays "1,1,1,10" `
    --seed $SEED

# -------------------------------
# 10. Straggler summary
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[8/9] Running straggler summary"
Write-Host "================================================================================"

python scripts/straggler_analysis.py `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --num-workers $NUM_WORKERS `
    --seed $SEED

# -------------------------------
# 11. Full summary
# -------------------------------

Write-Host ""
Write-Host "================================================================================"
Write-Host "[9/9] Running full experiment summary"
Write-Host "================================================================================"

python scripts/summary_all_experiments.py `
    --raw-dir $RAW_DIR `
    --tables-dir $TABLES_DIR `
    --figures-dir $FIGURES_DIR

Write-Host ""
Write-Host "================================================================================"
Write-Host "All experiments finished successfully."
Write-Host "Summary table: results/tables/summary.csv"
Write-Host "Straggler table: results/tables/straggler_summary.csv"
Write-Host "Figures dir: results/figures"
Write-Host "================================================================================"