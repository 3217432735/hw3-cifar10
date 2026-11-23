$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\asya-ai\hw3env\Scripts\python.exe"
$DataRoot = "D:\asya-ai\data"
$env:PYTHONUTF8 = "1"
$env:TORCH_HOME = Join-Path $ProjectRoot ".torch_cache"

Set-Location $ProjectRoot

$Experiments = @(
    @{
        Name = "cnn_basic"
        Model = "cnn"
        Aug = "basic"
        Epochs = 30
        Lr = 0.001
        WeightDecay = 0.0005
        Patience = 8
    },
    @{
        Name = "cnn_strong"
        Model = "cnn"
        Aug = "strong"
        Epochs = 30
        Lr = 0.001
        WeightDecay = 0.0005
        Patience = 8
    },
    @{
        Name = "resnet18"
        Model = "resnet18"
        Aug = "basic"
        Epochs = 12
        Lr = 0.0003
        WeightDecay = 0.0005
        Patience = 4
    }
)

foreach ($Experiment in $Experiments) {
    & $Python "src\train.py" `
        --experiment $Experiment.Name `
        --model $Experiment.Model `
        --aug $Experiment.Aug `
        --epochs $Experiment.Epochs `
        --lr $Experiment.Lr `
        --weight-decay $Experiment.WeightDecay `
        --early-stop-patience $Experiment.Patience `
        --batch-size 128 `
        --num-workers 0 `
        --data-root $DataRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($Experiment.Name)"
    }

    & $Python "src\evaluate.py" `
        --checkpoint "logs\$($Experiment.Name)_best.pt" `
        --experiment $Experiment.Name `
        --batch-size 128 `
        --num-workers 0 `
        --data-root $DataRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed: $($Experiment.Name)"
    }
}

& $Python "src\gradcam.py" `
    --checkpoint "logs\resnet18_best.pt" `
    --experiment "resnet18" `
    --data-root $DataRoot
if ($LASTEXITCODE -ne 0) {
    throw "Grad-CAM failed"
}

Write-Host "All experiments finished. Metrics are in logs/, figures are in results/."
