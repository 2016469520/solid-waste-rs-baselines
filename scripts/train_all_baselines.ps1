$ErrorActionPreference = "Stop"

$Config = "solid_waste_baselines/configs/baseline.yaml"
$Models = @(
  "small_cnn",
  "resnet18",
  "resnet50",
  "mobilenet_v3_small",
  "efficientnet_b0",
  "convnext_tiny"
)

foreach ($Model in $Models) {
  Write-Host "==== Training $Model ===="
  python -m solid_waste_baselines.src.swbaseline.train --config $Config --model $Model
  python -m solid_waste_baselines.src.swbaseline.evaluate --config $Config --checkpoint "solid_waste_baselines/outputs/$Model/best.pt" --split test
}

