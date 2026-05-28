param(
  [string]$Model = "resnet18",
  [string]$Config = "solid_waste_baselines/configs/baseline.yaml",
  [string]$Python = "F:\anaconda\envs\urbanvillage\python.exe"
)

$ErrorActionPreference = "Stop"

& $Python -m solid_waste_baselines.src.swbaseline.train --config $Config --model $Model
& $Python -m solid_waste_baselines.src.swbaseline.evaluate --config $Config --checkpoint "solid_waste_baselines/outputs/$Model/best.pt" --split test

