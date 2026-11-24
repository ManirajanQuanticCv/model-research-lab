# INT8 Conversion Guide (YOLO → TensorRT, on Jetson)

Standalone steps to build and validate a TensorRT INT8 engine for the `car_fall` vehicle model
before wiring INT8 into the web app.

> TensorRT engines are tied to the device and TensorRT version. **Build and run everything on the Jetson.**

## Inputs

| Item | Value |
|------|-------|
| Model | `best.pt` (from `runs/detect/car_fall/weights/`) |
| Dataset | `data/images/{train,valid}`, `data/labels/{train,valid}` (924 train / 246 valid) |
| Classes (order matters) | `car, truck, bike, bus, others, ambulance, car_fall` |
| Image size | 640 |

## 1. Copy files to the Jetson

- `best.pt`
- the `data/` folder (images and labels)
- `modelopt.py` from this repo

## 2. Check the environment

```bash
python3 -c "import tensorrt, torch, ultralytics; print(tensorrt.__version__, torch.cuda.is_available(), ultralytics.__version__)"
cat /etc/nv_tegra_release          # JetPack / L4T version
sudo nvpmodel -m 0 && sudo jetson_clocks   # max performance for stable benchmarks
```

`torch.cuda.is_available()` must print `True`. If not, torch is a CPU-only build and the export will fail.

## 3. Create `int8_data.yaml`

Use **absolute paths on the Jetson** and keep the **training class order**.

```yaml
path: /home/<user>/vehicle/data
train: images/train
val: images/valid          # INT8 calibration uses this split
names: {0: car, 1: truck, 2: bike, 3: bus, 4: others, 5: ambulance, 6: car_fall}
```

Delete old label caches after moving the dataset:

```bash
find data/labels -name "*.cache" -delete
```

### Calibration set tips

- Use 300–1000 images that look like deployment data (246 works, 500+ is better).
- Include the rare classes (`ambulance`, `car_fall`, `others`) or their ranges will be calibrated badly.
- To control the set, point `val` at a folder of your choice (for example all `car_fall` and `ambulance` images plus a random mix of the rest).

## 4. Build FP16 and INT8 engines

```bash
python modelopt.py convert --weights best.pt --data int8_data.yaml --imgsz 640 \
    --variants trt_fp16 trt_int8 --out runs/car_fall_int8
```

INT8 takes several minutes (it calibrates first). Output goes to `runs/car_fall_int8/models/`:
`trt_fp16.engine`, `trt_int8.engine`.

Minimal equivalent (this is what to port into the app):

```python
from ultralytics import YOLO
YOLO("best.pt").export(format="engine", int8=True, data="int8_data.yaml",
                       imgsz=640, batch=1, device=0, workspace=4)
```

Delete any stale `*.cache` next to the model if you change the calibration images or the model.

## 5. Benchmark and compare

```bash
python modelopt.py bench  --variants trt_fp16 trt_int8 --out runs/car_fall_int8 \
    --data int8_data.yaml --image data/images/valid/<any_image>.jpg
python modelopt.py report --out runs/car_fall_int8
```

Open `runs/car_fall_int8/report.html` for size, latency (p50/p95), FPS and mAP side by side.

## 6. Check per-class accuracy

Overall mAP can hide drops on rare classes.

```python
from ultralytics import YOLO
for f in ["trt_fp16.engine", "trt_int8.engine"]:
    r = YOLO(f"runs/car_fall_int8/models/{f}", task="detect").val(
        data="int8_data.yaml", imgsz=640, device=0)
    print(f, r.box.map, dict(zip(r.names.values(), r.box.maps)))
```

## 7. Decide

| Result | Action |
|--------|--------|
| mAP50-95 drop ≤ 2 pts, `car_fall` drop ≤ ~3 pts, INT8 clearly faster than FP16 | Use INT8 |
| Accuracy drops too much | Rebuild with a calibration set that over-represents `car_fall` / `ambulance` |
| No speedup over FP16 | Stay on FP16 |

## 8. Port to the app

1. [worker.py](worker.py): pass `data=<calibration yaml>` to `export` when the variant is `trt_int8`.
2. [app.py](app.py): add `trt_int8` to `CONVERTIBLE`.
3. UI: add a calibration images / dataset input to the Convert tab.
4. Update the "INT8 on hold" notes in `README.md`.

## Troubleshooting

- **`torch.cuda.is_available()` is False:** install the Jetson (JetPack) build of torch.
- **No images found / wrong labels:** check the yaml paths and delete `labels/*.cache`.
- **Out of memory during build:** lower `workspace` (e.g. 2) or close other processes.
- **Engine fails to load after an update:** rebuild it; JetPack, TensorRT or ultralytics version changes invalidate engines.
