"""Phase 4: export the vehicle detector to ONNX and try INT8 dynamic
quantization with ONNX Runtime. Reports real model size on disk for both.

Usage:
    python benchmarks/export_onnx.py
Writes:
    benchmarks/results/vehicle_detector_fp32.onnx
    benchmarks/results/vehicle_detector_int8.onnx
    benchmarks/results/onnx_export_report.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

RESULTS_DIR = ROOT / "benchmarks" / "results"


def main():
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()

    dummy = torch.rand(3, 320, 320)
    fp32_path = RESULTS_DIR / "vehicle_detector_fp32.onnx"

    print("[export] tracing + exporting FP32 ONNX (torchvision detection models export as a list-input graph)...")
    # dynamo=False: torchvision's Faster R-CNN uses data-dependent control
    # flow inside NMS that the newer torch.export/dynamo-based exporter
    # cannot symbolically trace (GuardOnDataDependentSymNode). The legacy
    # TorchScript-tracing exporter handles it fine and is what torchvision's
    # own ONNX export documentation/tutorials use for this model family.
    torch.onnx.export(
        model,
        ([dummy],),
        str(fp32_path),
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        opset_version=17,
        dynamo=False,
    )
    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024)
    print(f"[export] FP32 ONNX written: {fp32_path} ({fp32_size_mb:.1f} MB)")

    report = {"fp32_onnx_path": str(fp32_path), "fp32_size_mb": fp32_size_mb}

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = RESULTS_DIR / "vehicle_detector_int8.onnx"
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
        int8_size_mb = int8_path.stat().st_size / (1024 * 1024)
        print(f"[export] INT8 dynamic-quantized ONNX written: {int8_path} ({int8_size_mb:.1f} MB)")
        report["int8_onnx_path"] = str(int8_path)
        report["int8_size_mb"] = int8_size_mb
        report["size_reduction_pct"] = 100 * (1 - int8_size_mb / fp32_size_mb)
    except Exception as e:
        print(f"[warn] INT8 quantization failed: {e}")
        report["int8_error"] = str(e)

    (RESULTS_DIR / "onnx_export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
