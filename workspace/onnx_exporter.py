import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.onnx")
    parser.add_argument("--load", default="model.pt")
    args = parser.parse_args()
    
    yolov5 = torch.jit.load(args.load)
    dummy_input = torch.randn(1, 3, 512, 512)
    yolov5 = yolov5.eval()

    torch.onnx.export(
        yolov5,
        dummy_input,
        args.save,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("Saved {}".format(args.save))
