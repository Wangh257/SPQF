from __future__ import annotations

import argparse
from pathlib import Path


def export_camera_params(
    input_path: str | Path,
    output_path: str | Path,
    matlab_version: str | None = None,
) -> Path:
    """
    通过 MATLAB Engine 读取 scipy 无法解析的 cameraParameters 类对象，
    导出为只包含数值数组的 MATLAB v7 ``.mat`` 文件。

    ``matlab_version`` 仅作为日志备注；Python Engine 使用当前 Python 环境已安装的版本。
    """
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.suffix.lower() != ".mat":
        raise ValueError("输出文件必须使用 .mat 扩展名")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matlab.engine
    except ImportError as exc:
        raise ImportError(
            "当前 Python 环境没有 matlab.engine，请使用已安装 MATLAB Engine 的 Python。"
        ) from exc

    print(f"启动 MATLAB Engine{f' ({matlab_version})' if matlab_version else ''} ...")
    engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    try:
        engine.workspace["phase_field_input_path"] = str(source)
        engine.workspace["phase_field_output_path"] = str(output)
        engine.eval(
            "phase_field_source = load(phase_field_input_path); "
            "phase_field_names = fieldnames(phase_field_source); "
            "if isfield(phase_field_source, 'cameraParams'); "
            "  phase_field_cp = phase_field_source.cameraParams; "
            "elseif numel(phase_field_names) == 1; "
            "  phase_field_cp = phase_field_source.(phase_field_names{1}); "
            "else; "
            "  error('Input MAT does not contain cameraParams'); "
            "end;",
            nargout=0,
        )
        engine.eval(
            "K = phase_field_cp.IntrinsicMatrix'; "
            "R = phase_field_cp.RotationMatrices; "
            "T = phase_field_cp.TranslationVectors; "
            "WorldPoints = phase_field_cp.WorldPoints; "
            "RadialDistortion = phase_field_cp.RadialDistortion; "
            "TangentialDistortion = phase_field_cp.TangentialDistortion; "
            "save(phase_field_output_path, 'K', 'R', 'T', 'WorldPoints', "
            "     'RadialDistortion', 'TangentialDistortion', '-v7');",
            nargout=0,
        )
    finally:
        engine.quit()
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"相机参数导出失败：{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用 Python + MATLAB Engine 将 cameraParams 类对象导出为纯数值 MAT"
    )
    parser.add_argument("--input", required=True, help="cameraParams.mat")
    parser.add_argument("--output", required=True, help="纯数值 camera_numeric.mat")
    parser.add_argument("--matlab-version", help="可选的日志备注，例如 R2025b")
    args = parser.parse_args()
    result = export_camera_params(args.input, args.output, args.matlab_version)
    print(f"相机参数已导出：{result}")


if __name__ == "__main__":
    main()
