r"""
save_params.py
==============
参数识别结果保存工具(**identify 类脚本统一使用**)。
每次识别得到的参数 + 元数据 + 指标 写入 test/identify_param/ 下, 便于跨运行比较与复用。

用法
----
**保存**:
    from save_params import save_params

    save_params(
        script="identify_params.py",                     # 脚本名(必填)
        params={"R1": 2.175e-3, "C1": 7604.4, ...},      # 识别参数(必填, 数值)
        meta={"data_date": "20250701", ...},             # 可选元数据
        metrics={"rmse_unweighted_valid": 2.938, ...},   # 可选指标(数值)
        tag="a0.60",                                     # 可选后缀(如扫描的 alpha 值)
    )

**读取(再入变量)**:
    from save_params import load_params, param_values, load_summary

    rec = load_params(script="identify_params.py")       # 取该脚本最新一条(默认)
    P = rec["params"]                                    # dict: 参数名 -> 数值
    R1, C1, R2, C2 = P["R1"], P["C1"], P["R2"], P["C2"]  # 直接入变量

    P = param_values(script="identify_params.py")        # 也可一步只取参数字典
    print(rec["metrics"], rec["meta"])                   # 指标 / 元数据

    df = load_summary()                                  # 汇总表(跨运行比较)

    其它筛选: load_params(data_date="20250701", tag="a1.00")
              load_params(latest=False, script="verify_alpha_sweep.py")  # 返回全部匹配(list)

输出(目录不存在会自动创建)
--------------------------
    identify_param/<脚本名>_<数据标识>_<tag>_<时间戳>.json   每次运行一条完整记录
    identify_param/params_summary.csv                        一行一条, 便于横向比较
"""

import json
import os
from datetime import datetime

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_DIR = os.path.join(_HERE, "identify_param")


def save_params(script, params, meta=None, metrics=None, tag="", out_dir=PARAM_DIR):
    """保存一次参数识别的结果。

    Parameters
    ----------
    script : str
        产生该结果的脚本名(如 "identify_params.py")。
    params : dict
        识别得到的参数, 形如 {名称: 数值}。
    meta : dict, optional
        运行元数据(数据文件、掩膜、权重、固定项等); 值可以是任意 JSON 可序列化对象。
    metrics : dict, optional
        精度指标(如 RMSE、nfev); 值需为数值。
    tag : str, optional
        文件名后缀标签(如扫描时的 alpha 值)。
    out_dir : str, optional
        输出目录, 默认 test/identify_param/。

    Returns
    -------
    str
        写出的 JSON 文件路径。
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now()
    meta = dict(meta or {})
    stem = os.path.splitext(os.path.basename(script))[0]
    date_tag = str(meta.get("data_date", "")) or ""
    parts = [stem] + [p for p in (date_tag, tag) if p] + [ts.strftime("%Y%m%d-%H%M%S")]
    path = os.path.join(out_dir, "_".join(parts) + ".json")

    rec = {
        "script": stem,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {k: float(v) for k, v in params.items()},
        "metrics": {k: float(v) for k, v in (metrics or {}).items()},
        "meta": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    # 追加一行摘要 -> params_summary.csv (便于跨运行比较)
    row = {"timestamp": rec["timestamp"], "script": stem, "file": os.path.basename(path)}
    row.update({f"p_{k}": v for k, v in rec["params"].items()})
    row.update({f"m_{k}": v for k, v in rec["metrics"].items()})
    row.update({f"d_{k}": v for k, v in meta.items()
                if isinstance(v, (int, float, str, bool))})
    summ = os.path.join(out_dir, "params_summary.csv")
    df = pd.DataFrame([row])
    if os.path.exists(summ):
        df = pd.concat([pd.read_csv(summ), df], ignore_index=True)
    df.to_csv(summ, index=False, encoding="utf-8-sig")

    print(f"  参数已保存: {path}")
    return path


# --------------------------------------------------------------------------- #
# 读取接口
# --------------------------------------------------------------------------- #
def _iter_record_files(out_dir):
    """按文件名(时间戳在尾部)升序遍历记录 JSON(旧 -> 新)。"""
    if not os.path.isdir(out_dir):
        return []
    return [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
            if f.endswith(".json")]


def load_params(path=None, script=None, data_date=None, tag=None,
                latest=True, out_dir=PARAM_DIR):
    """读取已保存的识别参数记录。

    Parameters
    ----------
    path : str, optional
        直接指定 JSON 文件路径; 给出后忽略其余筛选条件。
    script : str, optional
        脚本名(含或不含 ".py")。
    data_date : str, optional
        数据日期标识(如 "20250701")。
    tag : str, optional
        保存时的标签(如 "a1.00")。
    latest : bool, default True
        True -> 返回最新一条(dict); False -> 返回全部匹配记录(list[dict])。
    out_dir : str, optional
        目录, 默认 test/identify_param/。

    Returns
    -------
    dict or list[dict] or None
        记录形如 {"script","timestamp","params","metrics","meta"}(另含 "file");
        无匹配时 latest=True 返回 None, latest=False 返回 []。
    """
    if path is not None:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    stem = os.path.splitext(script)[0] if script else None
    hits = []
    for fp in _iter_record_files(out_dir):
        name = os.path.basename(fp)
        if stem is not None and not name.startswith(stem + "_"):
            continue
        if data_date and f"_{data_date}" not in name:
            continue
        if tag and f"_{tag}_" not in name:
            continue
        with open(fp, encoding="utf-8") as f:
            rec = json.load(f)
        rec["file"] = name
        hits.append(rec)
    if not hits:
        return None if latest else []
    return hits[-1] if latest else hits


def param_values(path=None, **kwargs):
    """只取参数字典 {参数名: 数值}; 等价于 load_params(...)["params"]。

    便于直接入变量:
        P = param_values(script="identify_params.py")
        R1, C1 = P["R1"], P["C1"]
    """
    rec = load_params(path, **kwargs)
    if rec is None:
        raise FileNotFoundError(f"未找到匹配的识别参数记录: {kwargs}")
    return dict(rec["params"])


def load_summary(out_dir=PARAM_DIR):
    """读取 params_summary.csv 为 DataFrame(跨运行横向比较); 不存在返回 None。"""
    f = os.path.join(out_dir, "params_summary.csv")
    return pd.read_csv(f) if os.path.exists(f) else None
