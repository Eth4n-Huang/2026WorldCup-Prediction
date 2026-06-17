"""
权威球队名称映射表 —— 唯一维护点
step1_load_data.py 用它标准化 results.csv；
live_features.py 用它将赛程表名称转为训练集名称。
两处同步：只改这一个文件。
"""

# raw Kaggle 名称  →  训练集标准名
TEAM_NAME_MAP: dict[str, str] = {
    # 确认存在于 results.csv 的异名
    "Ivory Coast":            "Côte d'Ivoire",   # FIFA 官方名
    "Czech Republic":         "Czechia",          # 2016 年后 FIFA 官方名
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "North Macedonia":        "Macedonia",        # FIFA 注册名
    "Swaziland":              "Eswatini",         # 2018 年改名
    "Cape Verde Islands":     "Cape Verde",
    # 防御性映射（其他数据源可能使用的别名）
    "USA":             "United States",
    "IR Iran":         "Iran",
    "Korea Republic":  "South Korea",
    "Korea DPR":       "North Korea",
    "Türkiye":         "Turkey",
    "Kyrgyz Republic": "Kyrgyzstan",
}


def norm_team(name: str) -> str:
    """将任意来源的队名映射为训练集标准名（未命中则原样返回）。"""
    return TEAM_NAME_MAP.get(name, name)
