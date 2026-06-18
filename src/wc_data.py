"""
历届世界杯静态数据：分组、东道主、大洲映射
被 step3_features.py 导入使用
"""

# ── 历届WC分组（标准化后的球队名，与step1保持一致）──────────
WC_GROUPS = {
    1998: {
        "A": ["Brazil", "Norway", "Morocco", "Scotland"],
        "B": ["Italy", "Chile", "Cameroon", "Austria"],
        "C": ["France", "Denmark", "South Africa", "Saudi Arabia"],
        "D": ["Spain", "Nigeria", "Paraguay", "Bulgaria"],
        "E": ["Netherlands", "Belgium", "South Korea", "Mexico"],
        "F": ["Germany", "Serbia", "Iran", "United States"],  # 数据中Yugoslavia记为Serbia（后继国）
        "G": ["Romania", "Colombia", "England", "Tunisia"],
        "H": ["Argentina", "Japan", "Jamaica", "Croatia"],
    },
    2002: {
        "A": ["France", "Denmark", "Senegal", "Uruguay"],
        "B": ["Spain", "Paraguay", "Slovenia", "South Africa"],
        "C": ["Brazil", "Turkey", "China", "Costa Rica"],
        "D": ["South Korea", "United States", "Poland", "Portugal"],
        "E": ["Germany", "Republic of Ireland", "Cameroon", "Saudi Arabia"],
        "F": ["Argentina", "England", "Sweden", "Nigeria"],
        "G": ["Italy", "Ecuador", "Croatia", "Mexico"],
        "H": ["Japan", "Belgium", "Russia", "Tunisia"],
    },
    2006: {
        "A": ["Germany", "Costa Rica", "Poland", "Ecuador"],
        "B": ["England", "Paraguay", "Trinidad and Tobago", "Sweden"],
        "C": ["Argentina", "Netherlands", "Serbia", "Côte d'Ivoire"],
        "D": ["Mexico", "Iran", "Angola", "Portugal"],
        "E": ["Italy", "Ghana", "United States", "Czechia"],
        "F": ["Brazil", "Croatia", "Australia", "Japan"],
        "G": ["France", "Switzerland", "South Korea", "Togo"],
        "H": ["Spain", "Ukraine", "Tunisia", "Saudi Arabia"],
    },
    2010: {
        "A": ["South Africa", "Mexico", "Uruguay", "France"],
        "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
        "C": ["England", "United States", "Algeria", "Slovenia"],
        "D": ["Germany", "Australia", "Serbia", "Ghana"],
        "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
        "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
        "G": ["Brazil", "North Korea", "Côte d'Ivoire", "Portugal"],
        "H": ["Spain", "Switzerland", "Honduras", "Chile"],
    },
    2014: {
        "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "B": ["Spain", "Netherlands", "Chile", "Australia"],
        "C": ["Colombia", "Greece", "Côte d'Ivoire", "Japan"],
        "D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "F": ["Argentina", "Bosnia-Herzegovina", "Iran", "Nigeria"],
        "G": ["Germany", "Portugal", "Ghana", "United States"],
        "H": ["Belgium", "Algeria", "Russia", "South Korea"],
    },
    2018: {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
    2022: {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
    2026: {
        "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
        "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
        "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "D": ["United States", "Paraguay", "Australia", "Turkey"],
        "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
        "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
        "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
        "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
        "I": ["France", "Senegal", "Iraq", "Norway"],
        "J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
        "L": ["England", "Croatia", "Ghana", "Panama"],
    },
}

# 每队所在组的反向索引：{year: {team: group_letter}}
WC_TEAM_GROUP: dict[int, dict[str, str]] = {}
for _year, _groups in WC_GROUPS.items():
    WC_TEAM_GROUP[_year] = {}
    for _g, _teams in _groups.items():
        for _t in _teams:
            WC_TEAM_GROUP[_year][_t] = _g

# ── 东道主 ────────────────────────────────────────────────
WC_HOSTS: dict[int, set] = {
    1990: {"Italy"},
    1994: {"United States"},
    1998: {"France"},
    2002: {"South Korea", "Japan"},
    2006: {"Germany"},
    2010: {"South Africa"},
    2014: {"Brazil"},
    2018: {"Russia"},
    2022: {"Qatar"},
    2026: {"United States", "Canada", "Mexico"},
}

# 注：Ireland 在2002 WC中是 Republic of Ireland；
# Yugoslavia 在1998后不再参赛（塞黑/塞尔维亚）；
# 2006 Group C 用"Serbia"代表 Serbia and Montenegro（赛中解体，FIFA以Serbia延续）

# ── 小组赛日期范围（用于判断是否是小组赛阶段）────────────────
WC_GROUP_STAGE_END = {
    1998: "1998-06-26",
    2002: "2002-06-14",
    2006: "2006-06-23",
    2010: "2010-06-25",  # R3分两批：A-D组6月22-23日，E-H组6月24-25日
    2014: "2014-06-26",
    2018: "2018-06-28",
    2022: "2022-12-02",
    2026: "2026-06-28",  # 暂定，step7会精确化
}

# ── 大洲/联合会映射 ───────────────────────────────────────
# 以球队在 matches_clean.csv 中的标准名为键
CONFEDERATION: dict[str, str] = {
    # UEFA (欧洲)
    **{t: "UEFA" for t in [
        "Germany", "France", "Spain", "England", "Italy", "Portugal",
        "Netherlands", "Belgium", "Switzerland", "Croatia", "Denmark",
        "Sweden", "Norway", "Poland", "Czech Republic", "Czechia",
        "Hungary", "Romania", "Bulgaria", "Greece", "Turkey",
        "Ukraine", "Russia", "Austria", "Serbia", "Bosnia-Herzegovina",
        "Slovakia", "Slovenia", "Albania", "Kosovo", "North Macedonia",
        "Montenegro", "Finland", "Iceland", "Wales", "Scotland",
        "Northern Ireland", "Republic of Ireland", "Ireland",
        "Luxembourg", "Latvia", "Lithuania", "Estonia",
        "Belarus", "Moldova", "Georgia", "Armenia", "Azerbaijan",
        "Kazakhstan", "Liechtenstein", "Andorra", "San Marino",
        "Malta", "Cyprus", "Faroe Islands",
        "Yugoslavia", "Czechoslovakia",  # 历史球队
        "Bosnia and Herzegovina", "Bosnia-Herzegovina",
        "Austria", "Jordan",
    ]},
    # CONMEBOL (南美)
    **{t: "CONMEBOL" for t in [
        "Brazil", "Argentina", "Colombia", "Chile", "Uruguay",
        "Peru", "Ecuador", "Bolivia", "Paraguay", "Venezuela",
    ]},
    # CONCACAF (北中美+加勒比)
    **{t: "CONCACAF" for t in [
        "United States", "Mexico", "Canada", "Costa Rica",
        "Panama", "Honduras", "El Salvador", "Guatemala",
        "Jamaica", "Trinidad and Tobago", "Haiti", "Cuba",
        "Martinique", "Curaçao", "Bermuda", "Barbados",
        "Nicaragua", "Belize", "Dominican Republic",
        "Puerto Rico", "Antigua and Barbuda",
    ]},
    # CAF (非洲)
    **{t: "CAF" for t in [
        "Nigeria", "Ghana", "Senegal", "Cameroon", "Côte d'Ivoire",
        "Ivory Coast", "Egypt", "Morocco", "Tunisia", "Algeria",
        "South Africa", "Mali", "Burkina Faso", "Guinea",
        "DR Congo", "Congo", "Zambia", "Zimbabwe",
        "Kenya", "Tanzania", "Uganda", "Ethiopia",
        "Angola", "Mozambique", "Cape Verde", "Mauritania", "Ghana",
        "Gambia", "Sierra Leone", "Liberia", "Togo",
        "Benin", "Gabon", "Equatorial Guinea",
        "Central African Republic", "Comoros",
        "Namibia", "Botswana", "Eswatini",
        "Rwanda", "Burundi", "Malawi", "Madagascar",
    ]},
    # AFC (亚洲)
    **{t: "AFC" for t in [
        "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
        "Qatar", "Iraq", "Jordan", "United Arab Emirates", "Oman",
        "Bahrain", "Kuwait", "Syria", "Lebanon", "Palestine",
        "Yemen", "China", "China PR", "North Korea",
        "Uzbekistan", "Tajikistan", "Kyrgyzstan",
        "Turkmenistan", "Kazakhstan",  # 部分时期在AFC
        "India", "Thailand", "Vietnam", "Indonesia",
        "Malaysia", "Myanmar", "Philippines",
        "Singapore", "Cambodia", "Laos",
        "Bangladesh", "Pakistan", "Sri Lanka",
        "Hong Kong", "Macao", "Mongolia",
        "Taiwan", "Chinese Taipei",
    ]},
    # OFC (大洋洲)
    **{t: "OFC" for t in [
        "New Zealand", "Fiji", "Papua New Guinea",
        "Solomon Islands", "Vanuatu", "Tahiti",
        "New Caledonia", "American Samoa", "Samoa",
        "Tonga", "Cook Islands",
    ]},
}
