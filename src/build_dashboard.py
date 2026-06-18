"""
生成 outputs/index.html 三标签页看板
只读数据文件，不修改任何模型/预测逻辑
标签页：预测看板 / 积分榜 / 赛程
用法：python src/build_dashboard.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'outputs'


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────
def multiclass_brier(y_true, probs_list):
    if not y_true: return 0.0
    total = sum(
        (pa-(yt=='A'))**2 + (pd_-(yt=='D'))**2 + (ph-(yt=='H'))**2
        for yt,(pa,pd_,ph) in zip(y_true, probs_list)
    )
    return total / len(y_true)

def fmt_top_scores(raw: str) -> str:
    """把 '1-0:0.182,0-0:0.143' 格式化为 '1-0: 18%  0-0: 14%'"""
    if not raw or str(raw) in ('', 'nan'):
        return ''
    parts = []
    for token in str(raw).split(','):
        token = token.strip()
        if ':' not in token:
            continue
        score, prob_s = token.rsplit(':', 1)
        try:
            prob_pct = round(float(prob_s) * 100)
            parts.append(f"{score}: {prob_pct}%")
        except ValueError:
            parts.append(score)
    return '  '.join(parts)

def bla_predict(ph_val):
    try:
        ph = float(ph_val)
        return 'H' if ph > 0.5 else ('A' if ph < 0.5 else 'D')
    except Exception:
        return '?'


# ─── A. 读取预测数据 ──────────────────────────────────────────────────────────
pred_df  = pd.read_csv(OUT / 'live_predictions_2026.csv', encoding='utf-8')
current  = pred_df[pred_df['is_current'].astype(str).isin(['1', '1.0'])].copy()
settled  = current[current['actual_result'].astype(str).isin(['H','D','A'])].sort_values('match_date').reset_index(drop=True)
_today   = pd.Timestamp.now().normalize()
upcoming = current[
    ~current['actual_result'].astype(str).isin(['H','D','A']) &
    (pd.to_datetime(current['match_date']) >= _today)
].sort_values('match_date').reset_index(drop=True)
settled  = settled.copy()
settled['bla_pred']    = settled['bla_ph'].apply(bla_predict)
settled['bla_correct'] = (settled['bla_pred'] == settled['actual_result']).astype(float)


# ─── 中英文映射（全局共用）────────────────────────────────────────────────────
CN_TO_EN = {
    '墨西哥':'Mexico','南非':'South Africa','韩国':'South Korea','捷克':'Czech Republic',
    '加拿大':'Canada','波黑':'Bosnia and Herzegovina','卡塔尔':'Qatar','瑞士':'Switzerland',
    '巴西':'Brazil','摩洛哥':'Morocco','海地':'Haiti','苏格兰':'Scotland',
    '美国':'United States','巴拉圭':'Paraguay','澳大利亚':'Australia','土耳其':'Turkey',
    '德国':'Germany','库拉索':'Curaçao','科特迪瓦':'Ivory Coast','厄瓜多尔':'Ecuador',
    '荷兰':'Netherlands','日本':'Japan','瑞典':'Sweden','突尼斯':'Tunisia',
    '比利时':'Belgium','埃及':'Egypt','伊朗':'Iran','新西兰':'New Zealand',
    '西班牙':'Spain','佛得角':'Cape Verde','沙特阿拉伯':'Saudi Arabia','乌拉圭':'Uruguay',
    '法国':'France','塞内加尔':'Senegal','伊拉克':'Iraq','挪威':'Norway',
    '阿根廷':'Argentina','阿尔及利亚':'Algeria','奥地利':'Austria','约旦':'Jordan',
    '葡萄牙':'Portugal','民主刚果':'DR Congo','乌兹别克斯坦':'Uzbekistan','哥伦比亚':'Colombia',
    '英格兰':'England','克罗地亚':'Croatia','加纳':'Ghana','巴拿马':'Panama',
}
EN_TO_CN = {v: k for k, v in CN_TO_EN.items()}
def to_cn(name): return EN_TO_CN.get(str(name), str(name))

# 北京日期+时间查找表（小组赛全72场，key=中文主队|中文客队，value=(bj_date, bj_time)）
_MATCH_BJ = {
    # 第1轮 6/12-6/18
    '墨西哥|南非':('2026-06-12','03:00'),'韩国|捷克':('2026-06-12','10:00'),
    '加拿大|波黑':('2026-06-13','03:00'),'美国|巴拉圭':('2026-06-13','09:00'),
    '卡塔尔|瑞士':('2026-06-14','03:00'),'巴西|摩洛哥':('2026-06-14','06:00'),
    '海地|苏格兰':('2026-06-14','09:00'),'澳大利亚|土耳其':('2026-06-14','12:00'),
    '德国|库拉索':('2026-06-15','01:00'),'荷兰|日本':('2026-06-15','04:00'),
    '科特迪瓦|厄瓜多尔':('2026-06-15','07:00'),'瑞典|突尼斯':('2026-06-15','10:00'),
    '西班牙|佛得角':('2026-06-16','00:00'),'比利时|埃及':('2026-06-16','03:00'),
    '沙特阿拉伯|乌拉圭':('2026-06-16','06:00'),'伊朗|新西兰':('2026-06-16','09:00'),
    '法国|塞内加尔':('2026-06-17','03:00'),'伊拉克|挪威':('2026-06-17','06:00'),
    '阿根廷|阿尔及利亚':('2026-06-17','09:00'),'奥地利|约旦':('2026-06-17','12:00'),
    '葡萄牙|民主刚果':('2026-06-18','01:00'),'英格兰|克罗地亚':('2026-06-18','04:00'),
    '加纳|巴拿马':('2026-06-18','07:00'),'乌兹别克斯坦|哥伦比亚':('2026-06-18','10:00'),
    # 第2轮 6/19-6/24
    '捷克|南非':('2026-06-19','00:00'),'瑞士|波黑':('2026-06-19','03:00'),
    '加拿大|卡塔尔':('2026-06-19','06:00'),'墨西哥|韩国':('2026-06-19','09:00'),
    '美国|澳大利亚':('2026-06-20','03:00'),'苏格兰|摩洛哥':('2026-06-20','06:00'),
    '巴西|海地':('2026-06-20','09:00'),'土耳其|巴拉圭':('2026-06-20','12:00'),
    '荷兰|瑞典':('2026-06-21','01:00'),'德国|科特迪瓦':('2026-06-21','04:00'),
    '厄瓜多尔|库拉索':('2026-06-21','08:00'),'突尼斯|日本':('2026-06-21','12:00'),
    '西班牙|沙特阿拉伯':('2026-06-22','00:00'),'比利时|伊朗':('2026-06-22','03:00'),
    '乌拉圭|佛得角':('2026-06-22','06:00'),'新西兰|埃及':('2026-06-22','09:00'),
    '阿根廷|奥地利':('2026-06-23','01:00'),'法国|伊拉克':('2026-06-23','05:00'),
    '挪威|塞内加尔':('2026-06-23','08:00'),'约旦|阿尔及利亚':('2026-06-23','11:00'),
    '葡萄牙|乌兹别克斯坦':('2026-06-24','01:00'),'英格兰|加纳':('2026-06-24','04:00'),
    '巴拿马|克罗地亚':('2026-06-24','07:00'),'哥伦比亚|民主刚果':('2026-06-24','10:00'),
    # 第3轮 6/25-6/28
    '瑞士|加拿大':('2026-06-25','03:00'),'波黑|卡塔尔':('2026-06-25','03:00'),
    '苏格兰|巴西':('2026-06-25','06:00'),'摩洛哥|海地':('2026-06-25','06:00'),
    '捷克|墨西哥':('2026-06-25','09:00'),'南非|韩国':('2026-06-25','09:00'),
    '厄瓜多尔|德国':('2026-06-26','04:00'),'库拉索|科特迪瓦':('2026-06-26','04:00'),
    '日本|瑞典':('2026-06-26','07:00'),'突尼斯|荷兰':('2026-06-26','07:00'),
    '土耳其|美国':('2026-06-26','10:00'),'巴拉圭|澳大利亚':('2026-06-26','10:00'),
    '挪威|法国':('2026-06-27','03:00'),'塞内加尔|伊拉克':('2026-06-27','03:00'),
    '佛得角|沙特阿拉伯':('2026-06-27','08:00'),'乌拉圭|西班牙':('2026-06-27','08:00'),
    '新西兰|比利时':('2026-06-27','11:00'),'埃及|伊朗':('2026-06-27','11:00'),
    '巴拿马|英格兰':('2026-06-28','05:00'),'克罗地亚|加纳':('2026-06-28','05:00'),
    '哥伦比亚|葡萄牙':('2026-06-28','07:30'),'民主刚果|乌兹别克斯坦':('2026-06-28','07:30'),
    '阿尔及利亚|奥地利':('2026-06-28','10:00'),'约旦|阿根廷':('2026-06-28','10:00'),
}
def match_bj(home_cn, away_cn):
    return _MATCH_BJ.get(f'{home_cn}|{away_cn}', (None, ''))

# ─── B. 回测基准 ──────────────────────────────────────────────────────────────
bt_frames = [pd.read_csv(OUT / f'backtest_final_{yr}.csv') for yr in [2014,2018,2022]
             if (OUT / f'backtest_final_{yr}.csv').exists()]
if bt_frames:
    bt_all   = pd.concat(bt_frames, ignore_index=True)
    bt_acc   = float(bt_all['dc_correct'].mean())
    bt_brier = multiclass_brier(bt_all['result'].tolist(), bt_all[['dc_pa','dc_pd','dc_ph']].values.tolist())
else:
    bt_acc, bt_brier = 0.52, 0.60


# ─── C. 逐场滚动统计 ─────────────────────────────────────────────────────────
running = []
for i in range(len(settled)):
    sub = settled.iloc[:i+1]; row = settled.iloc[i]
    prbs = sub[['dc_pa','dc_pd','dc_ph']].values.astype(float).tolist()
    y    = sub['actual_result'].values.tolist()
    running.append({
        'n': i+1, 'label': f'#{i+1}',
        'dc_acc':  round(float(sub['dc_correct'].astype(float).mean()), 4),
        'xgb_acc': round(float(sub['xgb_correct'].astype(float).mean()), 4),
        'adj_acc': round(float(sub['adj_correct'].astype(float).mean()), 4),
        'bla_acc': round(float(sub['bla_correct'].mean()), 4),
        'dc_brier': round(multiclass_brier(y, prbs), 4),
        'dc_correct_this': int(float(row['dc_correct'])),
        'actual': str(row['actual_result']),
        'match': f"{to_cn(row['home_team'])} vs {to_cn(row['away_team'])}",
    })


# ─── D. 待赛/已赛数据 ────────────────────────────────────────────────────────
upcoming_data = []
for _, r in upcoming.iterrows():
    conf = str(r.get('high_conf','')).strip() in ('True','1','1.0')
    xadj = str(r.get('xgb_adj', r['xgb_pred'])) if pd.notna(r.get('xgb_adj')) else str(r['xgb_pred'])
    h_cn = to_cn(r['home_team']); a_cn = to_cn(r['away_team'])
    _bj_date, _bj_time = match_bj(h_cn, a_cn)
    upcoming_data.append({'date': _bj_date or str(r['match_date'])[:10],
        'home': h_cn, 'away': a_cn, 'bj_time': _bj_time,
        'ph': round(float(r['dc_ph']),3), 'pd': round(float(r['dc_pd']),3), 'pa': round(float(r['dc_pa']),3),
        'dc_pred': str(r['dc_pred']), 'xgb_pred': xadj, 'high_conf': conf,
        'top_scores': fmt_top_scores(r.get('dc_top_scores', ''))})

settled_data = []
for _, r in settled.iterrows():
    sh_cn = to_cn(r['home_team']); sa_cn = to_cn(r['away_team'])
    _sbj_date, _sbj_time = match_bj(sh_cn, sa_cn)
    settled_data.append({'date': _sbj_date or str(r['match_date'])[:10],
        'home': sh_cn, 'away': sa_cn, 'bj_time': _sbj_time,
        'actual': str(r['actual_result']), 'dc_pred': str(r['dc_pred']),
        'dc_correct': int(float(r['dc_correct'])), 'xgb_correct': int(float(r['xgb_correct'])),
        'adj_correct': int(float(r['adj_correct'])), 'bla_correct': int(float(r['bla_correct'])),
        'ph': round(float(r['dc_ph']),3), 'pd': round(float(r['dc_pd']),3), 'pa': round(float(r['dc_pa']),3),
        'top_scores': fmt_top_scores(r.get('dc_top_scores', ''))})


# ─── E. 蒙特卡洛（若有） ─────────────────────────────────────────────────────
mc_data = None
for mc_name in ['mc_champion_probs.csv', 'monte_carlo_champion.csv']:
    mc_path = OUT / mc_name
    if mc_path.exists():
        mc_df = pd.read_csv(mc_path)
        mc_df = mc_df.sort_values(mc_df.columns[1], ascending=False).head(12)
        mc_data = mc_df.to_dict(orient='records')
        break


# ─── F. 汇总指标 ─────────────────────────────────────────────────────────────
n_s = len(settled)
def smean(col): return round(float(settled[col].astype(float).mean()), 4) if n_s else 0.0
summary = {
    'n_settled': n_s, 'n_upcoming': len(upcoming),
    'dc_acc': smean('dc_correct'), 'xgb_acc': smean('xgb_correct'),
    'adj_acc': smean('adj_correct'), 'bla_acc': smean('bla_correct'),
    'dc_brier': round(multiclass_brier(
        settled['actual_result'].values.tolist(),
        settled[['dc_pa','dc_pd','dc_ph']].values.astype(float).tolist()
    ), 4) if n_s else 0.0,
    'draw_rate': round(float((settled['actual_result']=='D').mean()),4) if n_s else 0.0,
    'gen_time': datetime.now().strftime('%Y-%m-%d'),
}


# ─── G. 积分榜 ────────────────────────────────────────────────────────────────
from live_features import compute_all_group_standings
from wc_data import WC_GROUPS


live_res = pd.read_csv(ROOT / 'data/processed/wc2026_results.csv', encoding='utf-8', parse_dates=['date'])
played   = live_res[live_res['home_score'].notna()].copy()

all_st = compute_all_group_standings(WC_GROUPS[2026], played)

def wdl_ga(teams, df):
    w={t:0 for t in teams}; d={t:0 for t in teams}
    l={t:0 for t in teams}; ga={t:0 for t in teams}
    ts = set(teams)
    for _, r in df.iterrows():
        ht,at = str(r['home_team']),str(r['away_team'])
        if ht not in ts or at not in ts: continue
        hs,as_ = int(r['home_score']),int(r['away_score'])
        ga[ht]+=as_; ga[at]+=hs
        res = str(r['result'])
        if res=='H':   w[ht]+=1; l[at]+=1
        elif res=='D': d[ht]+=1; d[at]+=1
        elif res=='A': l[ht]+=1; w[at]+=1
    return w,d,l,ga

standings_data = {}
for grp in sorted(WC_GROUPS[2026].keys()):
    teams = WC_GROUPS[2026][grp]
    st    = all_st[grp]
    w,d,l,ga = wdl_ga(teams, played)
    standings_data[grp] = [
        {'team': EN_TO_CN.get(t, t), 'rank': st[t]['rank'], 'mp': st[t]['mp'],
         'w': w[t], 'd': d[t], 'l': l[t],
         'gf': st[t]['gf'], 'ga': ga[t], 'gd': st[t]['gd'], 'pts': st[t]['pts']}
        for t in sorted(teams, key=lambda t: st[t]['rank'])
    ]

best_thirds = []
for grp in sorted(WC_GROUPS[2026].keys()):
    teams = WC_GROUPS[2026][grp]; st = all_st[grp]
    third = next((t for t in teams if st[t]['rank']==3), None)
    if third:
        s = st[third]
        best_thirds.append({'group': grp, 'team': EN_TO_CN.get(third, third),
                            'mp': s['mp'], 'pts': s['pts'], 'gd': s['gd'], 'gf': s['gf']})

best_thirds.sort(key=lambda x: (x['pts'], x['gd'], x['gf']), reverse=True)
for i,row in enumerate(best_thirds):
    row['rank_among_thirds'] = i+1
    row['advancing'] = i < 8


# ─── H. 赛程：按球队名称的结果/预测索引 ────────────────────────────────────────
# CN_TO_EN / EN_TO_CN 已在 G 节定义

# 结果索引：'home_en|away_en' → {hs, as_, result}
res_by_teams = {}
for _, r in played.iterrows():
    k = f"{r['home_team']}|{r['away_team']}"
    res_by_teams[k] = {'hs': int(r['home_score']), 'as_': int(r['away_score']), 'result': str(r['result'])}

# 预测索引：'home_en|away_en' → {pred, actual, dc_ok, ph, pd, pa}
pred_by_teams = {}
for _, r in current.iterrows():
    k = f"{r['home_team']}|{r['away_team']}"
    actual_s = str(r['actual_result'])
    is_sett  = actual_s in ('H','D','A')
    pred_by_teams[k] = {
        'pred': str(r['dc_pred']),
        'actual': actual_s if is_sett else None,
        'dc_ok': int(float(r['dc_correct'])) if is_sett else None,
        'ph': round(float(r['dc_ph']),3),
        'pd': round(float(r['dc_pd']),3),
        'pa': round(float(r['dc_pa']),3),
    }


# ─── I. 序列化所有数据 ────────────────────────────────────────────────────────
DATA_JSON = json.dumps({
    'upcoming': upcoming_data, 'settled': settled_data, 'running': running,
    'bt_acc': round(bt_acc,4), 'bt_brier': round(bt_brier,4),
    'mc': mc_data, 'summary': summary,
    'standings': standings_data, 'best_thirds': best_thirds,
    'cn_to_en': CN_TO_EN,
    'res_by_teams': res_by_teams,
    'pred_by_teams': pred_by_teams,
}, ensure_ascii=False)


# ─── J. HTML 模板 ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2026WC-Prediction</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0f172a;--card:#1e293b;--border:#2d3f52;--txt:#e2e8f0;--muted:#64748b;--dim:#94a3b8;
  --blue:#38bdf8;--green:#4ade80;--red:#f87171;--yellow:#fbbf24;--purple:#c084fc;--orange:#fb923c}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px}

/* ── 顶栏 ── */
.topbar{background:var(--card);border-bottom:1px solid var(--border);
  padding:.6rem 1.25rem;display:flex;flex-wrap:wrap;align-items:center;gap:.75rem}
.topbar h1{font-size:.95rem;font-weight:700;color:var(--blue);white-space:nowrap}
.pills{display:flex;flex-wrap:wrap;gap:.4rem;flex:1}
.pill{background:#0f172a;border:1px solid var(--border);border-radius:999px;
  padding:.2rem .65rem;font-size:.78rem;white-space:nowrap}
.pill span{font-weight:700}
.pill.good span{color:var(--green)}.pill.warn span{color:var(--yellow)}
.pill.info span{color:var(--blue)}.pill.muted span{color:var(--dim)}
.gentime{font-size:.72rem;color:var(--muted);white-space:nowrap}

/* ── 标签页导航 ── */
.tabnav{background:var(--card);border-bottom:1px solid var(--border);
  display:flex;padding:0 1.25rem;gap:0}
.tabbtn{background:none;border:none;border-bottom:2px solid transparent;
  padding:.55rem 1.1rem;color:var(--muted);cursor:pointer;font-size:.83rem;font-weight:600;
  white-space:nowrap;transition:color .15s}
.tabbtn:hover{color:var(--txt)}
.tabbtn.active{color:var(--blue);border-bottom-color:var(--blue)}
.tabpage{display:none}.tabpage.active{display:block}

/* ── 卡片 ── */
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem;overflow:hidden}
.card-title{font-size:.78rem;font-weight:600;color:var(--dim);letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:.75rem;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.badge{background:var(--border);border-radius:4px;padding:.1rem .4rem;font-size:.7rem;color:var(--txt)}
.footnote{color:var(--muted);font-size:.72rem;margin-top:.5rem;line-height:1.5}
.empty{text-align:center;padding:1.5rem;color:var(--muted);font-size:.88rem}

/* ── 预测看板网格 ── */
.main{padding:1rem;display:grid;gap:1rem;
  grid-template-columns:1fr 1fr;
  grid-template-areas:"pred pred" "chart perf" "champ champ"}
@media(max-width:860px){.main{grid-template-columns:1fr;
  grid-template-areas:"pred" "chart" "perf" "champ"}}
.area-pred{grid-area:pred}.area-chart{grid-area:chart}
.area-perf{grid-area:perf}.area-champ{grid-area:champ}

/* ── 预测表 ── */
.ptbl{width:100%;border-collapse:collapse}
.ptbl th{text-align:left;padding:.38rem .5rem;color:var(--muted);font-size:.73rem;font-weight:600;border-bottom:1px solid var(--border)}
.ptbl td{padding:.42rem .5rem;border-bottom:1px solid #1a2a3a;vertical-align:middle}
.ptbl tr:last-child td{border-bottom:none}
.team-cell{font-weight:500}.vs{color:var(--muted);font-size:.78rem;margin:0 .2rem}
.date-cell{color:var(--dim);font-size:.76rem;white-space:nowrap}
.conf-badge{display:inline-block;font-size:.63rem;padding:.1rem .3rem;border-radius:3px;
  background:#fbbf2420;color:var(--yellow);border:1px solid #fbbf2440;vertical-align:middle;margin-left:.25rem}
.prob-bar{display:flex;height:20px;border-radius:3px;overflow:hidden;min-width:130px;gap:1px}
.pb-h{background:#1d4ed8;display:flex;align-items:center;justify-content:center;font-size:.67rem;font-weight:600;color:#bfdbfe;min-width:20px}
.pb-d{background:#44403c;display:flex;align-items:center;justify-content:center;font-size:.67rem;font-weight:600;color:#d6d3d1;min-width:20px}
.pb-a{background:#7f1d1d;display:flex;align-items:center;justify-content:center;font-size:.67rem;font-weight:600;color:#fca5a5;min-width:20px}
.pred-badge{display:inline-block;font-size:.73rem;font-weight:700;padding:.12rem .38rem;border-radius:4px;text-align:center;min-width:26px}
.pred-H{background:#1e3a5f;color:var(--blue)}.pred-D{background:#2d2926;color:#d6d3d1}.pred-A{background:#3b1515;color:var(--red)}

/* ── 折线图 ── */
.chart-wrap{position:relative;height:210px}
.chart-sub{position:relative;height:140px;margin-top:.75rem}
.chart-legend{display:flex;flex-wrap:wrap;gap:.4rem .9rem;margin-bottom:.5rem}
.legend-item{display:flex;align-items:center;gap:.3rem;font-size:.72rem;color:var(--dim)}
.legend-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.legend-dash{width:16px;height:2px;border-top:2px dashed;flex-shrink:0}

/* ── 战绩对比 ── */
.score-bar{display:inline-block;height:4px;border-radius:2px;vertical-align:middle;margin-right:.3rem}
.model-label{font-weight:600;font-size:.78rem}
.acc-val{font-weight:700;font-size:.88rem}
.settled-scroll{max-height:260px;overflow-y:auto}
.settled-scroll::-webkit-scrollbar{width:4px}
.settled-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.stbl{width:100%;border-collapse:collapse}
.stbl th{text-align:left;padding:.32rem .4rem;color:var(--muted);font-size:.7rem;font-weight:600;border-bottom:1px solid var(--border)}
.stbl td{padding:.32rem .4rem;border-bottom:1px solid #1a2a3a;font-size:.8rem}
.stbl tr:last-child td{border-bottom:none}
.ok{color:var(--green);font-weight:700}.ng{color:var(--red)}

/* ── 夺冠概率 ── */
.champ-list{display:flex;flex-direction:column;gap:.35rem}
.champ-row{display:flex;align-items:center;gap:.5rem}
.champ-name{width:130px;text-align:right;font-size:.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.champ-bar-wrap{flex:1;background:#1a2a3a;border-radius:4px;height:20px;overflow:hidden}
.champ-bar{height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));
  border-radius:4px;display:flex;align-items:center;justify-content:flex-end;
  padding-right:.35rem;font-size:.7rem;font-weight:700;color:#fff;min-width:26px}

/* ══ 积分榜页 ══ */
.stand-page{padding:1rem;display:flex;flex-direction:column;gap:1rem}
.best3-card{max-width:100%}
.b3tbl{width:100%;border-collapse:collapse}
.b3tbl th{padding:.32rem .55rem;text-align:left;color:var(--muted);font-size:.72rem;font-weight:600;border-bottom:1px solid var(--border)}
.b3tbl td{padding:.32rem .55rem;border-bottom:1px solid #1a2a3a;font-size:.8rem}
.b3tbl tr:last-child td{border-bottom:none}
.adv-yes{color:var(--green);font-weight:700}.adv-no{color:var(--muted)}

.grp-grid{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fill,minmax(270px,1fr))}
.grp-card .card-title{font-size:.85rem;color:var(--txt);letter-spacing:0;text-transform:none;font-weight:700}
.grp-tbl{width:100%;border-collapse:collapse}
.grp-tbl th{padding:.28rem .4rem;text-align:right;color:var(--muted);font-size:.7rem;font-weight:600;border-bottom:1px solid var(--border)}
.grp-tbl th:nth-child(2){text-align:left}
.grp-tbl td{padding:.3rem .4rem;border-bottom:1px solid #151f2d;font-size:.8rem;text-align:right}
.grp-tbl td:nth-child(2){text-align:left;font-weight:500}
.grp-tbl tr:last-child td{border-bottom:none}
/* 出线状态行色 */
.r1 td,.r2 td{} /* 仅左侧排名着色 */
.r1 td:first-child,.r2 td:first-child{color:var(--green);font-weight:700}
.r3 td:first-child{color:var(--yellow);font-weight:600}
.r4 td{color:var(--muted)}
/* 行背景渐变 */
.r1{background:rgba(74,222,128,.06)}
.r2{background:rgba(74,222,128,.04)}
.r3{background:rgba(251,191,36,.04)}
.grp-tbl tr.r1:hover,.grp-tbl tr.r2:hover,.grp-tbl tr.r3:hover,.grp-tbl tr.r4:hover{background:#253040}
.out-zone-label{font-size:.65rem;color:var(--green);margin-left:.3rem;vertical-align:middle}
.b3-zone-label{font-size:.65rem;color:var(--yellow);margin-left:.3rem;vertical-align:middle}

/* ══ 赛程页 ══ */
.sched-page{padding:1rem;max-width:1020px;margin:0 auto}
.sched-notice{background:#1c2a40;border:1px solid #2d4a6a;border-radius:8px;
  padding:.55rem 1rem;color:var(--blue);font-size:.8rem;margin-bottom:1rem;line-height:1.6}
.sched-day{margin-bottom:1.4rem}
.sched-day-header{font-size:.83rem;font-weight:700;color:var(--dim);padding:.4rem 0 .35rem;
  border-bottom:1px solid var(--border);margin-bottom:.4rem;display:flex;align-items:center;gap:.5rem}
.sched-day-header .day-badge{background:#1a2a3a;border-radius:4px;padding:.1rem .45rem;
  font-size:.7rem;color:var(--muted)}
.sched-match{display:grid;grid-template-columns:54px 1fr auto;align-items:center;
  gap:.35rem .6rem;padding:.5rem .65rem;
  border-radius:7px;margin-bottom:.22rem;background:var(--card);border:1px solid var(--border)}
.sched-match:hover{border-color:#3a5068;background:#1a2535}
.sched-match.focus-match{border-color:#fbbf2480;border-left:3px solid #fbbf24}
.sched-match.ko-match{border-color:#818cf840}
.sched-time-col{display:flex;flex-direction:column;align-items:center;flex-shrink:0}
.sched-bj{font-size:.88rem;font-weight:700;color:var(--txt)}
.sched-utc{font-size:.65rem;color:var(--muted);margin-top:1px}
.sched-main{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem .4rem;min-width:0}
.sched-meta{display:flex;align-items:center;gap:.25rem;flex-wrap:wrap;width:100%}
.sched-stage-badge{font-size:.65rem;padding:.1rem .35rem;border-radius:3px;
  background:#1a2a3a;color:var(--dim);white-space:nowrap}
.sched-grp-badge{font-size:.65rem;padding:.1rem .3rem;border-radius:3px;
  background:#0f172a;color:var(--blue);border:1px solid #1e3a5f;white-space:nowrap;font-weight:700}
.sched-focus-tag{font-size:.65rem;color:var(--yellow);white-space:nowrap}
.sched-teams-row{display:flex;align-items:center;gap:.3rem;flex-wrap:wrap;width:100%}
.sched-home,.sched-away{font-weight:500;font-size:.88rem}
.sched-score{font-weight:700;color:var(--txt);background:#0f172a;padding:.15rem .55rem;
  border-radius:4px;font-size:.92rem;white-space:nowrap;letter-spacing:.03em}
.sched-vs{color:var(--muted);font-size:.75rem}
.sched-pending{color:var(--muted);font-size:.75rem;font-style:italic}
.sched-venue-txt{font-size:.68rem;color:var(--muted);width:100%}
.sched-right{display:flex;flex-direction:column;align-items:flex-end;gap:.22rem;flex-shrink:0}
.sched-pred-row{display:flex;align-items:center;gap:.25rem}
.sched-pred-lbl{font-size:.67rem;color:var(--dim)}
.winner-h{border-left:3px solid var(--green)}
.winner-a{border-left:3px solid var(--green)}
.winner-h .sched-home{color:var(--green)}
.winner-a .sched-away{color:var(--green)}
.ko-home,.ko-away{color:var(--muted);font-style:italic;font-size:.82rem}

/* ══ 窄屏适配（仅手机，不影响桌面端任何样式）══ */
@media(max-width:600px){
  /* 关键：窄屏下解除卡片的overflow:hidden，否则内部无法横向滚动 */
  .card{overflow:visible}
  /* 问题1：预测看板——让表格本身在一个可滚动的包裹层里 */
  #upcoming-body{overflow-x:auto;-webkit-overflow-scrolling:touch}
  #upcoming-body .ptbl{min-width:520px}
  #upcoming-body .team-cell{white-space:nowrap}
  #upcoming-body .date-cell{white-space:nowrap}
  /* 问题2：最佳第三名表——同理 */
  #best3-body{overflow-x:auto;-webkit-overflow-scrolling:touch}
  #best3-body .b3tbl{min-width:560px}
  #best3-body .b3tbl th,
  #best3-body .b3tbl td{white-space:nowrap}
}
</style>
</head>
<body>

<div class="topbar">
  <h1>⚽ 2026WC-Prediction</h1>
  <div class="pills" id="pills"></div>
  <div class="gentime" id="gentime"></div>
</div>
<nav class="tabnav">
  <button class="tabbtn active" data-tab="dash">⚽ 预测看板</button>
  <button class="tabbtn" data-tab="stand">📊 积分榜</button>
  <button class="tabbtn" data-tab="sched">📅 赛程</button>
</nav>

<!-- ══ 页面1: 预测看板 ══ -->
<section id="pg-dash" class="tabpage active">
<div class="main">
  <div class="card area-pred">
    <div class="card-title">待赛预测 <span class="badge" id="upcoming-count">—</span>
      <span style="margin-left:auto;font-size:.7rem;color:var(--muted)">H &nbsp;D &nbsp;A 概率</span></div>
    <div id="upcoming-body"></div>
    <p class="footnote">金框 = DC置信度 &gt;60%</p>
  </div>
  <div class="card area-chart">
    <div class="card-title">滚动战绩曲线 <span class="badge" id="chart-n">—</span></div>
    <div class="chart-legend" id="chart-legend"></div>
    <div class="chart-wrap"><canvas id="chartAcc"></canvas></div>
    <div class="chart-sub"><canvas id="chartBrier"></canvas></div>
    <p class="footnote" id="chart-note"></p>
  </div>
  <div class="card area-perf">
    <div class="card-title">模型表现对比 <span class="badge" id="perf-n">—</span></div>
    <div id="perf-body"></div>
    <div class="settled-scroll" style="margin-top:.75rem">
      <table class="stbl"><thead><tr>
        <th>日期</th><th>对阵</th><th>结果</th><th>DC</th><th>XGB</th><th>Adj</th><th>BLa</th><th>DC概率</th>
      </tr></thead><tbody id="settled-tbody"></tbody></table>
    </div>
  </div>
  <div class="card area-champ">
    <div class="card-title">夺冠/晋级概率 <span class="badge">蒙特卡洛</span></div>
    <div id="champ-body"></div>
  </div>
</div>
</section>

<!-- ══ 页面2: 积分榜 ══ -->
<section id="pg-stand" class="tabpage">
<div class="stand-page">
  <div class="card best3-card">
    <div class="card-title">
      最佳第三名实时排名
      <span class="badge">12组第3名 → 前8名晋级 32强</span>

    </div>
    <div id="best3-body"></div>
  </div>
  <div class="grp-grid" id="grp-grid"></div>
  <p class="footnote" style="padding:0 .25rem">
    绿色 = 直接出线区（前2名）；黄色 = 最佳第三名争夺区；灰色 = 目前已被淘汰区。
    积分相同时按净胜球 / 进球数排序（不含头对头等FIFA完整规则）。仅统计已结算比赛。
  </p>
</div>
</section>

<!-- ══ 页面3: 赛程 ══ -->
<section id="pg-sched" class="tabpage">
<div class="sched-page">
  <div id="sched-body"></div>
</div>
</section>

<script>
const DATA = __DATA_JSON__;

// ── 工具函数 ─────────────────────────────────────────────────────────────────
const pct  = v => (v*100).toFixed(1)+'%';
const pct0 = v => (v*100).toFixed(0)+'%';
const esc  = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const tick = v => v ? '<span class="ok">✓</span>' : '<span class="ng">✗</span>';
const badge = pred => {
  const cls = {H:'pred-H',D:'pred-D',A:'pred-A'}[pred]||'pred-D';
  return `<span class="pred-badge ${cls}">${pred||'?'}</span>`;
};
const sgn = n => n > 0 ? '+'+n : String(n);

// ── 标签页切换 ────────────────────────────────────────────────────────────────
document.querySelectorAll('.tabbtn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabbtn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tabpage').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('pg-' + btn.dataset.tab).classList.add('active');
  });
});

// ── 顶栏摘要 ─────────────────────────────────────────────────────────────────
(function(){
  const s = DATA.summary;
  document.getElementById('pills').innerHTML = [
    {l:'已结算', v:s.n_settled+' 场', c:'info'},
    {l:'DC准确率', v:pct(s.dc_acc), c: s.dc_acc>=DATA.bt_acc?'good':'warn'},
    {l:'Brier',    v:s.dc_brier.toFixed(3), c: s.dc_brier<=DATA.bt_brier?'good':'warn'},
    {l:'平局率',   v:pct(s.draw_rate), c:'warn'},
    {l:'待赛',     v:s.n_upcoming+' 场', c:'muted'},
  ].map(p=>`<div class="pill ${p.c}"><span>${p.v}</span> ${p.l}</div>`).join('');
  document.getElementById('gentime').textContent = '生成于 ' + s.gen_time;
})();

// ── Panel 1: 待赛预测 ────────────────────────────────────────────────────────
(function(){
  const cnt  = document.getElementById('upcoming-count');
  const body = document.getElementById('upcoming-body');
  cnt.textContent = DATA.upcoming.length + ' 场';
  if(!DATA.upcoming.length){
    body.innerHTML='<div class="empty">今日暂无待赛场次<br><small>运行 daily_update.py 后刷新</small></div>'; return;
  }
  body.innerHTML = `<table class="ptbl"><thead><tr>
    <th>日期/北京时间</th><th>对阵</th><th>H &nbsp;D &nbsp;A 概率</th><th>DC</th><th>XGB</th>
  </tr></thead><tbody>${DATA.upcoming.map(m=>{
    const highRow = m.high_conf ? 'style="box-shadow:inset 3px 0 0 #fbbf24"':'';
    const dc = m.bj_time ? `${m.date}<br><small style="color:var(--muted)">${m.bj_time}</small>` : m.date;
    return `<tr ${highRow}>
      <td class="date-cell">${dc}</td>
      <td class="team-cell">${esc(m.home)}<span class="vs">vs</span>${esc(m.away)}</td>
      <td><div class="prob-bar">
        <div class="pb-h" style="width:${pct0(m.ph)}">${pct0(m.ph)}</div>
        <div class="pb-d" style="width:${pct0(m.pd)}">${pct0(m.pd)}</div>
        <div class="pb-a" style="width:${pct0(m.pa)}">${pct0(m.pa)}</div>
      </div>${m.top_scores?`<div style="color:var(--muted);font-size:.68rem;margin-top:.18rem;white-space:nowrap">${esc(m.top_scores)}</div>`:''}</td>
      <td>${badge(m.dc_pred)}${m.high_conf?'<span class="conf-badge">高</span>':''}</td>
      <td>${badge(m.xgb_pred)}</td>
    </tr>`;
  }).join('')}</tbody></table>`;
})();

// ── Panel 2: 折线图 ──────────────────────────────────────────────────────────
(function(){
  const r = DATA.running;
  document.getElementById('chart-n').textContent = r.length + ' 场';
  if(!r.length){ document.getElementById('chartAcc').parentElement.innerHTML='<div class="empty">暂无结算数据</div>'; return; }
  const labels   = r.map(d=>d.label);
  const ptColors = r.map(d=>d.dc_correct_this?'#4ade80':'#f87171');
  const tips     = r.map(d=>d.match+' ['+d.actual+']');
  const cOpts = {responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{callbacks:{title:i=>tips[i[0].dataIndex]},bodyFont:{size:11}}},
    elements:{point:{radius:5,hoverRadius:7}},
    scales:{x:{ticks:{font:{size:10}},grid:{color:'#1a2a3a'}}}};
  new Chart(document.getElementById('chartAcc'),{type:'line',data:{labels,datasets:[
    {label:'DC',data:r.map(d=>+(d.dc_acc*100).toFixed(1)),borderColor:'#38bdf8',
     pointBackgroundColor:ptColors,pointBorderColor:ptColors,tension:.3,fill:false},
    {label:'Adj',data:r.map(d=>+(d.adj_acc*100).toFixed(1)),borderColor:'#c084fc',
     pointStyle:'triangle',pointRadius:4,tension:.3,fill:false},
    {label:'BLa',data:r.map(d=>+(d.bla_acc*100).toFixed(1)),borderColor:'#fb923c',pointRadius:0,tension:.3,fill:false},
    {label:'基准',data:r.map(_=>+(DATA.bt_acc*100).toFixed(1)),borderColor:'#4ade8066',
     borderDash:[5,4],pointRadius:0,fill:false},
  ]},options:{...cOpts,scales:{...cOpts.scales,y:{min:0,max:100,
    title:{display:true,text:'累计准确率(%)',color:'#64748b',font:{size:11}},
    grid:{color:'#1a2a3a'},ticks:{color:'#64748b',font:{size:10}}}}}});
  new Chart(document.getElementById('chartBrier'),{type:'line',data:{labels,datasets:[
    {label:'DC Brier',data:r.map(d=>+d.dc_brier.toFixed(3)),borderColor:'#38bdf8',
     backgroundColor:'#38bdf810',pointRadius:3,tension:.3,fill:true},
    {label:'基准',data:r.map(_=>+DATA.bt_brier.toFixed(3)),borderColor:'#4ade8066',
     borderDash:[5,4],pointRadius:0,fill:false},
  ]},options:{...cOpts,scales:{...cOpts.scales,y:{
    title:{display:true,text:'Brier (↓越小越好)',color:'#64748b',font:{size:10}},
    grid:{color:'#1a2a3a'},ticks:{color:'#64748b',font:{size:10}}}}}});
  document.getElementById('chart-legend').innerHTML=[
    {c:'#38bdf8',l:'DC（绿点=命中/红点=失误）'},{c:'#c084fc',l:'XGB+调整'},
    {c:'#fb923c',l:'BLa基线'},{c:'#4ade8066',l:'回测基准',dash:true}
  ].map(({c,l,dash})=>`<div class="legend-item">${dash
    ?`<div class="legend-dash" style="border-color:${c}"></div>`
    :`<div class="legend-dot" style="background:${c}"></div>`}<span>${l}</span></div>`).join('');
  const dc=DATA.settled.filter(s=>s.actual==='D').length;
  document.getElementById('chart-note').textContent=
    `已结算${r.length}场，含${dc}场平局(${pct(dc/r.length)})；回测基准 ACC=${pct(DATA.bt_acc)} Brier=${DATA.bt_brier.toFixed(3)}`;
})();

// ── Panel 4: 模型表现 ────────────────────────────────────────────────────────
(function(){
  const s=DATA.summary, n=s.n_settled;
  document.getElementById('perf-n').textContent=n+' 场';
  if(!n){document.getElementById('perf-body').innerHTML='<div class="empty">暂无数据</div>';return;}
  const models=[
    {name:'DC Dixon-Coles',acc:s.dc_acc,brier:s.dc_brier,c:'#38bdf8'},
    {name:'XGB+平局调整', acc:s.adj_acc,brier:null,c:'#c084fc'},
    {name:'XGB原始',      acc:s.xgb_acc,brier:null,c:'#818cf8'},
    {name:'BLa Elo基线',  acc:s.bla_acc,brier:null,c:'#fb923c'},
  ];
  const maxA=Math.max(...models.map(m=>m.acc));
  document.getElementById('perf-body').innerHTML=models.map(m=>`
    <div style="display:flex;align-items:center;gap:.5rem;padding:.28rem 0;border-bottom:1px solid var(--border)">
      <div class="model-label" style="width:105px;color:${m.c}">${m.name}</div>
      <div class="score-bar" style="width:${(m.acc/maxA*80).toFixed(0)}px;background:${m.c}"></div>
      <div class="acc-val" style="color:${m.acc===maxA?'#4ade80':'var(--txt)'}">${pct(m.acc)}</div>
      ${m.brier!==null?`<span style="color:var(--muted);font-size:.73rem">Brier ${m.brier.toFixed(3)}</span>`:''}
    </div>`).join('');
  document.getElementById('settled-tbody').innerHTML=DATA.settled.slice().reverse().map(r=>{
    const dc = r.bj_time ? `${r.date}<br><small style="color:var(--muted)">${r.bj_time}</small>` : r.date;
    return `<tr>
      <td class="date-cell">${dc}</td>
      <td><span style="font-size:.76rem">${esc(r.home)}<span style="color:var(--muted)"> vs </span>${esc(r.away)}</span></td>
      <td>${badge(r.actual)}</td>
      <td>${tick(r.dc_correct)}</td><td>${tick(r.xgb_correct)}</td>
      <td>${tick(r.adj_correct)}</td><td>${tick(r.bla_correct)}</td>
      <td><span style="font-size:.7rem;color:var(--dim)">${pct0(r.ph)}/${pct0(r.pd)}/${pct0(r.pa)}</span>${r.top_scores?`<br><span style="font-size:.65rem;color:var(--muted)">${esc(r.top_scores)}</span>`:''}</td>
    </tr>`; }).join('');
})();

// ── Panel 3: 夺冠概率 ────────────────────────────────────────────────────────
(function(){
  const el=document.getElementById('champ-body');
  if(!DATA.mc||!DATA.mc.length){
    el.innerHTML='<div class="empty">🔬 蒙特卡洛模块尚未运行<br><small style="color:var(--muted)">运行 step7_predict_2026.py 生成后刷新</small></div>';return;
  }
  const tk=Object.keys(DATA.mc[0])[0],pk=Object.keys(DATA.mc[0])[1];
  const mx=Math.max(...DATA.mc.map(r=>r[pk]));
  el.innerHTML='<div class="champ-list">'+DATA.mc.map((r,i)=>{
    const w=Math.max(4,(r[pk]/mx*100)).toFixed(0);
    return `<div class="champ-row"><div class="champ-name">#${i+1} ${esc(r[tk])}</div>
      <div class="champ-bar-wrap"><div class="champ-bar" style="width:${w}%">${pct(r[pk])}</div></div></div>`;
  }).join('')+'</div>';
})();

// ══ 积分榜 ══════════════════════════════════════════════════════════════════
(function(){
  // ── 最佳第三名 ──
  const bt = DATA.best_thirds;
  const b3el = document.getElementById('best3-body');
  if(!bt||!bt.length){
    b3el.innerHTML='<div class="empty">小组赛尚未开始</div>';
  } else {
    b3el.innerHTML=`<table class="b3tbl"><thead><tr>
      <th>名次</th><th>组</th><th>球队</th><th>赛</th><th>积分</th><th>净胜球</th><th>进球</th><th>状态</th>
    </tr></thead><tbody>${bt.map(r=>`<tr>
      <td>${r.rank_among_thirds}</td>
      <td><span class="badge">${r.group}</span></td>
      <td style="font-weight:500">${esc(r.team)}</td>
      <td>${r.mp}</td>
      <td style="font-weight:700;color:var(--txt)">${r.pts}</td>
      <td>${sgn(r.gd)}</td><td>${r.gf}</td>
      <td class="${r.advancing?'adv-yes':'adv-no'}">${r.advancing?'✓ 晋级':'— 淘汰'}</td>
    </tr>`).join('')}</tbody></table>
    `;
  }

  // ── 12组积分榜 ──
  const grid = document.getElementById('grp-grid');
  const grpOrder = 'ABCDEFGHIJKL'.split('');
  grid.innerHTML = grpOrder.map(g => {
    const rows = DATA.standings[g];
    if(!rows) return '';
    const hasPlayed = rows.some(r=>r.mp>0);
    const tbody = rows.map(r=>{
      const cls = 'r'+r.rank;
      const rankLabel = r.rank<=2
        ? `${r.rank}<span class="out-zone-label">✓</span>`
        : r.rank===3
          ? `${r.rank}<span class="b3-zone-label">★</span>`
          : `${r.rank}`;
      return `<tr class="${cls}">
        <td style="text-align:center">${rankLabel}</td>
        <td>${esc(r.team)}</td>
        <td>${r.mp}</td><td>${r.w}</td><td>${r.d}</td><td>${r.l}</td>
        <td>${r.gf}</td><td>${r.ga}</td>
        <td style="color:${r.gd>0?'#4ade80':r.gd<0?'#f87171':'var(--dim)'}">${sgn(r.gd)}</td>
        <td style="font-weight:700;color:var(--txt)">${r.pts}</td>
      </tr>`;
    }).join('');
    return `<div class="card grp-card">
      <div class="card-title">小组 ${g}${hasPlayed?'':' <span style="color:var(--muted);font-size:.7rem;font-weight:400">待赛</span>'}</div>
      <table class="grp-tbl"><thead><tr>
        <th style="text-align:center">名</th><th>球队</th>
        <th>赛</th><th>胜</th><th>平</th><th>负</th>
        <th>进</th><th>失</th><th>净</th><th>分</th>
      </tr></thead><tbody>${tbody}</tbody></table>
    </div>`;
  }).join('');
})();

// ══ MATCHES 全局定义（赛程页 + 北京时间查询）════════════════════════════════
const MATCHES = [
  // === 小组赛 第1轮 ===
  { id:1,  date:'2026-06-12', time:'03:00', group:'A', home:'墨西哥',     away:'南非',     round:1, venue:'墨西哥城·阿兹特克体育场', focus:'🎬揭幕战' },
  { id:2,  date:'2026-06-12', time:'10:00', group:'A', home:'韩国',       away:'捷克',     round:1, venue:'瓜达拉哈拉体育场' },
  { id:3,  date:'2026-06-13', time:'03:00', group:'B', home:'加拿大',     away:'波黑',     round:1, venue:'多伦多·BMO球场' },
  { id:4,  date:'2026-06-13', time:'09:00', group:'D', home:'美国',       away:'巴拉圭',   round:1, venue:'洛杉矶·SoFi体育场' },
  { id:5,  date:'2026-06-14', time:'03:00', group:'B', home:'卡塔尔',     away:'瑞士',     round:1, venue:'旧金山·Levi\'s体育场' },
  { id:6,  date:'2026-06-14', time:'06:00', group:'C', home:'巴西',       away:'摩洛哥',   round:1, venue:'纽约·大都会人寿体育场', focus:'⭐焦点战' },
  { id:7,  date:'2026-06-14', time:'09:00', group:'C', home:'海地',       away:'苏格兰',   round:1, venue:'波士顿·吉列体育场' },
  { id:8,  date:'2026-06-14', time:'12:00', group:'D', home:'澳大利亚',   away:'土耳其',   round:1, venue:'温哥华·BC Place' },
  { id:9,  date:'2026-06-15', time:'01:00', group:'E', home:'德国',       away:'库拉索',   round:1, venue:'休斯顿·NRG体育场' },
  { id:10, date:'2026-06-15', time:'04:00', group:'F', home:'荷兰',       away:'日本',     round:1, venue:'达拉斯·AT&T体育场', focus:'🔥死亡之组' },
  { id:11, date:'2026-06-15', time:'07:00', group:'E', home:'科特迪瓦',   away:'厄瓜多尔', round:1, venue:'费城·林肯金融球场' },
  { id:12, date:'2026-06-15', time:'10:00', group:'F', home:'瑞典',       away:'突尼斯',   round:1, venue:'蒙特雷体育场', focus:'🔥死亡之组' },
  { id:13, date:'2026-06-16', time:'00:00', group:'H', home:'西班牙',     away:'佛得角',   round:1, venue:'亚特兰大·梅赛德斯-奔驰体育场' },
  { id:14, date:'2026-06-16', time:'03:00', group:'G', home:'比利时',     away:'埃及',     round:1, venue:'西雅图·Lumen球场' },
  { id:15, date:'2026-06-16', time:'06:00', group:'H', home:'沙特阿拉伯', away:'乌拉圭',   round:1, venue:'迈阿密·硬石体育场' },
  { id:16, date:'2026-06-16', time:'09:00', group:'G', home:'伊朗',       away:'新西兰',   round:1, venue:'洛杉矶·SoFi体育场' },
  { id:17, date:'2026-06-17', time:'03:00', group:'I', home:'法国',       away:'塞内加尔', round:1, venue:'纽约·大都会人寿体育场', focus:'⚡姆巴佩vs马内' },
  { id:18, date:'2026-06-17', time:'06:00', group:'I', home:'伊拉克',     away:'挪威',     round:1, venue:'波士顿·吉列体育场', focus:'哈兰德世界杯首秀' },
  { id:19, date:'2026-06-17', time:'09:00', group:'J', home:'阿根廷',     away:'阿尔及利亚',round:1,venue:'堪萨斯城·箭头体育场', focus:'⭐梅西最后一届首秀' },
  { id:20, date:'2026-06-17', time:'12:00', group:'J', home:'奥地利',     away:'约旦',     round:1, venue:'旧金山·Levi\'s体育场' },
  { id:21, date:'2026-06-18', time:'01:00', group:'K', home:'葡萄牙',     away:'民主刚果', round:1, venue:'休斯顿·NRG体育场', focus:'⭐C罗最后一届首秀' },
  { id:22, date:'2026-06-18', time:'04:00', group:'L', home:'英格兰',     away:'克罗地亚', round:1, venue:'达拉斯·AT&T体育场', focus:'🔥欧洲强强对话' },
  { id:23, date:'2026-06-18', time:'07:00', group:'L', home:'加纳',       away:'巴拿马',   round:1, venue:'多伦多·BMO球场' },
  { id:24, date:'2026-06-18', time:'10:00', group:'K', home:'乌兹别克斯坦',away:'哥伦比亚',round:1,venue:'墨西哥城·阿兹特克体育场' },
  // === 第2轮 ===
  { id:25, date:'2026-06-19', time:'00:00', group:'A', home:'捷克',       away:'南非',     round:2, venue:'亚特兰大·梅赛德斯-奔驰体育场' },
  { id:26, date:'2026-06-19', time:'03:00', group:'B', home:'瑞士',       away:'波黑',     round:2, venue:'洛杉矶·SoFi体育场' },
  { id:27, date:'2026-06-19', time:'06:00', group:'B', home:'加拿大',     away:'卡塔尔',   round:2, venue:'温哥华·BC Place' },
  { id:28, date:'2026-06-19', time:'09:00', group:'A', home:'墨西哥',     away:'韩国',     round:2, venue:'瓜达拉哈拉体育场', focus:'🔥A组头名之争' },
  { id:29, date:'2026-06-20', time:'03:00', group:'D', home:'美国',       away:'澳大利亚', round:2, venue:'西雅图·Lumen球场' },
  { id:30, date:'2026-06-20', time:'06:00', group:'C', home:'苏格兰',     away:'摩洛哥',   round:2, venue:'波士顿·吉列体育场' },
  { id:31, date:'2026-06-20', time:'09:00', group:'C', home:'巴西',       away:'海地',     round:2, venue:'费城·林肯金融球场' },
  { id:32, date:'2026-06-20', time:'12:00', group:'D', home:'土耳其',     away:'巴拉圭',   round:2, venue:'旧金山·Levi\'s体育场' },
  { id:33, date:'2026-06-21', time:'01:00', group:'F', home:'荷兰',       away:'瑞典',     round:2, venue:'达拉斯·AT&T体育场', focus:'🔥死亡之组决战' },
  { id:34, date:'2026-06-21', time:'04:00', group:'E', home:'德国',       away:'科特迪瓦', round:2, venue:'多伦多·BMO球场' },
  { id:35, date:'2026-06-21', time:'08:00', group:'E', home:'厄瓜多尔',   away:'库拉索',   round:2, venue:'堪萨斯城·箭头体育场' },
  { id:36, date:'2026-06-21', time:'12:00', group:'F', home:'突尼斯',     away:'日本',     round:2, venue:'蒙特雷体育场' },
  { id:37, date:'2026-06-22', time:'00:00', group:'H', home:'西班牙',     away:'沙特阿拉伯',round:2,venue:'亚特兰大·梅赛德斯-奔驰体育场' },
  { id:38, date:'2026-06-22', time:'03:00', group:'G', home:'比利时',     away:'伊朗',     round:2, venue:'洛杉矶·SoFi体育场' },
  { id:39, date:'2026-06-22', time:'06:00', group:'H', home:'乌拉圭',     away:'佛得角',   round:2, venue:'迈阿密·硬石体育场' },
  { id:40, date:'2026-06-22', time:'09:00', group:'G', home:'新西兰',     away:'埃及',     round:2, venue:'温哥华·BC Place' },
  { id:41, date:'2026-06-23', time:'01:00', group:'J', home:'阿根廷',     away:'奥地利',   round:2, venue:'达拉斯·AT&T体育场', focus:'⭐J组头名之争' },
  { id:42, date:'2026-06-23', time:'05:00', group:'I', home:'法国',       away:'伊拉克',   round:2, venue:'费城·林肯金融球场' },
  { id:43, date:'2026-06-23', time:'08:00', group:'I', home:'挪威',       away:'塞内加尔', round:2, venue:'纽约·大都会人寿体育场', focus:'⚡哈兰德vs马内' },
  { id:44, date:'2026-06-23', time:'11:00', group:'J', home:'约旦',       away:'阿尔及利亚',round:2,venue:'旧金山·Levi\'s体育场' },
  { id:45, date:'2026-06-24', time:'01:00', group:'K', home:'葡萄牙',     away:'乌兹别克斯坦',round:2,venue:'休斯顿·NRG体育场' },
  { id:46, date:'2026-06-24', time:'04:00', group:'L', home:'英格兰',     away:'加纳',     round:2, venue:'波士顿·吉列体育场' },
  { id:47, date:'2026-06-24', time:'07:00', group:'L', home:'巴拿马',     away:'克罗地亚', round:2, venue:'多伦多·BMO球场' },
  { id:48, date:'2026-06-24', time:'10:00', group:'K', home:'哥伦比亚',   away:'民主刚果', round:2, venue:'墨西哥城·阿兹特克体育场' },
  // === 第3轮 (末轮同组同时开球) ===
  { id:49, date:'2026-06-25', time:'03:00', group:'B', home:'瑞士',       away:'加拿大',   round:3, venue:'温哥华·BC Place', focus:'B组出线决战' },
  { id:50, date:'2026-06-25', time:'03:00', group:'B', home:'波黑',       away:'卡塔尔',   round:3, venue:'西雅图·Lumen球场' },
  { id:51, date:'2026-06-25', time:'06:00', group:'C', home:'苏格兰',     away:'巴西',     round:3, venue:'迈阿密·硬石体育场', focus:'🔥巴西关键战' },
  { id:52, date:'2026-06-25', time:'06:00', group:'C', home:'摩洛哥',     away:'海地',     round:3, venue:'亚特兰大·梅赛德斯-奔驰体育场' },
  { id:53, date:'2026-06-25', time:'09:00', group:'A', home:'捷克',       away:'墨西哥',   round:3, venue:'墨西哥城·阿兹特克体育场', focus:'A组出线决战' },
  { id:54, date:'2026-06-25', time:'09:00', group:'A', home:'南非',       away:'韩国',     round:3, venue:'蒙特雷体育场' },
  { id:55, date:'2026-06-26', time:'04:00', group:'E', home:'厄瓜多尔',   away:'德国',     round:3, venue:'纽约·大都会人寿体育场', focus:'E组头名战' },
  { id:56, date:'2026-06-26', time:'04:00', group:'E', home:'库拉索',     away:'科特迪瓦', round:3, venue:'费城·林肯金融球场' },
  { id:57, date:'2026-06-26', time:'07:00', group:'F', home:'日本',       away:'瑞典',     round:3, venue:'达拉斯·AT&T体育场', focus:'🔥死亡之组终局' },
  { id:58, date:'2026-06-26', time:'07:00', group:'F', home:'突尼斯',     away:'荷兰',     round:3, venue:'堪萨斯城·箭头体育场' },
  { id:59, date:'2026-06-26', time:'10:00', group:'D', home:'土耳其',     away:'美国',     round:3, venue:'洛杉矶·SoFi体育场', focus:'D组头名战' },
  { id:60, date:'2026-06-26', time:'10:00', group:'D', home:'巴拉圭',     away:'澳大利亚', round:3, venue:'旧金山·Levi\'s体育场' },
  { id:61, date:'2026-06-27', time:'03:00', group:'I', home:'挪威',       away:'法国',     round:3, venue:'波士顿·吉列体育场', focus:'⚡哈兰德vs姆巴佩' },
  { id:62, date:'2026-06-27', time:'03:00', group:'I', home:'塞内加尔',   away:'伊拉克',   round:3, venue:'多伦多·BMO球场' },
  { id:63, date:'2026-06-27', time:'08:00', group:'H', home:'佛得角',     away:'沙特阿拉伯',round:3,venue:'休斯顿·NRG体育场' },
  { id:64, date:'2026-06-27', time:'08:00', group:'H', home:'乌拉圭',     away:'西班牙',   round:3, venue:'瓜达拉哈拉体育场', focus:'🔥H组冠军战' },
  { id:65, date:'2026-06-27', time:'11:00', group:'G', home:'新西兰',     away:'比利时',   round:3, venue:'温哥华·BC Place' },
  { id:66, date:'2026-06-27', time:'11:00', group:'G', home:'埃及',       away:'伊朗',     round:3, venue:'西雅图·Lumen球场' },
  { id:67, date:'2026-06-28', time:'05:00', group:'L', home:'巴拿马',     away:'英格兰',   round:3, venue:'纽约·大都会人寿体育场' },
  { id:68, date:'2026-06-28', time:'05:00', group:'L', home:'克罗地亚',   away:'加纳',     round:3, venue:'费城·林肯金融球场' },
  { id:69, date:'2026-06-28', time:'07:30', group:'K', home:'哥伦比亚',   away:'葡萄牙',   round:3, venue:'迈阿密·硬石体育场', focus:'⭐C罗小组终战' },
  { id:70, date:'2026-06-28', time:'07:30', group:'K', home:'民主刚果',   away:'乌兹别克斯坦',round:3,venue:'亚特兰大·梅赛德斯-奔驰体育场' },
  { id:71, date:'2026-06-28', time:'10:00', group:'J', home:'阿尔及利亚', away:'奥地利',   round:3, venue:'堪萨斯城·箭头体育场' },
  { id:72, date:'2026-06-28', time:'10:00', group:'J', home:'约旦',       away:'阿根廷',   round:3, venue:'达拉斯·AT&T体育场', focus:'⭐梅西小组终战' },
  // === 淘汰赛 32强 ===
  { id:73,  date:'2026-06-29', time:'待定', group:'KO', home:'A组第2',          away:'B组第2',          round:'R32',   venue:'洛杉矶·SoFi体育场' },
  { id:74,  date:'2026-06-30', time:'待定', group:'KO', home:'E组第1',          away:'第3(A/B/C/D)',    round:'R32',   venue:'波士顿·吉列体育场' },
  { id:75,  date:'2026-06-30', time:'待定', group:'KO', home:'F组第1',          away:'C组第2',          round:'R32',   venue:'蒙特雷体育场' },
  { id:76,  date:'2026-06-30', time:'待定', group:'KO', home:'C组第1',          away:'F组第2',          round:'R32',   venue:'休斯顿·NRG体育场' },
  { id:77,  date:'2026-07-01', time:'待定', group:'KO', home:'I组第1',          away:'第3(E/F/G/H)',    round:'R32',   venue:'纽约·大都会人寿体育场' },
  { id:78,  date:'2026-07-01', time:'待定', group:'KO', home:'E组第2',          away:'I组第2',          round:'R32',   venue:'达拉斯·AT&T体育场' },
  { id:79,  date:'2026-07-01', time:'待定', group:'KO', home:'A组第1',          away:'第3(C/D/E/F)',    round:'R32',   venue:'墨西哥城·阿兹特克体育场' },
  { id:80,  date:'2026-07-02', time:'待定', group:'KO', home:'L组第1',          away:'第3(I/J/K/L)',    round:'R32',   venue:'亚特兰大·梅赛德斯-奔驰体育场' },
  { id:81,  date:'2026-07-02', time:'待定', group:'KO', home:'D组第1',          away:'第3(B/E/F/G)',    round:'R32',   venue:'旧金山·Levi\'s体育场' },
  { id:82,  date:'2026-07-02', time:'待定', group:'KO', home:'G组第1',          away:'第3(A/E/H/I)',    round:'R32',   venue:'西雅图·Lumen球场' },
  { id:83,  date:'2026-07-03', time:'待定', group:'KO', home:'K组第2',          away:'L组第2',          round:'R32',   venue:'多伦多·BMO球场' },
  { id:84,  date:'2026-07-03', time:'待定', group:'KO', home:'H组第1',          away:'J组第2',          round:'R32',   venue:'洛杉矶·SoFi体育场' },
  { id:85,  date:'2026-07-03', time:'待定', group:'KO', home:'B组第1',          away:'第3(E/F/G/H)',    round:'R32',   venue:'温哥华·BC Place' },
  { id:86,  date:'2026-07-04', time:'待定', group:'KO', home:'J组第1',          away:'H组第2',          round:'R32',   venue:'迈阿密·硬石体育场' },
  { id:87,  date:'2026-07-04', time:'待定', group:'KO', home:'K组第1',          away:'第3(D/E/I/J)',    round:'R32',   venue:'堪萨斯城·箭头体育场' },
  { id:88,  date:'2026-07-04', time:'待定', group:'KO', home:'D组第2',          away:'G组第2',          round:'R32',   venue:'达拉斯·AT&T体育场' },
  // === 16强 ===
  { id:89,  date:'2026-07-05', time:'待定', group:'KO', home:'R32-73胜者',      away:'R32-74胜者',      round:'R16',   venue:'费城·林肯金融球场' },
  { id:90,  date:'2026-07-05', time:'待定', group:'KO', home:'R32-76胜者',      away:'R32-75胜者',      round:'R16',   venue:'休斯顿·NRG体育场' },
  { id:91,  date:'2026-07-06', time:'待定', group:'KO', home:'R32-77胜者',      away:'R32-78胜者',      round:'R16',   venue:'纽约·大都会人寿体育场' },
  { id:92,  date:'2026-07-06', time:'待定', group:'KO', home:'R32-79胜者',      away:'R32-80胜者',      round:'R16',   venue:'墨西哥城·阿兹特克体育场' },
  { id:93,  date:'2026-07-07', time:'待定', group:'KO', home:'R32-82胜者',      away:'R32-81胜者',      round:'R16',   venue:'达拉斯·AT&T体育场' },
  { id:94,  date:'2026-07-07', time:'待定', group:'KO', home:'R32-84胜者',      away:'R32-83胜者',      round:'R16',   venue:'西雅图·Lumen球场' },
  { id:95,  date:'2026-07-08', time:'待定', group:'KO', home:'R32-85胜者',      away:'R32-86胜者',      round:'R16',   venue:'亚特兰大·梅赛德斯-奔驰体育场' },
  { id:96,  date:'2026-07-08', time:'待定', group:'KO', home:'R32-88胜者',      away:'R32-87胜者',      round:'R16',   venue:'温哥华·BC Place' },
  // === 八强 ===
  { id:97,  date:'2026-07-10', time:'待定', group:'KO', home:'R16-89胜者',      away:'R16-90胜者',      round:'QF',    venue:'波士顿·吉列体育场' },
  { id:98,  date:'2026-07-11', time:'待定', group:'KO', home:'R16-91胜者',      away:'R16-92胜者',      round:'QF',    venue:'洛杉矶·SoFi体育场' },
  { id:99,  date:'2026-07-11', time:'待定', group:'KO', home:'R16-93胜者',      away:'R16-94胜者',      round:'QF',    venue:'迈阿密·硬石体育场' },
  { id:100, date:'2026-07-12', time:'待定', group:'KO', home:'R16-95胜者',      away:'R16-96胜者',      round:'QF',    venue:'堪萨斯城·箭头体育场' },
  // === 半决赛 ===
  { id:101, date:'2026-07-15', time:'待定', group:'KO', home:'QF-97胜者',       away:'QF-98胜者',       round:'SF',    venue:'达拉斯·AT&T体育场', focus:'🔥半决赛' },
  { id:102, date:'2026-07-16', time:'待定', group:'KO', home:'QF-99胜者',       away:'QF-100胜者',      round:'SF',    venue:'亚特兰大·梅赛德斯-奔驰体育场', focus:'🔥半决赛' },
  // === 季军赛 ===
  { id:103, date:'2026-07-19', time:'待定', group:'KO', home:'SF-101负者',      away:'SF-102负者',      round:'3RD',   venue:'迈阿密·硬石体育场', focus:'🥉季军战' },
  // === 决赛 ===
  { id:104, date:'2026-07-20', time:'03:00', group:'KO', home:'SF-101胜者',     away:'SF-102胜者',      round:'FINAL', venue:'纽约·大都会人寿体育场', focus:'🏆世界杯决赛' },
];

// ── 赛程渲染 ──────────────────────────────────────────────────────────────────
(function(){
const ROUND_LABEL = {1:'第1轮',2:'第2轮',3:'第3轮(末轮)',
  'R32':'32强','R16':'16强','QF':'八强','SF':'半决赛','3RD':'季军赛','FINAL':'决赛'};
const STAGE_COLOR = {'R32':'#818cf8','R16':'#c084fc','QF':'#fb923c',
  'SF':'#f87171','3RD':'#94a3b8','FINAL':'#fbbf24'};

function toBJ(bjTime) {
  return (!bjTime||bjTime==='待定') ? '待定' : bjTime;
}

// 按球队名（英文）查找结果和预测
const CN2EN = DATA.cn_to_en;
const RES   = DATA.res_by_teams;   // 'home_en|away_en' → {hs,as_,result}
const PRED  = DATA.pred_by_teams;  // 'home_en|away_en' → {pred,actual,dc_ok,ph,pd,pa}

const body = document.getElementById('sched-body');

// 按 UTC 日期分组
const byDate = {};
MATCHES.forEach(m => {
  if(!byDate[m.date]) byDate[m.date]=[];
  byDate[m.date].push(m);
});

body.innerHTML = Object.keys(byDate).sort().map(date => {
  const ms = byDate[date];
  // 数每天已赛场次
  const playedCnt = ms.filter(m => {
    const homeEn = CN2EN[m.home]||m.home;
    const awayEn = CN2EN[m.away]||m.away;
    return !!RES[homeEn+'|'+awayEn];
  }).length;

  const dayLabel = new Date(date+'T12:00:00Z').toLocaleDateString('zh-CN',
    {month:'long',day:'numeric',weekday:'short'});

  const items = ms.map(m => {
    const homeEn  = CN2EN[m.home]||m.home;
    const awayEn  = CN2EN[m.away]||m.away;
    const lkpKey  = homeEn+'|'+awayEn;
    const res     = RES[lkpKey];
    const pred    = PRED[lkpKey];
    const isKO    = m.group==='KO';
    const isPlayed = !!res;
    const bjTime  = toBJ(m.time);
    const roundLbl= ROUND_LABEL[m.round]||String(m.round);
    const stageC  = STAGE_COLOR[m.round]||'var(--muted)';

    // 时间列
    const timeHtml = bjTime==='待定'
      ? `<div class="sched-bj" style="color:var(--muted);font-style:italic">待定</div>`
      : `<div class="sched-bj">${bjTime}</div>`;

    // 阶段/组标签
    const stagePart = isKO
      ? `<span class="sched-stage-badge" style="color:${stageC}">${roundLbl}</span>`
      : `<span class="sched-stage-badge">${roundLbl}</span>`
        + `<span class="sched-grp-badge">${m.group}组</span>`;
    const focusPart = m.focus ? `<span class="sched-focus-tag">${m.focus}</span>` : '';

    // 球队+比分行
    let winnerCls='', scoreHtml='';
    if(isPlayed){
      scoreHtml = `<span class="sched-score">${res.hs} - ${res.as_}</span>`;
      winnerCls = res.result==='H'?'winner-h':res.result==='A'?'winner-a':'';
    } else {
      scoreHtml = `<span class="sched-pending">${isKO?'待定阵容':'待赛'}</span>`;
    }
    const homeCls = isKO&&!isPlayed?'ko-home':'sched-home';
    const awayCls = isKO&&!isPlayed?'ko-away':'sched-away';

    // 预测行
    let predHtml='';
    if(pred){
      if(isPlayed && pred.dc_ok!==null){
        predHtml=`<div class="sched-pred-row"><span class="sched-pred-lbl">DC</span>${badge(pred.pred)}${pred.dc_ok?'<span class="ok"> ✓</span>':'<span class="ng"> ✗</span>'}</div>`;
      } else if(!isPlayed){
        predHtml=`<div class="sched-pred-row"><span class="sched-pred-lbl">DC</span>${badge(pred.pred)}</div>`;
      }
    }

    return `<div class="sched-match${m.focus?' focus-match':''}${isKO?' ko-match':''} ${winnerCls}">
      <div class="sched-time-col">${timeHtml}</div>
      <div class="sched-main">
        <div class="sched-meta">${stagePart}${focusPart}</div>
        <div class="sched-teams-row">
          <span class="${homeCls}">${esc(m.home)}</span>
          <span class="sched-vs">vs</span>
          <span class="${awayCls}">${esc(m.away)}</span>
          ${scoreHtml}
        </div>
        <div class="sched-venue-txt">📍 ${esc(m.venue)}</div>
      </div>
      <div class="sched-right">${predHtml}</div>
    </div>`;
  }).join('');

  return `<div class="sched-day">
    <div class="sched-day-header">
      ${dayLabel}
      <span class="day-badge">${playedCnt}/${ms.length} 已赛</span>
    </div>
    ${items}
  </div>`;
}).join('');
})();
</script>
</body>
</html>"""

html = HTML.replace('__DATA_JSON__', DATA_JSON)
out_path = OUT / 'index.html'
out_path.write_text(html, encoding='utf-8')
size_kb = out_path.stat().st_size / 1024
print(f'  [dashboard] → {out_path}  ({size_kb:.0f} KB)')
print(f'  已结算 {summary["n_settled"]} 场  DC_ACC={summary["dc_acc"]:.1%}  待赛 {summary["n_upcoming"]} 场')
print(f'  积分榜: {len(standings_data)} 个小组  最佳第三: {len(best_thirds)} 队')
print(f'  赛程: MATCHES数组104场（res_by_teams={len(res_by_teams)}已赛，pred_by_teams={len(pred_by_teams)}条预测）')
