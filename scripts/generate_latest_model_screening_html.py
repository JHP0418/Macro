from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "screening_dashboard_latest"


PATHS = {
    "driver_panel": ROOT / "outputs/macro_regime_asset_screener_latest/tables/driver_panel.csv",
    "v4_current": ROOT / "outputs/risk_off_v4_event_label_latest/tables/current_risk_off_v4_state.csv",
    "v4_predictions": ROOT / "outputs/risk_off_v4_event_label_latest/tables/risk_off_v4_walkforward_predictions.csv",
    "v4_comparison": ROOT / "outputs/risk_off_v4_event_label_latest/tables/risk_off_v3_v4_comparison.csv",
    "portfolio_latest": ROOT / "outputs/portfolio_rebalance_validator_latest/tables/latest_constrained_portfolio.csv",
    "portfolio_summary": ROOT / "outputs/portfolio_rebalance_validator_latest/tables/weekly_constrained_portfolio_summary.csv",
    "portfolio_constraints": ROOT / "outputs/portfolio_rebalance_validator_latest/tables/weekly_constraint_validation.csv",
    "v5_1w": ROOT / "outputs/etf_leadership_static_v5_ssl/v5_ssl_1w_rule_entry_predictions.csv",
    "v5_1m": ROOT / "outputs/etf_leadership_static_v5_ssl/v5_ssl_1m_ranker_entry_predictions.csv",
    "v5_basket_1w": ROOT / "outputs/etf_leadership_static_v5_ssl/v5_ssl_current_basket_scores_1w.csv",
    "v5_basket_1m": ROOT / "outputs/etf_leadership_static_v5_ssl/v5_ssl_current_basket_scores_1m.csv",
    "v5_summary": ROOT / "outputs/etf_leadership_static_v5_ssl/v5_ssl_backtest_summary.csv",
    "safe_summary": ROOT / "outputs/ssl_safe_asset_selector_latest/tables/ssl_safe_asset_summary.csv",
    "long_risk_gate": ROOT / "outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_strategy_summary.csv",
    "long_risk_crisis": ROOT / "outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_crisis_windows.csv",
    "long_selection": ROOT / "outputs/long_history_proxy_selection_backtest_latest/tables/proxy_selection_backtest_summary.csv",
    "long_selection_ic": ROOT / "outputs/long_history_proxy_selection_backtest_latest/tables/proxy_selection_rank_ic.csv",
    "gaps_long_lived_summary": ROOT / "outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_backtest_summary.csv",
    "gaps_long_lived_universe": ROOT / "outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_gaps_etf_universe.csv",
    "gaps_long_lived_importance": ROOT / "outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_feature_importance.csv",
    "selective_v3_summary": ROOT / "outputs/selective_leadership_safe_v3_latest/tables/etf_selective_v3_summary.csv",
    "selective_v3_trades": ROOT / "outputs/selective_leadership_safe_v3_latest/tables/etf_selective_v3_trades.csv",
    "safe_v3_summary": ROOT / "outputs/selective_leadership_safe_v3_latest/tables/safe_v3_summary.csv",
    "safe_v3_importance_1m": ROOT / "outputs/selective_leadership_safe_v3_latest/tables/safe_v3_importance_1m.csv",
}


LABELS = {
    "date": "날짜",
    "Date": "날짜",
    "horizon": "기간",
    "risk_off_v4_prob": "V4 확률",
    "risk_off_v4_watch_threshold": "Watch 기준",
    "risk_off_v4_derisk_threshold": "De-risk 기준",
    "risk_off_v4_cash_threshold": "Cash 기준",
    "risk_off_v4_stage": "V4 단계",
    "risk_off_v4_watch": "Watch",
    "risk_off_v4_alert": "De-risk",
    "risk_off_v4_cash": "Cash",
    "risk_off_score": "Risk-Off 원점수",
    "axis1_vol_credit_stress": "변동성/신용 스트레스",
    "axis2_fx_liquidity_stress": "달러/유동성 스트레스",
    "axis3_peak_fragility_stress": "고점취약성 스트레스",
    "dominant_axis": "주도 위험축",
    "model": "모델",
    "auc": "AUC",
    "daily_recall": "일간 포착률",
    "precision": "정밀도",
    "false_alarm_rate": "오경보율",
    "watch_event_recall_20d": "Watch 20일 선제포착",
    "derisk_event_recall_20d": "De-risk 20일 선제포착",
    "cash_event_recall_20d": "Cash 20일 선제포착",
    "caught_loss_ratio": "잡은 손실비율",
    "symbol": "코드",
    "name": "이름",
    "basket": "바스켓",
    "weight": "비중",
    "portfolio_score_pct": "포트폴리오 점수",
    "applied_risk_off_prob": "적용 Risk-Off 확률",
    "applied_risk_off_stage": "적용 단계",
    "target_risk_weight": "목표 위험자산",
    "target_safe_weight": "목표 안전자산",
    "periods": "검증 횟수",
    "start": "시작",
    "end": "종료",
    "cumulative_return": "누적수익률",
    "total_return": "누적수익률",
    "CAGR": "연환산 수익률",
    "MDD": "최대낙폭",
    "Sharpe": "샤프",
    "hit_positive": "양수 비율",
    "hit_rate_positive": "양수 비율",
    "avg_weekly_return": "평균 주간수익률",
    "avg_risk_weight": "평균 위험자산",
    "avg_safe_weight": "평균 안전자산",
    "avg_cash_weight": "평균 현금",
    "total_weight": "총비중",
    "risk_total": "위험자산",
    "safe_total": "안전자산",
    "cash_weight": "현금",
    "max_single": "단일 최대",
    "strategy": "전략",
    "days": "일수",
    "ann_vol": "연환산 변동성",
    "Calmar": "Calmar",
    "positive_day_rate": "양수 일수비율",
    "frequency": "주기",
    "top_k": "Top K",
    "avg_excess_return": "평균 초과수익",
    "hit_rate_excess_positive": "초과수익 승률",
    "score": "점수",
    "target": "타깃",
    "dates": "날짜 수",
    "mean_rank_ic": "평균 Rank IC",
    "median_rank_ic": "중앙 Rank IC",
    "positive_ic_rate": "양수 IC 비율",
    "asset_basket": "자산 바스켓",
    "ranking_group": "랭킹 그룹",
    "etf_count": "ETF 수",
    "top_score": "최고 점수",
    "top_entry_prob": "최고 진입확률",
    "top_etfs": "상위 ETF",
    "top_names": "상위 ETF명",
    "basket_score_0_100": "바스켓 점수",
    "etf_ticker": "ETF 코드",
    "group": "그룹",
    "rule_5d_score": "1주 룰 점수",
    "rule_20d_score": "1개월 룰 점수",
    "ranker_score": "Ranker 점수",
    "entry_prob_20d": "1개월 진입확률",
    "entry_adjusted_20d_score": "1개월 진입보정 점수",
    "Final_Rule_Score_0_100": "리더십 룰 점수",
    "ETF_RS_20D": "20일 상대강도",
    "ETF_RS_60D": "60일 상대강도",
    "ETF_RS_120D": "120일 상대강도",
    "weighted_HP": "구성종목 고점근접도",
    "HP90_share": "고점 90% 이상 비중",
    "MA60_breadth": "60일선 위 비중",
    "MA200_breadth": "200일선 위 비중",
    "effective_N": "유효 구성종목 수",
    "top5_weight_share": "상위5 비중",
    "label": "라벨",
    "coverage": "진입비율",
    "hit_excess": "초과수익 적중률",
    "hit_positive": "상승 적중률",
    "avg_return": "평균수익률",
    "avg_excess": "평균 초과수익",
    "risk_off_periods": "Risk-Off 표본",
    "avg_picked_target": "선택 평균 타깃",
    "avg_safe_target": "안전자산 평균 타깃",
    "beat_safe_average_rate": "안전자산 평균 초과율",
    "topk_overlap_rate": "TopK 겹침률",
    "invested_periods": "투자 구간",
    "entry_threshold": "진입 기준",
    "regression_features": "회귀 피처 수",
    "first_price_date": "최초 가격일",
    "price_obs": "가격 관측치",
    "benchmark_ticker": "기준지수",
    "importance": "중요도",
    "importance_gain": "Gain 중요도",
    "importance_split": "Split 중요도",
    "feature": "피처",
    "threshold": "기준값",
    "valid_objective": "검증 목적함수",
    "valid_sharpe": "검증 샤프",
    "valid_trade_hit_excess": "검증 초과수익 적중률",
    "valid_entry_auc": "진입 AUC",
    "trade_hit_excess": "거래 초과수익 적중률",
    "trade_hit_positive": "거래 상승 적중률",
    "avg_trade_return": "평균 거래 수익률",
    "avg_trade_excess": "평균 거래 초과수익",
    "avg_top_k": "평균 Top-K",
    "picked_return": "선택 수익률",
    "picked_target": "선택 타깃",
    "safe_avg_target": "안전자산 평균 타깃",
    "actual_top_target": "실제 Top 타깃",
    "beat_safe_average": "안전자산 평균 초과",
    "overlap": "실제 Top 겹침률",
    "selected": "선택 ETF",
    "selected_names": "선택 ETF명",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latest-model-only screening dashboard.")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.output)


def generate(output: Path = OUT_DIR) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    charts_dir = output / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    charts = create_latest_charts(data, charts_dir)
    html_text = render(data, charts)
    out = output / "screening_dashboard.html"
    out.write_text(html_text, encoding="utf-8")
    print(out.resolve())
    return out


def load_data() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, path in PATHS.items():
        if not path.exists():
            out[key] = pd.DataFrame()
            continue
        parse_dates = ["date"] if key not in {"driver_panel"} else ["Date"]
        try:
            out[key] = pd.read_csv(path, parse_dates=parse_dates)
        except Exception:
            out[key] = pd.read_csv(path)
    return out


def create_latest_charts(data: dict[str, pd.DataFrame], charts_dir: Path) -> dict[str, Path | list[Path]]:
    panel = build_latest_market_panel(data)
    paths: dict[str, Path | list[Path]] = {}
    if panel.empty:
        return paths
    for months in [12, 6]:
        y = panel[panel["date"].ge(panel["date"].max() - pd.DateOffset(months=months))].copy()
        path = charts_dir / f"latest_v4_market_{months}m.png"
        plot_recent_v4_market(y, months, path)
        paths[f"market_{months}m"] = path
    yearly = create_yearly_v4_charts(panel, charts_dir / "latest_v4_yearly")
    paths.update(yearly)
    return paths


def build_latest_market_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    driver = data["driver_panel"].copy()
    pred = data["v4_predictions"].copy()
    if driver.empty or pred.empty:
        return pd.DataFrame()
    driver = driver.rename(columns={"Date": "date"})
    driver["date"] = pd.to_datetime(driver["date"])
    pred["date"] = pd.to_datetime(pred["date"])
    one_m = pred[pred["horizon"].astype(str).eq("1m")].copy()
    keep = [
        "date",
        "risk_off_v4_prob",
        "risk_off_v4_stage",
        "risk_off_score",
        "axis1_vol_credit_stress",
        "axis2_fx_liquidity_stress",
        "axis3_peak_fragility_stress",
        "risk_3d_dominant_axis",
    ]
    one_m = one_m[[c for c in keep if c in one_m.columns]]
    out = driver.merge(one_m, on="date", how="inner").sort_values("date").ffill()
    return out


def plot_recent_v4_market(frame: pd.DataFrame, months: int, path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    setup_korean_font()
    fig, axes = plt.subplots(4, 1, figsize=(17, 12), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.25, 1.25, 1.05, 1.05]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    ax.plot(frame["date"], frame["risk_off_v4_prob"] * 100, color="#b42318", lw=2.3, label="Risk-Off V4 확률")
    ax.plot(frame["date"], frame["axis1_vol_credit_stress"], color="#7030a0", lw=1.5, label="변동성/신용")
    ax.plot(frame["date"], frame["axis2_fx_liquidity_stress"], color="#2563eb", lw=1.5, label="달러/유동성")
    ax.plot(frame["date"], frame["axis3_peak_fragility_stress"], color="#d97706", lw=1.5, label="고점취약성")
    for y, label in [(25, "Watch"), (30, "De-risk"), (38, "Cash")]:
        ax.axhline(y, color="#9ca3af", ls="--", lw=0.9)
        ax.text(frame["date"].min(), y + 1, label, fontsize=8, color="#666")
    ax.set_ylim(0, 100)
    ax.set_ylabel("최신 V4 위험")
    ax.legend(loc="upper left", ncol=4, fontsize=9)

    plot_normalized(axes[1], frame, {"NASDAQ100": "나스닥100", "SP500": "S&P500", "SOX": "SOX", "RUSSELL2000": "러셀2000"})
    axes[1].set_ylabel("미국지수\n시작=100")
    plot_normalized(axes[2], frame, {"CSI300": "CSI300", "KOSDAQ_KOSPI": "KOSDAQ/KOSPI", "USDKRW": "원/달러", "USDJPY": "달러/엔"})
    axes[2].set_ylabel("아시아/환율\n시작=100")
    plot_normalized(axes[3], frame, {"GOLD": "금", "WTI": "WTI", "COPPER": "구리", "DXY": "달러지수"})
    axes[3].set_ylabel("원자재/달러\n시작=100")
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in axes:
        ax.grid(True, axis="y", color="#e5e7eb")
        ax.set_facecolor("#fbfbfb")
        if ax is not axes[0]:
            ax.legend(loc="upper left", ncol=4, fontsize=9)
    fig.suptitle(f"최신 Risk-Off V4 기준 최근 {months}개월 시장 상태", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_normalized(ax, frame: pd.DataFrame, mapping: dict[str, str]) -> None:
    colors = ["#1f77b4", "#c00000", "#7030a0", "#548235", "#ed7d31", "#595959"]
    for i, (col, label) in enumerate(mapping.items()):
        if col not in frame:
            continue
        s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = s.dropna()
        if valid.empty or valid.iloc[0] == 0:
            continue
        ax.plot(frame.loc[valid.index, "date"], valid / valid.iloc[0] * 100, label=label, color=colors[i % len(colors)], lw=1.8)


def create_yearly_v4_charts(panel: pd.DataFrame, out_dir: Path) -> dict[str, Path | list[Path]]:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    setup_korean_font()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = panel.dropna(subset=["date", "NASDAQ100"]).copy()
    frame["year"] = frame["date"].dt.year
    chart_paths: list[Path] = []
    for year, y in frame.groupby("year"):
        if y.shape[0] < 20:
            continue
        path = out_dir / f"latest_v4_nasdaq_{year}.png"
        plot_yearly_v4(y, int(year), path)
        chart_paths.append(path)
    if not chart_paths:
        return {}
    contact = out_dir / "latest_v4_nasdaq_yearly_contact_sheet.png"
    cols = 2
    rows = int(np.ceil(len(chart_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, max(4.4 * rows, 8)), dpi=120)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, img_path in zip(axes_arr, chart_paths):
        ax.imshow(mpimg.imread(img_path))
        ax.axis("off")
    for ax in axes_arr[len(chart_paths) :]:
        ax.axis("off")
    fig.suptitle("최신 Risk-Off V4 기준 연도별 나스닥100과 위험축", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(contact, bbox_inches="tight")
    plt.close(fig)
    return {"yearly_contact": contact, "yearly_charts": chart_paths}


def plot_yearly_v4(y: pd.DataFrame, year: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_korean_font()
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.2), dpi=140, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax = axes[0]
    nasdaq = pd.to_numeric(y["NASDAQ100"], errors="coerce")
    ax.plot(y["date"], nasdaq / nasdaq.dropna().iloc[0] * 100, color="#1f77b4", lw=2.2, label="나스닥100")
    ax2 = ax.twinx()
    ax2.plot(y["date"], y["risk_off_v4_prob"] * 100, color="#b42318", lw=1.7, label="Risk-Off V4 확률")
    ax2.plot(y["date"], y["axis3_peak_fragility_stress"], color="#d97706", lw=1.2, alpha=0.85, label="고점취약성")
    ax2.set_ylim(0, 100)
    ax.set_ylabel("나스닥=100")
    ax2.set_ylabel("위험점수")
    ax.grid(True, axis="y", color="#e5e7eb")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", ncol=3, fontsize=8)

    ax = axes[1]
    stages = y["risk_off_v4_stage"].astype(str).fillna("Normal")
    colors = {"Normal": "#d9ead3", "Watch": "#fff2cc", "De-risk": "#f4cccc", "Cash": "#d9d2e9"}
    for stage in ["Normal", "Watch", "De-risk", "Cash"]:
        mask = stages.eq(stage)
        ax.fill_between(y["date"], 0, 1, where=mask.to_numpy(), color=colors.get(stage, "#eee"), alpha=0.8, step="pre", label=stage)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("V4 단계")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    fig.suptitle(f"{year} 나스닥100과 최신 Risk-Off V4", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def setup_korean_font() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def latest_etf_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    one_w = data["v5_1w"].copy()
    one_m = data["v5_1m"].copy()
    if one_w.empty:
        latest_1w = pd.DataFrame()
    else:
        one_w["date"] = pd.to_datetime(one_w["date"])
        latest_1w = one_w[one_w["date"].eq(one_w["date"].max())].copy()
        latest_1w = latest_1w.sort_values("rule_5d_score", ascending=False).head(35)
    if one_m.empty:
        latest_1m = pd.DataFrame()
    else:
        one_m["date"] = pd.to_datetime(one_m["date"])
        latest_1m = one_m[one_m["date"].eq(one_m["date"].max())].copy()
        latest_1m = latest_1m.sort_values("entry_adjusted_20d_score", ascending=False).head(35)
    return latest_1w, latest_1m


def render(data: dict[str, pd.DataFrame], charts: dict[str, Path | list[Path]]) -> str:
    v4_current = data["v4_current"]
    v4_comparison = data["v4_comparison"]
    portfolio_latest = data["portfolio_latest"]
    portfolio_summary = data["portfolio_summary"]
    portfolio_constraints = data["portfolio_constraints"]
    latest_1w, latest_1m = latest_etf_tables(data)
    v5_summary = data["v5_summary"].copy()
    if not v5_summary.empty:
        v5_summary["Sharpe"] = pd.to_numeric(v5_summary["Sharpe"], errors="coerce")
        v5_summary = v5_summary.sort_values("Sharpe", ascending=False).head(16)
    long_risk = data["long_risk_gate"].copy()
    if not long_risk.empty:
        long_risk["Sharpe"] = pd.to_numeric(long_risk["Sharpe"], errors="coerce")
        long_risk = long_risk.sort_values("Sharpe", ascending=False)
    long_selection = data["long_selection"].copy()
    if not long_selection.empty:
        long_selection["Sharpe"] = pd.to_numeric(long_selection["Sharpe"], errors="coerce")
        long_selection = long_selection.sort_values("Sharpe", ascending=False).head(18)
    gaps_long_lived_summary = data["gaps_long_lived_summary"].copy()
    if not gaps_long_lived_summary.empty:
        gaps_long_lived_summary["Sharpe"] = pd.to_numeric(gaps_long_lived_summary["Sharpe"], errors="coerce")
        gaps_long_lived_summary = gaps_long_lived_summary.sort_values("Sharpe", ascending=False).head(20)
    gaps_long_lived_universe = data["gaps_long_lived_universe"].copy()
    if not gaps_long_lived_universe.empty:
        gaps_long_lived_universe = gaps_long_lived_universe.sort_values(["group", "etf_ticker"]).head(80)
    gaps_long_lived_importance = data["gaps_long_lived_importance"].copy()
    if not gaps_long_lived_importance.empty:
        if "importance_gain" in gaps_long_lived_importance.columns:
            gaps_long_lived_importance["importance_gain"] = pd.to_numeric(gaps_long_lived_importance["importance_gain"], errors="coerce")
        if "model" in gaps_long_lived_importance.columns:
            gaps_long_lived_importance = gaps_long_lived_importance[gaps_long_lived_importance["model"].astype(str).eq("ranker")]
        gaps_long_lived_importance = gaps_long_lived_importance.sort_values("importance_gain", ascending=False).head(25)
    selective_v3_summary = data["selective_v3_summary"].copy()
    if not selective_v3_summary.empty:
        selective_v3_summary["Sharpe"] = pd.to_numeric(selective_v3_summary["Sharpe"], errors="coerce")
        selective_v3_summary = selective_v3_summary.sort_values(["horizon", "Sharpe"], ascending=[True, False])
    selective_v3_trades = data["selective_v3_trades"].copy()
    if not selective_v3_trades.empty:
        selective_v3_trades["date"] = pd.to_datetime(selective_v3_trades["date"])
        selective_v3_trades = selective_v3_trades[selective_v3_trades["invested"].astype(str).eq("1")].sort_values("date", ascending=False).head(20)
    safe_v3_summary = data["safe_v3_summary"].copy()
    safe_v3_importance_1m = data["safe_v3_importance_1m"].copy()
    if not safe_v3_importance_1m.empty:
        safe_v3_importance_1m["importance_gain"] = pd.to_numeric(safe_v3_importance_1m["importance_gain"], errors="coerce")
        safe_v3_importance_1m = safe_v3_importance_1m.sort_values("importance_gain", ascending=False).head(18)

    asof = ""
    if not v4_current.empty and "date" in v4_current:
        asof = str(pd.to_datetime(v4_current["date"]).max().date())
    cards = build_cards(v4_current, portfolio_summary, long_risk)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>최신 모델 스크리닝 - {esc(asof)}</title>
  <style>
    :root {{ --bg:#f5f7fa; --panel:#fff; --text:#1f2937; --muted:#667085; --line:#d7dde8; --navy:#111827; --red:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:"Malgun Gothic","Segoe UI",Arial,sans-serif; }}
    header {{ padding:24px 30px; background:var(--navy); color:white; }}
    header h1 {{ margin:0 0 6px; font-size:25px; }}
    header p {{ margin:0; color:#cbd5e1; }}
    main {{ max-width:1780px; margin:0 auto; padding:22px 28px 48px; }}
    section {{ margin:0 0 24px; }}
    h2 {{ font-size:19px; margin:0 0 11px; }}
    h3 {{ font-size:15px; margin:14px 0 8px; }}
    .grid {{ display:grid; gap:15px; }}
    .cards {{ grid-template-columns:repeat(6,minmax(150px,1fr)); }}
    .two {{ grid-template-columns:1fr 1fr; align-items:start; }}
    .card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .label {{ color:var(--muted); font-size:12px; margin-bottom:7px; }}
    .value {{ font-size:24px; font-weight:800; }}
    .note,.muted {{ color:var(--muted); font-size:13px; }}
    img {{ width:100%; display:block; border:1px solid var(--line); border-radius:8px; background:white; }}
    .table-wrap {{ overflow:auto; border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); font-size:13px; }}
    th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child,td.left,th.left {{ text-align:left; }}
    th {{ background:#eef2f7; color:#374151; font-weight:700; position:sticky; top:0; }}
    @media (max-width:1200px) {{ .cards,.two {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>최신 모델 전용 스크리닝</h1>
  <p>기준일 {esc(asof)} · 구형 risk_vector/optimizer/analog 산출물 제거 · Risk-Off V4, ETF Leadership V5 SSL, 최신 제약 포트폴리오만 표시</p>
</header>
<main>
  <section class="grid cards">{''.join(card_html(*c) for c in cards)}</section>

  <section>
    <h2>최근 12개월 최신 Risk-Off V4와 시장</h2>
    {img_tag(charts.get("market_12m"))}
  </section>
  <section>
    <h2>최근 6개월 최신 Risk-Off V4와 시장</h2>
    {img_tag(charts.get("market_6m"))}
  </section>
  <section>
    <h2>연도별 나스닥100과 최신 Risk-Off V4</h2>
    <p class="muted">최신 V4 walk-forward 예측이 생성된 2003년 이후만 표시합니다. 1995~2002의 구형 위험점수 이미지는 제거했습니다.</p>
    {img_tag(charts.get("yearly_contact"))}
  </section>

  <section class="grid two">
    <div>
      <h2>Risk-Off V4 현재 상태</h2>
      <div class="table-wrap">{df_to_html(v4_current, ["horizon","date","risk_off_v4_prob","risk_off_v4_watch_threshold","risk_off_v4_derisk_threshold","risk_off_v4_cash_threshold","risk_off_v4_stage","risk_off_v4_watch","risk_off_v4_alert","risk_off_v4_cash","risk_off_score","axis1_vol_credit_stress","axis2_fx_liquidity_stress","axis3_peak_fragility_stress","dominant_axis"])}</div>
    </div>
    <div>
      <h2>Risk-Off V4 검증</h2>
      <div class="table-wrap">{df_to_html(v4_comparison, ["model","horizon","auc","daily_recall","precision","false_alarm_rate","watch_event_recall_20d","derisk_event_recall_20d","cash_event_recall_20d","caught_loss_ratio"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>ETF Leadership V5 SSL: 1주 상위 ETF</h2>
      <div class="table-wrap">{df_to_html(latest_1w, ["date","etf_ticker","name","asset_basket","ranking_group","rule_5d_score","Final_Rule_Score_0_100","ETF_RS_20D","ETF_RS_60D","weighted_HP","HP90_share","MA60_breadth","MA200_breadth","effective_N","top5_weight_share"])}</div>
    </div>
    <div>
      <h2>ETF Leadership V5 SSL: 1개월 상위 ETF</h2>
      <div class="table-wrap">{df_to_html(latest_1m, ["date","etf_ticker","name","asset_basket","ranking_group","entry_adjusted_20d_score","ranker_score","entry_prob_20d","rule_20d_score","Final_Rule_Score_0_100","ETF_RS_20D","ETF_RS_60D","ETF_RS_120D","weighted_HP","MA60_breadth","effective_N"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>ETF Leadership V5 SSL 바스켓 점수</h2>
      <h3>1주</h3>
      <div class="table-wrap">{df_to_html(data["v5_basket_1w"], ["asset_basket","ranking_group","etf_count","top_score","top_entry_prob","top_etfs","top_names","basket_score_0_100"])}</div>
      <h3>1개월</h3>
      <div class="table-wrap">{df_to_html(data["v5_basket_1m"], ["asset_basket","ranking_group","etf_count","top_score","top_entry_prob","top_etfs","top_names","basket_score_0_100"])}</div>
    </div>
    <div>
      <h2>ETF Leadership V5 SSL 백테스트</h2>
      <div class="table-wrap">{df_to_html(v5_summary, ["label","horizon","top_k","periods","coverage","cumulative_return","CAGR","MDD","Sharpe","hit_excess","hit_positive","avg_return","avg_excess","model"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>DB GAPS 장기상장 ETF 리더십 백테스트</h2>
      <p class="muted">2010년 이후 가격이 충분한 DB GAPS ETF 31개를 대상으로 현재 스크리닝 리더십 모델과 같은 구성종목 breadth, 고점근접도, 내부 회귀분석, LightGBM Ranker 계열 피처를 적용했습니다. 과거 holdings는 현재 구성종목 고정 근사입니다.</p>
      <div class="table-wrap">{df_to_html(gaps_long_lived_summary, ["label","horizon","top_k","periods","invested_periods","coverage","cumulative_return","CAGR","MDD","Sharpe","hit_excess","hit_positive","avg_return","avg_excess","model","entry_threshold","regression_features"])}</div>
    </div>
    <div>
      <h2>장기 ETF 리더십 Ranker 피처 중요도</h2>
      <div class="table-wrap">{df_to_html(gaps_long_lived_importance, ["feature","importance_gain","importance_split","model"])}</div>
    </div>
  </section>

  <section>
    <h2>DB GAPS 장기상장 ETF 유니버스</h2>
    <div class="table-wrap">{df_to_html(gaps_long_lived_universe, ["etf_ticker","name","group","benchmark_ticker","first_price_date","price_obs"])}</div>
  </section>

  <section class="grid two">
    <div>
      <h2>선택형 ETF 리더십 V3: Dynamic Top-K / Entry Gate</h2>
      <p class="muted">여러 Top-K와 점수 모델 후보를 만든 뒤 진입확률이 높은 조합만 투자하도록 학습했습니다. 이번 검증에서는 기존 1개월 룰 Top5보다 낮아 운영 기본값이 아니라 개선 후보로 표시합니다.</p>
      <div class="table-wrap">{df_to_html(selective_v3_summary, ["label","horizon","periods","invested_periods","coverage","cumulative_return","CAGR","MDD","Sharpe","trade_hit_excess","trade_hit_positive","avg_trade_return","avg_trade_excess","avg_top_k","threshold","valid_sharpe","valid_trade_hit_excess","valid_entry_auc"])}</div>
    </div>
    <div>
      <h2>선택형 ETF 리더십 최근 투자 신호</h2>
      <div class="table-wrap">{df_to_html(selective_v3_trades, ["date","horizon","score_col","top_k","entry_gate_prob","portfolio_return","excess_return","selected","selected_names"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>안전자산 Macro V3 Ranker 검증</h2>
      <p class="muted">금리, 달러/원달러, VIX/신용, 금, Risk-Off 축과 자산군 상호작용을 넣어 안전자산 선택 모델을 재학습했습니다. 데이터 구간은 현재 확보된 GAPS 주간 패널 기준입니다.</p>
      <div class="table-wrap">{df_to_html(safe_v3_summary, ["horizon","periods","avg_picked_return","avg_picked_target","avg_safe_target","beat_safe_average_rate","topk_overlap_rate"])}</div>
    </div>
    <div>
      <h2>안전자산 Macro V3 1개월 피처 중요도</h2>
      <div class="table-wrap">{df_to_html(safe_v3_importance_1m, ["feature","importance_gain","horizon"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>제약 반영 최신 포트폴리오</h2>
      <div class="table-wrap">{df_to_html(portfolio_latest, ["date","symbol","name","basket","weight","portfolio_score_pct","applied_risk_off_prob","applied_risk_off_stage","target_risk_weight","target_safe_weight"])}</div>
    </div>
    <div>
      <h2>포트폴리오 검증과 제약</h2>
      <div class="table-wrap">{df_to_html(portfolio_summary, ["periods","start","end","cumulative_return","CAGR","MDD","Sharpe","hit_positive","avg_weekly_return","avg_risk_weight","avg_safe_weight","avg_cash_weight"])}</div>
      <h3>최근 제약 검증</h3>
      <div class="table-wrap">{df_to_html(portfolio_constraints.tail(10), ["date","total_weight","risk_total","safe_total","cash_weight","max_single"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>장기 프록시 Risk-Off 게이트</h2>
      <div class="table-wrap">{df_to_html(long_risk, ["strategy","start","end","days","total_return","CAGR","ann_vol","Sharpe","MDD","Calmar","positive_day_rate"])}</div>
    </div>
    <div>
      <h2>장기 프록시 ETF 리더십/안전자산 선택</h2>
      <div class="table-wrap">{df_to_html(long_selection, ["strategy","frequency","top_k","start","end","periods","total_return","CAGR","Sharpe","MDD","hit_rate_positive","avg_excess_return","hit_rate_excess_positive"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>장기 프록시 Rank IC</h2>
      <div class="table-wrap">{df_to_html(data["long_selection_ic"], ["score","target","dates","mean_rank_ic","median_rank_ic","positive_ic_rate"])}</div>
    </div>
    <div>
      <h2>SSL 안전자산 선택 검증</h2>
      <div class="table-wrap">{df_to_html(data["safe_summary"], ["horizon","risk_off_periods","avg_picked_target","avg_safe_target","beat_safe_average_rate","topk_overlap_rate"])}</div>
    </div>
  </section>
</main>
</body>
</html>"""


def build_cards(v4_current: pd.DataFrame, portfolio_summary: pd.DataFrame, long_risk: pd.DataFrame) -> list[tuple[str, str, str]]:
    cards: list[tuple[str, str, str]] = []
    for horizon in ["1w", "1m"]:
        row = v4_current[v4_current["horizon"].astype(str).eq(horizon)].tail(1) if not v4_current.empty else pd.DataFrame()
        if not row.empty:
            r = row.iloc[0]
            cards.append((f"Risk-Off V4 {horizon}", esc(r.get("risk_off_v4_stage", "")), f"확률 {pct(r.get('risk_off_v4_prob'))}"))
    if not portfolio_summary.empty:
        r = portfolio_summary.iloc[0]
        cards += [
            ("GAPS 포트폴리오 CAGR", pct(r.get("CAGR")), f"MDD {pct(r.get('MDD'))}"),
            ("GAPS 포트폴리오 Sharpe", num(r.get("Sharpe")), f"누적 {pct(r.get('cumulative_return'))}"),
        ]
    if not long_risk.empty:
        r = long_risk.sort_values("Sharpe", ascending=False).iloc[0]
        cards += [
            ("장기 게이트 최고 모델", esc(r.get("strategy", "")), f"Sharpe {num(r.get('Sharpe'))}"),
            ("장기 게이트 CAGR", pct(r.get("CAGR")), f"MDD {pct(r.get('MDD'))}"),
        ]
    return cards


def card_html(label: str, value: str, note: str) -> str:
    return f'<div class="card"><div class="label">{esc(label)}</div><div class="value">{value}</div><div class="note">{esc(note)}</div></div>'


def img_tag(path: object) -> str:
    if not path:
        return "<p class='muted'>이미지가 없습니다.</p>"
    p = Path(path)
    if not p.exists():
        return "<p class='muted'>이미지가 없습니다.</p>"
    return f'<img src="{esc(p.resolve().as_uri())}" alt="{esc(p.name)}">'


def df_to_html(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "<p class='muted'>데이터가 없습니다.</p>"
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "<p class='muted'>표시할 컬럼이 없습니다.</p>"
    left_cols = {"model", "strategy", "symbol", "name", "basket", "asset_basket", "ranking_group", "top_etfs", "top_names", "etf_ticker", "score", "target", "label", "dominant_axis", "risk_off_v4_stage"}
    rows = ["<table><thead><tr>"]
    for c in cols:
        cls = "left" if c in left_cols else ""
        rows.append(f'<th class="{cls}">{esc(LABELS.get(c, c))}</th>')
    rows.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        rows.append("<tr>")
        for c in cols:
            cls = "left" if c in left_cols else ""
            rows.append(f'<td class="{cls}">{fmt_value(c, row.get(c))}</td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def fmt_value(col: str, value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, str):
        return esc(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if any(k in col.lower() for k in ["return", "prob", "rate", "ratio", "cagr", "mdd", "weight", "coverage", "hit", "precision", "recall", "alarm", "auc", "brier"]):
            return f"{float(value):.2%}"
        return f"{float(value):,.3f}"
    return esc(value)


def pct(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return ""


def num(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return ""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
