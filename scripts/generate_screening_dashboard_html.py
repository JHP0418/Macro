from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "screening_dashboard_latest"

KOREAN_LABELS = {
    "Date": "날짜",
    "date": "날짜",
    "symbol": "종목코드",
    "name": "이름",
    "group": "자산군",
    "rank": "순위",
    "basket_rank": "바스켓 순위",
    "n": "개수",
    "basket": "바스켓",
    "asset_count": "ETF 수",
    "basket_score_0_100": "바스켓 점수",
    "basket_upside_prob_1w": "바스켓 1주 상승확률",
    "basket_upside_prob_4w": "바스켓 1개월 상승확률",
    "basket_prob_1w": "바스켓 1주 확률",
    "basket_prob_1m": "바스켓 1개월 확률",
    "basket_return_20d": "바스켓 20일 수익률",
    "basket_risk_penalty": "바스켓 위험 패널티",
    "basket_realized_return_1w": "바스켓 실제 1주",
    "basket_realized_return_1m": "바스켓 실제 1개월",
    "pred_top_avg_return": "예측 1등 평균수익률",
    "actual_top1_in_pred_top3_rate": "실제 1등 예측 Top3 포함률",
    "top3_overlap_rate": "Top3 겹침률",
    "top_symbols": "상위 코드",
    "top_names": "상위 ETF",
    "score_0_100": "종합점수",
    "institutional_score_0_100": "기관형 점수",
    "calibrated_prob_1w": "보정 1주 확률",
    "calibrated_prob_4w": "보정 1개월 확률",
    "realized_return_1w": "실제 1주 수익률",
    "realized_return_4w": "실제 1개월 수익률",
    "avg_score": "평균점수",
    "top_score": "최고점수",
    "upside_prob_1w": "1주 상승확률",
    "upside_prob_4w": "4주 상승확률",
    "avg_upside_1w": "평균 1주 상승확률",
    "avg_upside_4w": "평균 4주 상승확률",
    "technical_score": "기술적 점수",
    "avg_technical": "평균 기술점수",
    "driver_fit_score": "매크로 적합도",
    "avg_driver_fit": "평균 매크로 적합도",
    "return_20d": "20일 수익률",
    "drawdown_252d": "1년 고점대비 낙폭",
    "risk_penalty": "위험 패널티",
    "max_risk_penalty": "최대 위험 패널티",
    "similarity_distance": "유사도 거리",
    "similarity_rank": "유사도 순위",
    "similarity_score": "유사도 점수",
    "risk_archetype": "위험 유형",
    "risk_phase": "위험 단계",
    "dominant_risk_vector": "주도 위험축",
    "next_1w_nasdaq_return": "이후 나스닥 1주",
    "next_1m_nasdaq_return": "이후 나스닥 1개월",
    "next_1w_sp500_return": "이후 S&P500 1주",
    "next_1m_sp500_return": "이후 S&P500 1개월",
    "next_1w_sox_return": "이후 SOX 1주",
    "next_1m_sox_return": "이후 SOX 1개월",
    "next_1w_russell2000_return": "이후 러셀2000 1주",
    "next_1m_russell2000_return": "이후 러셀2000 1개월",
    "next_1w_gold_return": "이후 금 1주",
    "next_1m_gold_return": "이후 금 1개월",
    "next_1w_wti_return": "이후 WTI 1주",
    "next_1m_wti_return": "이후 WTI 1개월",
    "next_1w_dxy_return": "이후 달러지수 1주",
    "next_1m_dxy_return": "이후 달러지수 1개월",
    "next_1w_usdkrw_return": "이후 원/달러 1주",
    "next_1m_usdkrw_return": "이후 원/달러 1개월",
    "macro_liquidity_axis_x": "X축 유동성/신용/환율",
    "market_breakdown_axis_y": "Y축 주가/변동성 붕괴",
    "external_supply_axis_z": "Z축 대외/원자재 충격",
    "peak_fragility": "고점 취약성",
    "analog_macro_risk": "유사환경 위험",
    "correction_pressure": "조정 압력",
    "rai_appetite_stress": "RAI 위험선호 붕괴",
    "universe_breadth_stress": "ETF 유니버스 breadth 악화",
    "safe_rotation_stress": "안전자산 로테이션",
    "RAI_z": "RAI z-score",
    "RAI_level_0_100": "RAI 수준",
    "RAI_shock_score": "RAI 충격점수",
    "RAI_overheat_score": "RAI 과열점수",
    "ETF_breadth_shock_score": "ETF breadth 충격점수",
    "correction_pressure_score_0_100": "조정 압력점수",
    "correction_pressure_state": "조정 압력 상태",
    "correction_1w_drop_prob": "1주 급락확률",
    "correction_1m_prob": "1개월 조정확률",
    "delayed_correction_prob": "지연 조정확률",
    "analog_risk_score_0_100": "유사환경 위험점수",
    "analog_state": "유사환경 상태",
    "analog_down_prob_1w_model": "유사환경 1주 하락확률",
    "analog_down_prob_1m_model": "유사환경 1개월 하락확률",
    "analog_tail_prob_1m_model": "유사환경 1개월 급락확률",
    "liquidity_credit_stress": "유동성/신용 스트레스",
    "equity_breakdown_stress": "주식 붕괴 스트레스",
    "fx_external_stress": "환율/대외 스트레스",
    "volatility_stress": "변동성 스트레스",
    "driver": "선행지표",
    "level": "현재값",
    "change_5d": "5일 변화",
    "change_20d": "20일 변화",
    "z_60d": "60일 z-score",
    "riskon_score": "Risk-On 기여도",
    "horizon": "기간",
    "prediction_used": "사용 예측값",
    "risk_off_weeks": "Risk-Off 주수",
    "top1_exact_hit_rate": "1등 적중률",
    "actual_top1_in_pred_top5_rate": "실제 1등 Top5 포함률",
    "top5_overlap_rate": "Top5 겹침률",
    "pred_top1_avg_return": "예측 1등 평균수익률",
    "pred_top5_avg_return": "예측 Top5 평균수익률",
    "safe_avg_return": "안전자산 평균수익률",
    "actual_top1_avg_return": "실제 1등 평균수익률",
    "actual_top5_avg_return": "실제 Top5 평균수익률",
    "criterion": "판단기준",
    "target": "목표",
    "precision": "정밀도",
    "recall": "포착률",
    "false_alarm_rate": "오경보율",
    "miss_rate": "미포착률",
    "accuracy": "정확도",
    "signal_rate": "신호발생률",
    "actual_down_rate": "실제하락률",
    "samples": "표본수",
    "positive_rate": "타깃 발생률",
    "threshold": "임계값",
    "roc_auc": "ROC-AUC",
    "brier": "Brier 점수",
}

KOREAN_LABELS.update(
    {
        "model": "모델",
        "description": "설명",
        "false_alarm_rate": "오경보율",
        "avg_forward_return_when_signal": "신호 후 평균수익률",
        "avg_forward_drawdown_when_signal": "신호 후 평균낙폭",
        "risk_off_avoidance_score": "위험회피 최적점수",
        "peak_correction_score": "고점/조정 점수",
        "crash_sentinel_score": "급락 Sentinel 점수",
        "optimized_action": "최적화 액션",
        "bucket": "오경보 분류",
        "count": "건수",
        "rate": "비율",
        "avg_forward_return": "평균 이후수익률",
        "avg_forward_drawdown": "평균 이후낙폭",
        "top1_hit_rate": "Top1 적중률",
        "top3_hit_rate": "Top3 적중률",
        "pred_top1_avg_return": "예측 Top1 평균수익률",
        "pred_top3_avg_return": "예측 Top3 평균수익률",
        "safe_universe_avg_return": "안전자산 평균수익률",
        "avg_topk_return": "TopK 평균수익률",
        "avg_universe_return": "유니버스 평균수익률",
        "topk_overlap_rate": "TopK 겹침률",
        "top1_exact_hit_rate": "Top1 정확 적중률",
        "actual_top1_in_pred_topk_rate": "실제 Top1 예측TopK 포함률",
        "family": "자산군 Calibration",
        "raw_brier": "원확률 Brier",
        "calibrated_brier": "보정확률 Brier",
        "raw_avg_prob": "원확률 평균",
        "calibrated_avg_prob": "보정확률 평균",
        "actual_rate": "실제 발생률",
        "model_regime": "모델 Regime",
        "safe_score": "안전자산 점수",
        "model_version": "모델 버전",
        "leadership_rank": "리더십 순위",
        "etf_ticker": "ETF 코드",
        "leadership_score_0_100": "리더십 점수",
        "Final_Rule_Score": "룰 점수",
        "Final_Rule_Score_0_100": "룰 점수",
        "holding_logic": "구성종목 처리",
        "ETF_RS_20D": "20일 상대강도",
        "ETF_RS_60D": "60일 상대강도",
        "ETF_RS_120D": "120일 상대강도",
        "RS_positive_share": "구성종목 RS 양수비중",
        "MA60_breadth": "60일선 위 비중",
        "effective_N": "유효 구성종목 수",
        "top5_weight_share": "상위5 비중",
        "weight_source_note": "비중 출처",
        "strategy_mode": "전략 모드",
        "score_label": "점수 방식",
        "top_k": "선택 ETF 수",
        "threshold_col": "진입 필터",
        "threshold": "진입 기준",
        "risk_block_limit": "허용 위험신호 수",
        "invested_periods": "진입 횟수",
        "coverage": "진입 비율",
        "Sharpe": "샤프",
        "trade_hit_ratio_excess_gt0": "초과수익 적중률",
        "trade_positive_return_ratio": "상승 적중률",
        "CAGR": "연환산 수익률",
        "MDD": "최대낙폭",
        "optimized_action": "현재 액션",
        "risk_off_v4_prob": "Risk-Off V4 확률",
        "risk_off_v4_watch_threshold": "Watch 기준",
        "risk_off_v4_derisk_threshold": "De-risk 기준",
        "risk_off_v4_cash_threshold": "Cash 기준",
        "risk_off_v4_stage": "V4 단계",
        "risk_off_v4_watch": "Watch 신호",
        "risk_off_v4_alert": "De-risk 신호",
        "risk_off_v4_cash": "Cash 신호",
        "axis1_vol_credit_stress": "변동성/신용 스트레스",
        "axis2_fx_liquidity_stress": "달러/유동성 스트레스",
        "axis3_peak_fragility_stress": "고점취약성 스트레스",
        "dominant_axis": "주된 위험축",
        "watch_event_recall_20d": "Watch 선제포착률",
        "derisk_event_recall_20d": "De-risk 선제포착률",
        "cash_event_recall_20d": "Cash 선제포착률",
        "caught_loss_ratio": "잡은 손실비율",
        "watch_alert_rate": "Watch 발생률",
        "derisk_alert_rate": "De-risk 발생률",
        "cash_alert_rate": "Cash 발생률",
        "daily_recall": "일간 Recall",
        "alert_rate": "신호발생률",
        "weight": "편입비중",
        "applied_risk_off_prob": "적용 Risk-Off 확률",
        "applied_risk_off_stage": "적용 Risk-Off 단계",
        "target_risk_weight": "목표 위험자산",
        "target_safe_weight": "목표 안전자산",
        "periods": "검증 주수",
        "start": "시작일",
        "end": "종료일",
        "cumulative_return": "누적수익률",
        "hit_positive": "주간 양수비율",
        "avg_weekly_return": "평균 주간수익률",
        "avg_risk_weight": "평균 위험자산",
        "avg_safe_weight": "평균 안전자산",
        "avg_cash_weight": "평균 현금",
        "total_weight": "총비중",
        "risk_total": "위험자산",
        "safe_total": "안전자산",
        "cash_weight": "현금",
        "max_single": "개별 최대",
    }
)

GROUP_LABELS = {
    "Korea semiconductor": "한국 반도체",
    "Korea broad equity": "한국 대표지수",
    "Korea cyclical": "한국 경기민감",
    "Korea growth": "한국 성장주",
    "Korea IT": "한국 IT",
    "Korea value": "한국 가치주",
    "Korea defensive": "한국 방어주",
    "Korea bonds": "한국 채권",
    "US growth": "미국 성장주",
    "US semiconductor": "미국 반도체",
    "US broad equity": "미국 대표지수",
    "US cyclical": "미국 경기민감",
    "US REIT": "미국 리츠",
    "US IG bonds": "미국 투자등급채",
    "US long bonds": "미국 장기채",
    "US high yield": "미국 하이일드",
    "Japan equity": "일본 주식",
    "China equity": "중국 본토",
    "China/HK growth": "중국/홍콩 성장",
    "India/EM": "인도/신흥국",
    "Gold": "금",
    "USD cash": "달러/달러성 현금",
    "Cash/short bonds": "현금/단기채",
}

DRIVER_LABELS = {
    "NASDAQ100": "나스닥100",
    "SP500": "S&P500",
    "SOX": "필라델피아 반도체",
    "RUSSELL2000": "러셀2000",
    "DXY": "달러 인덱스",
    "USDKRW": "원/달러",
    "USDCNH": "달러/위안",
    "USDJPY": "달러/엔",
    "US10Y": "미국 10년 금리",
    "US2Y": "미국 2년 금리",
    "US10Y_REAL": "미국 10년 실질금리",
    "HY_OAS": "미국 하이일드 스프레드",
    "IG_OAS": "미국 투자등급 스프레드",
    "VIX": "VIX",
    "VXN": "VXN",
    "MOVE": "MOVE 채권변동성",
    "COPPER": "구리",
    "WTI": "WTI 유가",
    "GOLD": "금",
    "COPPER_GOLD": "구리/금 비율",
    "CSI300": "CSI300",
    "HANGSENG_TECH": "항셍테크",
    "KOSDAQ_KOSPI": "코스닥/코스피",
}

REGIME_COLORS = {
    "Calm Risk-On": "#d9ead3",
    "Peak Warning": "#fff2cc",
    "Peak Fragility": "#eadcf8",
    "Mixed/Transition": "#e7edf7",
    "Credit/Liquidity Shock": "#f4cccc",
    "FX/External Stress": "#d9eaf7",
    "Inflation/Supply Shock": "#f9cb9c",
    "Technical Equity Breakdown": "#f4cccc",
    "Cyclical/China Stress": "#eadcf8",
    "Full Risk-Off": "#d9d2e9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Korean HTML screening dashboard.")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--top-assets", type=int, default=50)
    parser.add_argument("--similar-days", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The screening page is now latest-model-only. Keep this legacy entry point
    # so existing commands do not regenerate the old mixed V3/V4 dashboard.
    from generate_latest_model_screening_html import generate

    generate(args.output)


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "risk_vector": ROOT / "outputs/risk_vector_dashboard_latest/tables/daily_risk_vector.csv",
        "asset_scores": ROOT / "outputs/macro_regime_asset_screener_latest/tables/current_asset_scores.csv",
        "basket_scores": ROOT / "outputs/macro_regime_asset_screener_latest/tables/current_basket_scores.csv",
        "driver_state": ROOT / "outputs/macro_regime_asset_screener_latest/tables/driver_state.csv",
        "driver_panel": ROOT / "outputs/macro_regime_asset_screener_latest/tables/driver_panel.csv",
        "safe_eval": ROOT / "outputs/risk_off_short_horizon_model_latest/tables/safe_asset_top_pick_1w_3w_evaluation.csv",
        "risk_confusion": ROOT / "outputs/risk_off_short_horizon_model_latest/tables/severe_weekly_risk_off_confusion_matrix.csv",
        "peak_validation": ROOT / "outputs/peak_fragility_model_latest/tables/peak_fragility_validation.csv",
        "analog_validation": ROOT / "outputs/analog_macro_risk_model_latest/tables/analog_macro_validation.csv",
        "optimizer_current": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/current_optimized_risk_signal.csv",
        "optimizer_calibration": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/risk_probability_calibration_validation.csv",
        "optimizer_threshold": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/risk_threshold_optimization.csv",
        "optimizer_high_conf": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/high_confidence_rule_validation.csv",
        "optimizer_false_alarm": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/false_alarm_taxonomy.csv",
        "optimizer_safe": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/safe_asset_selector_validation.csv",
        "optimizer_current_safe": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/current_safe_asset_recommendations.csv",
        "optimizer_rank": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/fast_weekly_rank_summary.csv",
        "weekly_basket_summary": ROOT / "outputs/weekly_screening_rank_backtest_latest/tables/weekly_basket_backtest_summary.csv",
        "weekly_basket_current": ROOT / "outputs/weekly_screening_rank_backtest_latest/tables/latest_basket_scores.csv",
        "weekly_basket_constituents": ROOT / "outputs/weekly_screening_rank_backtest_latest/tables/latest_basket_constituent_scores.csv",
        "etf_leadership_current": ROOT / "outputs/etf_leadership_current/current_etf_leadership_scores.csv",
        "etf_leadership_coverage": ROOT / "outputs/etf_leadership_current/etf_holdings_ohlcv_coverage.csv",
        "etf_leadership_failed": ROOT / "outputs/etf_leadership_current/failed_component_ohlcv_tickers.csv",
        "etf_leadership_performance": ROOT / "outputs/etf_leadership_current/etf_leadership_performance_summary.csv",
        "etf_leadership_roadmap": ROOT / "outputs/etf_leadership_current/institutional_upgrade_roadmap.csv",
        "etf_leadership_selective": ROOT / "outputs/etf_leadership_selective_strategy_risk_gated/test_summary.csv",
        "current_optimized_risk_signal": ROOT / "outputs/risk_model_walkforward_optimizer_latest/tables/current_optimized_risk_signal.csv",
        "risk_off_v4_current": ROOT / "outputs/risk_off_v4_event_label_latest/tables/current_risk_off_v4_state.csv",
        "risk_off_v4_comparison": ROOT / "outputs/risk_off_v4_event_label_latest/tables/risk_off_v3_v4_comparison.csv",
        "portfolio_latest": ROOT / "outputs/portfolio_rebalance_validator_latest/tables/latest_constrained_portfolio.csv",
        "portfolio_summary": ROOT / "outputs/portfolio_rebalance_validator_latest/tables/weekly_constrained_portfolio_summary.csv",
        "portfolio_constraints": ROOT / "outputs/portfolio_rebalance_validator_latest/tables/weekly_constraint_validation.csv",
        "long_proxy_summary": ROOT / "outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_strategy_summary.csv",
        "long_proxy_crisis": ROOT / "outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_crisis_windows.csv",
        "long_proxy_exposure": ROOT / "outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_v4_stage_exposure.csv",
        "long_selection_summary": ROOT / "outputs/long_history_proxy_selection_backtest_latest/tables/proxy_selection_backtest_summary.csv",
        "long_selection_ic": ROOT / "outputs/long_history_proxy_selection_backtest_latest/tables/proxy_selection_rank_ic.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            out[name] = pd.DataFrame()
            continue
        parse_dates = ["Date"] if name in {"risk_vector", "driver_panel", "optimizer_current"} else None
        if name in {"risk_off_v4_current", "portfolio_latest", "portfolio_constraints"}:
            parse_dates = ["date"]
        out[name] = pd.read_csv(path, parse_dates=parse_dates)
    return out


def create_recent_market_charts(data: dict[str, pd.DataFrame], charts_dir: Path) -> dict[str, object]:
    import matplotlib.pyplot as plt

    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    panel = build_market_panel(data)
    paths: dict[str, object] = {}
    if panel.empty:
        return paths
    for months in (12, 6):
        cut = panel["Date"].max() - pd.DateOffset(months=months)
        y = panel[panel["Date"].ge(cut)].copy()
        if y.empty:
            continue
        path = charts_dir / f"market_risk_recent_{months}m.png"
        plot_recent_market_chart(y, months, path)
        paths[f"market_{months}m"] = path
    paths.update(create_yearly_nasdaq_risk_regime_charts(panel, charts_dir / "yearly_nasdaq_risk_regime"))
    return paths


def build_market_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rv = data["risk_vector"].copy()
    dp = data["driver_panel"].copy()
    if rv.empty or dp.empty:
        return pd.DataFrame()
    base_cols = [
        "Date",
        "composite_vector_risk",
        "risk_off_score",
        "peak_fragility",
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "risk_archetype",
        "risk_phase",
        "analog_macro_risk",
        "correction_pressure",
        "rai_appetite_stress",
        "universe_breadth_stress",
        "safe_rotation_stress",
        "analog_down_prob_1w_model",
        "analog_down_prob_1m_model",
        "analog_tail_prob_1m_model",
        "correction_1w_drop_prob",
        "correction_1m_prob",
        "delayed_correction_prob",
    ]
    out = rv[[c for c in base_cols if c in rv]].merge(dp, on="Date", how="left", suffixes=("", "_driver"))
    yahoo_indices = load_yahoo_indices()
    if not yahoo_indices.empty:
        out = out.merge(yahoo_indices, on="Date", how="left")
    for symbol, label in [("069500.KS", "한국 KOSPI200"), ("238720.KS", "일본 주식")]:
        if label in out and out[label].notna().any():
            continue
        px = load_cached_close(symbol, label)
        if not px.empty:
            out = out.merge(px, on="Date", how="left")
    out = out.sort_values("Date").ffill()
    return out


def load_cached_close(symbol: str, label: str) -> pd.DataFrame:
    path = ROOT / ".cache/prices" / f"{symbol.replace('.', '_')}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Date", label])
    px = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    return px[["Date", "Close"]].rename(columns={"Close": label})


def load_yahoo_indices() -> pd.DataFrame:
    try:
        import yfinance as yf

        raw = yf.download(["^KS11", "^N225"], period="15mo", progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        out = pd.DataFrame({"Date": pd.to_datetime(close.index)})
        if "^KS11" in close:
            out["한국 KOSPI"] = pd.to_numeric(close["^KS11"], errors="coerce")
            out["한국 KOSPI200"] = out["한국 KOSPI"]
        if "^N225" in close:
            out["일본 Nikkei225"] = pd.to_numeric(close["^N225"], errors="coerce")
            out["일본 주식"] = out["일본 Nikkei225"]
        return out.dropna(how="all", subset=[c for c in out.columns if c != "Date"])
    except Exception:
        return pd.DataFrame()


def plot_recent_market_chart(y: pd.DataFrame, months: int, path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.15, 1.3, 1.2, 1.05]})
    fig.patch.set_facecolor("white")
    colors = {
        "종합 위험": "#111827",
        "Risk-Off": "#c00000",
        "고점 취약성": "#7030a0",
        "유사환경 위험": "#0f766e",
        "조정 압력": "#d97706",
        "나스닥100": "#1f77b4",
        "S&P500": "#2f5597",
        "SOX 반도체": "#7030a0",
        "한국 KOSPI200": "#c00000",
        "일본 주식": "#548235",
        "금": "#c9a227",
        "WTI 유가": "#ed7d31",
        "구리": "#a65e2e",
        "달러 인덱스": "#595959",
        "원/달러": "#4472c4",
    }

    ax = axes[0]
    ax.plot(y["Date"], y["composite_vector_risk"], label="종합 위험", color=colors["종합 위험"], lw=2.2)
    ax.plot(y["Date"], y["risk_off_score"], label="Risk-Off", color=colors["Risk-Off"], lw=1.8)
    ax.plot(y["Date"], y["peak_fragility"], label="고점 취약성", color=colors["고점 취약성"], lw=1.8)
    if "analog_macro_risk" in y:
        ax.plot(y["Date"], y["analog_macro_risk"], label="유사환경 위험", color=colors["유사환경 위험"], lw=1.8)
    if "correction_pressure" in y:
        ax.plot(y["Date"], y["correction_pressure"], label="조정 압력", color=colors["조정 압력"], lw=1.8)
    for level, label in [(35, "주의"), (50, "위험"), (65, "현금")]:
        ax.axhline(level, color="#bfbfbf", lw=0.9, ls="--")
        ax.text(y["Date"].min(), level + 1, label, fontsize=9, color="#666666")
    ax.set_ylim(0, 100)
    ax.set_ylabel("위험 점수")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, axis="y", color="#e6e6e6")

    ax = axes[1]
    plot_normalized(ax, y, {"NASDAQ100": "나스닥100", "SP500": "S&P500", "SOX": "SOX 반도체"}, colors)
    ax.set_ylabel("미국지수\n(시작=100)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    ax = axes[2]
    plot_normalized(ax, y, {"한국 KOSPI200": "한국 KOSPI200", "일본 주식": "일본 주식", "RUSSELL2000": "러셀2000"}, colors)
    ax.set_ylabel("한국/일본/중소형\n(시작=100)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    ax = axes[3]
    plot_normalized(ax, y, {"GOLD": "금", "WTI": "WTI 유가", "COPPER": "구리", "DXY": "달러 인덱스", "USDKRW": "원/달러"}, colors)
    ax.set_ylabel("원자재/환율\n(시작=100)")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    for ax in axes:
        ax.set_facecolor("#fbfbfb")
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.8)
    fig.suptitle(f"최근 {months}개월 시장 위험 점수와 주요 지수/원자재", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_normalized(ax, frame: pd.DataFrame, mapping: dict[str, str], colors: dict[str, str]) -> None:
    for col, label in mapping.items():
        if col not in frame:
            continue
        s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty or s.iloc[0] == 0:
            continue
        dates = frame.loc[s.index, "Date"]
        ax.plot(dates, s / s.iloc[0] * 100.0, label=label, color=colors.get(label, None), lw=1.8)


def create_yearly_nasdaq_risk_regime_charts(panel: pd.DataFrame, yearly_dir: Path) -> dict[str, object]:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    yearly_dir.mkdir(parents=True, exist_ok=True)
    required = {"Date", "NASDAQ100"}
    if panel.empty or not required.issubset(panel.columns):
        return {}

    frame = panel.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["NASDAQ100"] = pd.to_numeric(frame["NASDAQ100"], errors="coerce")
    frame = frame.dropna(subset=["Date", "NASDAQ100"]).sort_values("Date")
    if frame.empty:
        return {}

    chart_paths: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    for year, y in frame.groupby(frame["Date"].dt.year):
        if year < 1995 or y["NASDAQ100"].notna().sum() < 20:
            continue
        path = yearly_dir / f"nasdaq_risk_regime_{year}.png"
        plot_yearly_nasdaq_risk_regime(y.copy(), int(year), path)
        chart_paths.append(path)
        risk_mean = safe_float(pd.to_numeric(y.get("composite_vector_risk", pd.Series(dtype=float)), errors="coerce").mean())
        risk_max = safe_float(pd.to_numeric(y.get("composite_vector_risk", pd.Series(dtype=float)), errors="coerce").max())
        peak_max = safe_float(pd.to_numeric(y.get("peak_fragility", pd.Series(dtype=float)), errors="coerce").max())
        regime_col = "risk_archetype" if "risk_archetype" in y else "risk_phase"
        dominant_regime = ""
        if regime_col in y and y[regime_col].notna().any():
            dominant_regime = translate_state(y[regime_col].dropna().astype(str).mode().iloc[0])
        summary_rows.append(
            {
                "year": int(year),
                "nasdaq_return": y["NASDAQ100"].iloc[-1] / y["NASDAQ100"].iloc[0] - 1.0,
                "avg_composite_risk": risk_mean,
                "max_composite_risk": risk_max,
                "max_peak_fragility": peak_max,
                "dominant_regime": dominant_regime,
            }
        )

    if not chart_paths:
        return {}

    pd.DataFrame(summary_rows).to_csv(yearly_dir / "yearly_nasdaq_risk_regime_summary.csv", index=False, encoding="utf-8-sig")
    contact_path = yearly_dir / "nasdaq_risk_regime_1995_to_now_contact_sheet.png"
    cols = 2
    rows = int(np.ceil(len(chart_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, max(4.2 * rows, 8)), dpi=120)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, img_path in zip(axes_arr, chart_paths):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
    for ax in axes_arr[len(chart_paths) :]:
        ax.axis("off")
    fig.suptitle("1995년 이후 연도별 나스닥100 · 위험점수 · Regime", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(contact_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "yearly_nasdaq_risk_contact_sheet": contact_path,
        "yearly_nasdaq_risk_charts": chart_paths,
    }


def plot_yearly_nasdaq_risk_regime(y: pd.DataFrame, year: int, path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    y = y.sort_values("Date").copy()
    regime_col = "risk_archetype" if "risk_archetype" in y else "risk_phase"
    if regime_col not in y:
        regime_col = "risk_archetype"
        y[regime_col] = "Mixed/Transition"
    else:
        y[regime_col] = y[regime_col].fillna("Mixed/Transition").astype(str)
    start = y["NASDAQ100"].dropna().iloc[0]
    y["NASDAQ_norm"] = y["NASDAQ100"] / start * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 8.6), dpi=145, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.45, 0.55]})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#fbfbfb")
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.8)

    shade_regime_background(axes[0], y, regime_col)
    axes[0].plot(y["Date"], y["NASDAQ_norm"], color="#1f77b4", lw=2.2, label="나스닥100")
    axes[0].set_ylabel("나스닥100\n(연초=100)")
    axes[0].legend(loc="upper left", fontsize=9)

    risk_lines = [
        ("composite_vector_risk", "종합 위험", "#111827", 2.2),
        ("risk_off_score", "Risk-Off", "#c00000", 1.7),
        ("peak_fragility", "고점 취약성", "#7030a0", 1.6),
        ("analog_macro_risk", "유사환경 위험", "#0f766e", 1.5),
        ("correction_pressure", "조정 압력", "#d97706", 1.5),
    ]
    for col, label, color, lw in risk_lines:
        if col not in y:
            continue
        s = pd.to_numeric(y[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        axes[1].plot(y["Date"], s, label=label, color=color, lw=lw)
    for level, label in [(35, "주의"), (50, "위험"), (65, "현금/방어")]:
        axes[1].axhline(level, color="#bfbfbf", lw=0.85, ls="--")
        axes[1].text(y["Date"].min(), level + 1.0, label, fontsize=8, color="#666666")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("위험 점수")
    axes[1].legend(loc="upper left", ncol=5, fontsize=8)

    plot_regime_band(axes[2], y, regime_col)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Regime")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ret = y["NASDAQ100"].iloc[-1] / y["NASDAQ100"].iloc[0] - 1.0
    max_risk = pd.to_numeric(y.get("composite_vector_risk", pd.Series(dtype=float)), errors="coerce").max()
    fig.suptitle(f"{year} 나스닥100과 Risk-Off Sentinel / 동적 Regime  |  연간 나스닥 {ret:.1%}, 최대 위험 {safe_float(max_risk):.1f}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def regime_spans(frame: pd.DataFrame, regime_col: str) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    if frame.empty:
        return []
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    dates = list(pd.to_datetime(frame["Date"]))
    regimes = list(frame[regime_col].fillna("Mixed/Transition").astype(str))
    start_idx = 0
    for i in range(1, len(frame)):
        if regimes[i] != regimes[start_idx]:
            spans.append((dates[start_idx], dates[i - 1], regimes[start_idx]))
            start_idx = i
    spans.append((dates[start_idx], dates[-1], regimes[start_idx]))
    return spans


def shade_regime_background(ax, frame: pd.DataFrame, regime_col: str) -> None:
    for start, end, regime in regime_spans(frame, regime_col):
        ax.axvspan(start, end, color=REGIME_COLORS.get(regime, "#eeeeee"), alpha=0.38, lw=0)


def plot_regime_band(ax, frame: pd.DataFrame, regime_col: str) -> None:
    for start, end, regime in regime_spans(frame, regime_col):
        color = REGIME_COLORS.get(regime, "#eeeeee")
        ax.axvspan(start, end, color=color, alpha=0.95, lw=0)
        days = max((end - start).days, 1)
        if days >= 18:
            mid = start + (end - start) / 2
            ax.text(mid, 0.5, translate_state(regime), ha="center", va="center", fontsize=8, color="#263238")
    ax.set_ylim(0, 1)


def find_similar_macro_risk_days(risk_vector: pd.DataFrame, driver_panel: pd.DataFrame, n: int) -> pd.DataFrame:
    if risk_vector.empty or driver_panel.empty:
        return pd.DataFrame()
    merged = risk_vector.merge(driver_panel, on="Date", how="left", suffixes=("", "_driver"))
    feature_cols = [
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "peak_fragility",
        "liquidity_credit_stress",
        "equity_breakdown_stress",
        "fx_external_stress",
        "volatility_stress",
        "NASDAQ100",
        "SOX",
        "DXY",
        "USDKRW",
        "US10Y",
        "VIX",
        "HY_OAS",
        "COPPER_GOLD",
    ]
    feature_cols = [c for c in feature_cols if c in merged]
    x = merged[["Date", "risk_archetype", "risk_phase", "dominant_risk_vector"] + feature_cols].copy()
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=feature_cols)
    if x.shape[0] < 30:
        return pd.DataFrame()
    latest = x.iloc[-1]
    hist = x.iloc[:-20].copy()
    means = hist[feature_cols].mean()
    stds = hist[feature_cols].std().replace(0, np.nan)
    z_hist = ((hist[feature_cols] - means) / stds).astype(float)
    z_latest = ((latest[feature_cols] - means) / stds).astype(float)
    hist["similarity_distance"] = np.sqrt(((z_hist - z_latest) ** 2).mean(axis=1).astype(float))
    forward_assets = {
        "NASDAQ100": "nasdaq",
        "SOX": "sox",
        "SP500": "sp500",
        "RUSSELL2000": "russell2000",
        "GOLD": "gold",
        "WTI": "wti",
        "DXY": "dxy",
        "USDKRW": "usdkrw",
    }
    for asset_col, label in forward_assets.items():
        if asset_col not in merged:
            continue
        series = merged[["Date", asset_col]].dropna().sort_values("Date")
        if series.empty:
            continue
        series[f"next_1w_{label}_return"] = series[asset_col].shift(-5) / series[asset_col] - 1.0
        series[f"next_1m_{label}_return"] = series[asset_col].shift(-20) / series[asset_col] - 1.0
        hist = hist.merge(series[["Date", f"next_1w_{label}_return", f"next_1m_{label}_return"]], on="Date", how="left")
    hist = hist.sort_values("similarity_distance").head(max(n, 30)).copy()
    hist["similarity_rank"] = np.arange(1, len(hist) + 1)
    hist["similarity_score"] = 100.0 / (1.0 + pd.to_numeric(hist["similarity_distance"], errors="coerce"))
    cols = [
        "similarity_rank",
        "Date",
        "similarity_score",
        "similarity_distance",
        "risk_archetype",
        "risk_phase",
        "dominant_risk_vector",
        "next_1w_nasdaq_return",
        "next_1m_nasdaq_return",
        "next_1w_sox_return",
        "next_1m_sox_return",
        "next_1w_sp500_return",
        "next_1m_sp500_return",
        "next_1w_russell2000_return",
        "next_1m_russell2000_return",
        "next_1w_gold_return",
        "next_1m_gold_return",
        "next_1w_dxy_return",
        "next_1m_dxy_return",
        "next_1w_usdkrw_return",
        "next_1m_usdkrw_return",
    ] + feature_cols[:8]
    return hist.loc[:, [c for c in cols if c in hist]]


def group_summary(asset_scores: pd.DataFrame) -> pd.DataFrame:
    if asset_scores.empty:
        return pd.DataFrame()
    df = asset_scores.copy()
    for col in ["score_0_100", "upside_prob_1w", "upside_prob_4w", "technical_score", "driver_fit_score", "risk_penalty"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    summary = (
        df.groupby("group", as_index=False)
        .agg(
            n=("symbol", "count"),
            avg_score=("score_0_100", "mean"),
            top_score=("score_0_100", "max"),
            avg_upside_1w=("upside_prob_1w", "mean"),
            avg_upside_4w=("upside_prob_4w", "mean"),
            avg_technical=("technical_score", "mean"),
            avg_driver_fit=("driver_fit_score", "mean"),
            max_risk_penalty=("risk_penalty", "max"),
        )
        .sort_values("avg_score", ascending=False)
    )
    return summary


def render_html(data: dict[str, pd.DataFrame], similar: pd.DataFrame, grouped: pd.DataFrame, chart_paths: dict[str, object], top_assets: int) -> str:
    risk = data["risk_vector"]
    current = risk.tail(1).iloc[0] if not risk.empty else pd.Series(dtype=object)
    assets = data["asset_scores"].copy()
    if not assets.empty:
        assets = assets.sort_values("score_0_100", ascending=False).head(top_assets)
        assets["group"] = assets["group"].map(lambda x: GROUP_LABELS.get(str(x), str(x)))
    basket_scores = data.get("basket_scores", pd.DataFrame()).copy()
    weekly_basket_current = data.get("weekly_basket_current", pd.DataFrame()).copy()
    weekly_basket_summary = data.get("weekly_basket_summary", pd.DataFrame()).copy()
    weekly_basket_constituents = data.get("weekly_basket_constituents", pd.DataFrame()).copy()
    etf_leadership_current = data.get("etf_leadership_current", pd.DataFrame()).copy()
    etf_leadership_coverage = data.get("etf_leadership_coverage", pd.DataFrame()).copy()
    etf_leadership_failed = data.get("etf_leadership_failed", pd.DataFrame()).copy()
    etf_leadership_performance = data.get("etf_leadership_performance", pd.DataFrame()).copy()
    etf_leadership_roadmap = data.get("etf_leadership_roadmap", pd.DataFrame()).copy()
    etf_leadership_selective = data.get("etf_leadership_selective", pd.DataFrame()).copy()
    current_optimized_risk_signal = data.get("current_optimized_risk_signal", pd.DataFrame()).copy()
    risk_off_v4_current = data.get("risk_off_v4_current", pd.DataFrame()).copy()
    risk_off_v4_comparison = data.get("risk_off_v4_comparison", pd.DataFrame()).copy()
    portfolio_latest = data.get("portfolio_latest", pd.DataFrame()).copy()
    portfolio_summary = data.get("portfolio_summary", pd.DataFrame()).copy()
    portfolio_constraints = data.get("portfolio_constraints", pd.DataFrame()).copy()
    long_proxy_summary = data.get("long_proxy_summary", pd.DataFrame()).copy()
    long_proxy_crisis = data.get("long_proxy_crisis", pd.DataFrame()).copy()
    long_proxy_exposure = data.get("long_proxy_exposure", pd.DataFrame()).copy()
    long_selection_summary = data.get("long_selection_summary", pd.DataFrame()).copy()
    long_selection_ic = data.get("long_selection_ic", pd.DataFrame()).copy()
    if not long_selection_summary.empty and "Sharpe" in long_selection_summary:
        long_selection_summary["Sharpe"] = pd.to_numeric(long_selection_summary["Sharpe"], errors="coerce")
        long_selection_summary = long_selection_summary.sort_values("Sharpe", ascending=False)
    if not weekly_basket_constituents.empty:
        weekly_basket_constituents = weekly_basket_constituents.sort_values(["basket", "basket_rank"]).groupby("basket").head(8)
    if not etf_leadership_failed.empty:
        etf_leadership_failed = etf_leadership_failed.sort_values(["etf_ticker", "weight"], ascending=[True, False]).groupby("etf_ticker").head(6)
    if not etf_leadership_selective.empty:
        for col in ["Sharpe", "trade_hit_ratio_excess_gt0", "trade_positive_return_ratio", "CAGR", "MDD", "coverage"]:
            if col in etf_leadership_selective:
                etf_leadership_selective[col] = pd.to_numeric(etf_leadership_selective[col], errors="coerce")
        etf_leadership_selective = (
            etf_leadership_selective.sort_values(["horizon", "Sharpe"], ascending=[True, False])
            .groupby("horizon")
            .head(6)
        )
    grouped = grouped.copy()
    if not grouped.empty:
        grouped["group"] = grouped["group"].map(lambda x: GROUP_LABELS.get(str(x), str(x)))
    driver_state = data["driver_state"].copy()
    if not driver_state.empty and "driver" in driver_state:
        driver_state["driver"] = driver_state["driver"].map(lambda x: DRIVER_LABELS.get(str(x), str(x)))
    asof = fmt_date(current.get("Date", ""))
    old_chart_paths = {
        "risk_map": ROOT / "outputs/risk_vector_dashboard_latest/charts/risk_vector_2d_map.png",
        "risk_vector_sheet": ROOT / "outputs/risk_vector_dashboard_latest/charts/yearly_vector_vs_nasdaq/risk_vector_vs_nasdaq_yearly_contact_sheet.png",
        "peak_sheet": ROOT / "outputs/peak_fragility_model_latest/charts/yearly_peak_fragility_vs_nasdaq/peak_fragility_vs_nasdaq_yearly_contact_sheet.png",
    }
    cards = [
        ("종합 위험점수", num(current.get("composite_vector_risk")), "Risk Vector 전체 위험도"),
        ("Risk-Off 점수", num(current.get("risk_off_score")), "VIX·신용·환율 충격형 위험"),
        ("고점 취약성", num(current.get("peak_fragility")), "고점권 조정 가능성"),
        ("유사환경 위험", num(current.get("analog_macro_risk")), "과거 유사 국면 기반"),
        ("조정 압력", num(current.get("correction_pressure")), "1주/1개월 조정 타이밍"),
        ("RAI", num(current.get("rai_appetite_stress")), "위험선호 붕괴/공포"),
        ("위험 유형", translate_state(current.get("risk_archetype", "")), "현재 위험의 성격"),
        ("주도 위험축", translate_state(current.get("dominant_risk_vector", "")), "가장 큰 위험 원인"),
        ("위험 단계", translate_state(current.get("risk_phase", "")), "Normal/Warning/Risk-Off"),
    ]
    v4_1m = risk_off_v4_current[risk_off_v4_current["horizon"].astype(str).eq("1m")].tail(1) if not risk_off_v4_current.empty and "horizon" in risk_off_v4_current else pd.DataFrame()
    v4_1w = risk_off_v4_current[risk_off_v4_current["horizon"].astype(str).eq("1w")].tail(1) if not risk_off_v4_current.empty and "horizon" in risk_off_v4_current else pd.DataFrame()
    if not v4_1m.empty:
        row = v4_1m.iloc[0]
        cards.insert(0, ("Risk-Off V4 1개월", esc(row.get("risk_off_v4_stage", "")), f"확률 {safe_float(row.get('risk_off_v4_prob')):.1%}"))
    if not v4_1w.empty:
        row = v4_1w.iloc[0]
        cards.insert(1, ("Risk-Off V4 1주", esc(row.get("risk_off_v4_stage", "")), f"확률 {safe_float(row.get('risk_off_v4_prob')):.1%}"))
    axis = [
        ("X축 유동성·신용·환율", current.get("macro_liquidity_axis_x", 0)),
        ("Y축 주가·변동성 붕괴", current.get("market_breakdown_axis_y", 0)),
        ("Z축 대외·원자재 충격", current.get("external_supply_axis_z", 0)),
        ("유동성/신용 스트레스", current.get("liquidity_credit_stress", 0)),
        ("주식 붕괴 스트레스", current.get("equity_breakdown_stress", 0)),
        ("환율/대외 스트레스", current.get("fx_external_stress", 0)),
        ("변동성 스트레스", current.get("volatility_stress", 0)),
        ("원자재/공급 충격", current.get("inflation_supply_stress", 0)),
        ("유사환경 위험", current.get("analog_macro_risk", 0)),
        ("조정 압력", current.get("correction_pressure", 0)),
        ("RAI 위험선호 붕괴", current.get("rai_appetite_stress", 0)),
        ("ETF breadth 악화", current.get("universe_breadth_stress", 0)),
        ("안전자산 로테이션", current.get("safe_rotation_stress", 0)),
    ]
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>매크로 퀀트 스크리닝 - {asof}</title>
  <style>
    :root {{ --bg:#f4f6f8; --panel:#ffffff; --text:#1f2933; --muted:#667085; --line:#d7dde8; --navy:#111827; --blue:#2563eb; --red:#b42318; --purple:#6f42c1; --green:#228b5f; --amber:#b7791f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:"Malgun Gothic","Segoe UI",Arial,sans-serif; }}
    header {{ padding:24px 30px; background:var(--navy); color:white; }}
    header h1 {{ margin:0 0 6px; font-size:25px; }}
    header p {{ margin:0; color:#cbd5e1; font-size:14px; }}
    main {{ padding:22px 28px 44px; max-width:1780px; margin:0 auto; }}
    section {{ margin:0 0 24px; }}
    h2 {{ font-size:19px; margin:0 0 12px; }}
    h3 {{ font-size:15px; margin:14px 0 8px; }}
    .grid {{ display:grid; gap:15px; }}
    .cards {{ grid-template-columns:repeat(7,minmax(145px,1fr)); }}
    .two {{ grid-template-columns:1.08fr .92fr; align-items:start; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:15px; }}
    .card .label {{ color:var(--muted); font-size:12px; margin-bottom:8px; }}
    .card .value {{ font-size:24px; font-weight:800; }}
    .card .note {{ color:var(--muted); font-size:12px; margin-top:6px; }}
    .hero-chart {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .hero-chart img {{ min-height:520px; object-fit:contain; }}
    .year-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .year-grid article {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }}
    .year-grid h3 {{ margin:0 0 8px; font-size:14px; }}
    .barrow {{ display:grid; grid-template-columns:210px 1fr 58px; gap:10px; align-items:center; margin:9px 0; }}
    .bar {{ height:12px; border-radius:999px; background:#e5e7eb; overflow:hidden; }}
    .bar span {{ display:block; height:100%; background:linear-gradient(90deg,#60a5fa,#fbbf24,#dc2626); }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; font-size:13px; }}
    th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th:first-child, td:first-child, td.left, th.left {{ text-align:left; }}
    th {{ background:#eef2f7; color:#374151; font-weight:700; position:sticky; top:0; }}
    tr:last-child td {{ border-bottom:0; }}
    .table-wrap {{ overflow:auto; border-radius:8px; }}
    img {{ width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:8px; background:white; }}
    .muted {{ color:var(--muted); }}
    @media (max-width:1200px) {{ .cards,.two,.year-grid {{ grid-template-columns:1fr; }} .hero-chart img {{ min-height:0; }} }}
  </style>
</head>
<body>
<header>
  <h1>매크로 퀀트 스크리닝 대시보드</h1>
  <p>기준일 {asof} · 위험점수, 미국/한국/일본 지수, 원자재, Risk Vector, RAI, 유사환경 위험, Peak Fragility 통합 화면</p>
</header>
<main>
  <section class="grid cards">
    {''.join(card_html(*c) for c in cards)}
  </section>

  <section class="hero-chart">
    <h2>최근 12개월: 위험점수 + 미국지수 + 한국/일본지수 + 원자재</h2>
    {img_tag(chart_paths.get("market_12m"))}
  </section>

  <section class="hero-chart">
    <h2>최근 6개월: 위험점수 + 미국지수 + 한국/일본지수 + 원자재</h2>
    {img_tag(chart_paths.get("market_6m"))}
  </section>

  <section class="hero-chart">
    <h2>1995년 이후 연도별 나스닥100 · 위험점수 · Regime</h2>
    <p class="muted">각 연도별로 나스닥100을 연초 100으로 정규화하고, 종합 위험·Risk-Off·고점 취약성·유사환경 위험·조정 압력과 동적 Regime 배경을 함께 표시합니다.</p>
    {img_tag(chart_paths.get("yearly_nasdaq_risk_contact_sheet"))}
  </section>

  <section>
    <h2>연도별 상세 차트</h2>
    <div class="year-grid">{yearly_gallery_html(chart_paths.get("yearly_nasdaq_risk_charts"))}</div>
  </section>

  <section class="grid two">
    <div class="card">
      <h2>위험 벡터 축</h2>
      {''.join(bar_html(label, value) for label, value in axis)}
    </div>
    <div class="card">
      <h2>현재 해석</h2>
      <p>{current_interpretation(current)}</p>
      <p class="muted">단일 위험점수는 위험 크기만 보여줍니다. 이 화면은 유동성/신용, 주가 붕괴, 환율/대외, 원자재, 고점 취약성, 과거 유사환경 위험을 분리해서 봅니다.</p>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>2차원 위험 지도</h2>
      {img_tag(old_chart_paths["risk_map"])}
    </div>
    <div>
      <h2>연도별 Risk Vector와 나스닥</h2>
      {img_tag(old_chart_paths["risk_vector_sheet"])}
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>연도별 고점 취약성과 나스닥</h2>
      {img_tag(old_chart_paths["peak_sheet"])}
    </div>
    <div>
      <h2>세부 자산군 요약</h2>
      <div class="table-wrap">{df_to_html(grouped, ["group","n","avg_score","top_score","avg_upside_1w","avg_upside_4w","avg_technical","avg_driver_fit","max_risk_penalty"])}</div>
    </div>
  </section>

  <section>
    <h2>GAPS 바스켓 투자매력도</h2>
    <p class="muted">해외지수, 해외섹터, 국내지수, 국내섹터, FX및 원자재, 국내/해외 채권과 초단기채권으로 나눠 바스켓 점수를 계산했습니다.</p>
    <div class="table-wrap">{df_to_html(basket_scores if not basket_scores.empty else weekly_basket_current, ["basket_rank","basket","asset_count","basket_score_0_100","basket_upside_prob_1w","basket_upside_prob_4w","basket_prob_1w","basket_prob_1m","basket_return_20d","basket_risk_penalty","top_symbols","top_names"])}</div>
  </section>

  <section class="grid two">
    <div>
      <h2>바스켓 Walk-Forward 성능</h2>
      <div class="table-wrap">{df_to_html(weekly_basket_summary, ["horizon","weeks","pred_top_avg_return","basket_avg_return","top1_hit_rate","actual_top1_in_pred_top3_rate","top3_overlap_rate"])}</div>
    </div>
    <div>
      <h2>바스켓 내부 상위 ETF</h2>
      <div class="table-wrap">{df_to_html(weekly_basket_constituents, ["basket","basket_rank","symbol","name","group","institutional_score_0_100","calibrated_prob_1w","calibrated_prob_4w","realized_return_1w","realized_return_4w"])}</div>
    </div>
  </section>

  <section>
    <h2>ETF 리더십 현재 스크리닝</h2>
    <p class="muted">매크로/Regime을 제외하고 ETF 자체 상대강도와 구성종목 리더십만으로 계산한 현재 투자매력도입니다. 국내 ETF는 KOSPI200, 해외 ETF는 QQQ를 기준지수로 봅니다.</p>
    <div class="table-wrap">{df_to_html(etf_leadership_current.head(30), ["leadership_rank","etf_ticker","name","group","leadership_score_0_100","Final_Rule_Score","holding_logic","ETF_RS_20D","ETF_RS_60D","ETF_RS_120D","RS_positive_share","MA60_breadth","effective_N","top5_weight_share","weight_source_note"])}</div>
  </section>

  <section class="grid two">
    <div>
      <h2>ETF 리더십 실전 필터 성능</h2>
      <p class="muted">2023-2024 검증구간에서 진입 필터를 고르고, 2025년 이후 구간에서 평가한 결과입니다. 구성종목근사 모델은 현재 구성종목을 과거에도 쓴 근사치라 strict 모델보다 신뢰도를 낮게 봅니다.</p>
      <div class="table-wrap">{df_to_html(etf_leadership_selective, ["strategy_mode","model","score_label","horizon","top_k","threshold_col","threshold","risk_block_limit","invested_periods","coverage","Sharpe","trade_hit_ratio_excess_gt0","trade_positive_return_ratio","CAGR","MDD"])}</div>
    </div>
    <div>
      <h2>현재 위험자산 진입 게이트</h2>
      <p class="muted">ETF 리더십 랭킹은 매크로를 쓰지 않습니다. 다만 최종 진입 여부는 Risk-Off Sentinel의 현재 위험 신호로 별도 제어합니다.</p>
      <div class="table-wrap">{df_to_html(current_optimized_risk_signal.head(1), ["Date","model_regime","risk_off_score","composite_vector_risk","peak_fragility","analog_macro_risk","correction_pressure","RAI_z","optimized_action"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>Risk-Off V4 현재 상태</h2>
      <p class="muted">V4는 실전 손실 이벤트 라벨로 재학습한 최신 방어 게이트입니다. Watch는 감시, De-risk는 위험자산 축소, Cash는 강한 방어 상태입니다.</p>
      <div class="table-wrap">{df_to_html(risk_off_v4_current, ["horizon","date","risk_off_v4_prob","risk_off_v4_watch_threshold","risk_off_v4_derisk_threshold","risk_off_v4_cash_threshold","risk_off_v4_stage","risk_off_v4_watch","risk_off_v4_alert","risk_off_v4_cash","axis1_vol_credit_stress","axis2_fx_liquidity_stress","axis3_peak_fragility_stress","dominant_axis"])}</div>
    </div>
    <div>
      <h2>Risk-Off V3 대비 V4 검증</h2>
      <p class="muted">AUC보다 손실 이벤트 선제 포착률과 잡은 손실비율을 우선합니다. De-risk 선제포착률이 80% 이상인지가 핵심입니다.</p>
      <div class="table-wrap">{df_to_html(risk_off_v4_comparison, ["model","horizon","auc","daily_recall","precision","false_alarm_rate","watch_event_recall_20d","derisk_event_recall_20d","cash_event_recall_20d","caught_loss_ratio","watch_alert_rate","derisk_alert_rate","cash_alert_rate"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>제약 반영 최신 리밸런싱 포트폴리오</h2>
      <p class="muted">개별 ETF 20%, 위험자산 70%, 세부 바스켓 상한을 모두 적용한 결과입니다. V4 stage가 Cash면 위험자산 목표를 15%까지 낮춥니다.</p>
      <div class="table-wrap">{df_to_html(portfolio_latest, ["date","symbol","name","basket","weight","portfolio_score_pct","applied_risk_off_prob","applied_risk_off_stage","target_risk_weight","target_safe_weight"])}</div>
    </div>
    <div>
      <h2>리밸런싱 백테스트와 제약 검증</h2>
      <div class="table-wrap">{df_to_html(portfolio_summary, ["periods","start","end","cumulative_return","CAGR","MDD","Sharpe","hit_positive","avg_weekly_return","avg_risk_weight","avg_safe_weight","avg_cash_weight"])}</div>
      <h3>최근 제약 검증</h3>
      <div class="table-wrap">{df_to_html(portfolio_constraints.tail(8), ["date","total_weight","risk_total","safe_total","cash_weight","max_single","total_weight_violation","single_cap_violation","risk_total_violation","safe_total_violation"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>장기 프록시 백테스트: Risk-Off 게이트</h2>
      <p class="muted">ETF 상장 전 기간까지 보려면 실제 ETF 대신 Nasdaq/S&P500/SOX/Russell 위험자산 프록시와 금·달러·장기채·현금 안전자산 프록시를 사용합니다. V4 adaptive는 고점취약성만 높은 Cash 신호를 완화한 보강 버전입니다.</p>
      <div class="table-wrap">{df_to_html(long_proxy_summary, ["strategy","start","end","days","total_return","CAGR","ann_vol","Sharpe","Sortino","MDD","Calmar","positive_day_rate"])}</div>
    </div>
    <div>
      <h2>장기 위기구간 MDD 검증</h2>
      <div class="table-wrap">{df_to_html(long_proxy_crisis.sort_values(["window","strategy"]).head(80) if not long_proxy_crisis.empty else long_proxy_crisis, ["window","strategy","start","end","total_return","MDD","vol"])}</div>
      <h3>V4 단계별 노출</h3>
      <div class="table-wrap">{df_to_html(long_proxy_exposure, ["v4_stage_combined","days","avg_risk_return_next_day","avg_safe_return_next_day","avg_v4_combined_return","avg_risk_weight","avg_adaptive_risk_weight","peak_only_relaxed_days"])}</div>
    </div>
  </section>

  <section class="grid two">
    <div>
      <h2>장기 프록시 백테스트: ETF 리더십/안전자산 선택</h2>
      <p class="muted">실제 GAPS ETF의 과거 상장/holdings 제약을 분리하기 위해, 2010년대부터 상장된 글로벌 프록시 ETF로 리더십과 안전자산 선택 로직을 검증한 결과입니다.</p>
      <div class="table-wrap">{df_to_html(long_selection_summary.head(16), ["strategy","frequency","top_k","start","end","periods","total_return","CAGR","Sharpe","MDD","hit_rate_positive","avg_excess_return","hit_rate_excess_positive"])}</div>
    </div>
    <div>
      <h2>장기 프록시 Rank IC</h2>
      <p class="muted">점수가 높았던 ETF가 이후 1주/1개월 수익률 또는 QQQ 대비 초과수익 순위에서도 높았는지 보는 순위상관입니다.</p>
      <div class="table-wrap">{df_to_html(long_selection_ic, ["score","target","dates","mean_rank_ic","median_rank_ic","positive_ic_rate"])}</div>
    </div>
  </section>

  <section>
    <h2>상위 스크리닝 자산과 핵심 지표</h2>
    <div class="table-wrap">{asset_table(assets)}</div>
  </section>

  <section>
    <h2>현재와 유사했던 매크로/Risk 환경</h2>
    <p class="muted">현재 Risk Vector와 주요 선행지표를 표준화한 뒤 가까운 과거 날짜를 찾았습니다. 이후 나스닥 1주/1개월 수익률을 같이 표시합니다.</p>
    <div class="table-wrap">{similar_table(similar)}</div>
  </section>

  <section class="grid two">
    <div>
      <h2>선행지표 현재 상태</h2>
      <div class="table-wrap">{driver_table(driver_state)}</div>
    </div>
    <div>
      <h2>모델 검증 요약</h2>
      {validation_html(data)}
    </div>
  </section>
</main>
</body>
</html>"""


def card_html(label: str, value: str, note: str) -> str:
    return f'<div class="card"><div class="label">{esc(label)}</div><div class="value">{value}</div><div class="note">{esc(note)}</div></div>'


def bar_html(label: str, value: object) -> str:
    v = safe_float(value)
    return f'<div class="barrow"><div>{esc(label)}</div><div class="bar"><span style="width:{max(0,min(100,v)):.1f}%"></span></div><div>{v:.1f}</div></div>'


def current_interpretation(current: pd.Series) -> str:
    arch = translate_state(current.get("risk_archetype", ""))
    phase = translate_state(current.get("risk_phase", ""))
    peak = safe_float(current.get("peak_fragility"))
    ro = safe_float(current.get("risk_off_score"))
    x = safe_float(current.get("macro_liquidity_axis_x"))
    y = safe_float(current.get("market_breakdown_axis_y"))
    if phase == "정상" and peak >= 48 and ro < 25:
        return "현재는 충격형 Risk-Off가 아니라 고점 취약성 경고에 가깝습니다. 유동성/신용/주가 붕괴는 낮지만, 고점권 추격 매수는 보수적으로 봐야 합니다."
    if x >= 55 and y >= 55:
        return "유동성/신용 스트레스와 가격 붕괴가 동시에 높습니다. 전형적인 전면 Risk-Off 조합입니다."
    if x >= 45 and y < 45:
        return "가격은 아직 버티지만 유동성·신용·환율 스트레스가 먼저 올라오는 구간입니다. 선행 경고로 해석합니다."
    return f"현재 위험 유형은 {arch}, 위험 단계는 {phase}입니다. 단일 점수보다 주도 위험축과 X/Y/Z 조합을 같이 봐야 합니다."


def asset_table(df: pd.DataFrame) -> str:
    cols = ["rank", "symbol", "name", "group", "score_0_100", "upside_prob_1w", "upside_prob_4w", "technical_score", "driver_fit_score", "return_20d", "drawdown_252d", "risk_penalty"]
    return df_to_html(df, cols)


def similar_table(df: pd.DataFrame) -> str:
    cols = [
        "similarity_rank",
        "Date",
        "similarity_score",
        "similarity_distance",
        "risk_archetype",
        "risk_phase",
        "dominant_risk_vector",
        "next_1w_nasdaq_return",
        "next_1m_nasdaq_return",
        "next_1w_sox_return",
        "next_1m_sox_return",
        "next_1w_sp500_return",
        "next_1m_sp500_return",
        "next_1w_russell2000_return",
        "next_1m_russell2000_return",
        "next_1w_gold_return",
        "next_1m_gold_return",
        "next_1w_dxy_return",
        "next_1m_dxy_return",
        "next_1w_usdkrw_return",
        "next_1m_usdkrw_return",
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "peak_fragility",
        "liquidity_credit_stress",
        "equity_breakdown_stress",
        "fx_external_stress",
    ]
    return df_to_html(df, cols)


def driver_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='muted'>선행지표 데이터가 없습니다.</p>"
    cols = ["driver", "level", "change_5d", "change_20d", "z_60d", "riskon_score"]
    use = df.copy()
    if "riskon_score" in use:
        use["riskon_score"] = pd.to_numeric(use["riskon_score"], errors="coerce")
    return df_to_html(use.sort_values("driver"), cols)


def validation_html(data: dict[str, pd.DataFrame]) -> str:
    chunks = []
    opt_current = data.get("optimizer_current", pd.DataFrame())
    if not opt_current.empty:
        chunks.append("<h3>Walk-Forward 최적화 현재 신호</h3>")
        chunks.append(
            df_to_html(
                opt_current,
                [
                    "Date",
                    "risk_off_avoidance_score",
                    "peak_correction_score",
                    "crash_sentinel_score",
                    "peak_fragility",
                    "analog_macro_risk",
                    "correction_pressure",
                    "RAI_z",
                    "RAI_overheat_score",
                    "ETF_breadth_shock_score",
                    "model_regime",
                    "optimized_action",
                ],
            )
        )
    opt_cal = data.get("optimizer_calibration", pd.DataFrame())
    if not opt_cal.empty:
        chunks.append("<h3>자산군/Regime별 확률 Calibration</h3>")
        chunks.append(
            df_to_html(
                opt_cal,
                ["target", "family", "samples", "actual_rate", "raw_avg_prob", "calibrated_avg_prob", "raw_brier", "calibrated_brier"],
            )
        )
    opt_threshold = data.get("optimizer_threshold", pd.DataFrame())
    if not opt_threshold.empty:
        chunks.append("<h3>Walk-Forward Threshold 최적화</h3>")
        chunks.append(
            df_to_html(
                best_optimizer_rows(opt_threshold),
                [
                    "target",
                    "model",
                    "samples",
                    "positive_rate",
                    "signal_rate",
                    "accuracy",
                    "precision",
                    "recall",
                    "false_alarm_rate",
                    "roc_auc",
                    "avg_forward_drawdown_when_signal",
                ],
            )
        )
    opt_false = data.get("optimizer_false_alarm", pd.DataFrame())
    if not opt_false.empty:
        chunks.append("<h3>오경보 분해</h3>")
        chunks.append(df_to_html(opt_false.head(16), ["target", "bucket", "count", "rate", "avg_forward_return", "avg_forward_drawdown"]))
    opt_safe = data.get("optimizer_safe", pd.DataFrame())
    if not opt_safe.empty and "row_type" in opt_safe:
        chunks.append("<h3>Risk-Off 안전자산 선택 검증</h3>")
        chunks.append(
            df_to_html(
                opt_safe[opt_safe["row_type"].eq("summary")],
                ["horizon", "signal_weeks", "top1_hit_rate", "top3_hit_rate", "pred_top1_avg_return", "pred_top3_avg_return", "safe_universe_avg_return"],
            )
        )
    opt_current_safe = data.get("optimizer_current_safe", pd.DataFrame())
    if not opt_current_safe.empty:
        chunks.append("<h3>현재 안전자산 추천</h3>")
        chunks.append(df_to_html(opt_current_safe, ["rank", "symbol", "name", "group", "safe_score", "return_20d"]))
    opt_rank = data.get("optimizer_rank", pd.DataFrame())
    if not opt_rank.empty:
        chunks.append("<h3>Risk-On 빠른 랭킹 백테스트</h3>")
        chunks.append(
            df_to_html(
                opt_rank,
                ["horizon", "model_regime", "weeks", "avg_topk_return", "avg_universe_return", "topk_overlap_rate", "top1_exact_hit_rate", "actual_top1_in_pred_topk_rate"],
            )
        )
    for title, key in [("안전자산 1주/3주 선택 검증", "safe_eval"), ("주간 급락 Risk-Off 4분면 검증", "risk_confusion"), ("고점 취약성 모델 검증", "peak_validation"), ("유사환경 Analog 모델 검증", "analog_validation")]:
        df = data.get(key, pd.DataFrame())
        chunks.append(f"<h3>{esc(title)}</h3>")
        chunks.append(df_to_html(df, list(df.columns)) if not df.empty else "<p class='muted'>데이터가 없습니다.</p>")
    return "\n".join(chunks)


def best_optimizer_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["recall", "precision", "signal_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["objective"] = 1.6 * out["recall"] + out["precision"] - 0.35 * out["signal_rate"]
    idx = out.groupby("target")["objective"].idxmax()
    return out.loc[idx].drop(columns=["objective"]).sort_values("target")


def df_to_html(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "<p class='muted'>데이터가 없습니다.</p>"
    cols = [c for c in cols if c in df.columns]
    left_cols = {"symbol", "name", "group", "basket", "top_symbols", "top_names", "driver", "risk_archetype", "risk_phase", "dominant_risk_vector", "prediction_used", "target", "criterion", "horizon", "model", "description", "optimized_action", "bucket", "model_regime", "family", "model_version"}
    rows = ["<table><thead><tr>"]
    for c in cols:
        cls = "left" if c in left_cols else ""
        rows.append(f'<th class="{cls}">{esc(KOREAN_LABELS.get(c, c))}</th>')
    rows.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        rows.append("<tr>")
        for c in cols:
            cls = "left" if c in left_cols else ""
            rows.append(f'<td class="{cls}">{fmt_value(c, row.get(c))}</td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def img_tag(path: object | None) -> str:
    if path is None or not isinstance(path, Path):
        return "<p class='muted'>차트가 없습니다.</p>"
    return f'<img src="{path.resolve().as_posix()}" alt="{esc(path.name)}">'


def yearly_gallery_html(paths: object | None) -> str:
    if not isinstance(paths, list) or not paths:
        return "<p class='muted'>연도별 차트가 없습니다.</p>"
    chunks = []
    for item in paths:
        if not isinstance(item, Path):
            continue
        year = item.stem.rsplit("_", 1)[-1]
        chunks.append(f"<article><h3>{esc(year)}</h3>{img_tag(item)}</article>")
    return "".join(chunks) if chunks else "<p class='muted'>연도별 차트가 없습니다.</p>"


def fmt_value(column: str, value: object) -> str:
    if pd.isna(value):
        return ""
    if column in {"risk_archetype", "risk_phase", "dominant_risk_vector"}:
        return esc(translate_state(value))
    if column == "group":
        return esc(GROUP_LABELS.get(str(value), str(value)))
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, str):
        if value.startswith("{") and value.endswith("}"):
            try:
                return esc(json.dumps(json.loads(value), ensure_ascii=False))
            except Exception:
                return esc(value)
        return esc(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if column.endswith("return") or "return" in column or "prob" in column or "rate" in column or "ratio" in column or column in {"drawdown_252d", "change_5d", "change_20d", "coverage", "CAGR", "MDD"}:
            return f"{value:.2%}"
        if abs(value) < 1e-4 and value != 0:
            return f"{value:.2e}"
        return f"{value:,.2f}"
    return esc(value)


def translate_state(value: object) -> str:
    text = str(value)
    mapping = {
        "Peak Warning": "고점 경고",
        "Peak Fragility": "고점 취약성",
        "Calm Risk-On": "평온한 Risk-On",
        "Mixed/Transition": "혼합/전환",
        "Full Risk-Off": "전면 Risk-Off",
        "Credit/Liquidity Shock": "신용/유동성 충격",
        "FX/External Stress": "환율/대외 스트레스",
        "Inflation/Supply Shock": "물가/공급 충격",
        "Technical Equity Breakdown": "주식 기술적 붕괴",
        "Cyclical/China Stress": "경기/중국 스트레스",
        "Normal": "정상",
        "Fragile": "취약",
        "Warning": "경고",
        "Risk-Off": "Risk-Off",
        "Crisis": "위기",
        "Liquidity Credit Stress": "유동성/신용 스트레스",
        "Equity Breakdown Stress": "주식 붕괴 스트레스",
        "Fx External Stress": "환율/대외 스트레스",
        "Volatility Stress": "변동성 스트레스",
        "Inflation Supply Stress": "원자재/공급 충격",
        "Hedge Demand": "헤지 수요",
    }
    return mapping.get(text, text)


def num(value: object) -> str:
    return f"{safe_float(value):.1f}"


def fmt_date(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return esc(value)


def safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
