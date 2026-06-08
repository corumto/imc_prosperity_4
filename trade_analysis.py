import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


def analyze_trade_distribution_by_mid_price(
    trades: pd.DataFrame,
    alt_mid_price: float,
    product: str | None = None,
    figsize: Tuple[int, int] = (14, 5),
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    Analyze the distribution of trade sizes split by whether trades occurred
    above or below the alt_mid_price.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with columns at minimum: 'price', 'quantity'.
        Optionally 'symbol' or 'product' to filter.
    alt_mid_price : float
        Threshold price to split trades.
    product : str, optional
        If provided, filter trades to this product/symbol first.
    figsize : Tuple[int, int]
        Figure size for plots.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]
        - above_mid: stats for trades at price >= alt_mid_price
        - below_mid: stats for trades at price < alt_mid_price
        - fig: matplotlib figure with distribution plots
    """
    # Normalize column names
    trades = trades.copy()
    trades.columns = [str(c).strip().lower() for c in trades.columns]

    # Filter by product if specified
    if product is not None:
        symbol_col = "symbol" if "symbol" in trades.columns else "product"
        if symbol_col in trades.columns:
            trades = trades[trades[symbol_col].astype(str).str.upper() == product.upper()].copy()

    # Ensure numeric columns
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")

    # Drop rows with NaN price or quantity
    trades = trades.dropna(subset=["price", "quantity"])

    # Split by mid price
    above_mid = trades[trades["price"] >= alt_mid_price]
    below_mid = trades[trades["price"] < alt_mid_price]

    # Compute statistics
    above_stats = {
        "count": len(above_mid),
        "size_mean": above_mid["quantity"].mean(),
        "size_median": above_mid["quantity"].median(),
        "size_std": above_mid["quantity"].std(),
        "size_min": above_mid["quantity"].min(),
        "size_max": above_mid["quantity"].max(),
        "total_volume": above_mid["quantity"].sum(),
    }

    below_stats = {
        "count": len(below_mid),
        "size_mean": below_mid["quantity"].mean(),
        "size_median": below_mid["quantity"].median(),
        "size_std": below_mid["quantity"].std(),
        "size_min": below_mid["quantity"].min(),
        "size_max": below_mid["quantity"].max(),
        "total_volume": below_mid["quantity"].sum(),
    }

    above_df = pd.DataFrame([above_stats], index=["above_mid"])
    below_df = pd.DataFrame([below_stats], index=["below_mid"])

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if len(above_mid) > 0:
        axes[0].hist(
            above_mid["quantity"],
            bins=30,
            color="green",
            alpha=0.7,
            edgecolor="black",
        )
        axes[0].set_title(f"Trade sizes >= {alt_mid_price:.0f} (n={len(above_mid)})")
        axes[0].set_xlabel("Trade quantity")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, alpha=0.3)

    if len(below_mid) > 0:
        axes[1].hist(
            below_mid["quantity"],
            bins=30,
            color="red",
            alpha=0.7,
            edgecolor="black",
        )
        axes[1].set_title(f"Trade sizes < {alt_mid_price:.0f} (n={len(below_mid)})")
        axes[1].set_xlabel("Trade quantity")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)

    fig.suptitle("Trade Size Distribution Split by alt_mid_price", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return above_df, below_df, fig
