"""
Candlestick Chart Renderer

Renders OHLCV data into candlestick chart images using mplfinance.
"""

import logging
import io
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import mplfinance as mpf
from PIL import Image

logger = logging.getLogger(__name__)


class ChartRenderer:
    """Renders OHLCV data into candlestick chart images"""
    
    def __init__(self, style: str = "yahoo", figsize: tuple = (10, 6)):
        """
        Initialize chart renderer
        
        Args:
            style: Chart style (yahoo, binance, etc.)
            figsize: Figure size (width, height) in inches
        """
        self.style = style
        self.figsize = figsize
        self.logger = logger
    
    def ohlcv_to_dataframe(self, ohlcv_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert OHLCV data list to pandas DataFrame
        
        Args:
            ohlcv_data: List of OHLCV dictionaries with keys:
                - timestamp (Unix timestamp or datetime)
                - open, high, low, close (float)
                - volume (float, optional)
        
        Returns:
            DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
        """
        if not ohlcv_data:
            raise ValueError("OHLCV data is empty")
        
        # Convert to list of dicts if needed
        records = []
        for candle in ohlcv_data:
            # Handle different timestamp formats
            timestamp = candle.get("timestamp") or candle.get("time") or candle.get("t")
            if isinstance(timestamp, str):
                # Try parsing as ISO format
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    timestamp = datetime.fromtimestamp(int(timestamp))
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            
            records.append({
                "Open": float(candle.get("open", candle.get("o", 0))),
                "High": float(candle.get("high", candle.get("h", 0))),
                "Low": float(candle.get("low", candle.get("l", 0))),
                "Close": float(candle.get("close", candle.get("c", 0))),
                "Volume": float(candle.get("volume", candle.get("v", 0)))
            })
        
        df = pd.DataFrame(records)
        df.index = pd.DatetimeIndex([pd.to_datetime(candle.get("timestamp") or candle.get("time") or candle.get("t")) 
                                     for candle in ohlcv_data])
        
        return df
    
    def render_chart(
        self,
        ohlcv_data: List[Dict[str, Any]],
        symbol: str = "UNKNOWN",
        interval: str = "1h",
        save_path: Optional[str] = None,
        show_volume: bool = True,
        chart_type: str = "candle"
    ) -> Image.Image:
        """
        Render OHLCV data as candlestick chart image
        
        Args:
            ohlcv_data: List of OHLCV dictionaries
            symbol: Trading symbol (for title)
            interval: Time interval (for title)
            save_path: Optional path to save image file
            show_volume: Whether to show volume subplot
            chart_type: Chart type ('candle', 'line', 'ohlc', 'renko', 'pnf')
        
        Returns:
            PIL Image object
        """
        try:
            # Convert to DataFrame
            df = self.ohlcv_to_dataframe(ohlcv_data)
            
            if df.empty:
                raise ValueError("DataFrame is empty after conversion")
            
            # Validate data
            if df.isnull().any().any():
                self.logger.warning("DataFrame contains NaN values, filling with forward fill")
                df = df.ffill().bfill()
            
            # Configure mplfinance style
            mc = mpf.make_marketcolors(
                up='green',
                down='red',
                edge='inherit',
                wick={'upcolor': 'green', 'downcolor': 'red'},
                volume='in'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='gray',
                y_on_right=False
            )
            
            # Prepare volume subplot
            volume_config = {}
            if show_volume and 'Volume' in df.columns and df['Volume'].sum() > 0:
                volume_config = {'volume': True}
            
            # Create chart
            fig, axes = mpf.plot(
                df,
                type=chart_type,
                style=style,
                volume=show_volume and 'Volume' in df.columns and df['Volume'].sum() > 0,
                figsize=self.figsize,
                title=f"{symbol} - {interval}",
                ylabel="Price",
                ylabel_lower="Volume" if show_volume else None,
                returnfig=True,
                savefig=dict(fname=save_path) if save_path else None
            )
            
            # Convert to PIL Image
            if save_path:
                # If saved to file, load it
                img = Image.open(save_path)
            else:
                # Convert figure to image in memory
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img = Image.open(buf)
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error rendering chart: {e}", exc_info=True)
            raise
    
    def render_chart_to_bytes(
        self,
        ohlcv_data: List[Dict[str, Any]],
        symbol: str = "UNKNOWN",
        interval: str = "1h",
        format: str = "PNG",
        show_volume: bool = True,
        chart_type: str = "candle"
    ) -> bytes:
        """
        Render chart and return as bytes
        
        Args:
            ohlcv_data: List of OHLCV dictionaries
            symbol: Trading symbol
            interval: Time interval
            format: Image format (PNG, JPEG, etc.)
            show_volume: Whether to show volume
            chart_type: Chart type
        
        Returns:
            Image bytes
        """
        img = self.render_chart(
            ohlcv_data=ohlcv_data,
            symbol=symbol,
            interval=interval,
            show_volume=show_volume,
            chart_type=chart_type
        )
        
        buf = io.BytesIO()
        img.save(buf, format=format)
        return buf.getvalue()
    
    def render_chart_from_fks_data(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
        data_service_url: str = "http://fks_data:8003"
    ) -> Image.Image:
        """
        Fetch OHLCV data from fks_data and render chart
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            limit: Number of candles
            data_service_url: fks_data service URL
        
        Returns:
            PIL Image object
        """
        import httpx
        import asyncio
        
        async def fetch_and_render():
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{data_service_url}/api/v1/data/ohlcv",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "limit": limit,
                        "use_cache": True
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                ohlcv_list = data.get("data", [])
                if not ohlcv_list:
                    raise ValueError(f"No OHLCV data returned for {symbol}")
                
                return self.render_chart(
                    ohlcv_data=ohlcv_list,
                    symbol=symbol,
                    interval=interval
                )
        
        return asyncio.run(fetch_and_render())
