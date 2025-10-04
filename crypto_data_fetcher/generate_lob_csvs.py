"""
generate_lob_csvs.py

Fetch real crypto data (Binance) using ccxt and save in LOB Intensity Simulator format:
  - covariates.csv: time, S, q1, Q10
  - events.csv: time, event_type, price, side

Dependencies:
  pip install ccxt pandas numpy
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone
import argparse
import json

# CONFIGURATION
SYMBOL = "BTC/USDT"
DEPTH_LEVELS = 10
N_SNAPSHOTS = 200         # how many order book snapshots to fetch
SNAPSHOT_INTERVAL = 5.0   # seconds between snapshots
OUT_EVENTS = "events.csv"
OUT_COVARS = "covariates.csv"

class CryptoDataFetcher:
    """Enhanced crypto data fetcher for LOB Intensity Simulator."""
    
    def __init__(self, symbol="BTC/USDT", exchange_name="binance"):
        """
        Initialize the data fetcher.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            exchange_name: Exchange name (binance, coinbase, etc.)
        """
        self.symbol = symbol
        self.exchange_name = exchange_name
        
        # Initialize exchange
        if exchange_name.lower() == "binance":
            self.exchange = ccxt.binance()
        elif exchange_name.lower() == "coinbase":
            self.exchange = ccxt.coinbasepro()
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        
        # Verify exchange supports required methods
        if not hasattr(self.exchange, 'fetch_order_book'):
            raise ValueError(f"Exchange {exchange_name} does not support order book fetching")
        
        print(f"Initialized {exchange_name} exchange for {symbol}")

    def get_orderbook(self, limit=DEPTH_LEVELS):
        """Fetch order book and return simplified structure."""
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=limit)
            bids = np.array(ob['bids'])
            asks = np.array(ob['asks'])
            
            if len(bids) == 0 or len(asks) == 0:
                raise ValueError("Empty order book")
            
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            spread = best_ask - best_bid
            
            # q1: volume at best quote (minimum of best bid/ask volumes)
            q1 = min(bids[0, 1], asks[0, 1])
            
            # Q10: total volume in first 10 levels
            Q10 = bids[:, 1].sum() + asks[:, 1].sum()
            
            timestamp = ob.get('timestamp')
            if timestamp is None:
                timestamp = time.time() * 1000  # Use current time if no timestamp
            
            return {
                "timestamp": timestamp / 1000.0,
                "S": spread,
                "q1": q1,
                "Q10": Q10,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bids": bids,
                "asks": asks
            }
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return None

    def get_recent_trades(self, limit=100):
        """Fetch recent trades (real trades -> 'market' events)."""
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            data = []
            for t in trades:
                timestamp = t.get('timestamp')
                if timestamp is None:
                    timestamp = time.time() * 1000
                
                data.append({
                    "time": timestamp / 1000.0,
                    "event_type": "market",
                    "price": float(t['price']),
                    "side": t['side']
                })
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return pd.DataFrame()

    def generate_synthetic_events(self, ob_data, start_time):
        """Generate synthetic limit and cancel events for realism."""
        events = []
        
        # Add synthetic limit orders
        if np.random.rand() < 0.3:
            # Random limit order near best bid
            price_offset = np.random.uniform(-0.001, 0.001) * ob_data['best_bid']
            events.append([
                ob_data['timestamp'] - start_time,
                "limit", 
                ob_data['best_bid'] + price_offset, 
                "buy"
            ])
        
        if np.random.rand() < 0.3:
            # Random limit order near best ask
            price_offset = np.random.uniform(-0.001, 0.001) * ob_data['best_ask']
            events.append([
                ob_data['timestamp'] - start_time,
                "limit", 
                ob_data['best_ask'] + price_offset, 
                "sell"
            ])
        
        # Add synthetic cancellations
        if np.random.rand() < 0.2:
            events.append([
                ob_data['timestamp'] - start_time,
                "cancel", 
                ob_data['best_ask'], 
                "sell"
            ])
        
        if np.random.rand() < 0.2:
            events.append([
                ob_data['timestamp'] - start_time,
                "cancel", 
                ob_data['best_bid'], 
                "buy"
            ])
        
        return events

    def fetch_data(self, n_snapshots=N_SNAPSHOTS, snapshot_interval=SNAPSHOT_INTERVAL):
        """
        Main data fetching function.
        
        Args:
            n_snapshots: Number of order book snapshots to fetch
            snapshot_interval: Seconds between snapshots
            
        Returns:
            Tuple of (covariates_df, events_df)
        """
        print(f"Collecting {n_snapshots} snapshots from {self.exchange_name} for {self.symbol}...")
        print(f"Snapshot interval: {snapshot_interval} seconds")
        
        cov_rows = []
        event_rows = []
        
        start_time = time.time()
        successful_snapshots = 0
        
        for i in range(n_snapshots):
            try:
                print(f"Snapshot {i+1}/{n_snapshots}...", end=" ")
                
                # Fetch order book data
                ob_data = self.get_orderbook()
                if ob_data is None:
                    print("Failed")
                    continue
                
                # Add to covariates
                cov_rows.append([
                    ob_data['timestamp'] - start_time,
                    ob_data['S'], 
                    ob_data['q1'], 
                    ob_data['Q10']
                ])
                
                # Add real trades as market events
                trades_df = self.get_recent_trades()
                for _, trade in trades_df.iterrows():
                    rel_time = trade['time'] - start_time
                    # Only add trades that occurred after the start time
                    if rel_time >= 0:
                        event_rows.append([
                            rel_time, 
                            trade['event_type'], 
                            trade['price'], 
                            trade['side']
                        ])
                
                # Add synthetic events for realism
                synthetic_events = self.generate_synthetic_events(ob_data, start_time)
                event_rows.extend(synthetic_events)
                
                successful_snapshots += 1
                print(f"Success (S={ob_data['S']:.6f}, q1={ob_data['q1']:.2f})")
                
            except Exception as e:
                print(f"Error: {e}")
            
            # Sleep between snapshots
            if i < n_snapshots - 1:  # Don't sleep after last snapshot
                time.sleep(snapshot_interval)
        
        print(f"\nSuccessfully collected {successful_snapshots}/{n_snapshots} snapshots")
        
        # Build DataFrames
        cov_df = pd.DataFrame(cov_rows, columns=["time", "S", "q1", "Q10"])
        cov_df = cov_df.sort_values("time").drop_duplicates("time")
        
        evt_df = pd.DataFrame(event_rows, columns=["time", "event_type", "price", "side"])
        evt_df = evt_df.sort_values("time")
        
        return cov_df, evt_df

    def save_data(self, cov_df, evt_df, output_dir="."):
        """Save data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        cov_path = os.path.join(output_dir, OUT_COVARS)
        evt_path = os.path.join(output_dir, OUT_EVENTS)
        
        cov_df.to_csv(cov_path, index=False)
        evt_df.to_csv(evt_path, index=False)
        
        print(f"\nSaved:")
        print(f"  {cov_path}  ({len(cov_df)} rows)")
        print(f"  {evt_path}  ({len(evt_df)} rows)")
        
        # Save metadata
        metadata = {
            "symbol": self.symbol,
            "exchange": self.exchange_name,
            "timestamp": datetime.now().isoformat(),
            "n_covariates": len(cov_df),
            "n_events": len(evt_df),
            "time_range": [cov_df['time'].min(), cov_df['time'].max()],
            "spread_stats": {
                "mean": float(cov_df['S'].mean()),
                "std": float(cov_df['S'].std()),
                "min": float(cov_df['S'].min()),
                "max": float(cov_df['S'].max())
            }
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  {metadata_path}  (metadata)")
        
        return cov_path, evt_path, metadata_path


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Fetch real crypto data for LOB Intensity Simulator")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--exchange", default="binance", help="Exchange name")
    parser.add_argument("--snapshots", type=int, default=N_SNAPSHOTS, help="Number of snapshots")
    parser.add_argument("--interval", type=float, default=SNAPSHOT_INTERVAL, help="Snapshot interval (seconds)")
    parser.add_argument("--output", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize fetcher
        fetcher = CryptoDataFetcher(symbol=args.symbol, exchange_name=args.exchange)
        
        # Fetch data
        cov_df, evt_df = fetcher.fetch_data(
            n_snapshots=args.snapshots,
            snapshot_interval=args.interval
        )
        
        # Save data
        cov_path, evt_path, metadata_path = fetcher.save_data(cov_df, evt_df, args.output)
        
        print("\n✅ Done! You can now upload these CSVs into your LOB Intensity Simulator.")
        print(f"\nTo use with the web interface:")
        print(f"1. Go to http://localhost:8080")
        print(f"2. Upload {cov_path} as covariates.csv")
        print(f"3. Upload {evt_path} as events.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
