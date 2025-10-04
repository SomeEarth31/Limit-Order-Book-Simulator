# Crypto Data Fetcher

A tool to fetch real-time cryptocurrency market data and convert it to the format required by the LOB Intensity Simulator.

## Features

- **Real-time Data**: Fetches live order book and trade data from major exchanges
- **Multiple Exchanges**: Supports Binance, Coinbase Pro, and other CCXT-compatible exchanges
- **LOB Format**: Outputs data in the exact format required by the LOB Intensity Simulator
- **Synthetic Events**: Generates realistic limit orders and cancellations for complete datasets
- **Metadata**: Includes detailed statistics and metadata about the collected data

## Installation

```bash
cd crypto_data_fetcher
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

```bash
python generate_lob_csvs.py
```

This will fetch 200 snapshots of BTC/USDT data from Binance with 5-second intervals.

### 2. Custom Parameters

```bash
python generate_lob_csvs.py --symbol ETH/USDT --snapshots 100 --interval 3.0 --output my_data
```

### 3. Different Exchange

```bash
python generate_lob_csvs.py --exchange coinbase --symbol BTC/USD
```

## Command Line Options

- `--symbol`: Trading pair symbol (default: "BTC/USDT")
- `--exchange`: Exchange name (default: "binance")
- `--snapshots`: Number of snapshots to fetch (default: 200)
- `--interval`: Seconds between snapshots (default: 5.0)
- `--output`: Output directory (default: current directory)

## Output Files

The tool generates three files:

1. **covariates.csv**: Order book snapshots with columns `[time, S, q1, Q10]`
2. **events.csv**: Market events with columns `[time, event_type, price, side]`
3. **metadata.json**: Statistics and metadata about the collected data

## Data Format

### Covariates CSV
```csv
time,S,q1,Q10
0.0,0.01,1.5,150.0
5.0,0.012,1.8,180.0
10.0,0.011,1.6,160.0
```

### Events CSV
```csv
time,event_type,price,side
1.2,market,50000.0,buy
3.5,limit,50001.0,sell
7.8,cancel,50000.5,buy
```

## Supported Exchanges

- **Binance** (recommended): High liquidity, reliable API
- **Coinbase Pro**: Good for USD pairs
- **Any CCXT-compatible exchange**: Kraken, Bitfinex, etc.

## Example Usage in Python

```python
from generate_lob_csvs import CryptoDataFetcher

# Initialize fetcher
fetcher = CryptoDataFetcher(symbol="BTC/USDT", exchange_name="binance")

# Fetch data
cov_df, evt_df = fetcher.fetch_data(n_snapshots=50, snapshot_interval=2.0)

# Save data
fetcher.save_data(cov_df, evt_df, "my_data")
```

## Integration with LOB Intensity Simulator

1. **Fetch data**:
   ```bash
   python generate_lob_csvs.py --snapshots 200 --interval 5.0
   ```

2. **Start the simulator**:
   ```bash
   cd ..
   python app.py
   ```

3. **Upload data**:
   - Go to http://localhost:8080
   - Upload `covariates.csv` and `events.csv`
   - Run simulations with real market data

## Data Quality Notes

- **Real trades**: All market events are real trades from the exchange
- **Synthetic events**: Limit orders and cancellations are generated for realism
- **Time synchronization**: All timestamps are relative to the start of data collection
- **Data validation**: Automatic filtering of invalid or duplicate data points

## Troubleshooting

### Common Issues

1. **Rate limiting**: Reduce snapshot frequency if you get rate limit errors
2. **Network issues**: Check internet connection and exchange API status
3. **Empty order books**: Some exchanges may have temporary data issues

### Performance Tips

- Use longer intervals for large datasets
- Start with fewer snapshots for testing
- Monitor exchange API limits

## Example Output

```
Collecting 200 snapshots from binance for BTC/USDT...
Snapshot interval: 5.0 seconds
Snapshot 1/200... Success (S=0.010000, q1=1.50)
Snapshot 2/200... Success (S=0.012000, q1=1.80)
...
Successfully collected 200/200 snapshots

Saved:
  covariates.csv  (200 rows)
  events.csv  (1250 rows)
  metadata.json  (metadata)

✅ Done! You can now upload these CSVs into your LOB Intensity Simulator.
```

## Advanced Usage

### Custom Event Generation

You can modify the `generate_synthetic_events` method to create different types of synthetic events based on your needs.

### Multiple Symbols

Run multiple instances with different symbols to collect data for various trading pairs simultaneously.

### Historical Data

For historical analysis, consider using exchange-specific historical data APIs or modify the fetcher to work with historical datasets.

## License

This tool is part of the LOB Intensity Simulator project and follows the same license terms.
