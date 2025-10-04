"""
Flask Web Application for LOB Intensity Simulator
Provides web interface for CSV upload and backtesting.
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import tempfile
import zipfile

# Import our core modules
from core.intensity_models import MarketOrderIntensityModel, LimitOrderIntensityModel
from core.placement_model import LimitOrderPlacementModel
from core.simulator import OrderFlowSimulator
from core.data_handler import DataHandler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store current session data
current_simulator = None
current_results = None

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj


@app.route('/')
def index():
    """Main page with file upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle CSV file uploads."""
    global current_simulator, current_results
    
    try:
        # Check if files were uploaded
        if 'events_file' not in request.files or 'covariates_file' not in request.files:
            return jsonify({'error': 'Both events.csv and covariates.csv files are required'})
        
        events_file = request.files['events_file']
        covariates_file = request.files['covariates_file']
        
        if events_file.filename == '' or covariates_file.filename == '':
            return jsonify({'error': 'No files selected'})
        
        # Save uploaded files temporarily
        events_path = os.path.join(tempfile.gettempdir(), 'events.csv')
        covariates_path = os.path.join(tempfile.gettempdir(), 'covariates.csv')
        
        events_file.save(events_path)
        covariates_file.save(covariates_path)
        
        # Load and validate data
        handler = DataHandler()
        events_df, covariates_df = handler.load_csv_files(events_path, covariates_path)
        
        # Initialize simulator
        current_simulator = OrderFlowSimulator(random_state=42)
        
        # Fit models
        current_simulator.fit_models(events_df, covariates_df)
        
        # Get model parameters (convert numpy arrays to lists for JSON serialization)
        market_params = current_simulator.market_model.beta.tolist()
        limit_params = current_simulator.limit_model.beta.tolist()
        
        ask_params = current_simulator.ask_placement_model.get_parameters()
        bid_params = current_simulator.bid_placement_model.get_parameters()
        
        # Convert numpy arrays in placement parameters to lists
        for param_dict in [ask_params, bid_params]:
            for key, value in param_dict.items():
                if hasattr(value, 'tolist'):  # Check if it's a numpy array
                    param_dict[key] = value.tolist()
        
        # Get data summary
        data_summary = handler.get_data_summary()
        
        data_summary = convert_numpy_types(data_summary)
        
        # Store the original covariates for simulation use
        current_simulator._original_covariates = covariates_df
        
        # Store results
        current_results = {
            'data_summary': data_summary,
            'market_params': market_params,
            'limit_params': limit_params,
            'ask_placement_params': ask_params,
            'bid_placement_params': bid_params,
            'upload_time': datetime.now().isoformat()
        }
        
        # Clean up temporary files
        os.remove(events_path)
        os.remove(covariates_path)
        
        return jsonify({
            'success': True,
            'message': 'Files uploaded and models fitted successfully',
            'results': current_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing files: {str(e)}'})


@app.route('/simulate', methods=['POST'])
def simulate():
    """Run simulation with fitted models."""
    global current_simulator, current_results
    
    if current_simulator is None:
        return jsonify({'error': 'No models fitted. Please upload files first.'})
    
    try:
        data = request.get_json()
        T = float(data.get('simulation_time', 100.0))
        initial_price = float(data.get('initial_price', 100.0))
        
        # Use the original covariates data instead of generating new ones
        # This ensures we use the same market conditions as the fitted models
        if not hasattr(current_simulator, '_original_covariates'):
            # If no original covariates stored, create sample data
            handler = DataHandler()
            _, covariates_df = handler.create_sample_data(T=T, n_covariates=int(T/10))
            current_simulator._original_covariates = covariates_df
        
        # Use original covariates, but scale time to match simulation duration
        original_covariates = current_simulator._original_covariates.copy()
        
        # Scale the time to match the requested simulation duration
        original_time_range = original_covariates['time'].max() - original_covariates['time'].min()
        if original_time_range > 0:
            time_scale = T / original_time_range
            original_covariates['time'] = original_covariates['time'] * time_scale
        else:
            # If no time range, create simple covariates
            handler = DataHandler()
            _, original_covariates = handler.create_sample_data(T=T, n_covariates=int(T/10))
        
        # Run simulation
        simulated_events = current_simulator.simulate_order_flow(
            original_covariates, T=T, initial_mid_price=initial_price
        )
        
        # Prepare results
        simulation_results = {
            'simulated_events': simulated_events.to_dict('records'),
            'covariates': original_covariates.to_dict('records'),
            'simulation_time': T,
            'n_events': len(simulated_events),
            'event_counts': simulated_events['event_type'].value_counts().to_dict(),
            'simulation_timestamp': datetime.now().isoformat()
        }
        
        # Convert any numpy types in simulation results
        simulation_results = convert_numpy_types(simulation_results)
        
        return jsonify({
            'success': True,
            'simulation_results': simulation_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Simulation error: {str(e)}'})


@app.route('/results')
def results():
    """Display results page."""
    return render_template('results.html')


@app.route('/api/results')
def api_results():
    """Get current results via API."""
    if current_results is None:
        return jsonify({'error': 'No results available'})
    
    return jsonify(current_results)


@app.route('/download/<file_type>')
def download_file(file_type):
    """Download generated files."""
    global current_results
    
    if current_results is None:
        return jsonify({'error': 'No results available'})
    
    try:
        if file_type == 'parameters':
            # Create parameters JSON file
            params_data = {
                'market_order_intensity_params': current_results['market_params'],
                'limit_order_intensity_params': current_results['limit_params'],
                'ask_placement_params': current_results['ask_placement_params'],
                'bid_placement_params': current_results['bid_placement_params'],
                'fitted_at': current_results['upload_time']
            }
            
            filename = f"lob_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            with open(filepath, 'w') as f:
                json.dump(params_data, f, indent=2)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        elif file_type == 'sample_data':
            # Create sample CSV files
            handler = DataHandler()
            events_df, covariates_df = handler.create_sample_data(T=100.0, n_events=100, n_covariates=20)
            
            # Create zip file
            zip_filename = f"sample_lob_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add events.csv
                events_csv = os.path.join(tempfile.gettempdir(), 'sample_events.csv')
                events_df.to_csv(events_csv, index=False)
                zipf.write(events_csv, 'events.csv')
                
                # Add covariates.csv
                covariates_csv = os.path.join(tempfile.gettempdir(), 'sample_covariates.csv')
                covariates_df.to_csv(covariates_csv, index=False)
                zipf.write(covariates_csv, 'covariates.csv')
            
            return send_file(zip_path, as_attachment=True, download_name=zip_filename)
        
        else:
            return jsonify({'error': 'Invalid file type'})
            
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'})


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    print("Starting LOB Intensity Simulator...")
    print("Access the application at: http://localhost:8080")
    
    app.run(debug=True, host='0.0.0.0', port=8080)
