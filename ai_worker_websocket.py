"""
AI WORKER WEBSOCKET SERVER
===========================

WebSocket server that bridges Node.js server with Python prediction worker.

This server:
1. Listens for WebSocket connections on port 8765
2. Receives JSON prediction requests from Node.js server
3. Converts JSON to DataFrame
4. Calls ProductionPredictor for predictions
5. Formats response for MQ5
6. Returns via WebSocket

Author: WebSocket Integration Layer
Version: 1.0
"""
# Di bagian import (sekitar line 20-30), TAMBAHKAN:
import sys
import os

# Add parent directory to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import traceback

# Import our production predictor
from ai_predict_worker_ultra import ProductionPredictor, generate_signals_simple
from utils.production_model import ProductionModel
class DataConverter:
    """Convert between MQ5 JSON format and DataFrame"""
    
    @staticmethod
    def json_to_dataframe(candles_data: list) -> pd.DataFrame:
        """
        Convert MQ5 candle JSON to DataFrame
        
        Input format from MQ5:
        [
            {"time": "2024.11.15 10:00", "open": 2650.23, "high": 2651.50, ...},
            ...
        ]
        
        Output: DataFrame with OHLCV columns
        """
        if not candles_data or len(candles_data) == 0:
            raise ValueError("No candle data provided")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles_data)
        
        # Rename columns to match our format
        column_map = {
            'time': 'Datetime',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_map)
        
        # Convert time to datetime
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')
        
        # Ensure correct dtypes
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate
        if df.isnull().any().any():
            print("⚠️  Warning: Found NaN values in data, dropping...")
            df = df.dropna()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]


class ResponseFormatter:
    """Format worker predictions for MQ5"""
    
    @staticmethod
    def format_for_mq5(results: dict, use_last_signal: bool = True) -> dict:
        """
        Convert worker output to MQ5 expected format
        
        Args:
            results: Dictionary from ProductionPredictor
            use_last_signal: If True, return only last prediction
            
        Returns:
            Dictionary matching MQ5 expected format
        """
        
        # Get the last prediction if requested
        if use_last_signal:
            idx = -1
        else:
            # Find last non-zero signal
            signals = results['signals']
            non_zero = np.where(signals != 0)[0]
            if len(non_zero) > 0:
                idx = non_zero[-1]
            else:
                idx = -1
        
        # Extract values
        signal_value = int(results['signals'][idx])
        prediction = float(results['predictions'][idx])
        regime_value = int(results['regime'][idx])
        q25 = float(results['q25'][idx])
        q75 = float(results['q75'][idx])
        close = float(results['close'][idx])
        atr = float(results['atr'][idx])
        position_size = float(results['sizes'][idx]) if results['sizes'][idx] > 0 else 0.01
        
        # Map signal to string
        signal_map = {
            -1: "SELL",
            0: "HOLD",
            1: "BUY"
        }
        signal_str = signal_map.get(signal_value, "HOLD")
        
        # Map regime to string
        regime_map = {
            0: "TREND",
            1: "RANGE",
            2: "NEUTRAL"
        }
        regime_str = regime_map.get(regime_value, "NEUTRAL")
        
        # Calculate confidence based on IQR
        iqr = q75 - q25
        median_iqr = np.median(results['q75'] - results['q25'])
        confidence = float(np.clip(1 - (iqr / (median_iqr * 2 + 1e-10)), 0, 1))
        
        # Calculate expected return (prediction is log return)
        expected_return = float(prediction)
        
        # Determine risk level based on regime and confidence
        if regime_value == 1:
            risk_level = "HIGH"  # RANGE selalu toxic
        elif confidence > 0.75:
            risk_level = "LOW"
        elif 0.55 < confidence <= 0.75:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Calculate stop loss and take profit
        if signal_value > 0:  # BUY
            stop_loss = float(close - atr * 2.0)
            take_profit = float(close + atr * 3.0)
        elif signal_value < 0:  # SELL
            stop_loss = float(close + atr * 2.0)
            take_profit = float(close - atr * 3.0)
        else:  # HOLD
            stop_loss = float(close - atr * 1.5)
            take_profit = float(close + atr * 1.5)
        
        # Round to reasonable precision
        stop_loss = round(stop_loss, 2)
        take_profit = round(take_profit, 2)
        confidence = round(confidence, 4)
        expected_return = round(expected_return, 6)
        print(f"  SL(raw): {stop_loss} | TP(raw): {take_profit} | ATR: {atr} | Close: {close}")

        # Build response
        response = {
            "signal": signal_str,
            "confidence": confidence,
            "expected_return": expected_return,
            "regime": regime_str,
            "risk_level": risk_level,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "timestamp": datetime.now().isoformat(),
            "error": None,
            "metadata": {
                "prediction": prediction,
                "q25": q25,
                "q75": q75,
                "atr": atr,
                "close": close,
                "regime_value": regime_value,
                "signal_value": signal_value
            }
        }
        
        return response


class AIWorkerWebSocket:
    """WebSocket server for AI predictions"""
    
    def __init__(self, model_path: str = "models/ensemble_7model.pkl", 
                 host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server
        
        Args:
            model_path: Path to pre-trained model PKL
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        
        # Initialize predictor
        print(f"\n{'='*60}")
        print("AI WORKER WEBSOCKET SERVER")
        print(f"{'='*60}")
        print(f"Initializing predictor with model: {model_path}")
        
        # Pastikan path absolut
        model_path = Path(model_path).resolve()

        self.predictor = ProductionPredictor(model_path=model_path)
        
        # Converters
        self.data_converter = DataConverter()
        self.response_formatter = ResponseFormatter()
        
        print(f"✅ Predictor initialized!")
        print(f"Starting WebSocket server on {host}:{port}...")
    
    async def handle_prediction_request(self, request_data: dict) -> dict:
        """
        Handle prediction request
        
        Args:
            request_data: Dictionary with prediction request
            
        Returns:
            Dictionary with prediction response
        """
        try:
            # Validate request
            if 'candles' not in request_data:
                return {
                    "error": "Missing 'candles' in request",
                    "signal": "HOLD",
                    "confidence": 0.0
                }
            
            candles = request_data['candles']
            
            if len(candles) < 50:
                return {
                    "error": f"Not enough candles (need ≥50, got {len(candles)})",
                    "signal": "HOLD",
                    "confidence": 0.0
                }
            
            # Convert JSON to DataFrame
            print(f"\n{'='*40}")
            print(f"Processing prediction request")
            print(f"{'='*40}")
            print(f"Candles received: {len(candles)}")
            
            df = self.data_converter.json_to_dataframe(candles)
            print(f"DataFrame created: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Make predictions
            results = self.predictor.process_and_predict(df)
            from ai_predict_worker_ultra import generate_signals_simple
            signals, sizes = generate_signals_simple(
            results['predictions'],
            results['q25'],
            results['q75'],
            results['regime'],
            results['close'],
            results['atr']
            )
            results['signals'] = signals
            results['sizes'] = sizes
            
            # Format response for MQ5
            response = self.response_formatter.format_for_mq5(results)
            
            print(f"\nPrediction complete:")
            print(f"  Signal: {response['signal']}")
            print(f"  Confidence: {response['confidence']:.2%}")
            print(f"  Regime: {response['regime']}")
            print(f"  Risk: {response['risk_level']}")
            print(f"  Stop Loss: {response['stop_loss']}")
            print(f"  Take Profit: {response['take_profit']}")
            print(f"  Position Size: {response['position_size']}")
            return response
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(f"\n❌ ERROR: {error_msg}")
            traceback.print_exc()
            
            return {
                "error": error_msg,
                "signal": "HOLD",
                "confidence": 0.0,
                "regime": "NEUTRAL",
                "risk_level": "HIGH",
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "position_size": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def handle_client(self, websocket, path=None):
        """Handle WebSocket client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"\n✅ Client connected: {client_id}")
        
        try:
            async for message in websocket:
                # Parse request
                try:
                    request = json.loads(message)
                except json.JSONDecodeError as e:
                    error_response = {
                        "error": f"Invalid JSON: {str(e)}",
                        "signal": "HOLD"
                    }
                    await websocket.send(json.dumps(error_response))
                    continue
                
                # Check request type
                action = request.get('action', 'predict')
                
                if action == 'predict':
                    # Handle prediction request
                    response = await self.handle_prediction_request(request.get('data', {}))
                    response["id"] = request.get("id")  # <<< tambahkan ini
                    response["type"] = "prediction"     # <<< biar Node tidak bingung
                    await websocket.send(json.dumps(response))
                
                elif action == 'ping':
                    # Health check
                    await websocket.send(json.dumps({
                        "status": "ok",
                        "timestamp": datetime.now().isoformat()
                    }))
                
                else:
                    # Unknown action
                    await websocket.send(json.dumps({
                        "error": f"Unknown action: {action}",
                        "signal": "HOLD"
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"❌ Client disconnected: {client_id}")
        
        except Exception as e:
            print(f"❌ Error handling client {client_id}: {e}")
            traceback.print_exc()
    
    async def start(self):
        """Start WebSocket server"""
        print(f"\n{'='*60}")
        print(f"✅ WebSocket server running on ws://{self.host}:{self.port}")
        print(f"{'='*60}")
        print("Waiting for connections...")
        print("Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Worker WebSocket Server')
    parser.add_argument('--model', default='models/ensemble_7model.pkl',
                       help='Path to model PKL file')
    parser.add_argument('--host', default='localhost',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8765,
                       help='Port to listen on')
    
    args = parser.parse_args()
    
    # Create and start server
    server = AIWorkerWebSocket(
        model_path=args.model,
        host=args.host,
        port=args.port
    )
    
    # Run server
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    """
    USAGE:
    
    # Start with default settings:
    python ai_worker_websocket.py
    
    # Custom model:
    python ai_worker_websocket.py --model models/my_model.pkl
    
    # Custom host/port:
    python ai_worker_websocket.py --host 0.0.0.0 --port 8765
    
    # The server will:
    # 1. Load pre-trained model
    # 2. Listen for WebSocket connections on specified port
    # 3. Process prediction requests from Node.js server
    # 4. Return formatted responses for MQ5
    """
    
    main()
