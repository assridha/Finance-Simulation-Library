"""
Service layer for handling simulation operations.
"""
import queue
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import yfinance as yf

@dataclass
class SimulationJob:
    """Data class for simulation job information."""
    id: str
    config: Dict[str, Any]
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SimulationService:
    """Service for managing option strategy simulations."""
    
    def __init__(self):
        """Initialize the simulation service."""
        self._job_queue = queue.Queue(maxsize=100)
        self._active_jobs: Dict[str, SimulationJob] = {}
        self._completed_jobs: Dict[str, SimulationJob] = {}
        self._user_configs: Dict[str, Dict[str, Any]] = {}
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for a symbol.
        
        Args:
            symbol: Stock symbol to get data for
            
        Returns:
            Dictionary containing market data
            
        Raises:
            ValueError: If symbol is invalid or data cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data for price trends and volatility calculation
            hist = ticker.history(period="1mo")
            if hist.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            latest_price = hist['Close'].iloc[-1]
            price_change = latest_price - hist['Close'].iloc[-2]
            price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
            
            # Calculate historical volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility in percentage
            
            return {
                'symbol': symbol,
                'price': latest_price,
                'change': price_change,
                'change_percent': price_change_pct,
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'high_52week': info.get('fiftyTwoWeekHigh', None),
                'low_52week': info.get('fiftyTwoWeekLow', None),
                'volatility': volatility,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            raise ValueError(f"Error fetching market data: {str(e)}")
    
    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict[str, Any]:
        """Get option chain data for a symbol.
        
        Args:
            symbol: Stock symbol to get options for
            expiry: Optional expiry date to filter by
            
        Returns:
            Dictionary containing option chain data
            
        Raises:
            ValueError: If symbol is invalid or data cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get all expiration dates if none specified
            expirations = ticker.options
            if not expirations:
                raise ValueError(f"No options available for symbol: {symbol}")
            
            if expiry:
                if expiry not in expirations:
                    raise ValueError(f"Invalid expiry date: {expiry}")
                expirations = [expiry]
            
            all_calls = []
            all_puts = []
            
            for exp in expirations:
                opt = ticker.option_chain(exp)
                if opt.calls.empty and opt.puts.empty:
                    continue
                
                # Process calls
                calls = opt.calls.to_dict('records')
                for call in calls:
                    call['type'] = 'call'
                    call['expiry'] = exp
                all_calls.extend(calls)
                
                # Process puts
                puts = opt.puts.to_dict('records')
                for put in puts:
                    put['type'] = 'put'
                    put['expiry'] = exp
                all_puts.extend(puts)
            
            return {
                'symbol': symbol,
                'expiry_dates': expirations,
                'calls': all_calls,
                'puts': all_puts,
                'underlying_price': ticker.history(period='1d')['Close'].iloc[-1],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ValueError(f"Error fetching option chain: {str(e)}")
    
    def validate_strategy_config(self, config: Dict[str, Any]) -> None:
        """Validate a strategy configuration.
        
        Args:
            config: Strategy configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['symbol', 'strategy_type', 'legs']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if config['strategy_type'] not in self.get_available_strategies():
            raise ValueError(f"Invalid strategy type: {config['strategy_type']}")
        
        # TODO: Add more validation logic
    
    def queue_simulation(self, simulation_id: str, config: Dict[str, Any]) -> None:
        """Queue a new simulation job.
        
        Args:
            simulation_id: Unique identifier for the simulation
            config: Strategy configuration dictionary
        """
        job = SimulationJob(
            id=simulation_id,
            config=config,
            status='queued',
            created_at=datetime.now()
        )
        self._active_jobs[simulation_id] = job
        self._job_queue.put(job)
    
    def get_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Get results of a simulation by ID.
        
        Args:
            simulation_id: ID of the simulation to get results for
            
        Returns:
            Dictionary containing simulation results or status
            
        Raises:
            ValueError: If simulation ID is not found
        """
        if simulation_id in self._completed_jobs:
            job = self._completed_jobs[simulation_id]
            return {
                'status': job.status,
                'results': job.results,
                'error': job.error,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None
            }
        elif simulation_id in self._active_jobs:
            job = self._active_jobs[simulation_id]
            return {
                'status': job.status,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'progress': {
                    'current_step': 'Initializing' if job.status == 'queued' else 'Running simulation',
                    'percent_complete': 0 if job.status == 'queued' else 50
                }
            }
        else:
            raise ValueError(f"Simulation not found: {simulation_id}")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy types.
        
        Returns:
            List of strategy type names
        """
        return [
            'single',
            'vertical_spread',
            'butterfly',
            'iron_condor',
            'calendar_spread',
            'diagonal_spread'
        ]
    
    def save_user_config(self, config: Dict[str, Any]) -> str:
        """Save a user's strategy configuration.
        
        Args:
            config: Configuration dictionary to save
            
        Returns:
            Configuration ID
            
        Raises:
            ValueError: If configuration is invalid
        """
        config_id = str(len(self._user_configs) + 1)
        self._user_configs[config_id] = config
        return config_id
    
    def get_user_config(self, config_id: str) -> Dict[str, Any]:
        """Get a saved user configuration.
        
        Args:
            config_id: ID of the configuration to retrieve
            
        Returns:
            Saved configuration dictionary
            
        Raises:
            ValueError: If configuration ID is not found
        """
        if config_id not in self._user_configs:
            raise ValueError(f"Configuration not found: {config_id}")
        return self._user_configs[config_id]
    
    def _process_queue(self) -> None:
        """Process jobs in the simulation queue."""
        while True:
            try:
                job = self._job_queue.get()
                job.status = 'running'
                job.started_at = datetime.now()
                
                try:
                    # TODO: Implement actual simulation execution
                    job.results = {'message': 'Simulation completed'}
                    job.status = 'completed'
                except Exception as e:
                    job.error = str(e)
                    job.status = 'failed'
                
                job.completed_at = datetime.now()
                self._completed_jobs[job.id] = job
                del self._active_jobs[job.id]
                
            except Exception as e:
                print(f"Error processing job: {e}")
            finally:
                self._job_queue.task_done() 