from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any

class GrowthModel(ABC):
    """Base class for all growth models."""
    
    @abstractmethod
    def calculate_growth_rate(self, t: float, dt: float, **kwargs) -> float:
        """
        Calculate the growth rate for a given time step.
        
        Args:
            t: Current time since reference date
            dt: Time step over which to calculate growth
            **kwargs: Additional parameters specific to the growth model
            
        Returns:
            float: Growth rate for the given time step
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters of the growth model."""
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Set the parameters of the growth model."""
        pass

class FixedGrowthModel(GrowthModel):
    """Simple exponential growth model with fixed growth rate."""
    
    def __init__(self, growth_rate: float = 0.0):
        """
        Initialize the fixed growth model.
        
        Args:
            growth_rate: Annual growth rate (e.g., 0.1 for 10% annual growth)
        """
        self.growth_rate = growth_rate
        self.metadata = {
            'model_type': 'fixed',
            'description': 'Fixed annual growth rate model'
        }
    
    def calculate_growth_rate(self, t: float, dt: float, **kwargs) -> float:
        """
        Calculate the growth rate for a given time step.
        
        Args:
            t: Current time since reference date (not used for fixed growth)
            dt: Time step over which to calculate growth
            
        Returns:
            float: Growth rate for the given time step
        """
        # Convert annual growth rate to growth rate for the time step (apply log growth)
        return np.log(1 + self.growth_rate) * dt
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "growth_rate": self.growth_rate,
            "metadata": self.metadata
        }
    
    def set_parameters(self, **kwargs) -> None:
        if "growth_rate" in kwargs:
            self.growth_rate = kwargs["growth_rate"]

class PowerLawGrowthModel(GrowthModel):
    """Power law growth model where log return = k*log(1+dt/t)."""
    
    def __init__(self, k: float = 0.1):
        """
        Initialize the power law growth model.
        
        Args:
            k: Growth constant that determines the strength of the power law
        """
        self.k = k
        self.metadata = {
            'model_type': 'power_law',
            'description': 'Power law growth model with diminishing returns'
        }
    
    def calculate_growth_rate(self, t: float, dt: float, **kwargs) -> float:
        """
        Calculate the growth rate using power law formula.
        
        Args:
            t: Current time since reference date
            dt: Time step over which to calculate growth
            
        Returns:
            float: Growth rate for the given time step
        """
        if t <= 0:
            return 0.0
        return self.k * np.log(1 + dt/t)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "metadata": self.metadata
        }
    
    def set_parameters(self, **kwargs) -> None:
        if "k" in kwargs:
            self.k = kwargs["k"]

class ExogenousGrowthModel(GrowthModel):
    """Growth model where growth rate depends on an exogenous variable."""
    
    def __init__(self, growth_function: callable):
        """
        Initialize the exogenous growth model.
        
        Args:
            growth_function: Function that takes time and exogenous variables as input
                           and returns the growth rate
        """
        self.growth_function = growth_function
        self.exogenous_data = {}
        self.metadata = {
            'model_type': 'exogenous',
            'description': 'Custom growth model based on external factors'
        }
    
    def calculate_growth_rate(self, t: float, dt: float, **kwargs) -> float:
        """
        Calculate the growth rate using the provided growth function.
        
        Args:
            t: Current time since reference date
            dt: Time step over which to calculate growth
            **kwargs: Additional parameters to pass to the growth function
            
        Returns:
            float: Growth rate for the given time step
        """
        # Pass all kwargs to the growth function
        return self.growth_function(t, dt, **kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "growth_function": self.growth_function,
            "exogenous_data": self.exogenous_data,
            "metadata": self.metadata
        }
    
    def set_parameters(self, **kwargs) -> None:
        if "growth_function" in kwargs:
            self.growth_function = kwargs["growth_function"]
        if "exogenous_data" in kwargs:
            self.exogenous_data = kwargs["exogenous_data"]

class CompositeGrowthModel(GrowthModel):
    """Combines multiple growth models with weights."""
    
    def __init__(self, models: Dict[str, GrowthModel], weights: Optional[Dict[str, float]] = None):
        """
        Initialize the composite growth model.
        
        Args:
            models: Dictionary of growth models
            weights: Optional dictionary of weights for each model
        """
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: weight/total_weight for name, weight in self.weights.items()}
        
        self.metadata = {
            'model_type': 'composite',
            'description': 'Combined growth model with weighted components',
            'components': {name: model.metadata for name, model in models.items()}
        }
    
    def calculate_growth_rate(self, t: float, dt: float, **kwargs) -> float:
        """
        Calculate the weighted average growth rate from all models.
        
        Args:
            t: Current time since reference date
            dt: Time step over which to calculate growth
            **kwargs: Additional parameters to pass to the growth models
            
        Returns:
            float: Combined growth rate for the given time step
        """
        total_growth = 0.0
        for name, model in self.models.items():
            total_growth += self.weights[name] * model.calculate_growth_rate(t, dt, **kwargs)
        return total_growth
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "models": self.models,
            "weights": self.weights,
            "metadata": self.metadata
        }
    
    def set_parameters(self, **kwargs) -> None:
        if "models" in kwargs:
            self.models = kwargs["models"]
        if "weights" in kwargs:
            self.weights = kwargs["weights"]
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {name: weight/total_weight for name, weight in self.weights.items()} 