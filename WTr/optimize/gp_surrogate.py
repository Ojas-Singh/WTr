"""
Optional Gaussian Process surrogate for Bayesian optimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm

class GPSurrogate:
    """
    Gaussian Process surrogate model for descriptor-barrier optimization.
    """
    
    def __init__(self, kernel=None, alpha=1e-6, normalize_y=True):
        """
        Initialize GP surrogate.
        
        Args:
            kernel: GP kernel (default: RBF with automatic relevance determination)
            alpha: Noise level
            normalize_y: Whether to normalize target values
        """
        if kernel is None:
            # Default kernel with ARD (different length scales per dimension)
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * 6, (1e-2, 1e2))
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=10
        )
        
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        self.gp.fit(self.X_train, self.y_train)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Make predictions with uncertainty.
        
        Args:
            X: Feature matrix for prediction
            return_std: Whether to return standard deviation
            
        Returns:
            Predictions and optionally standard deviations
        """
        if not self.is_fitted:
            raise RuntimeError("GP not fitted. Call fit() first.")
        
        return self.gp.predict(X, return_std=return_std)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score on test data."""
        if not self.is_fitted:
            raise RuntimeError("GP not fitted. Call fit() first.")
        
        return self.gp.score(X, y)

class AcquisitionFunction:
    """Base class for acquisition functions."""
    
    def __call__(self, X: np.ndarray, gp: GPSurrogate, y_best: float) -> np.ndarray:
        """Evaluate acquisition function."""
        raise NotImplementedError

class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def __init__(self, xi: float = 0.01):
        """
        Args:
            xi: Exploration parameter (higher = more exploration)
        """
        self.xi = xi
    
    def __call__(self, X: np.ndarray, gp: GPSurrogate, y_best: float) -> np.ndarray:
        """
        Calculate Expected Improvement.
        
        Args:
            X: Candidate points
            gp: Fitted GP model
            y_best: Best observed value
            
        Returns:
            EI values for each candidate point
        """
        mu, sigma = gp.predict(X, return_std=True)
        
        # Handle numerical precision
        sigma = np.maximum(sigma, 1e-9)
        
        # Calculate improvement
        improvement = y_best - mu - self.xi  # Minimization (lower is better)
        Z = improvement / sigma
        
        # Expected improvement
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei

class BayesianOptimizer:
    """
    Bayesian optimization for descriptor-based surface design.
    """
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 acquisition_func: AcquisitionFunction = None,
                 gp_surrogate: GPSurrogate = None):
        """
        Initialize Bayesian optimizer.
        
        Args:
            bounds: List of (min, max) bounds for each descriptor dimension
            acquisition_func: Acquisition function (default: EI)
            gp_surrogate: GP surrogate model
        """
        self.bounds = bounds
        self.n_dims = len(bounds)
        
        if acquisition_func is None:
            acquisition_func = ExpectedImprovement(xi=0.01)
        self.acquisition_func = acquisition_func
        
        if gp_surrogate is None:
            gp_surrogate = GPSurrogate()
        self.gp = gp_surrogate
        
        self.X_observed = []
        self.y_observed = []
        self.iteration = 0
    
    def add_observation(self, x: np.ndarray, y: float):
        """
        Add a new observation to the dataset.
        
        Args:
            x: Descriptor vector
            y: Objective value (barrier height)
        """
        self.X_observed.append(np.array(x))
        self.y_observed.append(y)
    
    def fit_gp(self):
        """Fit GP to current observations."""
        if len(self.X_observed) < 2:
            raise RuntimeError("Need at least 2 observations to fit GP")
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        self.gp.fit(X, y)
    
    def _optimize_acquisition(self, n_restarts: int = 10) -> np.ndarray:
        """
        Find point that maximizes acquisition function.
        
        Args:
            n_restarts: Number of random restarts for optimization
            
        Returns:
            Optimal point
        """
        if not self.gp.is_fitted:
            self.fit_gp()
        
        y_best = min(self.y_observed)  # Best = minimum barrier
        
        def objective(x):
            # Negative because we want to maximize acquisition
            return -self.acquisition_func(x.reshape(1, -1), self.gp, y_best)[0]
        
        best_x = None
        best_val = float('inf')
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            
            # Optimize
            res = minimize(objective, x0, bounds=self.bounds, method='L-BFGS-B')
            
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        
        if best_x is None:
            # Fallback to random point
            best_x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
        
        return best_x
    
    def suggest_next(self) -> np.ndarray:
        """
        Suggest next point to evaluate.
        
        Returns:
            Descriptor vector for next evaluation
        """
        if len(self.X_observed) < 2:
            # Random exploration for first few points
            return np.array([np.random.uniform(low, high) for low, high in self.bounds])
        
        return self._optimize_acquisition()
    
    def run_optimization(self, objective_func: callable, n_iterations: int,
                        initial_points: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Run Bayesian optimization loop.
        
        Args:
            objective_func: Function that takes descriptor vector and returns barrier
            n_iterations: Number of optimization iterations
            initial_points: Optional initial points to evaluate
            
        Returns:
            Optimization results dictionary
        """
        # Evaluate initial points
        if initial_points:
            for x in initial_points:
                y = objective_func(x)
                self.add_observation(x, y)
        
        # Main optimization loop
        for i in range(n_iterations):
            print(f"Bayesian optimization iteration {i+1}/{n_iterations}")
            
            # Suggest next point
            x_next = self.suggest_next()
            
            # Evaluate objective
            try:
                y_next = objective_func(x_next)
                self.add_observation(x_next, y_next)
                
                # Report progress
                current_best = min(self.y_observed)
                print(f"  Suggested point: {x_next}")
                print(f"  Objective value: {y_next:.3f} eV")
                print(f"  Current best: {current_best:.3f} eV")
                
            except Exception as e:
                print(f"  Warning: Evaluation failed: {e}")
                continue
        
        # Return results
        best_idx = np.argmin(self.y_observed)
        results = {
            'best_x': self.X_observed[best_idx],
            'best_y': self.y_observed[best_idx],
            'X_observed': np.array(self.X_observed),
            'y_observed': np.array(self.y_observed),
            'gp_model': self.gp,
            'n_iterations': len(self.X_observed)
        }
        
        return results

def gp_fit(X: np.ndarray, y: np.ndarray) -> GPSurrogate:
    """
    Fit Gaussian Process to data.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        
    Returns:
        Fitted GP surrogate model
    """
    gp = GPSurrogate()
    gp.fit(X, y)
    return gp

def gp_suggest(gp: GPSurrogate, bounds: List[Tuple[float, float]], n: int = 1) -> np.ndarray:
    """
    Suggest next points using Expected Improvement.
    
    Args:
        gp: Fitted GP model
        bounds: Bounds for each dimension
        n: Number of suggestions
        
    Returns:
        Suggested points (n, n_dims)
    """
    acquisition = ExpectedImprovement()
    optimizer = BayesianOptimizer(bounds, acquisition, gp)
    
    suggestions = []
    for _ in range(n):
        x_next = optimizer._optimize_acquisition()
        suggestions.append(x_next)
    
    return np.array(suggestions)
