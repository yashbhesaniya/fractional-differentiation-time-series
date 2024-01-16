'''
Synthetic Backtesting


This module implements the synthetic backtesting method proposed by Lopez de Prado (2018).
The method is based on the Ornstein-Uhlenbeck process, which is a stochastic process that
describes the evolution of a particle under the influence of friction. 

The model is under construction and can possible contain multiple stochastic processes, but
currently Ornstein-Uhlenbeck is the only one (yet).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import sys


def generate_paths_ornstein_uhlenbeck(
        initial_price: float, 
        forecast: float, 
        sigma: float, 
        phi: float,
        MAXIMUM_HOLDING_PERIOD: int = 100, 
        NUM_SCENARIOS: int = int(1e5),
        seed_value: int = 42):
    '''
    ---

    Generates paths P_t according to the Ornstein-Uhlenbeck process:

        P_t = (1 - ϕ)E[P_T] + ϕP_(t-1) + σε_t
    
    ---
    Args
    ---

        initial_price: float = P_0 at the current data point (the present).
        forecast: float = E[P_T]
        sigma: float = σ
        phi: float = ϕ
        MAXIMUM_HOLDING_PERIOD: int = T, corresponds to the vertical bar of exit the position 
        NUM_SCENARIOS: int = N, number of scenarios generated

    ---
    Output
    ---

        _generated_paths.T: pd.DataFrame = a dataframe with the generated paths. 
            Each column represents a scenario and each row represents a time period.
            The first row stands for a constant value, equals to the current price. 
    
    ---
    '''
    MAXIMUM_HOLDING_PERIOD, NUM_SCENARIOS = int(MAXIMUM_HOLDING_PERIOD), int(NUM_SCENARIOS)

    _generated_paths = pd.DataFrame(0, 
        index = range(NUM_SCENARIOS), 
        columns = range(MAXIMUM_HOLDING_PERIOD+1))
    
    np.random.seed(seed_value)

    _white_noise = pd.DataFrame(
        np.random.normal(loc = 0, scale = 1, size = (NUM_SCENARIOS, MAXIMUM_HOLDING_PERIOD+1))
    )

    # guarantee that the first value is the initial price
    _white_noise[0], _generated_paths[0] = 0, initial_price

    # Generate paths
    for t in range(1, MAXIMUM_HOLDING_PERIOD+1): 
        _generated_paths[t] = (1-phi)*forecast \
        + phi*_generated_paths[t-1] \
        + sigma*_white_noise[t]
    
    return _generated_paths.T

def estimate_parameters_ornstein_uhlenbeck(
        historical_prices: pd.Series,
        position: int, 
        forecast: float
    ):

    '''
    ---

    Estimates the parameters ϕ and σ of the Ornstein-Uhlenbeck process:

        P_t = (1 - ϕ)E[P_T] + ϕP_(t-1) + σε_t
    
    ---
    Args
    ---

        historical_prices: pd.Series the history of prices of a singular asset.
        position : index of historical_prices corresponding to when one took a position.
        forecast : forecasted value of the position

    ---
    Output
    ---

        a list of two floats:
            the first corresponds to the parameter sigma
            the second, to phi.
    
    ---
    '''

    _X = np.array(historical_prices[position :-1] - forecast)
    _Y = np.array(historical_prices[position + 1:])
    _Z = np.repeat(forecast, len(historical_prices[position :]) -1)

    # covariance operator    
    cov = lambda v1, v2: v1.dot(v2)

    _phi = cov(_Y-_Z,_X)/cov(_X,_X)
    xi = _Y - _Z - _phi*_X

    # mean operator with degrees of freedom
    mu = lambda x, degrees_of_freedom = 1: np.sum(x)/(len(x) - degrees_of_freedom)

    _sigma = np.sqrt( mu(x = xi**2) - mu(x = xi)**2 )

    return {'phi_hat': _phi, 'sigma_hat':_sigma}

def mesh_ornstein_uhlenbeck(
        initial_price: float,
        sigma: float, 
        MAX_MESH: int = 10,
        MESH_DIMENSION: int = 20):
    '''
    ---

    Creates a set of meshs to backtest with.
    The meshs are the horizontal bars to exit the position
    
    ---
    Args
    ---

        initial_price: float, the avarage price of the  bet.
        sigma: float, the sigma of Orstein-Uhlenbeck process.
        MAX_MESH: int, the maximum distance, in sigmas, of a bar and the forecasted value.
        MESH_DIMENSION: int, the amount of lower/upper horizontal bars.
    
    ---
    Output
    ---

        An array of MESH_DIMENSION² pairs of horizontal bars.
        The entry [0] of a pair corresponds to the lower bound,
            [1], to the upper bound.
    
    ---
    '''
    _mesh_seed = sigma * np.linspace(
        start = (MAX_MESH), 
        stop = (MAX_MESH/MESH_DIMENSION), 
        num = MESH_DIMENSION
    )
    _lower_pi, _upper_pi = [(t*_mesh_seed) + initial_price for t in [-1,1]]

    # Deals with negative lower bars
    if MAX_MESH > (initial_price/sigma):
        _lower_pi = np.linspace(
            start = initial_price/MESH_DIMENSION, 
            stop = initial_price,
            num = MESH_DIMENSION, 
            endpoint = False
        )
    
    return np.array(np.meshgrid(_lower_pi, _upper_pi)).T.reshape(-1,2)

def simulate_one_box(
        generated_paths: pd.DataFrame,
        trading_rule: np.array):
    '''
    ---
    
    Simulates the exit prices of a trading rule
    on the generated scenarios

    ---
    Args
    ---

        generated_paths: a pd.DataFrame with the generated paths.
            The rows must be the time ticks.
            The columns must be the diferent scenarios.
        trading_rule: np.array, must be a pair.
            The first entry, [0], must be pi underscore / lower barrier.
            The second, [1], must be pi bar / upper barrier.

    ---
    Output
    ---

        A list with length corresponding to the number of scenarios.
        The entry [i] is the result of simulating the trading rule
            on the scenario [i].
    
    ---
    '''
    # Get dimensions
    MAXIMUM_HOLDING_PERIOD, NUM_SCENARIOS = generated_paths.shape
    MAXIMUM_HOLDING_PERIOD -= 1
    _simulation = []
   
    pi_under, pi_over = trading_rule
   
    for j in range(NUM_SCENARIOS):
        _tick = 1
        while True:
            if (_tick > MAXIMUM_HOLDING_PERIOD) \
                or any([(generated_paths[j][_tick] < pi_under),
                        (generated_paths[j][_tick] > pi_over)]):
                _simulation.append(generated_paths[j][min(MAXIMUM_HOLDING_PERIOD, _tick)])
                break
            _tick += 1
    return _simulation 

def progressbar(it, prefix="", size=60, out=sys.stdout): 
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "█"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def simulate_trading_rules(
        generated_paths: pd.DataFrame,
        mesh: np.ndarray
    ):
    '''
    ---
    
    Simulates every trade rule on the generated scenarios

    ---
    Args
    ---

        generated_paths: a pd.DataFrame with the generated paths.
            The rows must be the time ticks.
            The columns must be the diferent scenarios.
        mesh: np.array, an array of pairs of horizontal bars.
            The entry [0] of a pair corresponds to the lower bound,
                and [1], to the upper bound.
    
    ---
    Output
    ---

        A dictionary with two entries:
            'results': pd.DataFrame, with three columns:
                'stop-loss', 'profit-taking', 'sharpe_ratio'.
                For each row, the entries 'stop-loss' and 'profit-taking'
                    correspond to the horizontal bars evaluated. The entrie
                    'sharpe_ratio' is the sharpe ratio associeted with these
                    horizontal bars.
            'simulates': pd.DataFrame, the columns correspond to a pair of
                horizontal bars. The rows, to each scenario.
                The entry [i][j] corresponds to the exit price of the pair [i]
                    of bars in the scenario [j].

    ---
    '''
    n = len(mesh)
    _simulates = {}

    for i,m in zip(range(len(mesh)), 
                   progressbar(mesh, "Simulate one box for each trading rule:")):
        _simulates[i] = simulate_one_box(generated_paths, m)
    
    _simulates = pd.DataFrame(_simulates)

    _results = pd.DataFrame({
        'stop-loss': mesh[:, 0].tolist(),
        'profit-taking': mesh[:,1].tolist(),
        'sharpe_ratio': _simulates.apply(sharpe_ratio, initial_price = generated_paths[0][0]).values.tolist()
    })
    return {'results':_results, 'simulates':_simulates}

def tau_to_phi(tau: float):
    '''
    ---

    Converts the half-life of the Ornstein-Uhlenbeck process, tau,
    to the parameter phi:
        tau = - log[2]/log[phi]
    '''
    return 2**(-1./tau)

def plot_ornstein_uhlenbeck(generated_paths: pd.DataFrame = None, **key_args):
    '''
    ---

    Auxiliary function to visualization_ornstein_uhlenbeck.

        P_t = (1 - ϕ)E[P_T] + ϕP_(t-1) + σε_t
    
    ---
    Args
    ---

        INITIAL_PRICE: float = 100, corresponds to the P_0 in the above formula.
        FORECAST: float = 110, corresponds to the E[P_T] in the above formula
        SIGMA: float = .1, corresponds to the σ in the above formula.
        PHI: float = .5, corresponds to the ϕ in the above formula.
        MAXIMUM_HOLDING_PERIOD: int = T, corresponds to the vertical bar of exit the position.
        NUM_SCENARIOS: float = N, number of scenarios to plot.

    ---
    Output
    ---
    
    Does NOT return anything. This is an auxiliary function to visualization_ornstein_uhlenbeck.
    ---
    '''
    if generated_paths is None:
        generated_paths = generate_paths_ornstein_uhlenbeck(**key_args)
    
    INITIAL_PRICE = key_args.get('INITIAL_PRICE')
    FORECAST = key_args.get('FORECAST')
    SIGMA = key_args.get('SIGMA')

    _ax = generated_paths.plot(
        color = ['yellow', 'orange', "goldenrod"], 
        alpha = 0.5, 
        linewidth = 2, 
        figsize = (12,5), 
        legend = False
    )
    _ax.axhline(y = INITIAL_PRICE, ls = ":", color = "black")
    _ax.axhline(y = FORECAST, ls = ":", color = "black")

    plt.show()

def plot_ornstein_uhlenbeck_inter(
        INITIAL_PRICE: float = 100,
        FORECAST: float = 110, 
        SIGMA: float = .1,
        PHI:float  = .5,
        MAXIMUM_HOLDING_PERIOD: int = 100,
        NUM_SCENARIOS: int = 100):
    '''
    ---

    Auxiliary function to visualization_ornstein_uhlenbeck.

        P_t = (1 - ϕ)E[P_T] + ϕP_(t-1) + σε_t
    
    ---
    Args
    ---

        INITIAL_PRICE: float = 100, corresponds to the P_0 in the above formula.
        FORECAST: float = 110, corresponds to the E[P_T] in the above formula
        SIGMA: float = .1, corresponds to the σ in the above formula.
        PHI: float = .5, corresponds to the ϕ in the above formula.
        MAXIMUM_HOLDING_PERIOD: int = T, corresponds to the vertical bar of exit the position.
        NUM_SCENARIOS: float = N, number of scenarios to plot.

    ---
    Output
    ---
    
    Does NOT return anything. This is an auxiliary function to visualization_ornstein_uhlenbeck.
    ---
    '''
    _generated_paths = generate_paths_ornstein_uhlenbeck(
        INITIAL_PRICE, 
        FORECAST, 
        SIGMA, 
        PHI,
        MAXIMUM_HOLDING_PERIOD, 
        NUM_SCENARIOS
    )
    
    _ax = _generated_paths.plot(
        color = ['yellow', 'orange', "goldenrod"], 
        alpha = 0.5, 
        linewidth = 2, 
        figsize = (12,5), 
        legend = False
    )
    _ax.axhline(y = INITIAL_PRICE, ls = ":", color = "black")
    _ax.axhline(y = FORECAST, ls = ":", color = "black")
    

def visualization_ornstein_uhlenbeck():
    '''
    ---

    Function that aids one to visualize the evolution of an Ornstein-Uhlenbeck process.
    The formula for this process is found below

        P_t = (1 - ϕ)E[P_T] + ϕP_(t-1) + σε_t
    
    Call this function on a notebook to obtain an interactive plot.
    
    ---
    Args
    ---
    
    Does not receive any argument.
    
    ---
    Output
    ---
    
    When you call this function on a notebook, it returns an interactive plot.
    
    ---
    '''
    _controls = widgets.interactive(plot_ornstein_uhlenbeck_inter,
        INITIAL_PRICE = widgets.IntSlider(min = 100, max = 100, step = 1, value = 100),
        FORECAST = widgets.IntSlider(min = 90, max = 110, step = 1, value = 105), 
        MAXIMUM_HOLDING_PERIOD = widgets.IntSlider(min = 10, max = 1000, step = 1, value = 50),
        PHI = (-1.1, 1.1, .01),
        SIGMA = (0, .5, .01),
        NUM_SCENARIOS = widgets.IntSlider(min = 10, max = 1000, step = 10, value = 50)
    )
    display(_controls)

def sharpe_ratio(
        simulation: list,
        initial_price: float
    ):
    '''
    ---
    
    Computes the Sharpe Ratio of the results obtained simulating a box.

    ---
    Args
    ---

        simulation: list, a list with length corresponding to the number of scenarios.
            The entry [i] is the result of simulating the trading rule on the scenario [i].
        initial_price: float, the avarage price of the position taken.
    
    ---
    Output
    ---

        A float corresponding to the Sharpe ratio of the simulated results.

    ---
    '''
    _returns = (simulation-initial_price)*100/initial_price

    _std_returns = np.std(_returns)
    
    if _std_returns == 0:
        return 0
    return np.mean(_returns)/_std_returns

def optimal_trading_rule(
        historical_prices: pd.Series, 
        position: int,
        forecast: float,
        MAXIMUM_HOLDING_PERIOD: int = 100,
        NUM_SCENARIOS: int = int(1e3),
        MAX_MESH: int = 10,
        MESH_DIMENSION: int = 20,
        TEST = None,
        TO_PRINT = True
    ) -> pd.DataFrame:    
    '''
    ---
    
    Simulates every trade rule on the generated scenarios

    ---
    Args
    ---
        historical_prices: pd.Series the history of prices of a singular asset.
        position: index of historical_prices corresponding to when one took a position.
        forecast: float, forecasted value of the position.
        MAXIMUM_HOLDING_PERIOD: int = T, corresponds to the vertical bar of exit the position.
        NUM_SCENARIOS: float = N, number of scenarios to plot.
        MAX_MESH: int, the maximum distance, in sigmas, of a bar and the forecasted value.
        MESH_DIMENSION: int, the amount of lower/upper horizontal bars.
        TEST: to conduct a test with specific parameters, must be a numeric list: [sigma, phi].
        TO_PRINT: True, if true will print the generated scenarios.
        
    
    ---
    Output
    ---
        Returns a pd.DataFrame with the horizontal bars that generated the best
        Sharpe ratio together with the correponding Sharpe ratio.
    
        If you are conducting a test, returns a dictionary with two entries:
            'results': pd.DataFrame, with three columns:
                'stop-loss', 'profit-taking', 'sharpe_ratio'.
                For each row, the entries 'stop-loss' and 'profit-taking'
                    correspond to the horizontal bars evaluated. The entrie
                    'sharpe_ratio' is the sharpe ratio associeted with these
                    horizontal bars.
            'simulates': pd.DataFrame, the columns correspond to a pair of
                horizontal bars. The rows, to each scenario.
                The entry [i][j] corresponds to the exit price of the pair [i]
                    of bars in the scenario [j].

    ---
    '''


    if TEST is not None:
        _sigma, _phi = TEST
        _initial_price = historical_prices[0]
    else:
        estimated_params = estimate_parameters_ornstein_uhlenbeck(
            historical_prices = historical_prices,
            position = position, 
            forecast = forecast
        )
        _sigma, _phi = [estimated_params.get(estimate) for estimate in ['sigma_hat', 'phi_hat']]
        _initial_price = historical_prices.iloc[-1]
    
    if TO_PRINT:
        print(f"> sigma: {np.round(_sigma, 4)}, phi: {np.round(_phi, 4)}")

    _paths = generate_paths_ornstein_uhlenbeck(
        initial_price = _initial_price, 
        forecast = forecast,
        sigma = _sigma, 
        phi = _phi,
        MAXIMUM_HOLDING_PERIOD = MAXIMUM_HOLDING_PERIOD,
        NUM_SCENARIOS = NUM_SCENARIOS
    )

    if TO_PRINT:
        plot_ornstein_uhlenbeck(
            generated_paths = _paths, 
            INITIAL_PRICE = _initial_price, 
            FORECAST = forecast, 
            SIGMA = _sigma)
    
    _mesh = mesh_ornstein_uhlenbeck(
        initial_price = _initial_price,
        sigma = _sigma, 
        MAX_MESH = MAX_MESH,
        MESH_DIMENSION = MESH_DIMENSION
    )

    # print("> Simulating trading rules...")
    _results = simulate_trading_rules(generated_paths = _paths, mesh = _mesh)

    if TEST is not None:
        return _results

    return _results['results'].sort_values('sharpe_ratio', ascending = False).head(1)
