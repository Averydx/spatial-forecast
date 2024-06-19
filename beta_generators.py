import numpy as np
from numpy.typing import NDArray

def switch_beta_value(
    current_value: float, high_value: float, low_value: float
) -> float:
    """
    Switches the beta value from high to low or vice versa.

    Args:
        current_value: Current beta value.
        high_value: High beta value.
        low_value: Low beta value.

    Returns:
        The switched beta value.
    """
    return low_value if current_value == high_value else high_value

def gen_step_beta(n: int, t: int, period: int = 31) -> NDArray[np.float_]:
    """
    Generates a step-function beta that switches value at each period.

    Args:
        n: Number of locations.
        t: Number of time steps.
        period: Time step at which beta switches value.

    Returns:
        A numpy array of beta values with shape (n, t).
    """
    beta_mean = np.random.uniform(0.5, 0.7, n)
    beta = np.zeros((n, t))

    for i in range(n):
        high_value = beta_mean[i] + 0.08
        low_value = beta_mean[i] - 0.08
        current_value = high_value

        for day in range(t):
            if day % period == 0 and day != 0:
                current_value = switch_beta_value(
                    current_value, high_value, low_value
                )
            beta[i, day] = current_value

    return beta

def gen_static_beta(n:int , t:int)-> NDArray[np.float_]: 
    """
    Generates a sequence of static betas.

    Args:
        n: Number of locations.
        t: Number of time steps.

    Returns:
        A numpy array of beta values with shape (n, t).
    """

    beta = np.zeros((n, t))

    for loc in range(n):
        beta[loc,:] = np.random.uniform(0,0.5)

    return beta

