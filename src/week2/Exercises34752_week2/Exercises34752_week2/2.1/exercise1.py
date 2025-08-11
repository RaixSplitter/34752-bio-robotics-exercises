import numpy as np
import matplotlib.pyplot as plt

def sim_run(K : float, delay : float, simlen : int = 30, y0 : float = 1, target : float = 0) -> np.ndarray:
    """
    Runs a simulation with a given control gain and delay. Returns the state trajectory.
    
    Args:
        K (float): Control gain.
        delay (float): Time delay in the system.
        simlen (int): Length of the simulation.
        y0 (float): Initial state.
        target (float): Target state.

    Returns:
        np.ndarray: The state trajectory over time.
    """

    ## Initialization
    y = np.zeros((simlen))
    y[0] = y0

    # Buffer to store delayed control inputs
    u_buffer = np.zeros(simlen + delay)

    for t in range(simlen-1):
        # Compute control input
        u = K * (target - y[t])
        u_buffer[t + delay] = u  # Store control input to be applied after 'delay' steps

        # Apply delayed control input
        y[t+1] = 0.5*y[t] + 0.4*u_buffer[t+1]  # 1st order dynamics

    return y

def batch_run(K_values : list, delay_values : list, simlen : int=30, y0 : float=1, target : float=0, show = True, output : str = "src/week2/Exercises34752_week2/Exercises34752_week2/2.1/2.1.figure.png") -> dict:
    """
    Runs a batch simulation for different control gains and delays.

    Args:
        K_values (list): List of control gains.
        delay_values (list): List of time delays.
        simlen (int): Length of the simulation.
        y0 (float): Initial state.
        target (float): Target state.
        show (bool): Whether to show the plots.

    Returns:
        dict: A dictionary with keys as (K, delay) tuples and values as the state trajectories.
    """
    results = {}
    for K in K_values:
        for delay in delay_values:
            y = sim_run(K, delay, simlen, y0, target)
            results[(K, delay)] = y

    if show or output:
        n_plots = int(np.ceil(np.sqrt(len(delay_values))))
        fig, ax = plt.subplots(n_plots, n_plots, figsize=(10, 8))
        for i, delay in enumerate(delay_values):
            for j, K in enumerate(K_values):
                y = results[(K, delay)]
                row = int(i // n_plots)
                col = int(i % n_plots)
                ax[row, col].plot(range(len(y)), y)
            ax[row, col].set_title(f'delay={delay}')
            ax[row, col].set_xlabel('Time step')
            ax[row, col].set_ylabel('y')
            ax[row, col].grid()
            ax[row, col].legend([f'K={K}' for K in K_values])

        fig.suptitle('State Trajectories for Different Control Gains and Delays', fontsize=16)
        plt.tight_layout()

        
        if output:
            plt.savefig(output)
        if show:
            plt.show()

        

K_VALUES = [0.1, 0.5, 1.0, 2.0]
DELAY_VALUES = [0, 1, 2, 3]

batch_run(K_VALUES, DELAY_VALUES)
