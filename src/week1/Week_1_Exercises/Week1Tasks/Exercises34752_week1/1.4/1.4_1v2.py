
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

#region Constants
Rm = 10*1e6  # Mega Ohm
Cm = 1*1e-9  # Nano Farad
U_THRESH = -50*1e-3  # milli Volt
U_REST = -65*1e-3  # milli Volt
#endregion

#region Tweaks
REFRACTION_PERIOD = 8e-3  # seconds
SPIKING_VOLTAGE_THRESH = +30*1e-3  # Spike voltage in volts
SPIKING_VOLTAGE_GAIN = 20  # Gain for spiking voltage
POTASSIUM_CHANNEL_CLOSE_OFFSET = -25e-4  # Offset for potassium channel close
#endregion

#region Parameters
DT = 1e-5
T = 0.1  # Total time in seconds
_I = 3e-9  # Current in Amperes
Um_0 = -65e-3  # Initial membrane potential in volts
#endregion

def dum_dt(um_t, _I):
    """
    Helper function to calculate the change in membrane potential over time.
    """
    return (U_REST - um_t + Rm * _I) / (Rm * Cm)


class NeuronState(Enum):
    READY = 1
    SODIUM_CHANNEL_OPEN = 2
    POTASSIUM_CHANNEL_OPEN = 3
    REFRACTORY = 4

    

class Neuron:
    def __init__(self):
        self.state = NeuronState.REFRACTORY
        self.state_last_changed = 0
        self.time = 0
        
    @staticmethod
    def find_spikes(membrane_potential: np.ndarray) -> list:
        """
        Finds the indices of spikes in the membrane potential time series.
        A spike is defined as a local maximum in the membrane potential.
        """
        spikes = []
        for i in range(1, len(membrane_potential) - 1):
            if membrane_potential[i-1] < membrane_potential[i] > membrane_potential[i+1]:
                spikes.append(i)
        return spikes

    @staticmethod
    def calculate_spiking_frequency(spike_indices: list, delta_t: float) -> float:
        """
        Calculate the average spiking frequency given spike indices and timestep.
        Parameters:
        spike_indices : list
            List of indices where spikes occur.
        delta_t : float
            Time step size.
        Returns:
        frequency : float
            Spiking frequency in Hz.
        """
        if len(spike_indices) < 2:
            return 0.0
        isi = np.diff(spike_indices) * delta_t  # Interspike intervals in seconds
        avg_isi = np.mean(isi)
        return 1.0 / avg_isi if avg_isi > 0 else 0.0
        
    def update_state(self, state : NeuronState):
        """
        Handles State change logic for the neuron.
        """
        self.state_last_changed = self.time
        self.state = state
    
    def time_step(self):
        """
        Advances the neuron's time by one step.
        """
        self.time += DT
    
    def LIF(self) -> np.ndarray:
        #region initialization
        um_t = np.zeros(int(T // DT))
        um_t[0] = Um_0 #Set initial membrane potential
        
        states = np.zeros(len(um_t), dtype=NeuronState)
        states[0] = self.state
        #endregion
        
        #region simulation loop
        for idx in range(1, len(um_t)):
            
            if self.state == NeuronState.REFRACTORY:
                #region State Refraction

                # During the refractory period, the membrane potential is resting
                if um_t[idx-1] < U_REST:
                    um_t[idx] = min(um_t[idx-1] + dum_dt(um_t[idx-1], _I/4) * DT, U_REST)
                else:
                    um_t[idx] = um_t[idx-1]
                
                # Check if the neuron can transition out of the refractory state
                if self.time - self.state_last_changed >= REFRACTION_PERIOD:
                    self.update_state(NeuronState.READY)
                #endregion
            
            elif self.state == NeuronState.READY:
                #region State Ready
                
                #Slowly increase membrane potential
                um_t[idx] = um_t[idx-1] + dum_dt(um_t[idx-1], _I) * DT
                
                # Check for spike condition
                if um_t[idx] >= U_THRESH:
                    self.update_state(NeuronState.SODIUM_CHANNEL_OPEN)
                #endregion
            
            elif self.state == NeuronState.SODIUM_CHANNEL_OPEN:
                #region State Sodium Channel Open
                
                # Increase membrane potential rapidly
                um_t[idx] = um_t[idx-1] + SPIKING_VOLTAGE_GAIN * DT
                
                # Check for spike peak
                if um_t[idx] >= SPIKING_VOLTAGE_THRESH:
                    self.update_state(NeuronState.POTASSIUM_CHANNEL_OPEN)
                #endregion
            
            elif self.state == NeuronState.POTASSIUM_CHANNEL_OPEN:
                #region State Potassium Channel Open

                # Decrease membrane potential rapidly
                if um_t[idx-1] > U_REST:
                    um_t[idx] = um_t[idx-1] + dum_dt(um_t[idx-1], -10e-9) * DT
                else:
                    um_t[idx] = um_t[idx-1] + dum_dt(um_t[idx-1], -5e-10) * DT

                # Check for return to resting state
                if um_t[idx] <= U_REST + POTASSIUM_CHANNEL_CLOSE_OFFSET:
                    self.update_state(NeuronState.REFRACTORY)
                #endregion
            
            
    
            
            self.time_step()
            states[idx] = self.state
        #endregion
        
        
        return um_t, states
    
    def plot_membrane_potential(self, membrane_potential: np.ndarray, output_file: str = None, spiking_frequency = None, states = None):
        """
        Plot the membrane potential of the LIF neuron model.
        Parameters:
        membrane_potential : numpy array
            Membrane potential time series.
        """
        
        
        plt.figure(figsize=(10,5))
        time_points = np.arange(0, len(membrane_potential) * DT, DT)

        # Define colors for each state
        state_colors = {
            NeuronState.READY: 'blue',
            NeuronState.SODIUM_CHANNEL_OPEN: 'red',
            NeuronState.POTASSIUM_CHANNEL_OPEN: 'orange',
            NeuronState.REFRACTORY: 'green'
        }

        # Plot membrane potential, changing color based on state
        for i in range(len(states) - 1):
            plt.plot(time_points[i:i+2], membrane_potential[i:i+2], color=state_colors.get(states[i], 'black'))

        # Add a dummy plot for the legend
        for state, color in state_colors.items():
            plt.plot([], [], color=color, label=state.name)
        plt.xlabel('Time (s)')
        plt.ylabel('Membrane Potential (V)')
        if spiking_frequency is not None:
            plt.title(f'Membrane Potential of LIF Neuron Model [{spiking_frequency:.2f} Hz]')
        else:
            plt.title('Membrane Potential of LIF Neuron Model')
        plt.axhline(y=-50e-3, color='r', linestyle='--', label='Threshold (-50 mV)')
        plt.axhline(y=-65e-3, color='purple', linestyle='--', label='Resting Potential (-65 mV)', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        plt.show()

        
    
    def simulate(self, T: float, _I: float) -> np.ndarray:
        """
        Simulates the neuron over a given time period with a constant input current.
        """
        
        membrane_potential, states = self.LIF()
        spike_indices = self.find_spikes(membrane_potential)
        spiking_frequency = self.calculate_spiking_frequency(spike_indices, delta_t=DT)
        print("Spiking frequency (Hz):", spiking_frequency)
        self.plot_membrane_potential(membrane_potential, spiking_frequency=spiking_frequency, states=states)

Neuron1 = Neuron()
Neuron1.simulate(T, _I)
        
