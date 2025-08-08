import numpy as np
import matplotlib.pyplot as plt

# Point 1.1
def LIF(um_0, _I, T, expansion_disable = True):
    """
    Calculate the membrane potential of a Leaky Integrate-and-Fire (LIF) neuron model.
    Parameters:
    um_0 : float
        Initial membrane potential in volts.
    I : float
        Constant input current in nano amperes [nA].
    T : float
        Total time of simulation in seconds.
    """

    Rm = 10*1e6 # Mega Ohm
    Cm = 1*1e-9 # Nano Farad
    u_thresh = -50*1e-3 # milli Volt
    u_rest = -65*1e-3 # milli Volt
    refraction_period = 5e-3 # seconds

    delta_t = 1e-5
    um_t = np.zeros((int(T//delta_t)))
    um_t[0] = um_0

    def dum_dt(um_t):
        return (u_rest - um_t + Rm*_I)/(Rm*Cm)
   
    # TODO Calculate the um_t from T = 0 until T in steps of delta-t

    _T = 0
    idx = 0
    refraction_toggle = False
    refraction_last_active = 0
    
    spiking_voltage_thresh = +30*1e-3 # Spike voltage in volts
    spiking_voltage_gain = 5 # Gain for spiking voltage
    spiking_toggle = False
    
    while idx < len(um_t):

        delta_u = dum_dt(um_t[idx]) # Calculate the change in membrane potential

        #region Refraction Logic
        if refraction_toggle and not expansion_disable:
            if _T >= refraction_last_active + refraction_period:
                refraction_toggle = False
                time_since_last_spike = _T - refraction_last_active
                if idx + 1 < len(um_t):
                    um_t[idx+1] = um_t[idx] - (delta_u/2) * delta_t # Update the membrane potential
            else:
                if idx + 1 < len(um_t):
                    um_t[idx+1] = um_t[idx] - (delta_u/2) * delta_t # Update the membrane potential
                idx += 1
                _T += delta_t
                continue
        #endregion

        if spiking_toggle:
            if idx + 1 < len(um_t):
                um_t[idx+1] = um_t[idx] + spiking_voltage_gain * delta_t # Update the membrane potential with spike gain
        
        elif idx + 1 < len(um_t):
            um_t[idx+1] = um_t[idx] + delta_u * delta_t # Update the membrane potential
        
        if um_t[idx] > spiking_voltage_thresh and not expansion_disable: # Check for spike peak
            spiking_toggle = False
            if idx + 1 < len(um_t):
              um_t[idx+1] = u_rest # Set the membrane potential to spike voltage
            refraction_toggle = True
            refraction_last_active = _T
        
        elif um_t[idx] > u_thresh: # Check for spike
            if not expansion_disable:
                spiking_toggle = True
            else:
                if idx + 1 < len(um_t):
                    um_t[idx+1] = u_rest # Reset the membrane potential after spike
            
            # #region Refraction Funnel
            # if not expansion_disable:
            #     refraction_toggle = True
            #     refraction_last_active = _T
            # #endregion
        _T += delta_t
        idx += 1

    return um_t


# Point 1.2

def plot_membrane_potential(membrane_potential, output_file='membrane_potential.png'):
    """
    Plot the membrane potential of the LIF neuron model.
    Parameters:
    membrane_potential : numpy array
        Membrane potential time series.
    """
    plt.figure(figsize=(10,5))
    plt.plot(list(range(int(0.1//1e-5))), membrane_potential)
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (V)')
    plt.title(f'Membrane Potential of LIF Neuron Model [{spiking_frequency:.2f} Hz]')
    plt.axhline(y=-50e-3, color='r', linestyle='--', label='Threshold (-50 mV)')
    plt.axhline(y=-65e-3, color='g', linestyle='--', label='Resting Potential (-65 mV)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig('membrane_potential.png')
    plt.show()

# TODO: Calculate the membrane potential using the LIF function from Point 1.1



# plt.figure(figsize=(10,5))
# plt.plot(list(range(int(0.1//1e-5))), membrane_potential)
# plt.xlabel('Time (s)')
# plt.ylabel('Membrane Potential (V)')
# plt.title('Membrane Potential of LIF Neuron Model')
# plt.axhline(y=-50e-3, color='r', linestyle='--', label='Threshold (-50 mV)')
# plt.axhline(y=-65e-3, color='g', linestyle='--', label='Resting Potential (-65 mV)')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid()
# plt.tight_layout()
# plt.savefig('membrane_potential.png')
# plt.show()


# Point 1.3
# TODO: Define a function to calculate the interspike intervals
def find_spikes(um_t):
    """
    Find the indices of spikes in the membrane potential time series.
    Parameters:
    um_t : numpy array
        Membrane potential time series.
    Returns:
    spikes : list
        List of indices where spikes occur.
    """
    spikes = []
    for i in range(1, len(um_t)-1):
        if um_t[i-1] < um_t[i] > um_t[i+1]:
            spikes.append(i)
    return spikes

# calculate_isi = 

# TODO: Define a function to calculate the spiking frequency of a whole experiment
def calculate_spiking_frequency(spike_indices, delta_t):
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


membrane_potential = LIF(um_0=-65e-3, _I=2e-9, T=0.1) # 1nA current for 0.1 seconds
spike_indices = find_spikes(membrane_potential)
print("Spike indices:", spike_indices)
spiking_frequency = calculate_spiking_frequency(spike_indices, delta_t=1e-5)
print("Spiking frequency (Hz):", spiking_frequency)
plot_membrane_potential(membrane_potential)


membrane_potential = LIF(um_0=-65e-3, _I=2e-9, T=0.1, expansion_disable=False) # 1nA current for 0.1 seconds
spike_indices = find_spikes(membrane_potential)
print("Spike indices:", spike_indices)
spiking_frequency = calculate_spiking_frequency(spike_indices, delta_t=1e-5)
print("Spiking frequency (Hz):", spiking_frequency)
plot_membrane_potential(membrane_potential, output_file='membrane_potential_refraction.png')

# membrane_potential2 = LIF(TODO)
#plt.figure(figsize=(7,5))
#plt.plot(list(range(int(0.1//1e-5))), membrane_potential2)
#plt.show()


# # Point 1.4
# plt.figure(figsize=(7,5))
spikes = []
# # TODO write the code to accumulate the spikes

start = 0
end = 5.5e-9
step_size = 0.5e-9/4
indices = np.arange(start, end, step_size)

for current in indices: # Loop through different current values
    membrane_potential = LIF(um_0=-65e-3, _I=current, T=0.1) # 1nA current for 0.1 seconds
    spike_indices = find_spikes(membrane_potential)
    spiking_frequency = calculate_spiking_frequency(spike_indices, delta_t=1e-5)
    spikes.append(spiking_frequency)

plt.plot(indices, spikes)
plt.scatter(indices, spikes, color='red', label='Spikes', s=50, marker='o')
plt.xticks(rotation=45)
plt.xlim(0, 5.5e-9)
plt.xlabel('Constant current')
plt.ylabel('Spiking frequency')
plt.title('Spiking Frequency vs Constant Current')
plt.grid()
plt.tight_layout()
plt.savefig('spiking_frequency_vs_current.png')
plt.show()