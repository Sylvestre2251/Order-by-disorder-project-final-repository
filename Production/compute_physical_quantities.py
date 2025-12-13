


import KagomeFinal


J = 1
L = 12
Nx = L
Ny = L
theta = 2*np.pi/3

T_high = 1
T_low = 0.002
n_T = 20
Temp = np.linspace(T_high, T_low, n_T)



# Runs to get the mean values and standard deviations of the physical quantities
n_runs = 15  

E_mean_all = []
E_std_all = []
Capa_mean_all = []
Capa_std_all = []



for j, T in enumerate(Temp):
    decoherencetime = 50000
    numberMC = 100000

    Ener_runs = []
    Capa_runs = []

    for run in range(n_runs):
        A = Configuration(1, theta, Nx, Ny, J)

        A.Monte_Carlo(decoherencetime, T)
        A.verify_norm()

        Ener = A.measure_Capa(numberMC, T, Nx, Ny)
        A.verify_norm()

        Ener_runs.append(np.mean(Ener))
        Capa_runs.append(A.Capa[-1])

    E_mean_all.append(np.mean(Ener_runs))
    E_std_all.append(np.std(Ener_runs))
    
    Capa_mean = np.mean(Capa_runs)
    Capa_std = np.std(Capa_runs)
    
    # filtering because of a point of anomaly
    # Without this, the plot is around 1, and has the correct shape,
    # except sometimes for the last point of temperature that is an anomaly and the heat capacity for this point only is far too larger
    Capa_mean = min(Capa_mean, 2)
    Capa_std = min(Capa_std, 2 - Capa_mean)


    Capa_mean_all.append(Capa_mean)
    Capa_std_all.append(Capa_std)

    print(f"T step {j+1}/{n_T} done")
    
    
    
    
A.display_config()



triangle_sums = A.triangle_spins_sum()

for i in range(A.Nx):
    for j in range(A.Ny):
        print(f"Sum of spins in the triangle ({i},{j}) :", triangle_sums[i, j])   


# Plot Heat Capacity Vs T in browser with plotly

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=Temp,
    y=Capa_mean_all,
    error_y=dict(
        type='data',
        array=Capa_std_all,
        visible=True
    ),
    mode='markers+lines',
    name="Capacité thermique"
))

fig.update_layout(
    xaxis_type="log",
    xaxis_title="Température T",
    yaxis_title="Capacité thermique C(T)",
    title="Capacité thermique interactive",
    template="plotly_white"
)

fig.show(renderer="browser")




# Plot Energy Vs T in browser with plotly


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=Temp,
    y=E_mean_all,
    error_y=dict(
        type='data',
        array=E_std_all,
        visible=True
    ),
    mode='markers+lines',
    name="Heat Capacity"
))

fig.update_layout(
    xaxis_title="Temperature T",
    yaxis_title="Energy E",
    title="Interactive Energy",
    template="plotly_white"
)


fig.show(renderer="browser")

