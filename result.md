[DQNAgent] Model loaded ← models/checkpoints/dqn_calibrated/dqn_best.pth

> > > def make_state(e1_cpu=0.05, e2_cpu=0.05, c_cpu=0.05,  
> > > ... task_cpu=0.3, task_ram=0.3, task_dl=0.5):  
> > > ... """Build realistic state vector."""  
> > > ... return np.array([  
> > > ... e1_cpu, 0.1, 0.001, 0.0, # edge_1: cpu, ram, lat, queue  
> > > ... make_state(0.05, 0.05, 0.05, 0.10, 0.10, 0.8))

[1] Cold cluster + light task (cpu=10, ram=10, dl=long)  
 edge_1 : Q=-25.968  
 edge_2 : Q=-25.923  
 cloud : Q=-25.526  
 reject : Q=-27.188  
... make_state(0.05, 0.05, 0.05, 0.60, 0.50, 0.3))

[2] Cold cluster + heavy task (cpu=60, ram=50, dl=tight)  
 edge_1 : Q=-27.365  
 edge_2 : Q=-26.737  
 cloud : Q=-26.234  
 reject : Q=-27.108  
... make_state(0.95, 0.05, 0.10, 0.20, 0.20, 0.6))

[3] edge_1 saturated, light task  
 edge_1 : Q=-26.565  
 edge_2 : Q=-26.465  
 cloud : Q=-27.244  
 reject : Q=-27.419  
... make_state(0.95, 0.95, 0.10, 0.20, 0.20, 0.6))

[4] Both edges saturated, light task  
 edge_1 : Q=-27.010  
 edge_2 : Q=-26.952  
 cloud : Q=-26.504  
 reject : Q=-27.754  
... make_state(0.05, 0.05, 0.05, 0.30, 0.25, 0.5))

[5] Real K8s-like (CPU all 5%, queue 0)  
 edge_1 : Q=-26.174  
 edge_2 : Q=-26.305  
 cloud : Q=-24.956  
 reject : Q=-26.691  
... make_state(0.05, 0.05, 0.05, 0.20, 0.50, 0.5))

[6] Cold cluster + data-heavy task  
 edge_1 : Q=-25.145  
 edge_2 : Q=-25.082  
 cloud : Q=-25.204  
 reject : Q=-25.974  
 → argmax = edge_2, gap = 0.063
