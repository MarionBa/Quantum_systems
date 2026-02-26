from ryd_numerov.rydberg import RydbergState
import numpy as np

### Transition |1> --> |2>
state_i = RydbergState("Rb", 5, 0, j_tot=1 / 2, m=1 / 2)
state_f = RydbergState("Rb", 5, 1, j_tot=3 / 2, m=3 / 2)

dipole = state_i.calc_matrix_element(state_f, "ELECTRIC", k_radial=1, k_angular=1, q=1) # q = [+1, 0, -1] - light polarization

print(f"Numerov dipole matrix element: {dipole}")

### Transition |2> --> |3>
state_i = RydbergState("Rb", 5, 1, j_tot=3 / 2, m=3 / 2)
state_f = RydbergState("Rb", 33, 2, j_tot=5 / 2, m=5 / 2)

dipole = state_i.calc_matrix_element(state_f, "ELECTRIC", k_radial=1, k_angular=1, q=1) # q = [+1, 0, -1] - light polarization

print(f"Numerov dipole matrix element: {dipole}")

