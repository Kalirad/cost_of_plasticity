"""
A matrix population model, which utilizes the experimental data to project the
population dynamics of two strains of Pristionchus pacificus, in isolation and 
together, on two different bacterial diets.
"""

__author__ = 'Ata Kalirad'

__version__ = '1.0.2'


import numpy as np

# A = Strain RSC017, C= RS5405, OP50 = E.coli, and Novo = Novosphingobium 

# predation paramters for each strain under different conditions.
att_prob = {'OP50':{'A':0.00017,'C':0.00033},
           'Novo':{'A':0.000064,'C':0.00047}}

# Probabilities of Eu mouth form for different strains.
mf_prob = {'OP50':{'A':0.02,'C':1.0},
           'Novo':{'A':0.9,'C':1.0}}

# The fecundity parameters to calculate fecundity values for five different breeding stages in the life cycle.
fec_pars = {'A': {'OP50': [22.65, 68.45, 57.05, 33.4,  4.97], 'Novo': [11.66, 62.53, 47.13, 13.94,  0.72]},
            'C': {'OP50': [19.8 , 60.3 , 43.02, 19.9,  6.6 ], 'Novo': [16.88, 80.77, 77.7 , 16.28,  1.4]}}

# consumption rates for each developmental stage
cons = np.array([0.0, 0.2, 0.0, 0.3, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

# different arrangements of food in the one-dimensional stepping-stone model.
env_arr = {'ONO': ['OP50' for i in range(6)] + ['Novo' for i in range(6)],
           'NON': ['Novo' for i in range(6)] + ['OP50' for i in range(6)],
            'ON': ['OP50', 'Novo'] * 6, 
            'NO': ['Novo', 'OP50'] * 6}

# The transition matrix in the absence of food
ge = g2 = g3 = g4  = gya  = 0.0
g2d = 0.1
gd4 = 0.0
gb1 = gb2 = gb3 = gb4 = gb4 = gb5 = 0.0415
doa = 0.995
U_starv = np.matrix([[(1. - ge), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [ge, (1. - g2) * (1. - g2d), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, g2d,(1. - gd4), 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, g2, 0, (1. - g3), 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, gd4, g3, (1. - g4), 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, g4, (1. - gya), 0, 0, 0, 0, 0, 0],  
                      [0, 0, 0, 0, 0, gya, (1. - gb1), 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, gb1, (1. - gb2), 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, gb2, (1 - gb3), 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, gb3, (1 - gb4), 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, gb4, (1 - gb5), 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gb5, doa]])

def get_transition_probs(strain, food_type, food):
    """Generate transition probability for a strain.

    Args:
        strain (str): A (RSC017) or C (RS5405)
        food_type (str): The bacterial diet (Novo or OP50)
        food (float): The amount of food available to the population.

    Returns:
        numpy matrix: transition probabilities.
    """
    if food == 0:
        return U_starv
    else:
        ge = 0.0415
        g2 = 0.055
        g3 = 0.085
        g4  = 0.07
        gya  = 0.1
        g2d = 0.0
        gd4 = 0.1
        gb1 = gb2 = gb3 = gb4 = gb4 = gb5= 0.0415
        doa = 0.995
        
        if food_type == 'Novo':
            if strain == 'C':
                gya  = 0.4 # ~ 6 hours
        #     if strain == 'A':
        #         gya  = 0.13 # ~8 hours
        #     elif strain == 'C':
        #         gya  = 0.4 # ~ 6 hours
            

        U = np.matrix([[(1. - ge), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [ge, (1. - g2) * (1. - g2d), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, g2d,(1. - gd4), 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, g2, 0, (1. - g3), 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, gd4, g3, (1. - g4), 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, g4, (1. - gya), 0, 0, 0, 0, 0, 0], 
                       [0, 0, 0, 0, 0, gya, (1. - gb1), 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, gb1, (1. - gb2), 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, gb2, (1 - gb3), 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, gb3, (1 - gb4), 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, gb4, (1 - gb5), 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gb5, doa]])
        return U

def gen_fec_matrix(strain, food_type):
    """Generate fecundity matrix.

    Args:
        strain (str): A (RSC017) or C (RS5405)
        food_type (str): The bacterial diet (Novo or OP50)

    Returns:
        numpy matrix: fecundity values for each breeding developmental stage.
    """
    F = np.identity(12) * 0
    count = 0
    for i in np.arange(6, 11, 1):
        F[0][i] = gb1*fec_pars[strain][food_type][count]
        count += 1
    return F

def simulate_pop_dynamic(strain_1, strain_2, food_type, inti_food=1e12, mig_rate=0.1, n_pop=12, t=1000, predation=True):
    """Simulate the population dynamics as a Markov process.

    Args:
        strain_1 (str): A (RSC017) or C (RS5405)
        strain_2 (str): A (RSC017) or C (RS5405)
        food_type (str): The bacterial diet (Novo or OP50)
        inti_food (float, optional): Initial amount of food available to the population. . Defaults to 1e12.
        mig_rate (float, optional): The proportion of dauer larvae that migrate from a source locality to a sink locality in each step.. Defaults to 0.1.
        n_pop (int, optional): The number of localities in the population. Defaults to 12.
        t (int, optional): The number of projection steps.. Defaults to 1000.
        predation (bool, optional): If True, adults prey upon J2 individuals. . Defaults to True.

    Returns:
        dictionary
    """

    pop_dims = (n_pop, 1)
    env = np.zeros(shape=(pop_dims[1], pop_dims[0])).tolist()
    index = []
    for i in range(pop_dims[1]):
        for j in range(pop_dims[0]):
            index.append((i,j))
    if food_type == 'OP50' or food_type =="Novo":
        for i in index:
            env[i[0]][i[1]] = food_type
    elif type(food_type) == list:
        for i,j in zip(index, food_type):
            env[i[0]][i[1]] = j
    else:
        for i,j in zip(index, env_arr[food_type]):
            env[i[0]][i[1]] = j
            
    food = np.zeros(shape=(pop_dims[1], pop_dims[0]))
    for i in index:
        food[i[0]][i[1]]=inti_food

    pop1 = np.zeros(shape=(pop_dims[1], pop_dims[0]))
    pop2 = np.zeros(shape=(pop_dims[1], pop_dims[0]))
    pop1[0][0] = 1
    pop2[0][n_pop-1] = 1
    n1 = np.array([[0], [0], [0], [0], [0], [50], [0], [0], [0], [0], [0], [0]])
    n2 = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    pop1_dem = np.zeros(shape=(pop_dims[1], pop_dims[0])).tolist()
    pop2_dem = np.zeros(shape=(pop_dims[1], pop_dims[0])).tolist()
    for i in index:
        if pop1[i[0]][i[1]]:
            pop1_dem[i[0]][i[1]] = n1
        else:
            pop1_dem[i[0]][i[1]] = n2
    for i in index:
        if pop2[i[0]][i[1]]:
            pop2_dem[i[0]][i[1]] = n1
        else:
            pop2_dem[i[0]][i[1]] = n2

    f_1 = np.zeros(shape=(pop_dims[1], pop_dims[0])).tolist()
    f_2 = np.zeros(shape=(pop_dims[1], pop_dims[0])).tolist()
    for i in index:
        f_1[i[0]][i[1]] = gen_fec_matrix(strain_1, env[i[0]][i[1]])
        f_2[i[0]][i[1]] = gen_fec_matrix(strain_2, env[i[0]][i[1]])
    data = {'pop1':{}, 'pop2':{}}
    mig_data = {'dauer_A':{}, 'dauer_C':{}}
    for i in index:
        data['pop1'][i] = []
        data['pop2'][i] = []
        mig_data['dauer_A'][i] = []
        mig_data['dauer_C'][i] = []
    constant_resource=False
    neighb_ind = {}
    for i in index:
        pot_neighbours = [(i[0]+1,i[1]), (i[0]-1,i[1]), (i[0],i[1]+1), (i[0],i[1]-1)]
        real_neighbours = [nex for nex in pot_neighbours if nex in index]
        neighb_ind[i] = real_neighbours
    for j in range(t):
        for i in index:
            data['pop1'][i].append(pop1_dem[i[0]][i[1]])
            data['pop2'][i].append(pop2_dem[i[0]][i[1]])
            pop1_dem[i[0]][i[1]] = np.array(np.matmul(get_transition_probs(strain_1, env[i[0]][i[1]], food[i[0]][i[1]]) + f_1[i[0]][i[1]], pop1_dem[i[0]][i[1]]))
            pop2_dem[i[0]][i[1]] = np.array(np.matmul(get_transition_probs(strain_2, env[i[0]][i[1]], food[i[0]][i[1]]) + f_2[i[0]][i[1]], pop2_dem[i[0]][i[1]]))
            if not constant_resource:
                food[i[0]][i[1]] -= np.sum(cons.reshape(len(cons),1)*(pop1_dem[i[0]][i[1]] + pop2_dem[i[0]][i[1]])) 
                if food[i[0]][i[1]] < 0:
                    food[i[0]][i[1]] = 0
        # migration
        for i in index:
            for nex in neighb_ind[i]:
                if np.round(food[nex[0]][nex[1]], decimals=1) > np.round(food[i[0]][i[1]], decimals=1):
                    # The amount of food is rounded to one decimal point to remove the effect of infinitesimal differences in food on dispersal
                    disp_rate = mig_rate
                else:
                    disp_rate = 0.0
                migrants = disp_rate * pop1_dem[i[0]][i[1]][2][0] 
                pop1_dem[i[0]][i[1]][2][0] = pop1_dem[i[0]][i[1]][2][0] - migrants
                pop1_dem[nex[0]][nex[1]][2][0] = pop1_dem[nex[0]][nex[1]][2][0] + migrants
                mig_data['dauer_A'][i].append(migrants)
                migrants = disp_rate * pop2_dem[i[0]][i[1]][2][0]
                pop2_dem[i[0]][i[1]][2][0] = pop2_dem[i[0]][i[1]][2][0] - migrants
                pop2_dem[nex[0]][nex[1]][2][0] = pop2_dem[nex[0]][nex[1]][2][0] + migrants      
                mig_data['dauer_C'][i].append(migrants)
            if predation:
                # Predation by RS5405
                n_predator = np.sum(pop2_dem[i[0]][i[1]][5:])  
                if n_predator > 0:
                    n_prey = pop1_dem[i[0]][i[1]][1]
                    pred_by_2 =  att_prob[env[i[0]][i[1]]][strain_2]*n_predator*n_prey
                    if pred_by_2 > n_prey:
                        pred_by_2 = n_prey
                else:
                    pred_by_2 = 0
                # Predation by RSC017
                n_predator = mf_prob[env[i[0]][i[1]]][strain_1]*np.sum(pop1_dem[i[0]][i[1]][5:])
                if n_predator > 0:
                    n_prey = pop2_dem[i[0]][i[1]][1]
                    pred_by_1 =  att_prob[env[i[0]][i[1]]][strain_1]*n_predator*n_prey
                    if pred_by_1 > n_prey:
                        pred_by_1 = n_prey
                else:
                    pred_by_1 =0 
                pop1_dem[i[0]][i[1]][1] = pop1_dem[i[0]][i[1]][1] - pred_by_2
                pop2_dem[i[0]][i[1]][1] = pop2_dem[i[0]][i[1]][1] - pred_by_1
    return data, index, mig_data