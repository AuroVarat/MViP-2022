import numpy as np


def glauber_dynamic_numpy():
    
    #select a random co-ordinate
    itrial, jtrial = np.random.randint(0,N, size=2)
    #select the corresponding spin and reverse it
    spin_new= -spin[itrial,jtrial]
    #find the particle number count in the nerighbour matrix 
    count = itrial*N+jtrial
    
    delE = np.multiply(-2*spin_new, np.sum(spin_neighbour_matrix[count]) )
    
    change_spin(delE,itrial,jtrial,count)
    
    
def kawasaki_dynamic():
    first_choice, second_choice = np.random.randint(0,N, size=(2,2))
    
    first_choice_spin= spin[tuple(first_choice)]
    second_choice_spin = spin[tuple(second_choice)]
    
    count_1 =  np.sum(first_choice*[N,1])
    count_2 = np.sum(second_choice*[N,1])

    neighbors = np.where(neighbour_matrix[count_1]!= 0)[0]
    
    if  first_choice_spin != second_choice_spin and neighbors.any()!=count_2 :
        first_choice_spin_new= -first_choice_spin
        second_choice_spin_new = -second_choice_spin
        e = calc_energy_change(first_choice_spin_new,count_1) + calc_energy_change(second_choice_spin_new,count_2)
        change_spin(e,first_choice[0],first_choice[1],count_1)
        change_spin(e,second_choice[0],second_choice[1],count_2)
        
    elif first_choice_spin != second_choice_spin and neighbors.any()== count_2:
        first_choice_spin_new= -first_choice_spin
        second_choice_spin_new = -second_choice_spin
        e = calc_energy_change(first_choice_spin_new,count_1) + calc_energy_change(second_choice_spin_new,count_2)+4
        change_spin(e,first_choice[0],first_choice[1],count_1)
        change_spin(e,second_choice[0],second_choice[1],count_2)