from data import TitForTat,ReverseTFT,Random,Cooperative,NonCooperative,Neutral,LearningPavlov,Population
from copy import deepcopy

# all the available strategies for the simualtion
all_strategies = {
    'Tit for Tat': TitForTat(),
    'ReverseTFT': ReverseTFT(),    
    'Random': Random(),    
    'Cooperative': Cooperative(),
    'NonCooperative': NonCooperative(),
    'Neutral': Neutral(),
    'LearningPavlov':LearningPavlov(),  
}

def create_population(input_dict):
    """creates a population from a given dictionary"""
    profile = []
    for strategy in input_dict:
        if strategy not in all_strategies:
            raise Exception('Specified strategy does not exist.')
        else:
            for i in range(input_dict[strategy]):
                profile.append(deepcopy(all_strategies[strategy]))
    return Population(profile)


