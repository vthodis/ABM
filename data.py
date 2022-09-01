from agents import *
import statistics 

game_states = []
p1_moves = []
p2_moves = []
mutual_cooperation_list = []

def describe(df,stats):
    d = df.describe()
    return d.append(df.reindex(d.columns,axis =1).agg(stats))

def round_dictionary(dictionary):
    res = dict()
    for key in dictionary:
        # rounding to K using round()
        res[key] = round(dictionary[key], 3)
    return res

def format_list(list):
    list = ["%.3f" % member for member in list]
    list = [float(x) for x in list]
    return list

def mad(arr):
    arr = np.ma.array(arr).compressed() 
    med = np.median(arr)
    return np.median(np.abs(arr - med))
        
class TitForTat(Agent):
    """Starts by cooperating. After that, always cooperates unless
    opponent's last move was defect."""

    def __init__(self, score=0):
        super().__init__()
        self.name = "Tit for Tat"
        self.is_first_move = True
        self.opponent_last_action = None

    def decide_action(self):
        if self.is_first_move is True:
            action =  True
            self.is_first_move = False
        else:
            action =  self.opponent.last_action
            
        return action

    def new_match_against(self, opponent):
        self.is_first_move = True
        super().new_match_against(opponent)

    def post_action(self):        
        self.opponent_last_action = self.opponent.this_action
        self.update_last_payoff()
        self.update_last_action()

class ReverseTFT(Agent):
    """Starts by defecting. After that, always defects unless
    opponent's last move was defect, in which case play the opposite of your last action"""

    def __init__(self, score=0):
        super().__init__()
        self.name = "ReverseTFT"
        self.is_first_move = True

    def decide_action(self):
        if self.is_first_move == True:
            action =  False
        else:
            if self.opponent.last_action == False:
                action =  not self.last_action
            else:
                action = False
        
        self.is_first_move = False
        return action

    def new_match_against(self, opponent):
        self.is_first_move = True
        super().new_match_against(opponent)

    def post_action(self):        
        self.update_last_action()
        self.update_last_payoff()    
    

class LearningPavlov(Pavlov1):
    def __init__(self, score=0):
        super().__init__(score)
        self.name = "LearningPavlov"  


class Cooperative(Pavlov2):
    def __init__(self, score=0):
        super().__init__(score)
        self.name = "Cooperative"
        
    def init_emotional_potential(self):
        
        self.emotional_potential_to_cooperate = 0.6
        self.emotional_potential_to_defect = 0.4
        #initial potentila are based on personality 
        self.potential_to_cooperate = self.emotional_potential_to_cooperate
        self.potential_to_defect = self.emotional_potential_to_defect
  

class NonCooperative(Pavlov2):
    def __init__(self, score=0):
        super().__init__(score)
        self.name = "NonCooperative"

    def init_emotional_potential(self):
        
        self.emotional_potential_to_cooperate = 0.4
        self.emotional_potential_to_defect = 0.6
        #initial potentila are based on personality 
        self.potential_to_cooperate = self.emotional_potential_to_cooperate
        self.potential_to_defect = self.emotional_potential_to_defect

class Neutral(Pavlov2):
    def __init__(self, score=0):
        super().__init__(score)
        self.name = "Neutral"
        # W(S)
    def init_emotional_potential(self):        
        self.emotional_potential_to_cooperate = 0.5
        self.emotional_potential_to_defect = 0.5
        #initial potentila are based on personality 
        self.potential_to_cooperate = self.emotional_potential_to_cooperate
        self.potential_to_defect = self.emotional_potential_to_defect
        
class Random(Agent):
    """ Cooperates/defects with 50% chance. """
    def __init__(self, score=0):
        super().__init__()
        self.name = "Random"

    def decide_action(self):
        return random.choice([True,False])

    def post_action(self):        
        self.update_last_action()
        self.update_last_payoff() 

#The class of the population of agents in the simulation
class Population(object):
    
    def __init__(self, members: list):
        self.members = members
    
    def __repr__(self):
        return "{}".format(self.members)

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return self.members[item]

    def append(self, member):
        self.members.append(member)

    def scores(self):
        return [member.score for member in self.members]

    def reset_all_scores(self):
        for member in self.members:
            member.reset_score()

    def first_member(self):
        return self.members[0]

    def excluding(self, excluded_member):
        copy = self.members[:]
        return Population([member for member in copy if member is not excluded_member])

    def is_empty(self):
        return not self.members

    def total_score(self):
        total = 0
        for member in self.members:
            total += member.score
        return total
    
    def score_differences(self):
        # total points gathered,median of points gathered and mad of points gathered
        points_gathered = self.scores()
        median_agent_score  = statistics.median(points_gathered)
        mad_agent_score = mad(points_gathered)

        # Median and Mean absolute deviation for the differences in agents' scores
        unique_diff = abs(np.diff(points_gathered))
        agent_point_differnces = (unique_diff).tolist()
        agent_point_differnces.sort()
        agent_point_differnces_median = statistics.median(agent_point_differnces)
        average_diff_mad  = mad(agent_point_differnces)
        # RETURN AVERAGE AGENT SCORE, THE MEADIAN FOR SCORE DIFFERENCES, THE STANDARD DEVIATION FOR tOTAL POINTS GATHERED
        # AND THE MEAN ABSOLUTE DEVIATION FOR THE SCORE DIFFERENCES
        return [median_agent_score ,agent_point_differnces_median, mad_agent_score, average_diff_mad, agent_point_differnces  ]
    
def play_round(p1, p2, turn,mutual_cooperation_counter):
    
    state = []
    # action_selection
    p1_action, p2_action = p1.action(), p2.action()   
    
    # (C,C)
    if p1_action == True and p2_action == True:
        state = [1,1]
        if p1.name == "Cooperative" or p1.name == "NonCooperative" or p1.name == "Neutral" or p1.name == 'LearningPavlov':
            mutual_cooperation_counter += 1
            mutual_cooperation_list.append(mutual_cooperation_counter)
        if p2.name == "Cooperative" or p2.name == "NonCooperative" or p2.name == "Neutral" or p2.name == 'LearningPavlov' :  
            mutual_cooperation_counter += 1
            mutual_cooperation_list.append(mutual_cooperation_counter)  
        
        p1.add_points(pmax_c)        
        p2.add_points(pmax_c)
    #(C,D) 
    elif p1_action == True and p2_action == False:
        state = [1,0]
        p1.add_points(pmin_c)
        p2.add_points(pmax_d)        
        if p1.name == "Cooperative" or p1.name == "NonCooperative" or p1.name == "Neutral"  or p1.name == 'LearningPavlov':
            mutual_cooperation_list.append(mutual_cooperation_counter)
        if p2.name == "Cooperative" or p2.name == "NonCooperative" or p2.name == "Neutral"  or p2.name == 'LearningPavlov':  
            mutual_cooperation_list.append(mutual_cooperation_counter)
    #(D,C) 
    elif p1_action == False and p2_action == True:
        state = [0,1]
        p2.add_points(pmin_c)
        p1.add_points(pmax_d)        
        if p1.name == "Cooperative" or p1.name == "NonCooperative" or p1.name == "Neutral" or p1.name == 'LearningPavlov':
            mutual_cooperation_list.append(mutual_cooperation_counter)
        if p2.name == "Cooperative" or p2.name == "NonCooperative" or p2.name == "Neutral"  or p2.name == 'LearningPavlov':  
            mutual_cooperation_list.append(mutual_cooperation_counter)
    #(D,D)
    elif p1_action == False and p2_action == False :
        state = [0,0]
        p1.add_points(pmin_d)
        p2.add_points(pmin_d)
        if p1.name == "Cooperative" or p1.name == "NonCooperative" or p1.name == "Neutral"   or p1.name == 'LearningPavlov' :
            mutual_cooperation_list.append(mutual_cooperation_counter)
        if p2.name == "Cooperative" or p2.name == "NonCooperative" or p2.name == "Neutral"  or p2.name == 'LearningPavlov' :  
            mutual_cooperation_list.append(mutual_cooperation_counter)
    
    p1_moves.append(p1.this_action)
    p2_moves.append(p2.this_action)
    p1.post_action()
    p2.post_action()
    p1.turn += 1
    p2.turn += 1
    game_states.append(state)
    return mutual_cooperation_counter

# several rounds between 2 playerss
def play_several_rounds(p1, p2, num_rounds,mutual_cooperation_counter):
    p1.new_match_against(p2)
    p2.new_match_against(p1)
    # print("NEW MATCH ")
    for i in range(num_rounds):
        # initialize each match with the players involved and the 
        # number of the current round in the timeline(ie how long is a player in the environment)
        mutual_cooperation_counter = play_round(p1, p2,i,mutual_cooperation_counter)        
    return mutual_cooperation_counter

# round robin tournament of the population
def round_robin(population, num_rounds, mutual_cooperation_counter,use_flag):
    """Competes every member of the population with every other member
    of the population for num_rounds each.
    """
    if not population.is_empty():
        other_members = population.excluding(population.first_member())
        for member in other_members:
            mutual_cooperation_counter = play_several_rounds(population.first_member(), member, num_rounds,mutual_cooperation_counter)
        if use_flag == 1:
            mutual_cooperation_counter = round_robin(other_members, num_rounds,mutual_cooperation_counter,1)
        else:
            pass
        
    return mutual_cooperation_counter