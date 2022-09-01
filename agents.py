from random import randint,uniform
import random
import numpy as np
import math as mt
import decimal

# data structures to be returned to the main program
cooperation_potential_list = []
defection_potential_list = []
mood_vector_list = []
mood_intensity_list = []
mood_categorization_list = []
desirability_of_state_list = []
worthiness_of_state_list = []
total_emotional_category_list = []
emotional_potential_to_cooperate_list = []
emotional_potential_to_defect_list = []
moves_played = []
history_of_payoffs = []
learning_coeffiecient_list = []
reward_list = [ ]
# List of cooperation potential in a certain state
s1_pc_list = [0]
s2_pc_list = [0]
s3_pc_list = [0]
s4_pc_list = [0]
# GLOBALS OF THE GAME
temp = []
# min/max payoffs from the game
pmin_c = 0
pmin_d = 1
pmax_c = 3
pmax_d = 5
s1 = [1,1]
s2 = [1,0]
s3 = [0,1]
s4 = [0,0]
# OCC emotions mapped to PAD space
admiration = [0.5, 0.3 , -0.2]
anger = [-0.51 , 0.59, 0.25]
distress = [-0.4, -0.2, -0.5]
gratitude = [0.4, 0.2 ,-0.3]
joy = [0.4 , 0.2 , 0.1]
reproach = [-0.3, -0.1 , 0.4 ]
# PAD emotions vectors
anger_vector = np.array(anger)
joy_vector = np.array(joy)
reproach_vector = np.array(reproach)
distress_vector = np.array(distress)
admiration_vector = np.array(admiration)
gratitude_vector =  np.array(gratitude)

def float_range(start, stop, step):
  while start <= stop:
    yield float(start)
    start += decimal.Decimal(step)

# Noramalization in range [0,1]
# used only for emotional potential
def normalize_list(list):
    array = np.array(list)
    k = array - min(array)
    l = max(array) - min(array)
    array = k/l
    return array.tolist()

# Normalization 
def normalize_list2(list):
    list = np.array(list)
    list_norm = list / np.linalg.norm(list)
    return list_norm


def mood_categorization(mood_list):
    """characterizes mood according to PAD model"""
    categorization = None
    c1 = 'positive'
    c2 = 'negative'
    mood_name = None
    mood_octant = ['Excuberant','Bored', 'Dependent','Disdainful', 'Relaxed', 'Anxious', 'Docile', 'Hostile']
    p = mood_list[0]
    a = mood_list[1]
    d = mood_list[2]
    
    #POSITIVE MOOD CASES 
    if p>=0 and a>=0 and d>=0:
        categorization = c1
        mood_name = mood_octant[0]
    if p>=0 and a>=0 and d<0:
        categorization = c1
        mood_name = mood_octant[2]
    if p>=0 and d>=0 and a<0:
        categorization = c1
        mood_name = mood_octant[4]

    if p>=0 and a<0 and d<0:
        categorization = c1
        mood_name = mood_octant[6]


    # NEGATIVE MOOD CASES
    if p<0 and a<0 and d<0:
        categorization = c2
        mood_name = mood_octant[1]

    if p<0 and a<0 and d>=0:
        categorization = c2
        mood_name = mood_octant[3]
    
    if p<0 and d<0 and a>=0:
        categorization = c2
        mood_name = mood_octant[5]

    if p<0 and a>=0 and d>=0:
        categorization = c2
        mood_name = mood_octant[7]


    if p ==0 and d==0 and a == 0:
        categorization = "default"
        mood_name = 'Neutral'
    
    return [categorization,mood_name]


# normalize the contents of a list to [-1,1]
def normalize_instensities(discrete_emotional_instensities):
    x = np.array(discrete_emotional_instensities)
    # take the maximum absolute value from the given array, as a float
    abs_max = float(np.amax(np.abs(x)))
    if abs_max == 0:
        normalized_array = x
    else:
        # divide each element of the given array with abs_max
        normalized_array = x*(1.000 /float( abs_max))
    normalized_list = normalized_array.tolist()
    return normalized_list


#returns an array that represents an agent's personality
#the function's output depends on the 'flag' variable
def personality_init(flag):
    personality = []
    # flag = 0 -> return a random personality
    if flag == 0:
        for i in range(0,5):
            personality.append(random.uniform(-1,1))
    #flag = 1 -> return a personality specified by the user 
    elif flag == 1:
        o = 2
        c = 2
        e = 2
        a = 2
        n = 2
        while float(o) not in float_range(-1,1,'0.01') :        
            try:
                o = float(input("O (must be in [-1,1]): "))
            except ValueError:
                print("Input must be a number!")

        while float(c) not in float_range(-1,1,'0.01'):   
            try:
                c = float(input("C (must be in [-1,1]): "))
            except ValueError:
                print("Input must be a number!")
        while float(e) not in float_range(-1,1,'0.01'):
            try:
                e = float(input("E (must be in [-1,1]): "))
            except ValueError:
                print("Input must be a number!")
        while float(a) not in float_range(-1,1,'0.01'):
            try:
                a = float(input("A (must be in [-1,1]): "))
            except ValueError:
                print("Input must be a number!")
        while float(n) not in float_range(-1,1,'0.01'):
            try:
                n = float(input("N (must be in [-1,1]): "))
            except ValueError:
                print("Input must be a number!")   

        personality = [o,c,e,a,n]

    return personality


# Coverts ocean personality to pad space and returns the pad personality, the intesity of the mood
# that it represents and it's characterization
#(ie negative/positive and it's mood octant) according to the PAD model. 
def ocean_to_pad_personality_init(personality):  
    # pad dimensions
    pad_personality = []
    moodbase = []
    mood_intensity = 0
    p = 0.21*personality[0] + 0.59*personality[3] + 0.19*personality[4]
    a = 0.15*personality[0] + 0.3*personality[3] - 0.57*personality[4]
    d = 0.25*personality[0] + 0.17*personality[1] + 0.6*personality[2] -0.32*personality[3]
    pad_personality.append(p)
    pad_personality.append(a)
    pad_personality.append(d)
    # categorize the default personality(positive/negative)
    categorization = mood_categorization(pad_personality)
    pad_personality = np.array(pad_personality)
    mood_intensity = np.linalg.norm(pad_personality)
    moodbase.append(pad_personality)    
    moodbase.append(mood_intensity)
    moodbase.append(categorization)
    return moodbase

# caclulates the agent's emotional thresholds
def calculate_thresholds(agent,thresholds):
    th_anger = np.linalg.norm(agent.current_mood[0] - anger_vector)        
    th_joy = np.linalg.norm(agent.current_mood[0] - joy_vector)
    th_reproach = np.linalg.norm(agent.current_mood[0] - reproach_vector)
    th_distress = np.linalg.norm(agent.current_mood[0] - distress_vector)
    th_gratitude =  np.linalg.norm(agent.current_mood[0] - gratitude_vector)
    th_admiration = np.linalg.norm(agent.current_mood[0] - admiration_vector)
    temp1 = [th_anger ,th_joy,th_distress,  th_reproach, th_gratitude, th_admiration]
    thresholds = normalize_list2(temp1)
    return thresholds

# A generic Agent architecture
class Agent(object):
    def __init__(self, score=0):
        self.name = None
        self.turn = 0
        self.score = score
        self.new_payoff = 0
        self.last_payoff = 0
        self.last_action = None
        self.this_action = None
        self.opponent = None
        self.state = []
        
    def update_last_payoff(self):
        self.last_payoff = self.new_payoff
        self.new_payoff = 0
    
    def new_match_against(self, opponent):
        self.last_action = None
        self.this_action = None
        self.opponent = opponent
        self.turn = 0

    def reset_score(self):
        self.score = 0

    def action(self):
        self.this_action = self.decide_action()
        return self.this_action
   
    def update_last_action(self):
        self.last_action = self.this_action
        self.this_action = None

    def add_points(self, points):
        if points == pmax_c:
            self.state = s1            
        elif points == pmin_c:
            self.state = s2
        elif points == pmax_d:
            self.state = s3
        elif points == pmin_d:
            self.state = s4    
        self.score += points


class Pavlov1(Agent):
    def __init__(self, score=0):
        super().__init__()
        self.is_first_move = True
        self.learning_coefficient = 0.4
        self.potential_to_defect = 0.5
        self.potential_to_cooperate = 0.5
        self.first_move = None
        self.opponent_new_payoff = 0
        self.opponent_last_payoff  = 0
        self.opponent_last_action = None
        self.response = 0
        self.cooperation_percentage  = 0
        self.defection_percentage = 0

    def add_points(self, points):
        if points == pmax_c:
            self.cooperation_percentage += 1
            self.opponent_new_payoff = pmax_c
            self.state = s1            
        elif points == pmin_c:
            self.defection_percentage += 1
            self.opponent_new_payoff = pmax_d
            self.state = s2
        elif points == pmax_d:
            self.cooperation_percentage += 1
            self.opponent_new_payoff = pmin_c
            self.state = s3
        elif points == pmin_d:
            self.opponent_new_payoff = pmin_d
            self.defection_percentage += 1 
            self.state = s4                   
        self.new_payoff = points
        self.score += points

    def update_last_payoff(self):
        super().update_last_payoff()
        self.opponent_last_payoff = self.opponent_new_payoff
        self.opponent_new_payoff = 0
    
    #Action selection based on a probability distribution    
    def decide_action(self):
        action = None
        
        # Cooperate in the first turn and then
        if self.is_first_move is True:
            action = True
        else:        
            # choose action based on the distribution 
            action = (random.choices([True,False], weights=(self.potential_to_cooperate,self.potential_to_defect), k=1))[0]
        
        self.is_first_move = False
        return action
        
    #Caclulates the probability of an action on the next round based on
    #the outcome of the current one and the congruency with the agent's preferences in the IPD (ie Pavlov(1,0,0,1))
    def my_move_prediction(self):
        if self.this_action is True:
            if self.new_payoff == pmax_c:
                self.potential_to_cooperate = self.potential_to_cooperate+(1-self.potential_to_cooperate)*self.learning_coefficient
                self.potential_to_defect = 1 - self.potential_to_cooperate
            else:
                self.potential_to_cooperate = (1-self.learning_coefficient)*self.potential_to_cooperate
                self.potential_to_defect = 1 - self.potential_to_cooperate
        if self.this_action is False:
            if self.new_payoff == pmax_d:
                self.potential_to_defect = self.potential_to_defect+(1-self.potential_to_defect)*self.learning_coefficient
                self.potential_to_cooperate = 1 - self.potential_to_defect
            else:
                self.potential_to_defect = (1-self.learning_coefficient)*self.potential_to_defect
                self.potential_to_cooperate = 1 - self.potential_to_defect
        
    # Learning coefficient adjustment(keeps track of the action variation of the agent) 
    def learning_coefficient_adjustment(self):
        if self.this_action is self.last_action:
            self.learning_coefficient = self.learning_coefficient + 0.10
        elif self.this_action is not self.last_action:
            self.learning_coefficient = self.learning_coefficient - 0.10
        if self.learning_coefficient <= 0:
            self.learning_coefficient = 0
        elif self.learning_coefficient >=1:
            self.learning_coefficient = 1

    def reward_calculation(self,state):
        reward = 0
        if self.state == [1,1]:
            reward = pmax_c
        if self.state == [1,0]:
            reward = pmin_c
        if self.state == [0,1]:
            reward = pmax_d
        if self.state == [0,0]:
            reward = pmin_d
        return reward
    
    #the agent's routine after each round 
    def post_action(self):
        self.response = abs(self.cooperation_percentage - self.defection_percentage)
        # Data structures update
        reward_list.append(self.new_payoff)
        moves_played.append(self.this_action)
        cooperation_potential_list.append(self.potential_to_cooperate)
        defection_potential_list.append(self.potential_to_defect)
        # keep track of cooperation potential in every possible game state
        if self.state == [1,1]:
            s1_pc_list.append(self.potential_to_cooperate)
        elif self.state == [1,0]:
            s2_pc_list.append(self.potential_to_cooperate)
        elif self.state == [0,0]:
            s3_pc_list.append(self.potential_to_cooperate)
        elif self.state == [0,1]:
            s4_pc_list.append(self.potential_to_cooperate)
        self.opponent_last_action = self.opponent.this_action
        learning_coeffiecient_list.append(self.learning_coefficient)
        # Update learning coefficient
        self.learning_coefficient_adjustment() 
        #Make move predictions for the next round 
        self.my_move_prediction()
        self.update_last_action()
        self.update_last_payoff() 
        
    def new_match_against(self, opponent):
        self.is_first_move = True
        self.opponent_last_payoff = 0
        self.first_move = None
        self.opponent_new_payoff = 0
        self.opponent_last_action = None
        super().new_match_against(opponent)        

# "Emotional" agent architecture
class Pavlov2(Pavlov1):
    def __init__(self,score = 0):
        super().__init__()
        self.is_first_move = True
        self.old_desired = 0
        self.desired  = 0 
        self.worth = 0
        self.ocean_personality = personality_init(0)
        # Pad default mood = [[(P,A,D), Im0, categorization]
        self.default_mood = ocean_to_pad_personality_init(self.ocean_personality) 
        # current mood = ([(P,A,D), (im) intensity, categorization]) ---> (Imt)
        self.current_mood = self.default_mood
        self.total_emotional_vector = []
        self.total_emotional_intensity = 0
        self.total_emotional_category = ''
        # all emotional intensities [idist, igratitude, iapprec, ijoy, ireproach, ianger]
        self.discrete_emotional_intensities = [0, 0 ,0 ,0 ,0, 0]
        self.emotional_potential_to_cooperate = 0
        self.emotional_potential_to_defect = 0
        self.emotional_bias_about_coplayer_cooperation = 0
        self.emotional_bias_about_coplayer_defection = 0
        self.delta = 0
        # INITIALIZED EMOTIONAL INTESITIES
        self.ireproach = 0
        self.idist = 0
        self.ianger = 0
        self.idist = 0
        self.igratitude = 0
        self.ijoy = 0
        self.iapprec = 0       
        # AGENT'S EMOTIONAL THRESHOLDS
        self.thesholds = []
        self.th_anger = 0
        self.th_joy = 0
        self.th_reproach = 0
        self.th_distress = 0
        self.th_gratitude = 0
        self.th_admiration = 0

    #Action selection based on action probabilities    
    def decide_action(self):
        action = None
        # the agent chooses an action based on a probability distribution
        action = (random.choices([True,False], weights=(self.potential_to_cooperate,self.potential_to_defect), k=1))[0]
        self.is_first_move = False
        return action

    # Reward "perception" 
    def reward_calculation(self, state):
        reward  = 0

        if self.current_mood[2][0] == 'positive':
            if state == [1,1]:
                reward = 3 + 1/2*mt.sqrt(3)*self.current_mood[1]*3
            if state == [1,0]:
                reward = 1/2*mt.sqrt(3)*self.current_mood[1]*5 - (3 + 1/2*mt.sqrt(3)*self.current_mood[1]*3)
            if state == [0,0]:
                reward = 1 + 1/2*mt.sqrt(3)*self.current_mood[1]
            if state == [0,1]:
                reward = 5 - (3 + 1/2*mt.sqrt(3)*self.current_mood[1]*3)

        elif self.current_mood[2][0] == 'negative':
            if state == [1,1]:
                reward = 3 - 1/2*mt.sqrt(3)*self.current_mood[1]*3
            if state == [1,0]:
                reward = -1/2*mt.sqrt(3)*self.current_mood[1]*5
            if state == [0,0]:
                reward = 1 - 1/2*mt.sqrt(3)*self.current_mood[1]
            if state == [0,1]:
                reward = 5 
        else:
            reward = self.new_payoff

        return reward
    
    def add_points(self, points):
        if points == pmax_c:
            self.cooperation_percentage += 1
            self.opponent_new_payoff = pmax_c
            self.state = s1            
        elif points == pmin_c:
            self.defection_percentage += 1
            self.opponent_new_payoff = pmax_d
            self.state = s2
        elif points == pmax_d:
            self.cooperation_percentage += 1
            self.opponent_new_payoff = pmin_c
            self.state = s3
        elif points == pmin_d:
            self.opponent_new_payoff = pmin_d
            self.defection_percentage += 1 
            self.state = s4                   
        self.new_payoff = self.reward_calculation(self.state)
        self.score += points

    # D(S)
    def desirability(self,state):      
        desired = 0   
        desires= []     
        # desirablity is calculated as the difference between the new payoff and the
        # worst acceptable outcome(the agent does not want to lose, so that would be (D,D))
        desired_s1 = self.reward_calculation([1,1]) - self.reward_calculation([0,0])
        desired_s2 = self.reward_calculation([1,0]) - self.reward_calculation([0,0])
        desired_s3 = self.reward_calculation([0,1]) - self.reward_calculation([0,0])
        desired_s4 = self.reward_calculation([0,0]) - self.reward_calculation([0,0])
        desires.append(desired_s1)
        desires.append(desired_s2)
        desires.append(desired_s3)
        desires.append(desired_s4)
        desires = normalize_list2(desires)
        if state == [1,1]:
            desired = desires[0]
        if state == [1,0]:
            desired = desires[1]
        if state == [0,0]:
            desired = desires[3]
        if state == [0,1]:
            desired = desires[2]
        desired_norm = desired
        return desired_norm    
    
    def my_move_prediction(self):
        maxc = self.reward_calculation([1,1])
        maxd = self.reward_calculation([0,1])
        if self.this_action is True:
            if self.new_payoff == maxc:
                self.potential_to_cooperate = self.emotional_potential_to_cooperate+(1-self.emotional_potential_to_cooperate)*self.learning_coefficient
                self.potential_to_defect = 1 - self.potential_to_cooperate
            else:
                self.potential_to_cooperate = (1-self.learning_coefficient)*self.emotional_potential_to_cooperate
                self.potential_to_defect = 1 - self.potential_to_cooperate
        if self.this_action is False:
            if self.new_payoff == maxd:
                self.potential_to_defect = self.emotional_potential_to_defect+(1-self.emotional_potential_to_defect)*self.learning_coefficient
                self.potential_to_cooperate = 1 - self.potential_to_defect
            else:
                self.potential_to_defect = (1-self.learning_coefficient)*self.emotional_potential_to_defect
                self.potential_to_cooperate = 1 - self.potential_to_defect

    # W(S)
    def worthiness(self):
        worth = 0
        if self.turn == 0:
            worth = self.desired 
        else:
            worth = 2*( 1/self.turn )*abs(self.response) * self.desired
        return worth

    def distress_function(self):
        im = 0
        ie = self.desired
        if self.current_mood[2][0] == 'negative':
            im = ie + random.uniform(0,1)
        elif self.current_mood[2][0] == 'positive':
            im = ie - random.uniform(0,1)
        if abs(im)>=self.th_distress:
            self.idist = abs(im)
        else:
            self.idist = 0

    def reproach_function(self):
        im = 0
        ie = 0
        if self.last_action is not None:
            ie = abs(self.worth)
        if self.current_mood[2][0] == 'negative':
            im = ie + random.uniform(0,1)
        elif self.current_mood[2][0] == 'positive':
            im = ie - random.uniform(0,1)
            if abs(im)>=self.th_reproach:
                self.ireproach = abs(im)
            else:
                self.ireproach = 0

    def appreciation_function(self):
        im = 0
        ie= 0
        if self.last_action is not None:
            ie = abs(self.worth)
        if self.current_mood[2][0] == 'negative':
            im = ie - random.uniform(0,1)
        elif self.current_mood[2][0] == 'positive':
            im = ie + random.uniform(0,1)
            if abs(im)>=self.th_admiration:
                self.iapprec = abs(im)
            else:
                self.iapprec = 0

    def joy_function(self):
        im = 0
        k = 1/mt.sqrt(3)
        ie = self.desired
        if self.current_mood[2][0] == 'negative':
            im = ie - random.uniform(0,1)
        elif self.current_mood[2][0] == 'positive':
            im = ie + random.uniform(0,1)
        if abs(im)>=self.th_joy:
            self.ijoy = abs(im)
        else:
            self.ijoy = 0

    def anger_function(self):
        im = 0
        if self.turn>=2:
            im  = 0.6*self.idist + 0.4*self.ireproach
        else:
            im = 0.4*self.idist + 0.6*self.ireproach
        if im >= self.th_anger:
            self.ianger = im
        else: 
            self.ianger = 0

    def gratitude_function(self):
        im = 0
        if self.turn<=2:
            im = 0.6*self.ijoy + 0.4*self.iapprec
        else:
            im = 0.4*self.ijoy + 0.6*self.iapprec
        if im >= self.th_gratitude:
                self.gratitude = im
        else: 
            self.gratitude = 0

    def appraisal_derivation(self):
        self.desired = self.desirability(self.state)
        self.worth = self.worthiness()        
        self.affect_derivation()

    def affect_derivation(self):
        # Distress/Joy 
        if self.desired > 0:             
            self.joy_function()        
        elif self.desired <= 0:   
            self.distress_function()         

        # Reproach/Appreciation
        if self.worth > 0:            
            self.appreciation_function() 
        elif self.worth <= 0: 
            self.reproach_function()            
        
        # # anger/gratitude
        if self.desired <= 0 and self.worth <= 0:        
            self.anger_function()     
        elif self.desired > 0 and self.worth > 0:       
            self.gratitude_function()     

        # calculate the new emotional vectors according to PAD
        distress = distress_vector * self.idist
        anger = anger_vector * self.ianger
        joy = joy_vector * self.ijoy
        reproach = reproach_vector * self.ireproach
        gratitude = gratitude_vector * self.igratitude
        appreciation = admiration_vector * self.iapprec
        self.total_emotional_vector = (distress + anger + joy+ reproach+gratitude+appreciation)
        self.total_emotional_intensity = np.linalg.norm(self.total_emotional_vector)
        self.total_emotional_category = mood_categorization(self.total_emotional_vector)

    def emotional_biases_update(self):
        # Mood state affects future expectations
        n = 1/2*mt.sqrt(3)
        if self.current_mood[2][0] == 'negative':
            self.emotional_potential_to_cooperate = self.potential_to_cooperate - n*self.current_mood[1]
        elif self.current_mood[2][0] == 'positive':
            self.emotional_potential_to_cooperate = self.emotional_potential_to_cooperate + n*self.current_mood[1]
        self.emotional_potential_to_defect = 1 -  self.emotional_potential_to_cooperate
        distribution = [self.emotional_potential_to_cooperate, self.emotional_potential_to_defect]
        if self.emotional_potential_to_cooperate >1 or self.emotional_potential_to_cooperate <0:
            distribution = normalize_list(distribution)
            self.emotional_potential_to_cooperate = distribution[0]
            self.emotional_potential_to_defect = distribution[1]
    
    def mood_update(self):
        distress = distress_vector * self.idist
        anger = anger_vector * self.ianger
        joy = joy_vector * self.ijoy
        reproach = reproach_vector * self.ireproach
        gratitude = gratitude_vector * self.igratitude
        appreciation = admiration_vector * self.iapprec
        pad = [distress, anger, joy, reproach, gratitude, appreciation]
        counter = 0
        average_emotions = np.array([0,0,0,0,0,0])
        sum_emot = distress + anger + joy + reproach +gratitude +appreciation
        for i in pad:
            if np.linalg.norm(i) != 0:
                counter += 1
        #if there are active emotions readjust mood accordingly
        if counter!= 0:    
        # new mood->average emotional state
            average_emotions = sum_emot/counter
            # average_emotions = average_emotions.tolist()
            average_emotions = normalize_list2(average_emotions.tolist())
            categorization = mood_categorization(average_emotions)
        new_mood_intesity = np.linalg.norm(average_emotions)
        if mt.isclose(0,new_mood_intesity,abs_tol= 0.05) ==  True:
            self.current_mood = self.default_mood
        else:    
            new_mood_lst = [average_emotions, new_mood_intesity, categorization]
            self.current_mood = new_mood_lst

        # update emotional thresholds 
        self.thesholds = calculate_thresholds(self,[])
        self.th_joy = self.thesholds[1]
        self.th_distress = self.thesholds[3]
        self.th_reproach = self.thesholds[2]
        self.th_anger = self.thesholds[0]
        self.th_gratitude = self.thesholds[4]
        self.th_admiration = self.thesholds[5]

    def reinitialize_intesities(self):
        self.ireproach = 0
        self.idist = 0
        self.ianger = 0
        self.idist = 0
        self.igratitude = 0
        self.ijoy = 0
        self.iapprec = 0

    #post-action routine 
    def post_action(self):
        self.old_desired = self.desired
        reward_list.append(self.desired)
        history_of_payoffs.append(self.new_payoff)
        self.response = abs(self.cooperation_percentage - self.defection_percentage)
        self.appraisal_derivation()  
        cooperation_potential_list.append(self.potential_to_cooperate)
        defection_potential_list.append(self.potential_to_defect)
        if self.state == [1,1]:
            s1_pc_list.append(self.potential_to_cooperate)
        elif self.state == [1,0]:
            s2_pc_list.append(self.potential_to_cooperate)
        elif self.state == [0,0]:
            s3_pc_list.append(self.potential_to_cooperate)
        elif self.state == [0,1]:
            s4_pc_list.append(self.potential_to_cooperate)
        mood_intensity_list.append(self.current_mood[1])
        mood_vector_list.append(self.current_mood[0])
        mood_categorization_list.append(str(self.current_mood[2][1]))
        desirability_of_state_list.append(self.desired)
        worthiness_of_state_list.append(self.worth)
        total_emotional_category_list.append(str(self.total_emotional_category[1]))
        emotional_potential_to_cooperate_list.append(self.emotional_potential_to_cooperate)
        emotional_potential_to_defect_list.append(self.emotional_potential_to_defect)
        moves_played.append(self.this_action)
        learning_coeffiecient_list.append(self.learning_coefficient)
        self.learning_coefficient_adjustment() 
        self.mood_update() 
        self.emotional_biases_update()
        self.my_move_prediction()
        self.update_last_action()
        self.update_last_payoff()          
        self.reinitialize_intesities()

    # reinitialize worth
    def new_match_against(self, opponent):
        self.emotional_bias_about_coplayer_cooperation = 0
        self.emotional_bias_about_coplayer_defection = 0
        super().new_match_against(opponent)

