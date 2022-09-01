# TEST ASPECTS OF AGENT'S BEHAVIOR IN A MULTI-AGENT ENVIRONMENT
# THE RESULTS FOR SIMULATION #2 ARE PRODUCED USING THIS SCRIPT
from profiles import *
from data import round_robin,format_list,round_dictionary, describe
import pandas as pd
from agents import *

NUM_ROUNDS = 0
mutual_cooperation_percent = 0
cooperation_percent = 0
times_cooperated = 0
mutual_cooperation_counter = 0
main_agent_type_code = 0
bot_type_code = 0
c1 = 0
bots = ['Tit for Tat','ReverseTFT', 'Random','LearningPavlov']
results1 = []
msg1 = "Specify main subject's strategy code (choose between integers (1):LearningPavlov,(2): Cooperative,(3):NonCooperative,(4): Neutral): "
msg2 = "Specify bot type (choose between (1)TitForTat, (2)ReverseTFT,(3)Random(type the number before the desired strategy):  "
bot_agent_type_code = 0

# SPECIFY THE ROUNDS FOR EACH MATCH
while NUM_ROUNDS <= int(0):
    try:
        NUM_ROUNDS = int( input("Specify the number of rounds for each match: "))
    except ValueError:
            print("Input must be an integer!")

# "Specify the agents for the simulation"
while main_agent_type_code not in range(1,6,1) :
    try:
        main_agent_type_code = int(input(msg1))
    except ValueError:
            print("Input must be an integer!")

if main_agent_type_code == 1:
    main_agent_type = 'LearningPavlov'
if main_agent_type_code == 2:
    main_agent_type = 'Cooperative'
if main_agent_type_code == 3:
    main_agent_type = 'NonCooperative'
if main_agent_type_code == 4:
    main_agent_type = 'Neutral'

# INITIALIZE PERSONALITY
if main_agent_type_code >= 2:
    personality = personality_init(1)
# "Specify the agents for the simulation"
while bot_agent_type_code not in range(1,4,1) :
    try:
        bot_agent_type_code = int(input(msg2))
    except ValueError:
            print("Input must be an integer!")

if bot_agent_type_code == 1:
    bot_agent_type = 'Tit for Tat'
if bot_agent_type_code == 2:
    bot_agent_type = 'ReverseTFT'
if bot_agent_type_code == 3:
    bot_agent_type = 'Random'

# Initialize the dictionary of strategies involved in the simulation
agent_profiles  = {
    main_agent_type : 1,
    bot_agent_type:8
}
# CREATE THE SIMULATION POPULATION
agents_population = create_population(agent_profiles)

if main_agent_type_code >=2:
    # INIT MAIN AGENT'S EMOTIONAL COMPONENTS
    main_agent = agents_population.first_member()
    main_agent.ocean_personality = personality
    main_agent.default_mood = ocean_to_pad_personality_init(main_agent.ocean_personality) 
    main_agent.current_mood = main_agent.default_mood
    main_agent.init_emotional_potential()
    main_agent.thesholds = calculate_thresholds(main_agent, temp)
    main_agent.th_anger = main_agent.thesholds[0]
    main_agent.th_joy = main_agent.thesholds[1]
    main_agent.th_reproach = main_agent.thesholds[2]
    main_agent.th_distress = main_agent.thesholds[3]
    main_agent.th_gratitude = main_agent.thesholds[4]
    main_agent.th_admiration = main_agent.thesholds[5]

if main_agent_type_code>=3:
    print("=== "+str(main_agent.name)+"===" + "\n" +"Default mood: " +str(main_agent.default_mood) 
                    +"\n Thesholds[th_anger ,th_joy,th_distress,  th_reproach, th_gratitude, th_admiration]: " +str(main_agent.thesholds) 
                    + "\n OCEAN = " + str(main_agent.ocean_personality))

# ROUND_ROBIN 
mutual_cooperation_counter = round_robin(agents_population,NUM_ROUNDS,mutual_cooperation_counter,0)

# is_first_tournament = False
# Final strategy developed
total_score1 = agents_population.total_score() 
total_rounds = NUM_ROUNDS*agents_population.__len__()
counter = 0
median_point_differences = 0
num_of_agents = 0

# CALCULATE THE AVERAGE DIFFERENCE BETWEEN THE SCORES GATHERED FROM EACH AGENT
point_results = agents_population.score_differences()
median_agent_score = point_results[0]
median_point_differences = point_results[1]
mad_agent_score = point_results[2]
mad_point_differences = point_results[3]
agent_point_differences = point_results[4]
    
# CALCULATE THE AVERAGE COOPERATION POTENTIAL
for i in cooperation_potential_list:
    c1 += i

for i in moves_played:
    if i == 1:
        times_cooperated += 1
scores = agents_population.scores()
main_agent_score = (agents_population.first_member()).score

# SORT DATA TO EXPORT
agent_point_differences.sort()
scores.sort()
# # Fomat data
cooperation_potential_list = format_list(cooperation_potential_list)
defection_potential_list = format_list(defection_potential_list)
agent_point_differences = format_list(agent_point_differences)

# DATAFRAME DESCRIPTIONS
pc = pd.DataFrame(cooperation_potential_list)
prob_d = pd.DataFrame(defection_potential_list)
pointd = pd.DataFrame(agent_point_differences)
score_distribution = pd.DataFrame(scores)
agent_point_differences_frame = pd.DataFrame(agent_point_differences)
reward_list_frame = pd.DataFrame(reward_list)

# EXPORTED DATAFRAMES AND ADDITIONAL SYSTEM INFO
describe_cooperation_probability = round_dictionary(describe(pc,['median','mad']))
describe_defection_probability = round_dictionary(describe(prob_d,['median','mad']))
describe_scores = round_dictionary(describe(score_distribution,['median','mad']))
describe_score_differences = round_dictionary(describe(agent_point_differences_frame,['median','mad']))
describe_reward_list = describe(reward_list_frame,['median','mad'])
x = NUM_ROUNDS*8
cooperation_percent = 100*(times_cooperated/total_rounds)
mutual_cooperation_percent  = 100*(mutual_cooperation_counter/total_rounds)

# RESULTING DATAFRAME
results = {
            # 'Strategy Developed': final_strategy,
            'Main agent score' : main_agent_score,
            'Main agent cooperation percentage': cooperation_percent, 
            'Main agent P(C)': describe_cooperation_probability,
            'Main agent P(D)':describe_defection_probability, 
            'Total population points': total_score1,                
            'Individual agent scores': describe_scores,
            'Mutual cooperation percentage': mutual_cooperation_percent,
            'Score differences': describe_score_differences, 
}

results1.append(results)
df = pd.DataFrame.from_dict(results1)
df = (df.T)
df.to_excel(main_agent_type+" VS" +bot_agent_type +'.xlsx')






