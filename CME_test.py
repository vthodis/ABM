#THIS SCRIPT IS MAINLY USED TO TEST THE C.M.E. FOR THE EMOTIONAL AGENT
#THE RESULTS FOR SIMULATION #1 ARE PRODUCED USING THIS SCRIPT
from profiles import *
import matplotlib.pyplot as plt
from data import play_several_rounds
from agents import *
import statistics

NUM_ROUNDS = 0
personality = []
is_first_tournament = True
mutual_cooperation_counter = 0
total_score1 = 0
times_cooperated = 0
s1_pc = 0
s2_pc = 0
s3_pc = 0
s4_pc = 0
# Inputs
main_agent_type_code = 0
bot_agent_type_code = 0
msg1 = "Specify main subject's strategy code (choose between integers (1):LearningPavlov,(2): Cooperative,(3):NonCooperative,(4): Neutral): "
msg2 = "Specify bot type (choose between (1)TitForTat, (2)ReverseTFT,(3)Random (type the number before the desired strategy): "

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
    bot_agent_type : 1
}

# create the agents
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

if main_agent_type_code>=2:
    print("\n=== "+str(main_agent.name)+"AGENT'S INITIAL EMOTIONAL STATE===" + "\n" +"1) Default mood: " +str(main_agent.default_mood) 
                    +"\n2) Thesholds[th_anger ,th_joy,th_distress,  th_reproach, th_gratitude, th_admiration]: " +str(main_agent.thesholds) 
                    + "\n3)OCEAN = " + str(main_agent.ocean_personality)+"\n")


mutual_cooperation_counter =play_several_rounds(agents_population[0],agents_population[1], NUM_ROUNDS,mutual_cooperation_counter)    
mutual_cooperation_percentage = 100*(mutual_cooperation_counter / NUM_ROUNDS)

for i in moves_played:
    if i == 1:
        times_cooperated += 1

cooperation_percentage = 100*(times_cooperated/NUM_ROUNDS)
s1_pc_list.sort()
s2_pc_list.sort()
s3_pc_list.sort()
s4_pc_list.sort()


# Mean potential to cooperate after every possible strategy
if s1_pc_list:
    s1_pc = statistics.mean(s1_pc_list)
else:
    s1_pc = "Never occured"
if s2_pc_list:
    s2_pc = statistics.mean(s2_pc_list)
else:
    s2_pc = "Never occured"

if s3_pc_list:
    s3_pc = statistics.mean(s3_pc_list)
else:
    s3_pc = "Never occured"

if s4_pc_list:
    s4_pc = statistics.mean(s4_pc_list)
else:
    s4_pc = "Never occured"

print("===SIMULATION RESULTS===\nTOTAL SCORE = " +str((agents_population.first_member()).score))
print("COOPERATION PERCENTAGE = " +str(cooperation_percentage)+"%")
print("MUTUAL COOPERATION PERCENTAGE = " +str(mutual_cooperation_percentage) +"%")

if main_agent_type_code >=2:
    fig3, axs = plt.subplots(1,2)
    fig3.suptitle('Affective states')
    axs[0].hist(mood_categorization_list,label = 'Mood octants',color = 'orange')
    axs[0].set_title('Mood octant')
    axs[0].legend()
    axs[0].yaxis.grid()
    axs[0].sharex(axs[0])
    axs[1].hist(total_emotional_category_list,label = 'Emotional PAD category',color = 'blue')
    axs[1].set_title('PAD category of OCC emotions')
    axs[1].legend()
    axs[1].yaxis.grid()
    axs[1].sharey(axs[0])

plt.show()