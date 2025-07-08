# sumo_rl

A helpful tutorial for RL in sumo:
https://www.youtube.com/watch?v=IwsrNWlX9Ag&list=PLAk8GOoajG6tKI74YID0hwjXVg8KBxNAD

Basic example of RL code with Q tables (not DQN) that is very interpretable:
https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9

DQN explanation:
https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae

I think we could recreate some realistic busy intersections in sydney (with lots of assumptions of course).


Additional paper on traffic signal RL recommended by the youtube Sumo guy:

https://www.researchgate.net/publication/333233222_A_Reinforcement_Learning_Approach_for_Intelligent_Traffic_Signal_Control_at_Urban_Intersections?enrichId=rgreq-4d2bed32409c031c7b9f2146dd4a903e-XXX&enrichSource=Y292ZXJQYWdlOzMzMzIzMzIyMjtBUzo3NzQ4MzgwODM5MTk4NzNAMTU2MTc0NzIwMzM5NQ%3D%3D&el=1_x_3&_esc=publicationCoverPdf



What we need for our project:

Figure out state space, action space, reward, when to terminate, DQN structure, hardest part is prob figuring out SUMO (the software is a bit antique but should be fine following tuts)

Will use TraCI to link SUMO and python

Stuff from the above paper:

Specific test scenarios (areas where traditional traffic light systems have issues with): 
(1) major/minor road traffic, 
(2) through/left-turn lane traffic, 
(3) tidal traffic, and 
(4) varying demand 

State Space:
Number of vehicles that are queueing for the traffic light (defined by the number of vehicles below a certain speed i.e. 0.1 m/s).
I.e. a 1D vector where q_i corresponds to number in queue for lane i

Action Space:
Each of the possible configurations of traffic lights at the intersection. If the sam econfiguration is repeated, then that state is extended.

Reward:
Number of vehicles that are no longer in queue after the action

Should also have benchmarks:

Fixed phase traffic lights (traditional)

Gap-based control:  prolong signal phases whenever 
a continuous (i.e. maximum time gap between 
successive vehicles ≤ 5s) stream of traffic is detected 
[18]. 

Time loss-based control: prolong signal phases 
whenever there exists vehicles with accumulated 
time loss (i.e. 1－v/vmax) exceeding 1s [18]. 






SUMO SIMULATION SETTINGS
Parameter                           Value 
Lane length                         150 meters 
Vehicle length                      5 meters 
Minimal gap between vehicles        2.5 meters 
Car-following model                 Krauss following model [22] 
Max vehicle speed                   13.42 m/s 
Acceleration ability of vehicles    2.6 m/s2 
Deceleration ability of vehicles    4.5 m/s2 
Duration of yellow signal           3 seconds 
Time span D of signal phase         10 seconds 


BTW need to download SUMO separately to this github repo
