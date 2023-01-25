### Regret matching 

This repo contains some basic regret matching code and some games to run it on. It can plot results from self-play in real-time. 

Algorithms:
1. Regret matching with strategies knows both players' mixed strategies and uses their product to compute regret.
2. Regret matching with actions only knows both players' realized strategies. Here realized means that both players observe the same actions rather than independent samples from their mixed strategies.  

Games:
1. **Matching Pennies**. Only has one Nash equilibrium which regret matching's average history of play approaches over time, unless the players start in equilibrium. 
2. **Chicken**. There are 3 Nash equilibria. The pure Nash equilibria are reached very quickly by the last iterate strategies of both players from almost any starting position. The only exception is when the players start in the unique mixed Nash equilibrium. 
3. **Shapley**. This game also has only one Nash equilibrium, but the average history of play does not converge to any stationary distribution. 