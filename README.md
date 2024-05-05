# text-based-games

## Install Dependencies
The required dependencies can be installed with `pip install -r requirements.txt`.

If additional issues are encountered, it could be helpful to refer to the `README.md` from the TextWorld repository: https://github.com/microsoft/TextWorld


## Games
The TextWorld games that we used are described below (taken from TextWorld documentation). We used detailed instructions for both training and testing and varied the density of rewards (dense, balanced, sparse). 

To setup the games required for training and testing, you can run the `setup.sh` script.

We can use TextWorld to generate a few simple games with the following handcrafted world
```
                     Bathroom
                        +
                        |
                        +
    Bedroom +-(d1)-+ Kitchen +--(d2)--+ Backyard
      (P)               +                  +
                        |                  |
                        +                  +
                   Living Room           Garden
```
where the goal is always to retrieve a hidden food item and put it on the stove which is located in the kitchen. One can lose the game if it eats the food item instead of putting it on the stove!

Using `tw-make tw-simple ...`, we are going to generate the following 7 games:

| gamefile | description |
| :------- | :---------- |
| `games/rewardsDense_goalDetailed.z8` | dense reward + detailed instructions |
| `games/rewardsBalanced_goalDetailed.z8` | balanced rewards + detailed instructions |
| `games/rewardsSparse_goalDetailed.z8` | sparse rewards + detailed instructions |
| | |
| `games/rewardsDense_goalBrief.z8` | dense rewards + no instructions but the goal is mentionned |
| `games/rewardsBalanced_goalBrief.z8` | balanced rewards + no instructions but the goal is mentionned |
| `games/rewardsSparse_goalBrief.z8` | sparse rewards + no instructions but the goal is mentionned |
| | |
| `games/rewardsSparse_goalNone.z8` | sparse rewards + no instructions/goal<br>_Hint: there's an hidden note in the game that describes the goal!_ |

## Training and Testing
A variety of training and testing functions are included in `main.py`. Some example code is shown in the file, but different combinations can be used. The primary agents are available in `lstm_dqn_agent.py` and `neural_agent.py`.