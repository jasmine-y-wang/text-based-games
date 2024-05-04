#!/bin/bash

# generate games with different configurations
tw-make tw-simple --rewards dense    --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsDense_goalDetailed.z8
tw-make tw-simple --rewards balanced --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsBalanced_goalDetailed.z8
tw-make tw-simple --rewards sparse   --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalDetailed.z8
tw-make tw-simple --rewards dense    --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsDense_goalBrief.z8
tw-make tw-simple --rewards balanced --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsBalanced_goalBrief.z8
tw-make tw-simple --rewards sparse   --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalBrief.z8
tw-make tw-simple --rewards sparse   --goal none     --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalNone.z8

tw-make tw-simple --rewards dense --goal detailed --seed 1 --output games/tw-another_game.z8 -v -f

# generate 100 games with different seeds for training (for each density level)
seq 1 100 | xargs -n1 -P4 tw-make tw-simple --rewards dense --goal detailed --format z8 --output training_games_dense/ --seed
seq 1 100 | xargs -n1 -P4 tw-make tw-simple --rewards balanced --goal detailed --format z8 --output training_games_balanced/ --seed
seq 1 100 | xargs -n1 -P4 tw-make tw-simple --rewards sparse --goal detailed --format z8 --output training_games_sparse/ --seed

# generate 20 games with different seeds for testing (for each density level)
seq 1 20 | xargs -n1 -P4 tw-make tw-simple --rewards dense --goal detailed --test --format z8 --output testing_games_dense/ --seed
seq 1 20 | xargs -n1 -P4 tw-make tw-simple --rewards balanced --goal detailed --test --format z8 --output testing_games_balanced/ --seed
seq 1 20 | xargs -n1 -P4 tw-make tw-simple --rewards sparse --goal detailed --test --format z8 --output testing_games_sparse/ --seed