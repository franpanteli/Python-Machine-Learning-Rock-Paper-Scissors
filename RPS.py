
"""
FILE CONTENTS 
1. ABOUT THIS FILE 
2. ABOUT THE STARTER CODE FOR THE FUNCTION IN THIS FILE
3. THE STARTER CODE FOR THE FUNCTION IN THIS FILE
4. ABOUT THE CODE FOR THE PLAYER FUNCTION IN THIS FILE
5. THE CODE FOR THE PLAYER FUNCTION IN THIS FILE
"""

"""
ABOUT THIS FILE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

THIS IS THE FILE YOU EDIT, AND MAIN.PY IS THE FILE YOU USE TO RUN THE TESTS
WE WANT IT TO BEAT IT FOUR TIMES AT A 60% (RATHER THAN 50% AVERAGE)

This file has a function:
  -> the player function
  -> this function takes "R" "S" or "P" for Rock, Paper, or Scissors
  -> the input of the function is the last move which the opponent played (input)

  -> the output of the function is the next move which the player we are playing should play for this round
  -> so we know the move which the opponent last played -> that is the input to this function
  -> the output of this function is the next move which we should play -> based off of this and in order to maximise the probability of winning
  -> it's learning the opponent's move as the game is played and predicting the next move which our player should play based off of the move 
        which the opponent last played
        -> so this list is empty at the start of the game 
        -> it starts off with the user getting an empty string

  -> the player function is telling us -> given that this was the last move of your opponent, this is the move which you should play in this round to 
        have the highest likelihood of winning this round

  -> this entire .py file is for the one function 
        -> and this function needs to be called after each round of the game 

  -> the aim of this program is to pass four 'opponents' / games or RPS, so four of these different functions / subclasses of them may have to 
        -> be called / used 
"""

"""
ABOUT THE STARTER CODE FOR THE FUNCTION IN THIS FILE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 

The starter code for this function is below. This starter code keeps track of the opponent's history and plays whatever the opponent played two plays ago

    -> For the first two rounds:
        -> You pick either R, P, or S
        -> At the same time, the opponent makes a play with R, P, or S
        
    -> For the first two rounds:
        -> You pick R, P, or S
        -> You record what their plays are for those first two rounds
        -> In the third round, you play what they played in the first round
        -> You record what they play in the third round because you are going to play this in the fifth round

    -> So, from the first to the second round:
        -> You are picking what your plays are and recording theirs
        
    -> Then, from the third round to the nth round:
        -> You are playing whatever their play was two plays ago
        -> You are continuing to update the function with whatever their play is in this round

    -> So, the inputs to the function are the previous play of the opponent:
        -> In each round of the game, we are recording the current play of the opponent
        -> Because this is what we are going to play back to them in two rounds' time
        -> We also have the entire list of all of their past moves in an array which is passed into the function
        -> These are the opponent's plays for each round
        -> We only start playing them back to them after the second round
        -> But the list includes all of their plays from all time

    -> The function is returning what our player should guess:
        -> It's a guessing function
        -> The function is outputting what our player should guess/play for the current round to have the highest likelihood of winning
        -> The function is returning the guess

    -> We have from the 0th play to the second play:
        -> At which point the function is guessing Rock <- this is a random guess
        -> It's not random; it's just been set to Rock
        -> So, for the first two rounds, it's guessing Rock

    -> Then, from the second play to the nth play:
        -> At which point the guess which the function outputs is the same as the opponent's guess from two rounds ago
        -> The entire array of the opponent's guesses is stored in the opponent_history array
        -> So, if we are operating in the regime from the second to the nth play:
            -> Then the guess which the function outputs is the penultimate element in the entire array of the opponent's plays
            -> It's always going to be the -2n'th index in that array (of all of the opponent's plays)
            -> Because we are updating it with their play each time
            -> From the first round which is played to the second:
                -> i.e when we are operating outside of the n=2 to n=n regime
                -> We are still updating this array with all of the opponent's plays
                -> So, the first two plays will be included
"""

"""
# THE STARTER CODE FOR THE FUNCTION IN THIS FILE 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3

def player(prev_play, opponent_history=[]):

    # Below - this is updating the array which contains all of the opponent's guesses each time they play 
    # -> This array contains the guesses which we fire back at them from the second round of the game 
    # -> The function also takes this array as an argument -> in case we are starting from the second round onwards
    # -> We are adding the previous play/guess of the opponent to the array that stores all of the values which it has played
    opponent_history.append(prev_play)

    # Below - this sets the guess of our player to Rock for the first two rounds of the game
    guess = "R"

    # -> Then from the second round to the nth round that the game is played
    # -> The guess (play) of our function is the same as what the player played two rounds ago 
    # -> We know what the opponent played in the past 
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    # And we are outputting the guess (R, P, or S) which our player should place
    # -> It's a guessing function -> it's telling our player what to output 
    return guess
"""

"""
4. ABOUT THE CODE FOR THE PLAYER FUNCTION IN THIS FILE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 

"""

# THE CODE FOR THE PLAYER FUNCTION IN THIS FILE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 

"""
IMPORT MODULES 
--------------------------------------------------------
        -> these modules are the same as the ones which were imported in the example for the TensorFlow section of the course content 
        -> the first half of the content for the course
"""

import random, os
import tensorflow as tf
import numpy as np
import pandas as pd

"""
INPUTS TO THE FUNCTION
--------------------------------------------------------
    -> prev_play 
        -> This function is to output the play which will be placed during one round of the game 
        -> We don't know what the opponent will play 
        -> But we do know all of their past plays
        -> The aim is to record their past plays and use these in a machine learning model with Python to output guesses which will increase the chance of our player winning the game 
        -> prev_play is the guess which the opponent played in the previous round of the game 
        -> We are adding this to the list of their plays each time 
        -> In other words, not only are we building the array which has all of their past plays, but we are also running machine learning on this to predict what our next move should be

    -> opponent_history
        -> This is the array that stores the previous plays/guesses of the opponent 
        -> This is updated each time a new round of the game is played 

    -> my_history
        -> This is the array that stores our previous plays/guesses during the game 
        -> This is again updated each time a new round of the game is played 

    -> So we have 
        -> The previous play of the opponent
        -> An array of all of the previous plays of the opponent
        -> An array of all of our previous plays 
"""

def player(prev_play, opponent_history=[], my_history=[]):

"""
TO UPDATE THE ARRAY WHICH STORES THE OPPONENT'S PLAYS:
--------------------------------------------------------

    This is in the context of the function: 
    -> opponent_history <- this is the array that stores the previous plays which the opponent had 
    -> After each round of the game, the function guesses either R, P, or S, and the opponent responds 
    -> We know the history of the responses which the opponent has replied with 
    -> This block of code updates the array that stores the plays of the opponent 
    -> The function is designed to either be used mid-game or at the start of a game:
        -> The function takes prev_play as an input so that if it's being used mid-game, we know the last play which was made 
        -> And all of the last plays which we made and which the opponent made 

    If the previous play didn't exist: 
        -> In other words, if we're starting the game from scratch 
        -> Then there is no previous play
        -> This is the argument to the function 
        -> In which case, we initialise the opponent's history -> because they haven't had the chance to play any responses yet 

    If the previous play did exist: 
        -> When the previous play existed, know the opponent's play and our play
        -> We are adding the previous play of the opponent to the array that stores all of their plays

    In other words: 
        -> We are playing the game of RPS against the opponent 
        -> We know the previous plays of the opponent -> and these are what we are storing in an array 
        -> This is the block of Python which updates that array -> with the history of plays which the opponent has played
        -> For the first round of the game -> the opponent hasn't played any before 
            -> So the array which stores their play history is empty
        -> Otherwise -> there is an opponent history 
            -> In this case, we are updating the array of previous plays by the opponent with the last play which they had 
        -> Either we are initialising the array as empty for the first round -> or we are updating it with the opponent's last play 
        -> This builds the array of their previous plays
            -> We can call on this when using machine learning to predict what our next move (guess) should be (this is the output of this function)
"""

    if prev_play == "":
        opponent_history = []
    else:
        opponent_history.append(prev_play)
 
"""
INITIALISING THE NEURAL NETWORK:
--------------------------------------------------------

        We are playing a best of 3 game:
                -> the game we are playing has three rounds 
                -> n being the number of rounds
                -> you play a round, you do that three times 
                -> and then whoever wins the most rounds wins the game 
                -> that is one game (three rounds)

        Between the start (n=0) of the game and the start of the third round of the game (n=3) <- so for n = 0, 1, 2:
                -> these are for the first two rounds of the game 
                -> opponent_history is an array that stores all of the previous plays of the opponent 
                -> we are operating in between when this array is empty and when it had two values

        Possibilities we have are:
                -> n = 0 <- i.e this is the start of the game
                -> we have no opponent history in this case <- the opponent history is stored in the array which was previously updated in the block of code above 
                -> the record of plays which the opponent has previously made is empty in this case 
                -> n = 1 or 2 <- these are the conditions which are under 'else' for this 

        If the game has just started:
                -> in other words if prev_play == "" <- if the array of plays which the opponent has previously played is empty then we are at the start of the game
                -> in this case -> we are initialising the machine learning model 
                -> this model will take the history of plays which the opponent and use it to predict what our play should be currently, to maximise the chances of winning 
                -> we are initialising the machine learning model which sets the different layers which we will train 
                -> this was done by following the TensorFlow examples in the first half of the content for the course
                -> notes for these are in the project repository 

        Initialising the neural network which the project uses: 
                -> we are using keras to create a neural network 
                -> this is a sequential model <- each node has one input and one output 
                -> (1, 12) <- we are passing an array into the model 
                -> the data which we are training it on is the past plays (ours and the opponent's)
                -> sigmoid activation function 
                -> this is being used for each neurone in the layer 
                -> this was suggested for use in the examples in the course content 
                -> we are using this to set the values of the nodes to between 0 and 1 in the model <- so that they outputs represent probability 
                -> this function looks like a tanh curve, but it's normalising the outputs 
                -> three output neurones 
                -> there are three output nodes 
                -> we want to predict the play which the player should place
                -> there are three different outcomes (R, P or S)
                -> each of the output nodes corresponds to one of these outcomes (so there are three) 
                -> the classes which this outputs are the ones which we want to play (R, P or S)
                -> it's returning values in between 0 and 1 because of the sigmoid squashing function 
                -> we have three of these output values 
                -> we are passing an array into the model 
                        -> this array is a history of the past plays, which it learns and performs gradient descent on 
                        -> we are then passing this to three output nodes -> one for R, one for P and one for S
                        -> each of these three nodes has a value between 0 and 1
                        -> this is the probability that the next play will be the value which that specific output node represents
                        -> we are saying 
                        -> given this array of all of the different past plays  
                        -> predict how likely the next play is to be R, P or S 
                        -> each of those different outputs is the probability that the next play will be that value 
                        -> this is taking that architecture (how the layers and nodes in the neural network are set up) and initialising it at the start of the game 
                        -> for the 0th round of the game 
                        -> we are setting up this architecture 
                        -> the neurones inside the model (36) were selected after experimenting with the model 
                        -> adding in a dense (hidden) layer to ensure that more connections / possibilities are considered by the model 
                        -> we then take the output (R, P or S) which has the highest probability
"""

        # between rounds 0 and 3 of the game 
    if len(opponent_history) < 3:
        # if the game has never been played before 
        if prev_play == "":
            my_prev_play = ""

        # initialise the machine learning model which we will later train 
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(1, 12)),
                tf.keras.layers.Dense(36, activation="sigmoid"),
                tf.keras.layers.Dense(3, activation="sigmoid")
            ])

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.save("model.h5")

        else:
            my_prev_play = my_history[-1]
            
"""
TO DETERMINE WHO WINS AFTER A MATCH:
--------------------------------------------------------

        Variables in this section of code:
                -> my_prev_play <- this is the variable which stores the value of our previous play
                -> prev_play <- this is the variable which stores the value of the opponent's previous play 
                -> winner_play <- this variable stores the value of the winning play 

                -> in other words, we have two variables 
                        -> the first is for our guess, and the second is for the opponent's guess
                        -> our aim is to figure out which beats which 
                        -> then to store the value of the winning play in the winner_play variable 
        
        Different possibilities:
                -> either we win or the opponent wins 
                -> the first block is all the conditions we would win 
                -> the second block is all the conditions the opponent would win 
                -> for us to win -> we would play R and the opponent would play S
                        -> we are encoding all of the different conditions under which the opponent vs us would win
                -> in each of these cases, we are updating the value of the winner_play variable to store the value of the winning play
                        -> we aren't storing who won (us or the opponent) -> just the value of the play which won 
                -> the final possibility is there is a draw -> in which case, the winning play is set to the value of our play 
                        -> this can also be set to the value of the opponent's play in this case (they are the same)
"""

        # we win 
        if (my_prev_play == "P" and prev_play == "R") or (my_prev_play == "R" and prev_play == "S") or (my_prev_play == "S" and prev_play == "P"):
            winner_play = my_prev_play
        
        # the opponent wins 
        elif (prev_play == "P" and my_prev_play == "R") or (prev_play == "R" and my_prev_play == "S") or (prev_play == "S" and my_prev_play == "P"):
            winner_play = prev_play

        # draw
        elif (my_prev_play == prev_play):
            winner_play = my_prev_play

"""
SETTING THE OUTPUT OF THE GUESS FUNCTION:
--------------------------------------------------------

        Setting the output of the guess function: 
                -> we want the model to output a guess -> this is what we will play in the next round 
                -> this requires the model to be trained
                -> training the model requires a past history of what our plays and our opponents' plays are
                -> this history takes time to build -> one game is made up of three rounds of RPS
                -> in between the start of the game and when we have enough information to sensibly train the model with, to make predictions about what our next guess should be, 
                        we are setting the value of what those (our) guesses should be 
                -> this is what the model is returning -> a guess about what our next move should be to maximise the odds of winning the game 
                        -> or increase them above 50% -> which is what it would be when playing against a random opponent 
        
        The context we are operating in: 
                -> we don't have enough data to train the model to make predictions yet -> so within the first game
                -> we are at the end of one game and moving onto the next 
                -> in other words, we've just played a round -> and in that round either us or them wins 
                -> we know what the winning play was (R, P or S) in the last round
                -> so if the winning play was R -> our next guess won't be an R 
                -> what we are doing is saying if the winning hand from the previous round was this value (stored in the variable winner_play), then our guess for the 
                        next round (this new round) is going to be something other than what the last winning guess was
                        -> and it doesn't matter in this case whether winner_play came from us or the opponent <- it was the winning hand 
                        -> we are taking that guess and storing it in the variable called guess 
                        -> this guess is the output of this entire function (it's our guess for this round)
                -> the other case is 
                        -> the winning hand in the previous game was a draw
                        -> in which case our guess for the next round is going to be random 
"""

        # if the winning hand in the last round was P -> we don't guess P again for the next round 
        if winner_play == "P":
            guess = "S"

        # if the winning hand in the last round was R -> we don't guess R again for the next round 
        elif winner_play == "R":
            guess = "P"
        
        # if the winning hand in the last round was S -> we don't guess S again for the next round 
        elif winner_play == "S":
            guess = "R"

        # if the winning hand in the last round was a draw, then our guess for the next round is random 
        else:
            guess = random.choice(["P", "R", "S"])

"""
CONVERTING THE RPS DATA OF THE OPPONENT INTO ARRAYS FOR TRAINING THE MODEL
--------------------------------------------------------

This code converts the opponent's RPS moves into arrays for model training:
   -> It formats the opponent's moves for our machine learning model.
   -> Operates in the three rounds and more regime.
      -> Each game has three rounds.
      -> 'else' implies operating in the three or more rounds regime.
         -> We have sufficient data at this point for predictions and model training.
   -> Converts opponent's moves (R, P, S) into arrays understood by the model.
   -> Handles not just the opponent's last move but also from multiple moves ago.

Storing the opponent's last move in an array:
   Explanation:
      -> Uses one-hot encoded arrays for the neural network.
      -> Each element represents R, P, or S.
      -> Targets the most recent play of the opponent (opponent_history[-1]).
      -> Encodes array_last based on the last move (P, R, or S).
      -> Outputs:
         -> array_last: Encodes the last move in a 1's and 0's array.
         -> last: Stores the value of R, P, or S as a number between 0 and 2.
"""
    else:
        # Encoding the opponent's recent plays into arrays and variables for model input
        # -> Converts strings to numbers
        if opponent_history[-1] == "P":
            last, array_last = 0, np.array([1, 0, 0])
        elif opponent_history[-1] == "R":
            last, array_last = 1, np.array([0, 1, 0])
        elif opponent_history[-1] == "S":
            last, array_last = 2, np.array([0, 0, 1])

        # Similar code as above for the opponent's penultimate play (n=-2)
        if opponent_history[-2] == "P":
            array_second_to_last = np.array([1, 0, 0])
        elif opponent_history[-2] == "R":
            array_second_to_last = np.array([0, 1, 0])
        elif opponent_history[-2] == "S":
            array_second_to_last = np.array([0, 0, 1])

        # Similar code for the opponent's play from three moves ago (n=-3)
        if opponent_history[-3] == "P":
            array_third_to_last = np.array([1, 0, 0])
        elif opponent_history[-3] == "R":
            array_third_to_last = np.array([0, 1, 0])
        elif opponent_history[-3] == "S":
            array_third_to_last = np.array([0, 0, 1])

        # Aim is to store opponent's plays from three matches ago for model training
        # -> Included in an else block after three games have been played
        # -> For the first three games, adapts strategy based on the previous game's outcome
        # -> Converts play history into matrices/numbers for the neural network initialised earlier
            
"""
CONVERTING OUR RPS DATA INTO ARRAYS FOR TRAINING THE MODEL
--------------------------------------------------------
   -> Exactly the same process as the previous block of code
   -> Converts R, P, S string values into values for model training
   -> Trains the model on the past three plays of both us and the opponent
   -> my_history stores our past moves
"""

        # Convert the value of our last play from a string (R, P, or S) into an array for the model to understand
            # -> n = -1 
        if my_history[-1] == "P":
            my_array_last = np.array([1, 0, 0])
        elif my_history[-1] == "R":
            my_array_last = np.array([0, 1, 0])
        elif my_history[-1] == "S":
            my_array_last = np.array([0, 0, 1])

        # Convert the value of our play from two moves ago from a string into an array for the model to understand
            # -> n = -2
        if my_history[-2] == "P":
            my_array_second_to_last = np.array([1, 0, 0])
        elif my_history[-2] == "R":
            my_array_second_to_last = np.array([0, 1, 0])
        elif my_history[-2] == "S":
            my_array_second_to_last = np.array([0, 0, 1])

        # Convert the value of our play from three moves ago from a string into an array for the model to understand
            # -> n = -3
        if my_history[-3] == "P":
            my_array_third_to_last = np.array([1, 0, 0])
        elif my_history[-3] == "R":
            my_array_third_to_last = np.array([0, 1, 0])
        elif my_history[-3] == "S":
            my_array_third_to_last = np.array([0, 0, 1])


































"""
UPDATE THIS 
--------------------------------------------------------

"""      
        row_train = np.concatenate((my_array_third_to_last, my_array_second_to_last, array_third_to_last, array_second_to_last)).reshape(1, 1, 12)
        row_prediction = np.concatenate((my_array_second_to_last, my_array_last, array_second_to_last, array_last)).reshape(1, 1, 12)
        label_train = np.array(last).reshape(1, 1, 1)
        



















"""
UPDATE THIS 
--------------------------------------------------------

"""
        model = tf.keras.models.load_model("model.h5")
        model.fit(row_train, label_train, epochs=100, verbose=0)















"""
UPDATE THIS 
--------------------------------------------------------

"""
        prediction = model.predict(row_prediction)
        pred = ["P", "R", "S"][np.argmax(prediction)]















"""
UPDATE THIS 
--------------------------------------------------------

"""
        model.save("model.h5")












"""
UPDATE THIS 
--------------------------------------------------------

"""        
        if pred == "P":
            guess = "S"
        elif pred == "R":
            guess = "P"
        elif pred == "S":
            guess = "R"
    












"""
UPDATE THIS 
--------------------------------------------------------

"""

    my_history.append(guess)
    return guess

# %%
