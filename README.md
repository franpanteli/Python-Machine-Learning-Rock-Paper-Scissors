# Python-Machine-Learning-Rock-Paper-Scissors
## Rock-Paper-Scissors AI Challenge

This repository contains my work from the first freeCodeCamp project in its course called [Machine Learning with Python](https://www.freecodecamp.org/learn/machine-learning-with-python/). This project uses neural networks and machine learning in Python for a Rock Paper Scissors game, which uses TensorFlow to beat a random opponent for more than 60% of the games played. [This machine learning challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors) involveed creating an AI player that competes against various opponents in the classic Rock-Paper-Scissors game, testing and refining strategies for success. 

## How to Play

1. **Player Strategies:**
   - The project includes several AI players, each with its unique strategy. These players are named `quincy`, `abbey`, `kris`, `mrugesh`, and `player`
   - The `player` AI is the focus of the challenge, and its performance is tested against other opponents

2. **Unit Tests:**
   - Unit tests in Python have been implemented to evaluate the `player`'s success against each opponent
   - Run the tests using the provided `test_module.py` to ensure that the player wins against each opponent at least 60% of the time

## Getting Started

1. **Installation:**
   - Clone the repository to your local machine, running Python
   - The starter code for this project was initially provided in Replit, although VSCode was used to develop it

2. **Run the Game:**
   - Execute the `main.py` script to run the Rock-Paper-Scissors game with the AI players
   - Observe the performance of the `player` against different opponents

## Understanding the Model

- **Player Function:**
  - The file `RPS.py` contains a player function with in-depth comments explaining the approach used to create and train this machine learning model
  - This uses a sigmoid activation function, the Adam algorithm to train the model, and sets its architecture
  - The comments in this file explain the approach used to design this architecture in more depth, with a view to the Rock Paper Scissors game

## Unit Tests

To run the unit tests, use the following command:

```bash
python test_module.py
