# rl

## Environments

### Battleship
Battleship is a strategy-based guessing game with incomplete information. Each player chooses initial locations to place their fleet and then alternate turns calling "shots" at the other player. The objective of the game is to destroy the opposing player's fleet.

<img align="center" src="https://github.com/Liberty3000/rl/blob/master/graphics/battleship.gif" width="1000"/>

### Dark Chess
Dark chess is a chess variant with incomplete information. Players cannot see the entire board, only their own pieces and the squares that they can legally move to.

The goal of this chess variant is not to checkmate the king, but to capture it. A player is not told if their king is in check. Failing to move out of check, or moving into check, are both legal, and can result in a capture and loss of the game.
<p align="center">
  <img align="center" src="https://github.com/Liberty3000/rl/blob/master/graphics/dark_chess.gif" width="300"/>
</p>

### BoxWorld
BoxWorld is a perceptually simple but combinatorially complex environment that requires abstract relational reasoning and planning. It consists of a 12 × 12 pixel room with keys and boxes randomly scattered. The room also contains an agent, represented by a single dark gray pixel, which can move in four directions: up, down, left, right.

Keys are represented by a single colored pixel. The agent can pick up a loose key (i.e., one not adjacent to any other colored pixel) by walking over it. Boxes are represented by two adjacent colored pixels – the pixel on the right represents the box’s lock and its color indicates which key can be used to open that lock; the pixel on the left indicates the content of the box which is inaccessible while the box is locked.

To collect the content of a box the agent must first collect the key that opens the box (the one that matches the lock’s color) and walk over the lock, which makes the lock disappear. At this point the content of the box becomes accessible and can be picked up by the agent. Most boxes contain keys that, if made accessible, can be used to open other boxes. One of the boxes contains a gem, represented by a single white pixel. The goal of the agent is to collect the gem by unlocking the box that contains it and picking it up by walking over it. Keys that an agent has in possession are depicted in the input observation as a pixel in the top-left corner.
<p align="center">
  <img align="center" src="https://github.com/Liberty3000/rl/blob/master/graphics/box_world.gif" width="450"/>
</p>
