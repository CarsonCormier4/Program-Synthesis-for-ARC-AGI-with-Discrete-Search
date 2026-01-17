# Program Synthesis for ARC-AGI Using Discrete Search

This project explores solving problems from the Abstraction and Reasoning Corpus (ARC) by automatically generating small programs that transform input grids into their correct outputs.

The system builds candidate solutions using a limited set of grid operations and searches through them using algorithms such as Breadth-First Search (BFS), Greedy Best-First Search (GBFS), and A*. Heuristics guide the search toward likely solutions while keeping the search space manageable.

## Features
- Program synthesis using a constrained set of grid operations
- Discrete search algorithms (BFS, GBFS, A*)
- Heuristic-guided symbolic reasoning
- Programs represented as abstract syntax trees (ASTs)

## Technologies Used
- Python
- NumPy

## Project Context
Developed as part of **COSC 3P71 – Artificial Intelligence** at Brock University.
