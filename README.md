# Program Synthesis for ARC-AGI with Discrete Search

This project implements a **program synthesis system** for solving tasks from the **Abstraction and Reasoning Corpus (ARC)** using **discrete search algorithms**. The system searches for grid transformation programs that map input examples to their correct outputs.

Candidate programs are constructed from a **contained domain-specific language (DSL)** of grid operations and explored using **Breadth-First Search (BFS)**, **Greedy Best-First Search (GBFS)**, and **A\***. Heuristics are used to guide the search toward promising transformations while keeping the search space manageable.

## Features
- Program synthesis over a constrained DSL
- Discrete search algorithms (BFS, GBFS, A*)
- Heuristic-guided symbolic reasoning
- Abstract syntax tree (AST) representation of programs

## Technologies Used
- Python
- NumPy

## Project Context
Developed as part of **COSC 3P71 – Artificial Intelligence** at Brock University.
