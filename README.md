# Autonomous sPOMDP Environment Modeling With Partial Model Exploitation

This is the code repository for Autonomous sPOMDP Environment Modeling With Partial Model Exploitation, which demonstrates the improved speed benefits of using our proposed path finding algorithm to more quickly learn an sPOMDP environment (Test 2 below). This repository also demonstrates the improved accuracy that can be obtained from using a frequency-independent transition posterior update equation (Test 1 below) and the robustness of our proposed SDE generation algorithm (Test 3 below).

## Generating Data

To generate data, run
> `python test.py`

in the home directory. To specify which experiment to run, the `main()` function in `test.py` can be modified. The following lists which variables can be modified to specify the test:

- `TestNum`: Which test type to run. Use only the test number (e.g. For Test 3, use 3)
    - Test 1: Test the transition posteriors update
    - Test 2: Test the agent navigation algorithm
    - Test 3: Test the invalid SDE splitting


- `versionNum`: Which test version to run. Use only the version number (e.g. For version 3, use 3)
  - For Test 1: 1 corresponds to frequency-dependent transition posteriors update equation, 3 corresponds to our proposed frequency-independent transition posteriors update equation
  - For Test 2:
  - For Test 3: 1 corresponds to previous SDE generation algorithms, 3 corresponds to our proposed SDE generation algorithm with "safety checks"

- `envNum`: The testing environment to test on
  - 1: ae-Shape fully built
  - 2: ae-Shape with initial observations
  - 3: ae-Litle Prince with initial observations
  - 32: ae-Little Prince fully built
  - 4: ae-1D Maze with initial observations
  - 42: ae-1D Maze fully built
  - 6: ae-Balance Beam fully built
  - 7: ae-Balance Beam with initial observations

- `numSubTests`: The number of tests to run consecutively

## Generating Graphs
TODO
