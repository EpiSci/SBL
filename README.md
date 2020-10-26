# Autonomous sPOMDP Environment Modeling With Partial Model Exploitation

This is the code repository for Autonomous sPOMDP Environment Modeling With Partial Model Exploitation, which demonstrates the improved speed benefits of using our proposed path finding algorithm to more quickly learn an sPOMDP environment (Test 2 below). This repository also demonstrates the improved accuracy that can be obtained from using a frequency-independent transition posterior update equation (Test 1 below) and the robustness of our proposed SDE generation algorithm (Test 3 below).

## Generating Data

To generate data, open a terminal and run
> `python test.py`

in the home directory. Generated data is printed to `.csv` files in the Testing Data directory. To specify which experiment to run, the `main()` function in `test.py` can be modified. The following lists which variables can be modified to specify the test:

- `TestNum`: Which test type to run. Use only the test number (e.g. For Test 3, use 3)
    - Test 1: Test the transition posteriors update
    - Test 2: Test the agent navigation algorithm
    - Test 3: Test the invalid SDE splitting


- `versionNum`: Which test version to run. Use only the version number (e.g. For version 3, use 3)
  - For Test 1: 1 corresponds to frequency-dependent transition posteriors update equation, 3 corresponds to our proposed frequency-independent transition posteriors update equation
  - For Test 2:  1 corresponds to frequency-independent transition posteriors update equation without control, 2 corresponds to frequency-independent transition posteriors update equation with control, 3 corresponds to frequency-dependent transition posteriors update equation without control
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

Graphs can be generated using data already present in the repository (in the *Testing Data* directory) or by data collected from the "**Generating Data**" section above.

First ensure that the correct data is present in the appropriately labelled subdirectories within the *Testing Data* directory. For example, if you are interested in generating a graph for the Invalid SDE splitting, then make sure the raw `.csv` data files are present in the `Testing Data/Test3_v1` and `Testing Data/Test3_v3` directories.

Next select which test to generate data for. The `generateGraphs.py` file has been formatted to facilitate this process. In the `main()` function, uncomment the test you would like to generate graphs for.

Finally, generate the graphs by opening a terminal and running
> `python generateGraphs.py`
