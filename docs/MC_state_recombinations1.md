# Explanation: What are Monte-Carlo States and Why Use Them?

## What are Monte-Carlo States?

Monte-Carlo (MC) states are discrete integer labels assigned to each pixel/grid point 
during a Monte-Carlo simulation of grain growth. In this simulation:

- The microstructure is represented on a discrete grid
- Each grid point is assigned a state number (e.g., 1, 2, 3, 4, 5)
- These states represent different crystallographic orientations or grain identities
- The simulation evolves using probabilistic rules (Monte-Carlo method) to minimize 
    energy and simulate grain boundary motion

## Original MC State Distribution in Your Data

Your MC stack contains multiple MC states, each representing a different grain orientation/identity in the simulated microstructure.

## Why Perform State Re-combinations?

In real materials, we often want to group multiple MC states into meaningful physical 
categories called "species". For example:

1. **Phase Separation**: Some states might represent one phase (e.g., austenite),
     while others represent another phase (e.g., ferrite)

2. **Variant Grouping**: Different MC states might represent crystallographic variants
     that should be analyzed together

3. **Feature Extraction**: By combining states, you can:
     - Extract morphological properties of specific phases
     - Calculate volume fractions of different constituents
     - Analyze spatial distribution and connectivity
     - Study interface characteristics between species

## State Re-combination Workflow

State re-combination groups selected MC states into a new species label:
- Pixels in combined states are grouped together as one species
- Other states are set to background (0)
- Individual connected regions (grains) are detected and relabeled
- Morphological properties are extracted for each grain in the species

## Benefits

This state re-combination approach allows:
- Flexible post-processing without re-running expensive simulations
- Multiple species definitions from the same MC simulation
- Statistical analysis of specific microstructural constituents
- Comparison of properties between different species/phases
