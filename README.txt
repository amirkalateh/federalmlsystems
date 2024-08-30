
# Vertical Federated Learning with Fair Resource Allocation

## Project Overview

This project implements a vertical federated learning system with a focus on fair resource allocation using the Talmud division method. The system simulates a federated learning environment where multiple participants collaborate to improve a machine learning model while preserving data privacy and ensuring fair distribution of resources.

## Key Features

1. **Vertical Federated Learning**: Implements a federated learning system where data is vertically partitioned among participants.
2. **Fair Resource Allocation**: Utilizes the Talmud division method to allocate bandwidth fairly based on participants' contributions.
3. **Performance Comparison**: Compares the performance of local, individual, and federated models.
4. **Shapley Value Calculation**: Implements Shapley value calculation for comparison with the Talmud division method.
5. **Dummy and Symmetry Property Tests**: Includes tests for the dummy and symmetry properties of fair allocation.
6. **Computational Efficiency Comparison**: Compares the computational efficiency of Talmud division and Shapley value methods.
7. **Comprehensive Logging and Visualization**: Provides detailed logs and visualizations of the simulation results.

## System Components

- **MainServer**: Represents the central server that coordinates the federated learning process.
- **ParticipantServer**: Represents individual participants in the federated learning system.
- **Logger**: Handles logging of simulation data and results.
- **Various Utility Functions**: For data generation, model training, resource allocation, and result analysis.

## Key Algorithms

1. **Talmud Division**: Implements fair resource allocation based on participants' claims.
2. **Shapley Value Calculation**: Computes the marginal contributions of each participant.
3. **Logistic Regression**: Used as the base model for both local and federated learning.

## Data

The project uses a synthetic dataset generated using sklearn's `make_classification` function. This allows for controlled experiments and easy reproduction of results.

## Visualizations

The project generates several plots to visualize the results:
- Bandwidth usage and computation time
- Claims and allocations comparison
- Shapley values vs Talmud allocations
- Performance metrics comparison
- Model performance comparison across different stages

## How to Run

1. Ensure all required libraries are installed (numpy, pandas, sklearn, matplotlib, seaborn, tqdm).
2. Run the main script:
   ```
   python main.py
   ```
3. Results will be saved in a timestamped directory named `run_TIMESTAMP`.

## Output

- Detailed console output showing the progress and results of each step.
- CSV files containing logs and calculated parameters.
- PNG files with various visualizations of the results.

## Analysis

The simulation demonstrates:
- The effectiveness of vertical federated learning in improving model performance.
- Fair allocation of resources (bandwidth) using the Talmud division method.
- Comparison with Shapley values for contribution assessment.
- Computational efficiency of different allocation methods.

## Future Work

- Implement more complex machine learning models.
- Explore other fair division algorithms and compare their performance.
- Extend the system to handle real-world datasets and scenarios.
- Implement privacy-preserving techniques such as differential privacy.

## Conclusion

This project provides a comprehensive simulation of a vertical federated learning system with a focus on fair resource allocation. It demonstrates significant improvements in model performance while ensuring fair distribution of resources among participants.