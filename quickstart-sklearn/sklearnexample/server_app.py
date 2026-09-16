"""sklearnexample: A Flower / sklearn app."""
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read Parameters
    num_rounds: int = context.run_config["num-server-rounds"]

    # Create and parameters
    model = LinearRegression()
    model.fit([[0, 0, 0, 0]], [0])

    arrays = ArrayRecord([
        model.coef_, 
        np.array([model.intercept_])
    ])

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    if context.run_config["save-model"]:

        print("\nSaving final model to disk...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        
        model.coef_ = ndarrays[0]
        model.intercept_ = ndarrays[1][0]
        
        joblib.dump(model, "model.joblib")
