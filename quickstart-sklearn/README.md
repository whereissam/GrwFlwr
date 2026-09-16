# GrwFlwr

## Project structure
```shell
quickstart-sklearn
├── sklearnexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```


## Prerequisites

```shell
# 1. Install flower
pip install flwr
```


## Installation
```bash 
# 1. Install dependencies
pip install -e .


```

## Execution

`flwr run .`

```bash 
~/.flwr/config.toml
```






### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `sklearnexample` package.

```bash

```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run . --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config penalty="'l1'" --stream
```

### Run with the Deployment Engine
