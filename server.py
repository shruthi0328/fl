import flwr as fl

# Define strategy (you can also tweak FedAvg/FedAvgM params)
strategy = fl.server.strategy.FedAvgM(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

# Start the server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
