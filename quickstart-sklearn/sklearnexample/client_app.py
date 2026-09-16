"""sklearnexample: A Flower / sklearn app."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Importiere hier deine eigenen Funktionen aus der task.py
from sklearnexample.task import prepare_data, train_model


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):

    # 1. Daten laden und mit Hilfsfunktion skalieren
    partition_id = context.node_config["partition-id"]
    X_train_scaled, _, y_train, _, _ = prepare_data("./data.csv", partition_id=partition_id)

    # 2. Empfangene Gewichte vom Server auslesen & im Modell setzen
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model = LinearRegression()
    model.coef_ = ndarrays[0]
    model.intercept_ = ndarrays[1][0]

    # 3. Modell lokal fiten
    model.fit(X_train_scaled, y_train)

    # 4. MSE für Regressionsbewertung berechnen
    y_pred = model.predict(X_train_scaled)
    train_mse = float(mean_squared_error(y_train, y_pred))

    # 5. Aktualisierte Gewichte + Metriken an den Server senden
    model_record = ArrayRecord([
        model.coef_, 
        np.array([model.intercept_])
    ])

    metric_record = MetricRecord({
        "num-examples": len(X_train_scaled),
        "train_mse": train_mse
    })

    return Message(
        content=RecordDict({"arrays": model_record, "metrics": metric_record}),
        reply_to=msg
    )



@app.evaluate()
def evaluate(msg: Message, context: Context):

    # 1. Testdaten laden (nimmt das zweite und vierte Element aus prepare_data)
    partition_id = context.node_config["partition-id"]
    _, X_test_scaled, _, y_test, *_ = prepare_data("./data.csv", partition_id=partition_id)

    # 2. Modell erstellen & empfangene Gewichte vom Server setzen
    model = LinearRegression()
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.coef_ = ndarrays[0]
    model.intercept_ = ndarrays[1][0]

    # 3. Vorhersage und Regressions-Metriken berechnen
    y_pred = model.predict(X_test_scaled)
    test_mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(model.score(X_test_scaled, y_test)) # R²-Score als Qualitätsmaß

    # 4. Metriken an den Server zurückschicken
    metrics = {
        "num-examples": len(X_test_scaled),
        "test_mse": test_mse,
        "r2_score": r2,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
