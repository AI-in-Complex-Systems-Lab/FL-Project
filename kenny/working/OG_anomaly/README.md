# PyTorch: From Centralized To Federated

This example demonstrates how an already existing centralized PyTorch-based machine learning project can be federated with Flower.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone git@github.com:anaskalt/anomaly-fl.git
cd anomaly-fl/
```

This will create a new directory called `anomaly-fl` containing the following files:

```shell
-- requirements.txt
-- data/
   -- cell_data.py
-- anomaly.py
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## From Centralized To Federated

You can simply start the centralized training as described in the tutorial by running `anomaly.py`:

```shell
python3 anomaly.py
```

The next step is to use our existing project code in `anomaly.py` and build a federated learning system based on it. The only things we need are a simple Flower server (in `server.py`) and a Flower client that connects Flower to our existing model and data (in `client.py`). The Flower client basically takes the already defined model and training code and tells Flower how to call it.

Start the server in a terminal as follows:

```shell
python3 server.py
```

Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py
```

Start client 2 in the second terminal:

```shell
python3 client.py
```

Start client 3 in the third terminal:

```shell
python3 client.py
```

Start client 4 in the fourth terminal:

```shell
python3 client.py
```

Start client 5 in the fifth terminal:

```shell
python3 client.py
```


You are now training a PyTorch-based NN anomaly detection model, federated across five clients. Each client has a different part of Dataset.
