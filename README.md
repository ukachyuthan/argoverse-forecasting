[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

For installation of ARGOVERSE API: https://github.com/argoai/argoverse-api

For the base code referred to in this implementation: https://github.com/jagjeet-singh/argoverse-forecasting


#### Train Models:

The options available are lstm_train, GRU_train, rnn_train

Using Map prior:
```
$ python lstm_train_test.py --train_features <path/to/train/features> --val_features <path/to/val/features> --test_features <path/to/test/features> --model_path <path/to/saved/checkpoint> --use_map --use_delta --obs_len 20 --pred_len 30 --traj_save_path <pkl/file/for/forecasted/trajectories>
```

Using Social features:
```
$ python lstm_train_test.py --train_features <path/to/train/features> --val_features <path/to/val/features> --test_features <path/to/test/features> --model_path <path/to/saved/checkpoint> --use_social --use_delta --normalize --obs_len 20 --pred_len 30  --traj_save_path <pkl/file/for/forecasted/trajectories>
```

Neither map nor social:
```
$ python lstm_train_test.py --train_features <path/to/train/features> --val_features <path/to/val/features> --test_features <path/to/test/features> --model_path <path/to/saved/checkpoint> --use_delta --normalize --obs_len 20 --pred_len 30 --model_path <pkl/file/path/for/model> --traj_save_path <pkl/file/for/forecasted/trajectories>
```

| Component | Mode | Baseline | Runtime |
| --- | --- | --- | --- |
| LSTM (`lstm_train_test.py`) | train | Map prior | 2 hrs |
| LSTM (`lstm_train_test.py`) | test | Map prior | 1.5 hrs |
| LSTM (`lstm_train_test.py`) | train | Social | 5.5 hrs |
| LSTM (`lstm_train_test.py`) | test | Social | 0.1 hr |
| LSTM (`lstm_train_test.py`) | train | Neither Social nor Map | 5.5 hrs |
| LSTM (`lstm_train_test.py`) | test | Neither Social nor Map | 0.1 hr |
---

### 3) Metrics and visualization

#### Evaluation metrics

Here we compute the metric values for the given trajectories. Since ground truth trajectories for the test set have not been released, you can run the evaluation on the val set. If doing so, make sure you don't train any of the above baselines on val set.

Some examples:

Evaluating a baseline that didn't use map and allowing 6 guesses
```
python eval_forecasting_helper.py --metrics --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --miss_threshold 2 --features <path/to/test/features> --max_n_guesses 6
```

Evaluating a baseline that used map prior and allowing 1 guesses along each of the 9 centerlines
```
python eval_forecasting_helper.py --metrics --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --miss_threshold 2 --features <path/to/test/features> --n_guesses_cl 1 --n_cl 9 --max_neighbors_cl 3
```

Evaluating a K-NN baseline that can use map for pruning and allowing 6 guesses
```
python eval_forecasting_helper.py --metrics --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --miss_threshold 2 --features <path/to/test/features> --prune_n_guesses 6
```

It will print out something like
```
------------------------------------------------
Prediction Horizon : 30, Max #guesses (K): 1
------------------------------------------------
minADE: 3.533317191869932
minFDE: 7.887520305278937
DAC: 0.8857479236783845
Miss Rate: 0.9290787402582446
------------------------------------------------
```

#### Visualization

Here we visualize the forecasted trajectories

```
python eval_forecasting_helper.py --viz --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --features <path/to/test/features>
```
Some sample results are shown below

| | |
|:-------------------------:|:-------------------------:|
| ![](images/lane_change.png) | ![](images/map_for_reference_1.png) |
| ![](images/right_straight.png) | ![](images/map_for_reference_2.png) |


---
