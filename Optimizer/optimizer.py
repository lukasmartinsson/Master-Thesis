import time
import optuna
import os
import pandas as pd
import torch
import json
from Preprocessing.preprocessing import preprocessing
from tsai.all import *
from fastai.callback.tracker import EarlyStoppingCallback

def optimize_model(model_type: str, preprocessing_params: dict, n_trials: int, n_epochs: int = 15, folder : str = 'full'):

    # Load or create a new results DataFrame
    global results_df
    
    CLF = preprocessing_params['CLF']

    results_file = f"models/{model_type}/{folder}/{preprocessing_params['index']}/{model_type}_hyperparameters_results.csv"
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        if model_type == 'lstm_fcn_class': results_df = pd.DataFrame(columns=['model', 'df_len', 'epochs', 'seq_length', 'batch_size', 'learning_rate', 'hidden_size', 'rnn_layers', 'rnn_dropout', 'fc_dropout', 'conv_layers', 'kss', 'val_accuracy', 'time'])
        if model_type == 'lstm_fcn_reg': results_df = pd.DataFrame(columns=['model', 'df_len', 'epochs', 'seq_length', 'batch_size', 'learning_rate', 'hidden_size', 'rnn_layers', 'rnn_dropout', 'fc_dropout', 'conv_layers', 'kss', 'val_mae', 'time'])
        if model_type == 'lstm_class': results_df = pd.DataFrame(columns=['model', 'df_len', 'epochs', 'seq_length','batch_size', 'learning_rate', 'hidden_size', 'n_layers', 'rnn_dropout', 'fc_dropout', 'val_accuracy', 'time'])
        if model_type == 'lstm_reg': results_df = pd.DataFrame(columns=['model', 'df_len', 'epochs', 'seq_length','batch_size', 'learning_rate', 'hidden_size', 'n_layers', 'rnn_dropout', 'fc_dropout', 'val_mae', 'time'])
        if model_type == 'tst_class': results_df = pd.DataFrame(columns=['model', 'df_len', 'epochs', 'seq_length','batch_size', 'learning_rate', 'd_model', 'n_layers', 'n_heads', 'd_ff', 'dropout', 'val_accuracy', 'time'])
        if model_type == 'tst_reg': results_df = pd.DataFrame(columns=['model', 'df_len', 'epochs', 'seq_length','batch_size', 'learning_rate', 'd_model', 'n_layers', 'n_heads', 'd_ff', 'dropout', 'val_mae', 'time'])
        if model_type == 'mini_rocket': results_df = pd.DataFrame(columns=['model', 'df_len', 'epochs', 'seq_length','batch_size', 'learning_rate', 'num_features', 'max_dilations_per_kernel', 'kernel_size', 'max_num_channels', 'dropout', 'val_accuracy', 'time'])

    def objective(trial:optuna.Trial):
       
        seq_length = trial.suggest_int('seq_length',3, 50) # Add seq_length as a hyperparameter with appropriate values
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]) # Add batch size as a hyperparameter with appropriate values
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)  # search through all float values between 1e-6 and 1e-2 in log increment steps

        # Changes the data into features and labels with the split used later in TSAI for modelling
        data_train, data_test, _ = preprocessing(**preprocessing_params, sequence_length=seq_length)

        X, y, splits = combine_split_data([data_train[0], data_test[0]],[data_train[1], data_test[1]])

        # Utilizes the GPU if possible
        if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()

        if CLF:
            # Load the data into dataloaders
            dsets = TSDatasets(X, y, splits=splits)
            dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=batch_size)
        else: 
            dls = get_ts_dls(X, y, splits=splits, bs=batch_size)

        if model_type == 'lstm_fcn_class' or model_type == 'lstm_fcn_reg':
            hidden_size = trial.suggest_categorical('hidden_size', [25, 50, 100, 200])
            rnn_layers = trial.suggest_categorical('rnn_layers', [1, 2, 4, 8])
            rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.5, step=.1) # search through all float values between 0.0 and 0.5 with 0.1 increment steps
            fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.5, step=.1) # search through all float values between 0.0 and 0.5 with 0.1 increment steps
            conv_layers = trial.suggest_categorical('conv_layers', [[64, 128, 64], [128, 256, 128], [256, 512, 256]]) # Add conv_layers as a hyperparameter with appropriate sizes
            kss = trial.suggest_categorical('kss', [[3,3,3], [5,3,3], [7,5,3], [7,7,5]]) # add kss to the search space

            # Initialize the LSTMPlus model
            nr_features = X.shape[1] # Number of features
            nr_labels = torch.unique(y).numel() if CLF else 1 # Number of labels

            model =LSTM_FCNPlus(c_in=nr_features, 
                            c_out=nr_labels, 
                            hidden_size=hidden_size,
                            rnn_layers=rnn_layers,
                            rnn_dropout= rnn_dropout,
                            fc_dropout=fc_dropout,
                            conv_layers=conv_layers,
                            kss = kss,
                            shuffle=False)
            
        if model_type == 'lstm_class' or model_type == 'lstm_reg':
            hidden_size = trial.suggest_categorical('hidden_size', [25, 50, 100, 200])
            n_layers = trial.suggest_categorical('n_layers', [1, 2, 4, 8])
            rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.5, step=.1) # search through all float values between 0.0 and 0.5 with 0.1 increment steps
            fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.5, step=.1) # search through all float values between 0.0 and 0.5 with 0.1 increment steps

            # Initialize the LSTM model
            nr_features = X.shape[1] # Number of features
            nr_labels = torch.unique(y).numel() if CLF else 1 # Number of labels

            model = LSTMPlus(c_in=nr_features, 
                        c_out=nr_labels, 
                        hidden_size=hidden_size,
                        n_layers=n_layers,
                        rnn_dropout= rnn_dropout,
                        fc_dropout=fc_dropout)
            
        if model_type == 'tst_class' or model_type == 'tst_reg':
            d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
            n_layers = trial.suggest_categorical('n_layers', [1, 2, 4, 8])
            n_heads = trial.suggest_categorical('n_heads', [8, 16, 32])
            d_ff = trial.suggest_categorical('d_ff', [128, 256, 512, 1024])
            dropout = trial.suggest_float("dropout", 0.0, 0.5, step=.1) # search through all float values between 0.0 and 0.5 with 0.1 increment steps

            # Initialize the TSTPlus model
            nr_features = X.shape[1] # Number of features
            nr_labels = torch.unique(y).numel() if CLF else 1 # Number of labels
            seq_len = X.shape[2] # Sequence length

            model = TSTPlus(c_in=nr_features,
                            c_out=nr_labels,
                            seq_len=seq_len,
                            n_layers=n_layers,
                            d_model=d_model,
                            n_heads=n_heads,
                            d_ff=d_ff,
                            dropout=dropout)
            
        if model_type == 'mini_rocket':
            num_features = trial.suggest_categorical('num_features', [1000,2500,5000,10000])
            max_dilations_per_kernel = trial.suggest_categorical('max_dilations_per_kernel', [8, 16, 32, 64])
            kernel_size = trial.suggest_int('kernal_size',2,20)
            max_num_channels = trial.suggest_int('max_num_channels ',2,20)
            dropout = trial.suggest_float("dropout", 0.0, 0.5, step=.1) # search through all float values between 0.0 and 0.5 with 0.1 increment steps

            mrf = MiniRocketFeaturesPlus(X.shape[1], X.shape[2],
                            num_features=num_features,
                            max_dilations_per_kernel=max_dilations_per_kernel,
                            kernel_size=kernel_size,
                            max_num_channels=max_num_channels).to(default_device())
            
            X_train = X[splits[0]]
            mrf.fit(X_train)
            X_feat = get_minirocket_features(X, mrf, chunksize=1024)

            dls = get_ts_dls(X_feat, y, splits=splits, bs = 32)

            nr_labels = torch.unique(y).numel() # Number of labels
            model = build_ts_model(MiniRocketHead, c_out=nr_labels, dls=dls)

        if CLF: 
            learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=accuracy, cbs=[EarlyStoppingCallback(patience=5)])
        else:
            learn = ts_learner(dls, model, metrics=[rmse, mae], cbs=[EarlyStoppingCallback(patience=5)])

        start = time.time()
        with ContextManagers([learn.no_logging(), learn.no_bar()]): # [Optional] this prevents fastai from printing anything during training
            learn.fit_one_cycle(n_epochs, lr_max=learning_rate)
        training_time = time.time() - start

        # Get the validation accuracy of the last epoch
        if CLF: 
            val_accuracy = learn.recorder.values[-1][2]
            acc = get_binary_accuracy_clf(learn, data_test[0], data_test[1], preprocessing_params['buckets'])
        else:
            val_mae = learn.recorder.values[-1][2]
            acc = get_binary_accuracy_reg(learn, data_test[0], data_test[1])

        # Save the hyperparameters and validation accuracy in a dictionary
        if model_type == 'lstm_fcn_class':
            trial_results = {
                'model': model_type,
                'df_len': len(preprocessing_params['df']),
                'epochs' : n_epochs,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'hidden_size': hidden_size,
                'rnn_layers': rnn_layers,
                'rnn_dropout': rnn_dropout,
                'fc_dropout': fc_dropout,
                'conv_layers': conv_layers,
                'kss': kss,
                'val_accuracy': val_accuracy,
                'time': training_time
            }
        
        if model_type == 'lstm_fcn_reg':
            trial_results = {
                'model': model_type,
                'df_len': len(preprocessing_params['df']),
                'epochs' : n_epochs,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'hidden_size': hidden_size,
                'rnn_layers': rnn_layers,
                'rnn_dropout': rnn_dropout,
                'fc_dropout': fc_dropout,
                'conv_layers': conv_layers,
                'kss': kss,
                'val_mae': val_mae,
                'time': training_time
            }
        
        if model_type == 'lstm_class':
            trial_results = {
                'model': model_type,
                'df_len': len(preprocessing_params['df']),
                'epochs' : n_epochs,
                'seq_len': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'hidden_size': hidden_size,
                'n_layers': n_layers,
                'rnn_dropout': rnn_dropout,
                'fc_dropout': fc_dropout,
                'val_accuracy': val_accuracy,
                'time': training_time
            }
        
        if model_type == 'lstm_reg':
            trial_results = {
                'model': model_type,
                'df_len': len(preprocessing_params['df']),
                'epochs' : n_epochs,
                'seq_len': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'hidden_size': hidden_size,
                'n_layers': n_layers,
                'rnn_dropout': rnn_dropout,
                'fc_dropout': fc_dropout,
                'val_mae': val_mae,
                'time': training_time
            }
        
        if model_type == 'tst_class':
            trial_results = {
                'model': model_type,
                'df_len': len(preprocessing_params['df']),
                'epochs' : n_epochs,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'd_model': d_model,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'd_ff': d_ff,
                'dropout': dropout,
                'val_accuracy': val_accuracy,
                'time': training_time
            }
        
        if model_type == 'tst_reg':
            trial_results = {
                'model': model_type,
                'df_len': len(preprocessing_params['df']),
                'epochs' : n_epochs,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'd_model': d_model,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'd_ff': d_ff,
                'dropout': dropout,
                'val_mae': val_mae,
                'time': training_time
            }
        
        if model_type == 'mini_rocket':
            trial_results = {
                'model': model_type,
                'df_len': len(preprocessing_params['df']),
                'epochs' : n_epochs,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_features': num_features,
                'max_dilations_per_kernel':max_dilations_per_kernel,
                'kernel_size':kernel_size,
                'max_num_channels':max_num_channels,
                'dropout': dropout,
                'val_accuracy': val_accuracy,
                'time': training_time
            }

        # Append the results to the dataframe 
        global results_df

        # Save the results DataFrame to a CSV file
        results_df = results_df.append(trial_results, ignore_index=True)
        results_df.to_csv(results_file, index=False)

        # Return the validation accuracy value of the last epoch
        return acc

    # Create the necessary folders if they don't exist
    os.makedirs(f"models/{model_type}/{folder}/{preprocessing_params['index']}", exist_ok=True)

    # Load or create a new study
    study_name = f"{model_type}_study"
    storage_name = f"sqlite:///models/{model_type}/{folder}/{preprocessing_params['index']}/{study_name}.db"
    if os.path.exists(f"models/{model_type}/{folder}/{preprocessing_params['index']}/{study_name}.db"):
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize') # if CLF else "minimize")
        
    study.optimize(objective, n_trials=n_trials)

    # Save the best parameters
    best_params = study.best_params
    best_params_path = f"models/{model_type}/{folder}/{preprocessing_params['index']}/{model_type}_best_params.json"
    with open(best_params_path, "w") as f:
            json.dump(best_params, f)


def optimize_data(model_type: str, preprocessing_params: dict, n_trials: int, n_epochs: int = 15, batch_size: int = 32, learning_rate: int = 0.001, index : str = None, folder : str = 'full'):

    global results_df
    
    CLF = preprocessing_params['CLF']
    
    results_file = f"Optimizer/data_optimization/{model_type}/{folder}/{model_type}_{index}_hyperparameters_results.csv"
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else: 
        results_df = pd.DataFrame(columns=['model','df_len', 'epochs', 'lag', 'dif_all', 'train_size', 'index', 'buckets', 'TI', 'bin_accuracy'])

    def objective(trial:optuna.Trial):
        
        if index == '1min': 
            lag = trial.suggest_categorical('lag',[1,5,15])
        elif index == '5min': 
            lag = trial.suggest_categorical('lag',[1,3])
        else:
            lag = trial.suggest_categorical('lag',[1])

        dif_all = trial.suggest_categorical('dif_all',[True, False])
        train_size = 0.95 # trial.suggest_categorical('train_size',[0.95, 0.975, 0.99])
        index_test = trial.suggest_categorical('index',[None, index])
        preprocessing_params.pop('index', None)

        if CLF:
            buckets = trial.suggest_categorical('buckets',[1,3,5,10,15])
        else:
            buckets = 'None'
        
        TI = trial.suggest_categorical('TI',[True, False])

        # Changes the data into features and labels with the split used later in TSAI for modelling
        data_train, data_test, _ = preprocessing(**preprocessing_params, 
                                                 lag = lag, 
                                                 dif_all = dif_all, 
                                                 train_size = train_size,
                                                 index = index_test,
                                                 buckets = buckets,
                                                 TI = TI)

        X, y, splits = combine_split_data([data_train[0], data_test[0]],[data_train[1], data_test[1]])
        
        #Convert to long
        y=y.long()
        
        # Utilizes the GPU if possible
        if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()
        
        if CLF:
            # Load the data into dataloaders
            dsets = TSDatasets(X, y, splits=splits)
            dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=batch_size)
        else: 
            dls = get_ts_dls(X, y, splits=splits, bs=batch_size)
        
        # Initialize the LSTMPlus model
        nr_features = X.shape[1] # Number of features
        nr_labels = torch.unique(y).numel() if CLF else 1 # Number of labels

        if model_type == 'tst_class' or model_type == 'tst_reg': model = TSTPlus(c_in=nr_features, c_out = nr_labels, seq_len = preprocessing_params['sequence_length'])
        
        if model_type == 'lstm_class' or model_type == 'lstm_reg': model = LSTMPlus(c_in=nr_features, c_out = nr_labels)
       
        if model_type == 'lstm_fcn_class' or model_type == 'lstm_fcn_reg': model = LSTM_FCNPlus(c_in=nr_features, c_out = nr_labels)
       
        if model_type == 'mini_rocket': 
            mrf = MiniRocketFeaturesPlus(X.shape[1], X.shape[2]).to(default_device())
            X_train = X[splits[0]]
            mrf.fit(X_train)
            X_feat = get_minirocket_features(X, mrf, chunksize=1024)
            dls = get_ts_dls(X_feat, y, splits=splits)
            model = build_ts_model(MiniRocketHead,c_out=torch.unique(y).numel(), dls=dls)
            X = X_feat

        if CLF: 
            learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=accuracy, cbs=[EarlyStoppingCallback(patience=5)])
        else:
            learn = ts_learner(dls, model, metrics=[rmse, mae], cbs=[EarlyStoppingCallback(patience=5)])

        with ContextManagers([learn.no_logging(), learn.no_bar()]): # [Optional] this prevents fastai from printing anything during training
            learn.fit_one_cycle(n_epochs, lr_max=learning_rate)

        if CLF: 
            #val_accuracy = learn.recorder.values[-1][2]
            acc = get_binary_accuracy_clf(learn, data_test[0], data_test[1], buckets)
        else:
            #val_mae = learn.recorder.values[-1][2]
            acc = get_binary_accuracy_reg(learn, data_test[0], data_test[1])
        
        trial_results = {
                'model': model_type,
                'df_len':len(preprocessing_params['df']),
                'epochs': n_epochs,
                'lag': lag,
                'dif_all': dif_all,
                'train_size': train_size,
                'index':index_test,
                'buckets':buckets,
                'TI':TI,
                'bin_accuracy': acc
            }
        
        # Append the results to the dataframe 
        global results_df

        # Save the results DataFrame to a CSV file
        results_df = results_df.append(trial_results, ignore_index=True)
        results_df.to_csv(results_file, index=False)
        return acc

    # Create the necessary folders if they don't exist
    os.makedirs(f"Optimizer/data_optimization/{model_type}/{folder}/{index}", exist_ok=True)

    study_name = f"{model_type}_study"
    storage_name = f"sqlite:///Optimizer/data_optimization/{model_type}/{folder}/{index}/{study_name}.db"
    if os.path.exists(f"Optimizer/data_optimization/{model_type}/{folder}/{index}/{study_name}.db"):
        study = optuna.load_study(study_name=study_name, storage=storage_name, direction='maximize')
    else:
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

       # Save the best parameters
    best_params = study.best_params
    best_params_path = f"Optimizer/data_optimization/{model_type}/{folder}/{index}/best_params.json"
    with open(best_params_path, "w") as f:
            json.dump(best_params, f)

def get_binary_accuracy_clf(learner,X_test,y_test,buckets):
    preds, _, y_preds = learner.get_X_preds(X_test)

    test_binary = [-1 if y < buckets else 1 for y in y_test]
    pred_binary = [-1 if y < buckets else 1 for y in y_preds]

    accuracy = sum(y1 * y2 > 0 for y1, y2 in zip(pred_binary, test_binary))/len(pred_binary)
    return accuracy

def get_binary_accuracy_reg(learner,X_test, y_test):
    preds, _, y_preds = learner.get_X_preds(X_test)

    test_y_converted = [1 if  y_test[i] >  y_test[i-1] else -1 for i in range(len(y_test))]
    preds_y_converted = [1 if y_preds[i][0] > y_preds[i-1][0] else -1 for i in range(len(y_preds))]

    accuracy = sum(y1 * y2 > 0 for y1, y2 in zip(preds_y_converted, test_y_converted))/len(preds_y_converted)
    return accuracy