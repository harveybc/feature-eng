import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log

def process_data(config):
    print(f"Loading data from CSV file: {config['x_train_file']}")
    x_train_data = load_csv(config['x_train_file'], headers=config['headers'])
    print(f"Data loaded with shape: {x_train_data.shape}")

    y_train_file = config['y_train_file']
    target_column = config['target_column']

    if isinstance(y_train_file, str):
        print(f"Loading y_train data from CSV file: {y_train_file}")
        y_train_data = load_csv(y_train_file, headers=config['headers'])
        print(f"y_train data loaded with shape: {y_train_data.shape}")
    elif target_column is not None:
        y_train_data = x_train_data.iloc[:, target_column]
        print(f"Using target column at index: {target_column}")
    else:
        raise ValueError("Either y_train_file or target_column must be specified in the configuration.")

    # Ensure input data is numeric
    x_train_data = x_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Apply input offset and time horizon
    offset = config['input_offset'] + config['time_horizon']
    y_train_data = y_train_data[offset:]
    x_train_data = x_train_data[:-config['time_horizon']]

    # Ensure the shapes match
    min_length = min(len(x_train_data), len(y_train_data))
    x_train_data = x_train_data[:min_length]
    y_train_data = y_train_data[:min_length]

    # Debugging messages to confirm types and shapes
    print(f"Returning data of type: {type(x_train_data)}, {type(y_train_data)}")
    print(f"x_train_data shape after adjustments: {x_train_data.shape}")
    print(f"y_train_data shape after adjustments: {y_train_data.shape}")
    
    return x_train_data, y_train_data

def run_prediction_pipeline(config, plugin):
    start_time = time.time()
    
    print("Running process_data...")
    x_train, y_train = process_data(config)
    print(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")
    
    time_horizon = config['time_horizon']
    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold_error = config['threshold_error']

    # Ensure x_train and y_train are DataFrame or Series
    if isinstance(x_train, (pd.DataFrame, pd.Series)) and isinstance(y_train, (pd.DataFrame, pd.Series)):
        x_train = x_train.to_numpy().astype(np.float32)
        y_train = y_train.to_numpy().astype(np.float32)

        # Ensure x_train is a 2D array
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        
        # Ensure y_train matches the first dimension of x_train
        y_train = y_train[:len(x_train)]

        # Debug messages for shapes
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        # Train the model
        plugin.build_model(input_shape=x_train.shape[1])
        plugin.train(x_train, y_train, epochs=epochs, batch_size=batch_size, threshold_error=threshold_error)

        # Save the trained model
        if config['save_model']:
            plugin.save(config['save_model'])
            print(f"Model saved to {config['save_model']}")

        # Predict using the trained model
        predictions = plugin.predict(x_train)

        # Reshape predictions to match y_train shape
        predictions = predictions.reshape(y_train.shape)

        # Evaluate the model
        mse = float(plugin.calculate_mse(y_train, predictions))
        mae = float(plugin.calculate_mae(y_train, predictions))
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        # Convert predictions to a DataFrame and save to CSV
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        output_filename = config['output_file']
        write_csv(output_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        # Save final configuration and debug information
        end_time = time.time()
        execution_time = end_time - start_time
        debug_info = {
            'execution_time': float(execution_time),
            'mse': mse,
            'mae': mae
        }

        # Save debug info
        if config.get('save_log'):
            save_debug_info(debug_info, config['save_log'])
            print(f"Debug info saved to {config['save_log']}.")

        # Remote log debug info and config
        if config.get('remote_log'):
            remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
            print(f"Debug info saved to {config['remote_log']}.")

        print(f"Execution time: {execution_time} seconds")

        # Validate the model if validation data is provided
        if config['x_validation_file'] and config['y_validation_file']:
            print("Validating model...")
            x_validation = load_csv(config['x_validation_file'], headers=config['headers']).to_numpy().astype(np.float32)
            y_validation = load_csv(config['y_validation_file'], headers=config['headers']).to_numpy().astype(np.float32)
            
            # Ensure x_validation is a 2D array
            if x_validation.ndim == 1:
                x_validation = x_validation.reshape(-1, 1)
            
            # Ensure y_validation matches the first dimension of x_validation
            y_validation = y_validation[:len(x_validation)]
            
            print(f"x_validation shape: {x_validation.shape}")
            print(f"y_validation shape: {y_validation.shape}")
            
            validation_predictions = plugin.predict(x_validation)
            validation_predictions = validation_predictions.reshape(y_validation.shape)
            
            validation_mse = float(plugin.calculate_mse(y_validation, validation_predictions))
            validation_mae = float(plugin.calculate_mae(y_validation, validation_predictions))
            print(f"Validation Mean Squared Error: {validation_mse}")
            print(f"Validation Mean Absolute Error: {validation_mae}")

    else:
        print(f"Invalid data type returned: {type(x_train)}, {type(y_train)}")
        raise ValueError("Processed data is not in the correct format (DataFrame or Series).")

def load_and_evaluate_model(config, plugin):
    # Load the model
    plugin.load(config['load_model'])

    # Load the input data
    x_train, _ = process_data(config)

    # Predict using the loaded model
    predictions = plugin.predict(x_train.to_numpy())

    # Save the predictions to CSV
    evaluate_filename = config['evaluate_file']
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    write_csv(evaluate_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Predicted data saved to {evaluate_filename}")
