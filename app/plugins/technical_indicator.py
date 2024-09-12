import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

class Plugin:
    """
    A predictor plugin using a simple LSTM network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 100,
        'batch_size': 256,
        'intermediate_layers': 1,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.00001,
        'dropout_rate': 0.1
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        self.params['input_dim'] = input_shape

        # Layer configuration
        layers = []
        current_size = self.params['initial_layer_size']
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
            int_layers += 1
        layers.append(1)  # Output layer size

        # Debugging message
        print(f"LSTM Layer sizes: {layers}")

        # Model
        model_input = Input(shape=(input_shape, 1), name="model_input")
        print(f"LSTM input_shape: {input_shape}")

        x = model_input
        for size in layers[:-1]:
            if size > 1:
                x = LSTM(size, activation='relu', kernel_initializer=HeNormal(), return_sequences=True)(x)
        x = LSTM(layers[-2], activation='relu', kernel_initializer=HeNormal())(x)
        model_output = Dense(layers[-1], activation='tanh', kernel_initializer=GlorotUniform(), name="model_output")(x)
        
        self.model = Model(inputs=model_input, outputs=model_output, name="predictor_model")
                # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate= self.params['learning_rate'],   # Set the learning rate
            beta_1=0.9,            # Default value
            beta_2=0.999,          # Default value
            epsilon=1e-7,          # Default value
            amsgrad=False          # Default value
        )

        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Predictor Model Summary:")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error):
        print(f"Training predictor model with data shape: {x_train.shape}")
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Training completed.")
        mse = history.history['loss'][-1]
        if mse > threshold_error:
            print(f"Warning: Model training completed with MSE {mse} exceeding the threshold error {threshold_error}.")

    def predict(self, data):
        print(f"Predicting data with shape: {data.shape}")
        predictions = self.model.predict(data)
        print(f"Predicted data shape: {predictions.shape}")
        return predictions

    def calculate_mse(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) between the true values and predicted values.

        Parameters:
        y_true (np.array): The true values.
        y_pred (np.array): The predicted values.

        Returns:
        float: The calculated MSE.
        """
        # Debugging the shapes of input arrays
        print(f"Calculating MSE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Flatten the predicted values to ensure it is a 1D array
        y_pred = y_pred.flatten()

        # Debugging the shapes after flattening
        print(f"Shapes after flattening: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Convert to numpy arrays to ensure they are in the correct format for calculations
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred)

        # Calculate the absolute difference between true and predicted values
        abs_difference = np.abs(y_true - y_pred)

        # Debugging the intermediate results
        print(f"Absolute differences: {abs_difference}")

        # Square the absolute differences
        squared_abs_difference = abs_difference ** 2

        # Debugging the squared differences
        print(f"Squared absolute differences: {squared_abs_difference}")

        # Calculate the mean of the squared differences
        mse = np.mean(squared_abs_difference)

        # Debugging the final MSE
        print(f"Calculated MSE: {mse}")

        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Calculate the Mean Absolute Error (MAE) between the true values and predicted values.

        Parameters:
        y_true (np.array): The true values.
        y_pred (np.array): The predicted values.

        Returns:
        float: The calculated MAE.
        """
        # Debugging the shapes of input arrays
        print(f"Calculating MAE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Flatten the predicted values to ensure it is a 1D array
        y_pred = y_pred.flatten()

        # Debugging the shapes after flattening
        print(f"Shapes after flattening: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Convert to numpy arrays to ensure they are in the correct format for calculations
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred)

        # Calculate the absolute difference between true and predicted values
        abs_difference = np.abs(y_true - y_pred)

        # Debugging the intermediate results
        print(f"Absolute differences: {abs_difference}")

        # Calculate the mean of the absolute differences
        mae = np.mean(abs_difference)

        # Debugging the final MAE
        print(f"Calculated MAE: {mae}")

        return mae


    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Predictor model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.build_model(input_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
