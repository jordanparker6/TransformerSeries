import matplotlib.pyplot as plt

def plot_teacher_forcing(dates, X, sampled_X, pred, epoch):
    """
    Plots the true values and the model input under teacher forcing with sampling
    against the forecast. Droput must be removed for this to work properly.
    """
    fig = plt.figure(figsize=(15,6))

    plt.plot(dates, sampled_X, 'o-.', color='red', label = 'sampled source', linewidth=1, markersize=10)
    plt.plot(dates, X, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(dates[1:], pred, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)
    plt.title("Training with sampled teacher forcing, Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Y")
    plt.legend()
    return fig

def plot_prediction(X, Y, pred, X_dates, Y_dates):
    """
    Plots the input and forecast against the true prediction.
    """
    dates = X_dates + Y_dates

    fig = plt.figure(figsize=(15,6))
    plt.plot(X_dates, X, '-', color = 'blue', label = 'Input', linewidth=2)
    plt.plot(Y_dates, Y, '-', color = 'indigo', label = 'Target', linewidth=2)
    plt.plot(dates[1: -1], pred,'--', color = 'limegreen', label = 'Forecast', linewidth=2)

    plt.title("Forecast over test sample")
    plt.xlabel("Time Elapsed")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Forecast")
    return fig

def plot_candlestick_predictions(X, Y, pred, X_dates, Y_dates):
    """Plots candlestick data for financial market forecasting."""
    return