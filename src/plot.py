import matplotlib.pyplot as plt

def plot_training(dates, X, sampled_X, pred, epoch):
    """Plots the input value, the sampled input and the prediction for the next timestep."""
    fig = plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    ## REMOVE DROPOUT FOR THIS PLOT TO APPEAR AS EXPECTED !! DROPOUT INTERFERES WITH HOW THE SAMPLED SOURCES ARE PLOTTED
    plt.plot(dates, sampled_X, 'o-.', color='red', label = 'sampled source', linewidth=1, markersize=10)
    plt.plot(dates, X, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(dates[1:], pred, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)
    plt.title("Training with sampled teacher forcing, Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Y")
    plt.legend()
    return fig

def plot_prediction(X, Y, pred, X_dates, Y_dates):
    dates = X_dates + Y_dates
    plt.rcParams.update({"font.size" : 16})

    # plotting
    fig = plt.figure(figsize=(15,6))
    plt.plot(X_dates, X, '-', color = 'blue', label = 'Input', linewidth=2)
    plt.plot(Y_dates, Y, '-', color = 'indigo', label = 'Target', linewidth=2)
    plt.plot(dates[1: -1], pred,'--', color = 'limegreen', label = 'Forecast', linewidth=2)

    #formatting
    plt.grid(b=True, which='major', linestyle = 'solid')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle = 'dashed', alpha=0.5)
    plt.xlabel("Time Elapsed")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Forecast")
    return fig