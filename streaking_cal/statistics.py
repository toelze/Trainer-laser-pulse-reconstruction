def weighted_avg_and_std(values, weights):
    from math import sqrt
    from numpy import average
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    averagev = average(values, weights=weights)
    # Fast and numerically precise:
    variance = average((values-averagev)**2, weights=weights)
    return (averagev, sqrt(variance))
