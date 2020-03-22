from scipy import stats


def uniform(start, end):
    loc = start
    scale = end - start
    return stats.uniform(loc, scale)
