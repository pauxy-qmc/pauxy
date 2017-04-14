def greens_function(walker, trial):

    gup = np.dot(np.dot(walker.phi[0], walker.ovlp[0]), np.transpose(trial[0]))
    gdown = np.dot(np.dot(walker.phi[1], walker.ovlp[1]), np.transpose(trial[1]))

    return np.array(gup, gdown)
