import numpy

def get_numeric_buffer(class_dict, buff_names, buff_size):
    """Get walker buffer for MPI communication

    Returns
    -------
    buff : dict
        Relevant walker information for population control.
    """
    s = 0
    buff = numpy.zeros(buff_size, dtype=numpy.complex128)
    for d in buff_names:
        data = class_dict[d]
        if isinstance(data, (numpy.ndarray)):
            buff[s:s+data.size] = data.ravel()
            s += data.size
        elif isinstance(data, list):
            for l in data:
                if isinstance(l, (numpy.ndarray)):
                    buff[s:s+l.size] = l.ravel()
                    s += l.size
                elif isinstance(l, (int, float, complex)):
                    buff[s:s+1] = l
                    s += 1
        else:
            buff[s:s+1] = data
            s += 1
    return buff

def set_numeric_buffer(class_dict, buff_names, buff_size, buff):
    """Set walker buffer following MPI communication

    Parameters
    -------
    buff : dict
        Relevant walker information for population control.
    """
    s = 0
    for d in buff_names:
        data = class_dict[d]
        if isinstance(data, numpy.ndarray):
            class_dict[d] = buff[s:s+data.size].reshape(data.shape).copy()
            s += data.size
        elif isinstance(data, list):
            for ix, l in enumerate(data):
                if isinstance(l, (numpy.ndarray)):
                    class_dict[d][ix] = buff[s:s+l.size].reshape(l.shape).copy()
                    s += l.size
                elif isinstance(l, (int, float, complex)):
                    class_dict[d][ix] = buff[s]
                    s += 1
        else:
            if isinstance(class_dict[d], int):
                class_dict[d] = int(buff[s].real)
            elif isinstance(class_dict[d], float):
                class_dict[d] = buff[s].real
            else:
                class_dict[d] = buff[s]
            s += 1
