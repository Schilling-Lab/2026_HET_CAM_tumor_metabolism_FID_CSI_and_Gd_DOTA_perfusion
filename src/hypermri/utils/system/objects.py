"""
Generic object manipulation utilities.

Contains functions for managing attributes and counters on arbitrary objects.
"""


def _init_attr(data_obj=None, attr_name: str = None, attr_type=dict):
    """
    Initialize an attribute on the data object if it doesn't exist.

    Parameters
    ----------
    data_obj : object
        The object to modify.
    attr_name : str
        The name of the attribute to initialize.
    attr_type : type
        The type of the attribute (default: dict).

    Returns
    -------
    bool
        True if initialization ran (or attribute existed).
    """
    if data_obj is None or attr_name is None:
        return False

    if not hasattr(data_obj, attr_name):
        setattr(data_obj, attr_name, attr_type())
    return True


def add_counter(data_obj=None, counter_name=None, n_counts=1):
    """
    Add count to a counter attribute on the object, keyed by object ID.

    Parameters
    ----------
    data_obj : object
        The object to modify.
    counter_name : str
        The name of the counter attribute.
    n_counts : int
        Number to add to the counter.

    Returns
    -------
    int
        The new count value.
    """
    if data_obj is None:
        raise Exception("data_obj can not be None!")
    if counter_name is None:
        raise Exception("counter_name can not be None!")

    obj_id = str(id(data_obj))
    _init_attr(data_obj=data_obj, attr_name=counter_name)

    attr = getattr(data_obj, counter_name)

    if isinstance(attr, dict):
        if obj_id in attr:
            attr[obj_id] += n_counts
        else:
            attr[obj_id] = n_counts
    else:
        # If it's not a dict, reset it to a dict (legacy handling)
        setattr(data_obj, counter_name, {obj_id: n_counts})

    return getattr(data_obj, counter_name)[obj_id]


def get_counter(data_obj=None, counter_name=None):
    """
    Get the current count from a counter attribute.

    Parameters
    ----------
    data_obj : object
        The object to query.
    counter_name : str
        The name of the counter attribute.

    Returns
    -------
    int
        The current count value.
    """
    if data_obj is None:
        raise Exception("No object was passed!")
    if counter_name is None:
        raise Exception("No Name was passed!")

    _init_attr(data_obj=data_obj, attr_name=counter_name, attr_type=dict)
    obj_id = str(id(data_obj))

    attr = getattr(data_obj, counter_name)

    if isinstance(attr, dict):
        if obj_id in attr:
            return attr[obj_id]
        else:
            attr[obj_id] = 0
            return 0
    else:
        setattr(data_obj, counter_name, {obj_id: 0})
        return 0
