import os, sys

def get_file_names(path, extension):

    """
    Function to get a list of names for files with a given extension in a directory

    Parameters:
        path (str): path to the directory
        extension (str): extension of the files to be searched
    
    Returns:
        file_names (list): list of file names
    """

    import os

    file_names = [os.path.basename(f) for f in os.listdir(path) if f.endswith(extension)]

    return file_names