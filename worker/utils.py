import os

def get_dir_size(path):
    """calculate size of directory"""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def get_oldest_file(path):
    """
    returns the file path of the oldest (last modified time) 
    file in the directory, or None if there are no files.
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not files:
        return None
    return min(files, key=os.path.getmtime) 
