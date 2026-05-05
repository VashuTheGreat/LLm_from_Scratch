import pickle

from exception import MyException

def object_save(obj:object, file_path:str)->None:
    try:
         with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise MyException(f"Error saving object to {file_path}: {str(e)}")
    

def load_object(file_path:str)->object:
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise MyException(f"Error loading object from {file_path}: {str(e)}")