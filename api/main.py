import os
from dotenv import load_dotenv
load_dotenv()


is_terminal=os.getenv("TERMINAL")

if is_terminal:
    pass

else:
    from api.main import app
    