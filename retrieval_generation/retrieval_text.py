import langchain_community 
import os
import getpass

# configuring google api
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("AIzaSyB-PQLnrQm2Z5UGXW28R24TXG99MLPICFw")