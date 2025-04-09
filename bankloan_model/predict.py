import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bankloan_model import __version__ as _version
from bankloan_model.config.core import config
from bankloan_model.pipeline import bankloan_pipe
from bankloan_model.processing.data_manager import load_pipeline
from bankloan_model.processing.data_manager import pre_pipeline_preparation
from bankloan_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bankloan_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bankloan_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = bankloan_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'PassengerId':[79],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
                'SibSp':[0],'Parch':[2],'Ticket':['248738'],'Cabin':[np.nan,],'Embarked':['S'],'Fare':[29]}
    
    make_prediction(input_data=data_in)
