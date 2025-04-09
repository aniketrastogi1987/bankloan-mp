import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from bankloan_model.config.core import config
from bankloan_model.processing.features import Mapper
from bankloan_model.processing.features import IntentOneHotEncoder

bankloan_pipe=Pipeline([
    
     ##==========Mapper======##
     ("map_gender", Mapper(config.model_config_.gender, config.model_config_.gender_mappings)
      ),
     ("map_education", Mapper(config.model_config_.education, config.model_config_.education_mappings )
     ),
     ("map_hometype", Mapper(config.model_config_.home-type, config.model_config_.home_type_mappings)
     ),
     ("map_previous_loan_defaults", Mapper(config.model_config_.previous_loan_defaults, config.model_config_.previous_loan_defaults_mappings)
     ),
     ('encode_intent', IntentOneHotEncoder(variable = config.model_config_.intent)),
     ('model_xg', XGBClassifier(**params))
          
     ])
