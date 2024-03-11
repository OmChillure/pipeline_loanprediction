import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions

@pytest.fixture
def single_predictions():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result
 
def test_single_pred_not_none(single_predictions): #output is not none
    assert single_predictions is not None

def test_single_pred_str_type(single_predictions):
    assert isinstance(single_predictions.get('prediction')[0],str)  #data type is string

def test_single_pred_vlidate(single_predictions):  #check output is not y
    assert single_predictions.get('prediction')[0] == 'Y'
