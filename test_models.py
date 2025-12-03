#!/usr/bin/env python3
"""
Test script to verify that all models can be loaded without errors
"""

import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def test_cnn_model():
    """Test loading the CNN model"""
    try:
        # Initialize model architecture
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        
        # Load model weights
        model.load_state_dict(torch.load('erythroblast_detector.pth', map_location=torch.device('cpu')))
        model.eval()
        print("+ CNN model loaded successfully")
        return True
    except Exception as e:
        print(f"- Error loading CNN model: {e}")
        return False

def test_risk_model():
    """Test loading the risk model"""
    try:
        with open('ef_risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("+ Risk model loaded successfully")
        return True
    except Exception as e:
        print(f"- Error loading risk model: {e}")
        return False

def test_encoder():
    """Test loading the encoder"""
    try:
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        print("+ Encoder loaded successfully")
        return True
    except Exception as e:
        print(f"- Error loading encoder: {e}")
        return False

def test_end_to_end():
    """Test the complete pipeline"""
    try:
        # Load all models
        cnn_model = models.resnet18(weights=None)
        num_features = cnn_model.fc.in_features
        cnn_model.fc = nn.Linear(num_features, 1)
        cnn_model.load_state_dict(torch.load('erythroblast_detector.pth', map_location=torch.device('cpu')))
        cnn_model.eval()
        
        with open('ef_risk_model.pkl', 'rb') as f:
            risk_model = pickle.load(f)
        
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        # Test with dummy data
        clinical_data = pd.DataFrame({
            'Maternal_Rh_Factor': ['Positive'],
            'Fetal_Rh_Factor': ['Negative'],
            'Coombs_Test_Result': ['Positive']
        })
        
        # One-hot encode categorical features
        encoded_cats = encoder.transform(clinical_data)
        feature_names = encoder.get_feature_names_out(['Maternal_Rh_Factor', 'Fetal_Rh_Factor', 'Coombs_Test_Result'])
        encoded_df = pd.DataFrame(encoded_cats, columns=feature_names)
        encoded_df['erythroblast_prob'] = 0.5  # Dummy erythroblast probability
        
        # Make prediction
        risk_score = risk_model.predict(encoded_df)[0]
        print(f"+ End-to-end test successful. Risk score: {risk_score:.3f}")
        return True
    except Exception as e:
        print(f"- Error in end-to-end test: {e}")
        return False

if __name__ == "__main__":
    print("Testing Erythroblastosis Fetalis Risk Predictor models...")
    print("=" * 60)
    
    tests = [
        test_cnn_model,
        test_risk_model,
        test_encoder,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! The application is ready to use.")
    else:
        print("- Some tests failed. Please check the errors above.")
