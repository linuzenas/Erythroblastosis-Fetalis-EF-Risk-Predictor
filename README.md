# Erythroblastosis Fetalis (EF) Risk Predictor

## 📋 Project Overview

The **Erythroblastosis Fetalis Risk Predictor** is a machine learning-powered web application that combines computer vision and clinical data analysis to predict the risk of Erythroblastosis Fetalis (EF) in fetuses. This condition occurs when there's an incompatibility between maternal and fetal blood types, particularly Rh factors.

## 🎯 What is Erythroblastosis Fetalis?

Erythroblastosis Fetalis (also known as Hemolytic Disease of the Newborn) is a serious condition that occurs when:
- A mother with Rh-negative blood carries a fetus with Rh-positive blood
- The mother's immune system produces antibodies against the fetus's red blood cells
- This leads to destruction of fetal red blood cells, causing anemia and other complications

## 🏗️ Project Architecture

The application uses a **hybrid approach** combining:
1. **Computer Vision (CNN)**: Analyzes blood smear images to detect erythroblasts
2. **Machine Learning (Random Forest)**: Processes clinical data to predict risk
3. **Web Interface (Streamlit)**: Provides an intuitive user interface

## 📁 Project Structure

```
EF/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── test_models.py                 # Model testing script
├── erythroblast_detector.pth      # CNN model for image analysis (43MB)
├── ef_risk_model.pkl             # Risk prediction model (8.6MB)
├── encoder.pkl                    # OneHotEncoder for categorical data (728B)
└── venv/                         # Virtual environment
    ├── Scripts/                  # Python executables
    ├── Lib/                      # Installed packages
    └── ...
```

## 🔧 Technical Components

### 1. **app.py** - Main Application (283 lines)
The core Streamlit application that provides:
- **Image Upload Interface**: For blood smear images
- **Clinical Data Form**: For maternal/fetal Rh factors and Coombs test results
- **Model Integration**: Loads and uses all three ML models
- **Visualization**: Grad-CAM heatmaps and SHAP explanations
- **Risk Assessment**: Provides interpretable risk scores

### 2. **Model Files**

#### **erythroblast_detector.pth** (43MB)
- **Type**: PyTorch CNN model
- **Architecture**: ResNet18 with custom final layer
- **Purpose**: Detects erythroblasts in blood smear images
- **Input**: 224x224 RGB images
- **Output**: Probability of erythroblast presence (0-1)

#### **ef_risk_model.pkl** (8.6MB)
- **Type**: Random Forest Regressor
- **Purpose**: Predicts overall EF risk based on clinical features
- **Features**: One-hot encoded clinical data + erythroblast probability
- **Output**: Risk score (0-1, where 1 = highest risk)

#### **encoder.pkl** (728B)
- **Type**: Scikit-learn OneHotEncoder
- **Purpose**: Converts categorical clinical data to numerical format
- **Categories**:
  - Maternal Rh Factor: Positive/Negative
  - Fetal Rh Factor: Positive/Negative
  - Coombs Test Result: Positive/Negative

### 3. **requirements.txt** - Dependencies
Lists all required Python packages with minimum versions:
- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **shap**: Model explainability
- **opencv-python**: Image processing
- **matplotlib**: Plotting
- **pillow**: Image handling

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download the project**
   ```bash
   # Navigate to your project directory
   cd "D:\Project WorkSpace\CSP\EF"
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and go to: `http://localhost:8501`
   - The application will load with all models ready to use

## 🎮 How to Use the Application

### Step 1: Upload Blood Smear Image
- Click "Choose an image..." in the sidebar
- Select a blood smear image (JPG, JPEG, or PNG format)
- The image will be automatically processed by the CNN model

### Step 2: Enter Clinical Data
Fill out the clinical data form in the sidebar:
- **Maternal Rh Factor**: Select Positive or Negative
- **Fetal Rh Factor**: Select Positive or Negative  
- **Coombs Test Result**: Select Positive or Negative

### Step 3: Get Risk Prediction
- Click "Predict Risk" button
- The application will:
  - Analyze the blood smear image
  - Process clinical data
  - Generate risk prediction
  - Display visualizations and explanations

### Step 4: Interpret Results
The application provides:
- **Risk Percentage**: Overall EF risk score
- **Erythroblast Probability**: Likelihood of erythroblasts in the image
- **Grad-CAM Visualization**: Shows which parts of the image influenced the decision
- **SHAP Values**: Explains which clinical factors contributed to the risk
- **Interpretation**: Text-based risk assessment and recommendations

## 🔍 Understanding the Results

### Risk Categories
- **0-25%**: Very low risk - Routine monitoring
- **25-50%**: Low risk - Routine monitoring recommended
- **50-75%**: Moderate risk - Close monitoring recommended
- **75-100%**: High risk - Immediate medical attention recommended

### Visualizations

#### **Grad-CAM Heatmap**
- Shows which regions of the blood smear image the model focused on
- Red/yellow areas indicate high attention (likely erythroblasts)
- Blue areas indicate low attention

#### **SHAP Force Plot**
- Shows how each clinical factor contributes to the final risk score
- Red bars push risk higher, blue bars push risk lower
- Length of bars indicates magnitude of contribution

## 🧪 Testing the Models

Run the test script to verify all models are working correctly:

```bash
python test_models.py
```

This will test:
- CNN model loading
- Risk model loading  
- Encoder loading
- End-to-end prediction pipeline

## ⚠️ Important Notes

### Model Limitations
- **Demo Models**: The current models are trained on synthetic data for demonstration purposes
- **Medical Disclaimer**: This application is for educational/research purposes only
- **Not for Clinical Use**: Do not use for actual medical diagnosis or treatment decisions

### Version Compatibility
- The models were created with scikit-learn 1.5.2
- Current environment uses scikit-learn 1.7.1
- This may cause version warnings but won't affect functionality

### Performance Considerations
- **Image Size**: Images are automatically resized to 224x224 pixels
- **Processing Time**: First prediction may take longer due to model loading
- **Memory Usage**: CNN model requires ~43MB of RAM

## 🛠️ Troubleshooting

### Common Issues

1. **"Model file not found" warnings**
   - Ensure all .pth and .pkl files are in the project directory
   - Run `python test_models.py` to verify model files

2. **Version compatibility warnings**
   - These are warnings, not errors
   - The application will still work correctly

3. **Streamlit deprecation warnings**
   - Update `use_column_width` to `use_container_width` in future versions
   - Current version works fine with existing code

4. **Memory issues**
   - Ensure sufficient RAM (at least 4GB recommended)
   - Close other applications if needed

### Getting Help
- Check the terminal output for error messages
- Verify all dependencies are installed correctly
- Ensure Python version compatibility

## 🔬 Technical Details

### Machine Learning Pipeline

1. **Image Processing**:
   - Resize to 224x224 pixels
   - Normalize using ImageNet statistics
   - Convert to PyTorch tensor

2. **Feature Engineering**:
   - One-hot encode categorical variables
   - Combine with erythroblast probability
   - Create feature matrix for risk prediction

3. **Model Inference**:
   - CNN processes image → erythroblast probability
   - Risk model processes features → risk score
   - Encoder transforms clinical data → numerical features

### Model Architecture

#### CNN (ResNet18)
```
Input: 3x224x224 RGB image
↓
ResNet18 backbone (pretrained weights removed)
↓
Global Average Pooling
↓
Linear layer (512 → 1)
↓
Sigmoid activation
↓
Output: Erythroblast probability (0-1)
```

#### Risk Model (Random Forest)
```
Input: 7 features (3 one-hot encoded + 1 probability)
↓
100 decision trees
↓
Average prediction
↓
Output: Risk score (0-1)
```

## 📈 Future Enhancements

### Potential Improvements
1. **Real Medical Data**: Train on actual clinical datasets
2. **Additional Features**: Include more clinical parameters
3. **Model Updates**: Regular retraining with new data
4. **Mobile App**: Create mobile-friendly interface
5. **API Integration**: Connect to hospital systems
6. **Multi-language**: Support multiple languages

### Research Opportunities
1. **Advanced Architectures**: Try Vision Transformers or newer CNN models
2. **Ensemble Methods**: Combine multiple models for better accuracy
3. **Uncertainty Quantification**: Provide confidence intervals
4. **Active Learning**: Improve models with user feedback

## 📚 Learning Resources

### For Beginners
- **Machine Learning**: [Scikit-learn documentation](https://scikit-learn.org/)
- **Deep Learning**: [PyTorch tutorials](https://pytorch.org/tutorials/)
- **Web Apps**: [Streamlit documentation](https://docs.streamlit.io/)
- **Medical AI**: [Medical imaging with deep learning](https://www.coursera.org/learn/medical-ai)

### Advanced Topics
- **Explainable AI**: [SHAP documentation](https://shap.readthedocs.io/)
- **Computer Vision**: [OpenCV tutorials](https://opencv.org/tutorials/)
- **Model Deployment**: [MLOps best practices](https://ml-ops.org/)

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical software regulations if used in clinical settings.

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Model improvements
- UI/UX enhancements
- Documentation updates
- Bug fixes
- Feature additions

## 📞 Support

For technical support or questions:
1. Check this documentation first
2. Run the test script to identify issues
3. Check the terminal output for error messages
4. Verify all dependencies are correctly installed

---

**Remember**: This application is for educational purposes only and should not be used for actual medical diagnosis or treatment decisions.
