import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
try:
    import shap  # optional; may not be installed in some environments
except Exception:
    shap = None
import cv2
from torchvision.models import ResNet18_Weights
import os
import io
from datetime import datetime
from db import init_db, upsert_patient, add_prediction, list_patients, list_predictions, list_predictions_for_patient, export_predictions_dataframe, delete_all_data

# App/model identity
MODEL_NAME = "RhGuard EF Risk Predictor"

# Set page configuration
st.set_page_config(
    page_title=f"{MODEL_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_db()

# Basic theming (lightweight CSS)
st.markdown(
    """
    <style>
    .ef-header { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
    .ef-sub { color: #6b7280; margin-bottom: 20px; }
    .ef-card { padding: 16px; border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff; }
    .ef-sep { margin: 12px 0; border-top: 1px solid #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to load the CNN model
@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path):
    try:
        # Initialize model architecture
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.warning(f"Model file {model_path} not found. Using a dummy model for demonstration.")
        # Create a dummy model for demonstration
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        model.eval()
        return model

# Function to load the LightGBM model
@st.cache_resource(show_spinner=False)
def load_risk_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning(f"Model file {model_path} not found. Using a dummy model for demonstration.")
        # Create a simple dummy model that returns random values between 0 and 1
        class DummyModel:
            def predict(self, X):
                return [0.5]  # Return a moderate risk score for demonstration
        return DummyModel()

# Function to load the encoder
@st.cache_resource(show_spinner=False)
def load_encoder(encoder_path):
    try:
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        return encoder
    except FileNotFoundError:
        st.warning(f"Encoder file {encoder_path} not found. Using a dummy encoder for demonstration.")
        # Create a simple dummy encoder
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        dummy_data = pd.DataFrame({
            'Maternal_Rh_Factor': ['Positive', 'Negative'],
            'Fetal_Rh_Factor': ['Positive', 'Negative'],
            'Coombs_Test_Result': ['Positive', 'Negative']
        })
        encoder.fit(dummy_data)
        return encoder

# Function to preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to generate Grad-CAM heatmap
def generate_gradcam(model, image, target_layer_name='layer4'):
    # Get the target layer
    target_layer = getattr(model, target_layer_name)
    
    # Register hooks for the target layer
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image)
    
    # Backward pass
    model.zero_grad()
    output.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get activations and gradients
    activation = activations[0].detach()
    gradient = gradients[0].detach()
    
    # Global average pooling of gradients
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    
    # Weighted sum of activation maps
    cam = torch.sum(weights * activation, dim=1, keepdim=True)
    
    # ReLU and normalization
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Resize to input image size
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    
    return cam

# Function to overlay heatmap on image
def overlay_heatmap(image, heatmap, alpha=0.5):
    # Convert PIL image to numpy array
    img_array = np.array(image.resize((224, 224)))
    
    # Normalize heatmap to 0-255 and apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlayed = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
    
    return overlayed


def save_uploaded_image(patient_id, pil_image) -> str:
    os.makedirs("data/images", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("data", "images", f"{patient_id}_{timestamp}.png")
    pil_image.save(file_path)
    return file_path


def save_heatmap_image(patient_id, base_image, heatmap) -> str:
    os.makedirs("data/heatmaps", exist_ok=True)
    overlayed = overlay_heatmap(base_image, heatmap)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("data", "heatmaps", f"{patient_id}_{timestamp}_heatmap.png")
    Image.fromarray(overlayed).save(file_path)
    return file_path

# Main function
def main():
    st.markdown(f"<div class='ef-header'>{MODEL_NAME}</div>", unsafe_allow_html=True)
    st.markdown("<div class='ef-sub'>Hybrid computer vision + clinical risk assessment</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Predict", "Patients", "Dashboard", "Settings"])

    # Predict Tab
    with tabs[0]:
        # Initialize wizard state and defaults
        if 'predict_step' not in st.session_state:
            st.session_state.predict_step = 1
        if 'form_defaults' not in st.session_state:
            st.session_state.form_defaults = {
                'patient_id': '', 'name': '', 'age': 30, 'sex': 'Female', 'notes': '',
                'maternal_rh': 'Positive', 'fetal_rh': 'Positive', 'coombs_test': 'Negative'
            }

        def reset_predict_form():
            st.session_state.predict_step = 1
            st.session_state.form_defaults = {
                'patient_id': '', 'name': '', 'age': 30, 'sex': 'Female', 'notes': '',
                'maternal_rh': 'Positive', 'fetal_rh': 'Positive', 'coombs_test': 'Negative'
            }
            if 'uploaded_image' in st.session_state:
                del st.session_state['uploaded_image']
            if 'last_prediction' in st.session_state:
                del st.session_state['last_prediction']
            if 'predict_button' in st.session_state:
                del st.session_state['predict_button']

        st.subheader("New Prediction")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Patient Information")
            # STEP 1: Patient details (then continue to Step 2)
            if st.session_state.predict_step == 1:
                with st.form("patient_details_form"):
                    patient_id = st.text_input("Patient ID", value=st.session_state.form_defaults['patient_id'], placeholder="e.g., PT-0001", key="patient_id_key")
                    name = st.text_input("Name", value=st.session_state.form_defaults['name'], key="name_key")
                    age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.form_defaults['age'], key="age_key")
                    sex = st.selectbox("Sex", ["Female", "Male", "Other"], index=["Female","Male","Other"].index(st.session_state.form_defaults['sex']), key="sex_key") 
                    notes = st.text_area("Notes", value=st.session_state.form_defaults['notes'], height=80, key="notes_key")
                    continue_btn = st.form_submit_button("Continue to Prediction →")
                if continue_btn:
                    # persist details to defaults and move to step 2
                    st.session_state.form_defaults.update({
                        'patient_id': st.session_state.patient_id_key,
                        'name': st.session_state.name_key,
                        'age': st.session_state.age_key,
                        'sex': st.session_state.sex_key,
                        'notes': st.session_state.notes_key,
                    })
                    st.session_state.predict_step = 2
                    st.rerun()
            else:
                # STEP 2: Clinical data + image + predict
                st.info("""
                Instructions:
                1. Upload a blood smear image
                2. Select clinical values
                3. Click Predict Risk to save and view results
                """)
                st.button("New Patient", key="new_patient_top", on_click=reset_predict_form)
                st.markdown("#### Clinical Data")
                maternal_rh = st.selectbox("Maternal Rh Factor", ["Positive", "Negative"], index=["Positive","Negative"].index(st.session_state.form_defaults['maternal_rh']), key="maternal_key") 
                fetal_rh = st.selectbox("Fetal Rh Factor", ["Positive", "Negative"], index=["Positive","Negative"].index(st.session_state.form_defaults['fetal_rh']), key="fetal_key") 
                coombs_test = st.selectbox("Coombs Test Result", ["Positive", "Negative"], index=["Positive","Negative"].index(st.session_state.form_defaults['coombs_test']), key="coombs_key") 
                st.markdown("#### Blood Smear Image")
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploaded_image") 
                predict_button = st.button("Predict Risk", type="primary")

        with col_right:
            if st.session_state.get('uploaded_image') is not None:
                st.image(st.session_state.get('uploaded_image'), caption="Uploaded Blood Smear Image")

        if st.session_state.predict_step == 2 and predict_button:
            if st.session_state.get('uploaded_image') is None:
                st.warning("Please upload a blood smear image.")
            elif not st.session_state.form_defaults['patient_id']:
                st.warning("Please enter a Patient ID.")
            else:
                try:
                    with st.spinner("Loading models..."):
                        cnn_model = load_cnn_model("erythroblast_detector.pth")
                        risk_model = load_risk_model("ef_risk_model.pkl")
                        encoder = load_encoder("encoder.pkl")

                    image = Image.open(st.session_state.get('uploaded_image')).convert('RGB')
                    processed_image = preprocess_image(image)

                    with torch.no_grad():
                        output = cnn_model(processed_image)
                        erythroblast_prob = torch.sigmoid(output).item()

                    clinical_df = pd.DataFrame({
                        'Maternal_Rh_Factor': [maternal_rh],
                        'Fetal_Rh_Factor': [fetal_rh],
                        'Coombs_Test_Result': [coombs_test]
                    })
                    encoded_cats = encoder.transform(clinical_df)
                    feature_names = encoder.get_feature_names_out(['Maternal_Rh_Factor', 'Fetal_Rh_Factor', 'Coombs_Test_Result'])
                    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names)
                    encoded_df['erythroblast_prob'] = erythroblast_prob

                    risk_score = float(risk_model.predict(encoded_df)[0])
                    risk_percentage = risk_score * 100.0

                    # Save patient and prediction
                    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    upsert_patient(
                        st.session_state.form_defaults['patient_id'],
                        st.session_state.form_defaults['name'],
                        int(st.session_state.form_defaults['age']) if st.session_state.form_defaults['age'] else None,
                        st.session_state.form_defaults['sex'],
                        st.session_state.form_defaults['notes'],
                        timestamp,
                    )
                    heatmap = generate_gradcam(cnn_model, processed_image)
                    saved_img_path = save_uploaded_image(st.session_state.form_defaults['patient_id'], image)
                    saved_heatmap_path = save_heatmap_image(st.session_state.form_defaults['patient_id'], image, heatmap)
                    add_prediction(
                        st.session_state.form_defaults['patient_id'],
                        maternal_rh,
                        fetal_rh,
                        coombs_test,
                        float(erythroblast_prob),
                        float(risk_score),
                        saved_img_path,
                        saved_heatmap_path,
                        timestamp,
                    )

                    # Display results
                    st.success("Prediction complete and saved.")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Predicted Risk", f"{risk_percentage:.1f}%")
                        st.metric("Erythroblast Probability", f"{erythroblast_prob:.2%}")
                        st.image(saved_heatmap_path, caption="Grad-CAM Heatmap")
                    with res_col2:
                        st.markdown("#### SHAP Feature Importance")
                        try:
                            if shap is None:
                                raise RuntimeError("SHAP not installed")
                            explainer = shap.TreeExplainer(risk_model)
                            shap_values = explainer.shap_values(encoded_df)
                            vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                            feature_imp_df = pd.DataFrame({
                                'feature': list(encoded_df.columns),
                                'shap_value': list(vals)
                            }).sort_values('shap_value', key=np.abs, ascending=False)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.barh(feature_imp_df['feature'][:10][::-1], feature_imp_df['shap_value'][:10][::-1], color="#1f77b4")
                            ax.set_xlabel("SHAP value (impact on model output)")
                            ax.set_ylabel("")
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception:
                            if hasattr(risk_model, 'feature_importances_'):
                                imp = risk_model.feature_importances_
                                imp_df = pd.DataFrame({'feature': encoded_df.columns, 'importance': imp}).sort_values('importance', ascending=False)
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.barh(imp_df['feature'][:10][::-1], imp_df['importance'][:10][::-1], color="#ff7f0e")
                                ax.set_xlabel("Feature importance")
                                ax.set_ylabel("")
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("Explainability not available for this model.")

                    # Downloads
                    st.markdown("#### Download Results")
                    csv_bytes = io.StringIO()
                    pd.DataFrame({
                        'patient_id': [st.session_state.form_defaults['patient_id']],
                        'name': [st.session_state.form_defaults['name']],
                        'age': [st.session_state.form_defaults['age']],
                        'sex': [st.session_state.form_defaults['sex']],
                        'maternal_rh': [maternal_rh],
                        'fetal_rh': [fetal_rh],
                        'coombs_test': [coombs_test],
                        'erythroblast_prob': [erythroblast_prob],
                        'risk_score': [risk_score],
                        'created_at': [timestamp]
                    }).to_csv(csv_bytes, index=False)
                    st.download_button("Download CSV", key="dl_csv_single", data=csv_bytes.getvalue(), file_name=f"{st.session_state.form_defaults['patient_id']}_prediction.csv", mime="text/csv")
                    with open(saved_heatmap_path, 'rb') as f:
                        st.download_button("Download Heatmap PNG", key="dl_heatmap_single", data=f.read(), file_name=os.path.basename(saved_heatmap_path), mime="image/png")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    pass

                # Post-prediction actions
                st.markdown("---")
                st.button("New Patient", key="new_patient_bottom", on_click=reset_predict_form)
    
        if st.session_state.predict_step == 1:
            st.info("""
            Instructions:
            1. Enter patient details
            2. Click Continue to proceed to prediction
            """)

    # Patients Tab
    with tabs[1]:
        st.subheader("Patient Profiles")
        search = st.text_input("Search by Patient ID or Name", "")
        patients = list_patients(search)
        if len(patients) == 0:
            st.info("No patients found.")
        else:
            df = pd.DataFrame(patients, columns=["Patient ID", "Name", "Age", "Sex", "Created At"]) 
            st.dataframe(df, hide_index=True)
            selected_patient = st.selectbox("Select Patient", [p[0] for p in patients], key="patients_select")
            if selected_patient:
                history = list_predictions_for_patient(selected_patient)
                if len(history) == 0:
                    st.info("No predictions for this patient yet.")
                else:
                    hist_df = pd.DataFrame(history, columns=[
                        "ID", "Risk Score", "Erythroblast Prob", "Maternal Rh", "Fetal Rh", "Coombs", "Image Path", "Heatmap Path", "Created At"
                    ])
                    hist_df_display = hist_df[["Created At", "Risk Score", "Erythroblast Prob", "Maternal Rh", "Fetal Rh", "Coombs"]]
                    st.dataframe(hist_df_display, hide_index=True)
                    latest = hist_df.iloc[0]
                    st.markdown("#### Latest Prediction")
                    cols = st.columns(2)
                    with cols[0]:
                        st.image(latest["Image Path"], caption="Image")
                    with cols[1]:
                        st.image(latest["Heatmap Path"], caption="Heatmap")

    # Dashboard Tab
    with tabs[2]:
        st.subheader("Analytics Dashboard")
        preds = list_predictions(limit=500)
        if len(preds) == 0:
            st.info("No predictions yet.")
        else:
            preds_df = pd.DataFrame(preds, columns=[
                "Patient ID", "Risk", "EryProb", "Maternal Rh", "Fetal Rh", "Coombs", "Created At"
            ])
            preds_df["Risk %"] = preds_df["Risk"] * 100
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Total Predictions", len(preds_df))
            with k2:
                st.metric("Avg Risk %", f"{preds_df['Risk %'].mean():.1f}%")
            with k3:
                st.metric("Avg Erythroblast Prob", f"{preds_df['EryProb'].mean()*100:.1f}%")

            st.markdown("#### Risk Distribution")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(preds_df['Risk %'], bins=20, color="#1f77b4", edgecolor="white")
            ax.set_xlabel("Risk (%)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            st.markdown("#### Recent Activity")
            st.dataframe(preds_df.head(20), hide_index=True)

            st.markdown("#### Export All Predictions")
            exp_df = export_predictions_dataframe()
            csv_buf = io.StringIO()
            exp_df.to_csv(csv_buf, index=False)
            st.download_button("Download Predictions CSV", key="dl_all_predictions", data=csv_buf.getvalue(), file_name="all_predictions.csv", mime="text/csv")

    # Settings Tab
    with tabs[3]:
        st.subheader("Settings & About")
        st.write("Version: 1.0.0")
        st.write("Models: ResNet18 (CNN), RandomForest (Risk)")
        st.write("Data stored in ef_app.db and data/ folder")

        st.markdown("#### Maintenance")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear All Data", type="secondary"):
                delete_all_data()
                st.success("All data cleared from database.")
        with col_b:
            st.write("Make sure to back up data/ if needed.")

if __name__ == "__main__":
    main()