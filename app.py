import streamlit as st
from PIL import Image
import torch
import torchvision  # Missing import added here
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Visualize predictions on the image
def visualize_predictions(image, boxes, labels, scores, threshold=0.5):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    class_colors = {1: (255, 165, 0), 2: (255, 255, 0)}  # Marine Animal - Orange, Trash - Yellow
    class_names = {1: 'Marine Animal', 2: 'Trash'}

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            box = box.astype(np.int32)
            color = class_colors.get(label, (255, 255, 255))
            label_name = class_names.get(label, 'Unknown')

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(image, f"{label_name}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Streamlit UI
def main():
    st.title("Marine Animal and Trash Detection")
    st.write("Upload an image to detect marine animals and trash.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Transform the image for model
        transform = T.Compose([T.ToTensor()])
        input_image = transform(image).unsqueeze(0)

        # Perform inference
        with st.spinner("Detecting objects..."):
            outputs = model(input_image)

        # Extract predictions
        pred_boxes = outputs[0]["boxes"].detach().numpy()
        pred_labels = outputs[0]["labels"].detach().numpy()
        pred_scores = outputs[0]["scores"].detach().numpy()

        # Visualize predictions
        visualized_image = visualize_predictions(image, pred_boxes, pred_labels, pred_scores)

        # Display results
        st.image(visualized_image, caption="Predictions", use_column_width=True)

        # Show raw predictions if needed
        st.write("Detected Objects:")
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > 0.5:
                label_name = "Marine Animal" if label == 1 else "Trash"
                st.write(f"Label: {label_name}, Score: {score:.2f}, Box: {box}")

if __name__ == "__main__":
    model_path = "model/faster_rcnn_final.pth"  # Path to your saved model
    num_classes = 3  # 2 classes (marine animal, trash) + background
    model = load_model(model_path, num_classes)
    main()
