<h1>FireWatchAI: Wildfire Detection with Computer Vision</h1>

In this project, we developed a state-of-the-art wildfire detection system based on computer vision principles. The system employs two powerful object detection models - Faster RCNN and YOLO - to analyze real-time aerial and ground-based imagery for wildfire smoke detection.

<h2>Introduction</h2>

FireWatchAI is an innovative wildfire detection system that harnesses the power of computer vision and deep learning to combat the increasing threat of devastating wildfires. The system utilizes state-of-the-art object detection models - Faster RCNN and You Only Look Once (YOLO) - to autonomously detect and classify wildfire smoke from aerial and ground-based imagery.

The escalating frequency and intensity of wildfires pose severe environmental and economic risks, making early detection crucial for effective firefighting and damage mitigation. FireWatchAI bridges the gap between technology and wildfire prevention, providing stakeholders with rapid and precise detection capabilities. By distinguishing between benign natural occurrences and potentially hazardous fire events, the system reduces false alarms and response time, enabling timely interventions.

<h2>Key Components</h2>

**Data Preprocessing:** We curated a dataset comprising 747 annotated images of wildfire smoke. The dataset was split into training, validation, and testing subsets, ensuring proper evaluation of the models' performance.

**Model Training:** We trained the Faster RCNN and YOLO models on the curated dataset using transfer learning and data augmentation techniques. The models learned to detect and localize instances of wildfire smoke accurately.

**Evaluation:** We evaluated the performance of the models based on mean average precision (mAP) at an Intersection over Union (IoU) threshold of 50%. YOLO outperformed Faster RCNN with an mAP@50 of 0.67 compared to 0.57, demonstrating its superior accuracy.

**Streamlit App:** To showcase the system's capabilities, we developed an interactive Streamlit app. Users can upload images to the app, and the YOLO model performs real-time smoke detection, generating dynamic news articles for detected wildfire events.

<h2>Conclusion</h2>

FireWatchAI represents a significant stride towards safeguarding communities and natural habitats from the ever-looming threat of wildfires. The system's rapid and precise detection capabilities empower stakeholders with timely information, fostering proactive measures and strategic firefighting approaches to minimize damage, preserve ecosystems, and save lives.

Through the integration of advanced computer vision technologies, FireWatchAI demonstrates how technology can play a crucial role in wildfire prevention and disaster response. The interactive Streamlit app provides users with a user-friendly interface to experience real-time smoke detection and gain valuable insights into detected wildfire events.

By leveraging computer vision and deep learning, FireWatchAI exemplifies how innovation can make a profound impact in tackling environmental challenges. The project serves as a stepping stone towards building resilient communities and protecting natural environments in the face of an escalating wildfire crisis.

<h3>Demo</h3>

[FireWatchAI App Demo](https://firewatch-ai-fbqfb4dawupukndsmmv65c.streamlit.app/)

<h3>Github Pages Link</h3>

[Interactive HTML](https://terry-ab.github.io/FireWatch-AI/Final_project.html)

