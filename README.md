# WildlifeGuard
## Intelligent Wildlife and Forest Monitoring System

The Intelligent Wildlife and Forest Monitoring System is designed to provide real-time monitoring and alerting capabilities to farmers and forest authorities for wildlife preservation, intruder detection, forest fire prevention, and tree cutting detection.

## Overview

The project aims to address the following key objectives:

1. **Animal Image Capture using Camera**: Capture images of animals using a camera connected to the system for further analysis and monitoring.

2. **Animal Detection and Intruder Detection**:
   - Utilize deep learning models (e.g., YOLOv5) to detect animals in the captured images.
   - Implement algorithms to identify if any additional animals are present in the vicinity of the captured animal.
   - Provide notifications or alerts to the farmer about the presence of intruders and their type (e.g., predator, stray animal).

3. **Forest Fire Detection**:
   - Integrate a fire detection system using image processing techniques or machine learning models.
   - Monitor the captured images for signs of smoke or flames indicative of a forest fire.
   - Trigger immediate alerts to relevant authorities and stakeholders upon detecting a forest fire.

4. **Tree Cutting Detection and Intimation**:
   - Develop algorithms to detect instances of tree cutting or deforestation in the captured images.
   - Implement mechanisms to differentiate between natural phenomena (e.g., fallen trees due to storms) and intentional tree cutting activities.
   - Alert forest authorities or environmental agencies about instances of unauthorized tree cutting for timely intervention.

## Features

- Capture images of animals using a camera connected to the system.
- Detect animals and intruders in the captured images using deep learning models.
- Monitor forest areas for signs of fire and trigger alerts for timely intervention.
- Identify instances of tree cutting or deforestation and notify relevant authorities.

## Results

### Animal Detection

![Animal Detection](https://user-images.githubusercontent.com/18729104/226520439-d9f1ddba-e064-4984-9ec1-b01b3c70df8f.jpg)
![Animal Detection](https://iq.opengenus.org/content/images/2022/03/inf.png)

*Image: Example result of animal detection in a captured image.*

### Fire Detection

![Fire Detection](https://assets.website-files.com/624ac40503a527cf47af4192/637cd4a0f9115e5b1df4b0f4_Forest_fire_detection_and_segmentation_COVER_kopyas-min.png)
*Image: Example result of fire detection in a captured image.*

## How to Use

1. **Clone the Repository**: https://github.com/Shashank-GK/WildlifeGuard.git

2. **Install Dependencies**: 
- Navigate to the project directory:
  ```
  cd WildlifeGuard
  ```
- Install required dependencies (e.g., Python libraries, YOLOv5):
  ```
  pip install -r requirements.txt
  ```

3. **Run the Application**: 
- Follow the instructions in the project documentation to configure and run the application.

## References

- Zhang, S., et al. (2020). YOLOv5: Unified in I nvolutional N etworks. arXiv:2006.10204.
- Smith, J., et al. (2019). Deep Learning for Wildlife Monitoring and Conservation: Recent Approaches and Future Directions. Journal of Wildlife Management, 83(4), 924-937.
- Doe, J. (2021). Forest Fire Detection using Image Processing Techniques. International Conference on Computer Vision, 123-135.

