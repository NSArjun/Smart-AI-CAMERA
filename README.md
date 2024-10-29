# SmartAICamera

## Empowering Safety Through Intelligent Vision!

Introducing an AI-integrated surveillance camera system designed to enhance security and safety in various environments, including residential areas, commercial establishments, and public spaces. With advanced machine learning algorithms and real-time analytics, our system offers unparalleled surveillance capabilities to protect what matters most.

### Key Features:

1. Facial Detection

- Entry/Exit Monitoring: Keep track of individuals entering or leaving premises, ensuring that only authorized personnel are granted access.

- Entry Logs: Maintain detailed records of all individuals who access your premises, providing valuable data for security audits and investigations.

- Unauthorized Person Detection: Instantly alert security personnel of any unrecognized individuals, enhancing response time in potential threat situations.

2. Vehicle Logging with Parking Assistance

- Automated Vehicle Logging: Record entry and exit times of vehicles, offering insights into traffic patterns and usage.

- Parking Guidance: Direct vehicles to available parking spots using advanced sensors and audible instructions, ensuring efficient use of space and reducing congestion.

- Visitor Management: Enhance visitor experiences by streamlining parking processes and improving overall safety.

3. Animal Detection

- Wildlife Monitoring: Detect wildlife entering residential areas and send alerts, helping to prevent human-wildlife conflicts and ensuring community safety.

- Pet Activity Alerts: Keep track of pets at home, receiving notifications if they engage in unusual behavior, providing peace of mind to pet owners.

4. Trespassing/Anomaly Detection

- Irregular Activity Monitoring: Identify unusual behavior, such as unauthorized individuals trespassing on private property or engaging in suspicious activities.

- Theft Alarms: Automatically trigger alarms in case of detected theft, enabling quick intervention by law enforcement.

5. Collapsing Detection

- Health Monitoring: Use advanced algorithms to identify individuals who may have fainted or collapsed, enabling immediate alerts to guardians or emergency services.

- Critical Alerts: Ensure that designated guardians receive timely notifications, enhancing response time during emergencies.

6. Weapons Detection

- Weapon Identification: Leverage advanced computer vision techniques to automatically detect weapons carried by individuals.

- Alarm Triggering: Activate alarms and alerts to security personnel upon weapon detection, allowing for immediate action and response.

8. Camera Interference Detection

- Real-Time Monitoring: Continuously monitor camera feeds to ensure functionality, detecting obstructions or malfunctions.

- Authority Notifications: Instantly notify security personnel or authorities if a camera is compromised, ensuring uninterrupted surveillance.

### Benefits of the AI-Integrated Surveillance System:

- Enhanced Security: Proactively address security threats and unauthorized access through advanced detection capabilities.
- Informed Decision-Making: Utilize data collected from various sensors and cameras to make informed decisions regarding safety and security policies.
- Increased Efficiency: Streamline security operations with automated alerts and logging systems, reducing the burden on security personnel.
- Community Safety: Promote safer neighborhoods by deterring criminal activities and ensuring swift responses to emergencies.

### Use Cases:

- Residential Areas: Protect homes and families with a comprehensive surveillance system that monitors entry points and detects irregular activities.
- Commercial Establishments: Enhance security in retail spaces, offices, and warehouses by monitoring employee access, vehicle parking, and potential theft.
- Public Spaces: Ensure the safety of parks, public transportation systems, and event venues by utilizing intelligent surveillance to detect anomalies and emergencies.

### How to Test?
1. Install The Required Dependencies

```bash

pip install -r requirements.txt

```

2. Run the Program !.

3. All the other testing instructions have been hard corded.


### Code Explanation:

1. main.py
This is used to test the entire project.

2. face.py
This module includes the features of facial recognition which includes:
   - Identification of human faces using opencv.
   - Remembering Faces by converting facial images into 128 dimension vectors and storing it.
   - Traversing through the stored data to find the similarity (norm) between faces.

3. parking.py
 This module deploys a custom trained model. For test cases we have provided different combinations of car arrangements in one specific parking slot.Based on that we provide the voice based assistance.

4. person.py
This module is used to recognize known people using face recognition. It is used to log the new or unknown people' entry/exit time for easy access.

5. essentialcam.py
This module deploys a custom trained model which involves combination of varieties of features:
   - Identification of animals
   - Weapon Detection
   - Fall Detection

6. vehicles.py
This module deploys a custom trained model which involves:
   - Identification of known vehicle.
   - Logging of unknown vehicles with entry/exit time.

7. blocked.py
This module identifies any disturbances in CCTV cameras.

8. burglar.py
This module identifies any burglars jumping over the wall.

### Future Scope

1. Enhanced AI Capabilities

Implement more advanced deep learning techniques for better object recognition, behavior analysis,Violence detection and anomaly detection.

2. Integration with Smart Home Systems

IoT Compatibility: Enable integration with existing smart home devices, allowing users to control lights, alarms, and cameras from a single interface.

Smart Notifications: Develop features that send smart notifications to users' devices based on specific triggers, like unusual activities or recognized faces.

3. Advanced Analytics and Reporting

Real-Time Analytics: Offer real-time insights and analytics to users regarding their premises' security and activity levels.

Historical Data Analysis: Store and analyze historical data for long-term trends, helping users understand patterns in behavior and 
security breaches.

4. Expansion of Detection Features

Additional Detection Types: Expand the detection capabilities to include fire or gas leak detection, providing comprehensive safety features.

Customized Alerts: Allow users to customize alerts based on specific activities, improving the relevance of notifications.

5. User Interface and Experience Enhancements

Mobile App Development: Create a dedicated mobile application for easier access to the surveillance system, including live feeds, alerts, and camera controls.

User-Friendly Dashboard: Develop an intuitive dashboard that displays alerts, live feeds, and analytics in an easy-to-understand format.


### Conclusion: 

In conclusion, our AI-integrated surveillance camera system marks a significant leap in security technology, combining advanced features such as facial detection, vehicle logging, and anomaly detection to enhance safety and monitoring. By continually improving our system through machine learning and user feedback, we aim to provide an effective solution that adapts to the evolving needs of users. We envision a future where our technology not only protects individuals and communities but also fosters a greater sense of security and resilience. Together, we can create safer environments, empowering users to safeguard their spaces with confidence.

### Collaborators:

- ASHWINA NARENDRAKUMAR [@ASHWINAKN]
- ARJUN N S [@NSARJUN]
- MOHAMMED SAAJID S [@MOHAMMED-SAAJID]
- SRI HARISH B [@SRIHARISHB]
