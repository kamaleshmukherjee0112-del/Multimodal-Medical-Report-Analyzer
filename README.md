 
 
 In recent years, the healthcare sector has witnessed a rapid increase in the use of digital medical 
records, diagnostic reports, and imaging data. Hospitals and diagnostic centers routinely generate large 
volumes of medical reports in the form of PDFs, which may include scanned documents, structured 
tables, unstructured clinical text, and embedded medical images such as X-rays, CT scans, MRI scans, 
ultrasound images, and ECG reports. Interpreting these reports manually is time-consuming, error
prone, and often challenging for patients who lack medical expertise. 
The Multimodal Medical Report Analyzer proposed in this project aims to address these challenges 
by providing an end-to-end automated system for analyzing medical PDF reports. The system is 
designed to process multiple data modalities present in a single report, including textual content, 
tabular laboratory results, and medical images. It employs a combination of direct PDF text extraction 
and Optical Character Recognition (OCR) techniques to handle both digital and scanned reports. 
Extracted tables are further processed to identify laboratory data, clean inconsistencies, and flag 
abnormal values based on reference ranges. 
To structure unorganized medical text, the system utilizes a hybrid rule-based Natural Language 
Processing (NLP) approach, which performs sentence-level intent detection and organizes 
information into a unified, structured report format. In parallel, embedded medical images are 
extracted from PDF files and classified using Convolutional Neural Network (CNN)–based models to 
identify the image modality, such as X-ray, CT, MRI, ultrasound, ECG, or non-medical images. 
To enhance usability and understanding, the structured report is passed through a controlled Large 
Language Model (LLM)–based explanation layer, which generates patient-friendly and clinician
oriented summaries while strictly enforcing medical safety constraints. The system is deployed through 
a user-friendly Streamlit web interface, enabling users to upload reports, view extracted data, analyze 
results, and obtain simplified explanations in real time. 
The proposed system demonstrates how multimodal data processing, rule-based NLP, computer vision, 
and controlled language models can be effectively integrated into a single framework to support 
medical report interpretation. This project provides a scalable, explainable, and safety-aware solution 
that can assist patients, clinicians, and healthcare institutions in understanding complex medical reports 
more efficiently. 
 
 
 
 
 Tools and Technologies Used:

The following tools and technologies were used in the development of the 
proposed system: 
1. Programming Language: Python 
2. Web Framework: Streamlit (for user interface and interaction) 
3. PDF Processing: PyMuPDF, pdf2image 
4. Optical Character Recognition: Tesseract OCR 
5. Natural Language Processing: Rule-based and heuristic NLP techniques 
6. Table Extraction: Camelot and OCR-based table parsing 
 
7. Computer Vision: Convolutional Neural Networks for image classification 
8. Language Model Integration: Controlled LLM for report explanation and 
summarization 
9. Libraries and Frameworks: PyTorch, OpenCV, Pandas, NumPy 
These technologies were selected based on their reliability, community support, and 
suitability for medical document analysis.