# Vertex AI Model Comparison Tool

A Streamlit-based web application for comparing responses from different Vertex AI models, evaluating their performance, and generating optimised responses.

## Features

- Compare responses from multiple Vertex AI models simultaneously
- Interactive model selection and parameter configuration
- Automated evaluation of model responses
- Response optimisation based on evaluation results
- Real-time progress tracking
- Detailed visualisation of results with metrics and insights

## Prerequisites

- Python 3.9+
- Google Cloud Platform account with Vertex AI API enabled
- Project credentials configured

## Development Environment
1. Install [Python](https://www.python.org/downloads/) (v3.10 or higher)
2. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs)
3. Authenticate with Google Cloud SDK using the following commands:
   1. `gcloud auth login` (This will open a browser window to authenticate with your Google Account)
   2. `gcloud config set project <PROJECT_ID>` (replace `<PROJECT_ID>` with your Google Cloud Project ID you created earlier)
   3. `gcloud auth application-default login` (This sets up the application default credentials for your project)
   4. `gcloud auth application-default set-quota-project <PROJECT_ID>` (This sets the quota project for your project)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vertex-ai-comparison
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

The application uses the following default settings:
- Project ID: "benjaminwestern-test-genai"
- Default region: "us-central1"
- Available models:
  - Gemini 1.5 Flash 002
  - Gemini 1.5 Pro 002
  - Gemini 1.5 Flash 001
  - Gemini 1.5 Pro 001

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser (typically http://localhost:8501)

3. Using the interface:
   - Select exactly 3 models for comparison
   - Adjust temperature and max token parameters
   - Enter your prompt
   - Click "Generate & Compare" to start the process

## Features in Detail

### Model Selection
- Choose from available Vertex AI models
- Limited to exactly 3 models for optimal comparison

### Parameters
- Temperature: Controls response creativity (0.0 to 1.0)
- Max Tokens: Sets maximum response length (64 to 8192)

### Response Generation
- Parallel processing of model responses
- Real-time progress tracking
- Error handling and safety checks

### Evaluation Metrics
- Model accuracy percentage
- Traffic light status indicator
- Potential misses identification
- Areas for refinement
- Additional insights

### Optimisation
- Combines best elements from all responses
- Addresses identified weaknesses
- Maintains accuracy while improving quality

## Error Handling

The application includes comprehensive error handling for:
- Model initialisation failures
- Response generation errors
- Evaluation process issues
- Safety blocks
- Invalid JSON responses

## License

[Insert your chosen license here]

## Support

For support, please [insert support contact information]