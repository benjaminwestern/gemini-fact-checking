import streamlit as st
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting
import vertexai
from typing import Dict, List
import json
from datetime import datetime
import asyncio

def initialise_vertex_ai(project_id: str, location: str = "us-central1"):
    """Initialise Vertex AI with project settings"""
    vertexai.init(project=project_id, location=location)

def get_available_models() -> List[Dict[str, str]]:
    """Get list of available Vertex AI models"""
    models = [
        {
            'display_name': 'Gemini 1.5 Flash 002',
            'model_name': 'gemini-1.5-flash-002',
            'type': 'flash',
            'family': 'gemini',
            'description': 'Lightweight model'
        },
        {
            'display_name': 'Gemini 1.5 Pro 002',
            'model_name': 'gemini-1.5-pro-002',
            'type': 'pro',
            'family': 'gemini',
            'description': 'Professional model'
        },
        {
            'display_name': 'Gemini 1.5 Flash 001',
            'model_name': 'gemini-1.5-flash-001',
            'type': 'flash',
            'family': 'gemini',
            'description': 'Lightweight model'
        },
        {
            'display_name': 'Gemini 1.5 Pro 001',
            'model_name': 'gemini-1.5-pro-001',
            'type': 'pro',
            'family': 'gemini',
            'description': 'Professional model'
        }
    ]
    return models

def get_generation_config(temperature: float, max_output_tokens: int, structured_output: bool = False):
    config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    
    if structured_output:
        config.update({
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "modelAccuracy": {"type": "NUMBER"},
                    "potentialMisses": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "areasToRefine": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "trafficLightStatus": {"type": "STRING"},
                    "additionalInsights": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["modelAccuracy", "potentialMisses", "areasToRefine", "trafficLightStatus", "additionalInsights"]
            }
        })
    
    return GenerationConfig(**config)

def get_safety_settings():
    return [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED
        ),
    ]

def parse_response(response, expect_json=False) -> Dict:
    """Parse response from model to dictionary"""
    try:
        # Convert response to dictionary
        response_dict = response.to_dict()
        candidates = response_dict.get("candidates", [])
        
        # Check for safety blocks
        if candidates and candidates[0].get("finish_reason") == "SAFETY":
            safety_message = "I apologise, but I cannot provide a response due to safety concerns. Here's why:\n\n"
            for rating in candidates[0].get("safety_ratings", []):
                if rating.get("blocked"):
                    safety_message += f"- {rating['category']}: {rating['probability']} probability\n"
            return {"error": safety_message}

        # Try to extract text from candidates
        if candidates and "content" in candidates[0]:
            content = candidates[0]["content"]
            if isinstance(content, dict) and "parts" in content:
                parts = content["parts"]
                if parts and isinstance(parts[0], dict):
                    text = parts[0].get("text", "")
                    if text:
                        if expect_json:
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                return {"error": "Invalid JSON in response"}
                        return {"text": text}

        return {"error": "I apologise, but I was unable to generate a response. Please try rephrasing your question."}
        
    except Exception as e:
        return {"error": f"Error parsing response: {str(e)}"}

async def get_model_responses(prompt: str, models: List[Dict], parameters: Dict) -> List[Dict]:
    """Get responses from all models in parallel"""
    async def get_single_response(model_info):
        try:
            model = GenerativeModel(model_info['model_name'])
            response = await model.generate_content_async(
                [prompt],
                generation_config=get_generation_config(
                    parameters['temperature'],
                    parameters['max_output_tokens']
                ),
                safety_settings=get_safety_settings()
            )
            
            parsed_response = parse_response(response)
            if "error" in parsed_response:
                return {
                    'model_name': model_info['display_name'],
                    'error': parsed_response["error"],
                    'model_info': model_info
                }
                
            return {
                'model_name': model_info['display_name'],
                'response': parsed_response["text"],
                'model_info': model_info
            }
        except Exception as e:
            return {
                'model_name': model_info['display_name'],
                'error': str(e),
                'model_info': model_info
            }

    tasks = [get_single_response(model) for model in models]
    return await asyncio.gather(*tasks)

async def evaluate_responses(prompt: str, responses: List[Dict]) -> List[Dict]:
    """Evaluate all responses in parallel"""
    async def evaluate_single_response(response_data):
        if 'error' in response_data:
            return response_data
            
        try:
            evaluation_prompt = f"""
            Analyse the following model response against the original prompt.
            Provide a structured evaluation focusing on accuracy, completeness, and areas for improvement.

            Original Prompt: {prompt}
            Model Name: {response_data['model_name']}
            Model Response: {response_data['response']}
            """

            model = GenerativeModel("gemini-1.5-pro-002")
            response = await model.generate_content_async(
                [evaluation_prompt],
                generation_config=get_generation_config(0.1, 8192, True),
                safety_settings=get_safety_settings()
            )
            
            parsed_evaluation = parse_response(response, expect_json=True)
            if "error" in parsed_evaluation:
                return {**response_data, 'evaluation_error': parsed_evaluation["error"]}
                
            return {**response_data, 'evaluation': parsed_evaluation}
            
        except Exception as e:
            return {**response_data, 'evaluation_error': str(e)}

    tasks = [evaluate_single_response(response) for response in responses]
    return await asyncio.gather(*tasks)

async def optimise_prompt(original_prompt: str, evaluated_responses: List[Dict]) -> Dict:
    """Generate optimised response based on evaluations"""
    try:
        valid_responses = [
            resp for resp in evaluated_responses 
            if 'error' not in resp and 'evaluation_error' not in resp
        ]
        
        if not valid_responses:
            return {'optimisation_error': 'No valid responses to optimise'}

        optimisation_prompt = f"""
        Original Prompt: {original_prompt}

        I have received multiple responses and their evaluations:
        {json.dumps(valid_responses, indent=2)}

        Based on these responses and evaluations:
        1. Analyse the strengths and weaknesses
        2. Identify the best elements
        3. Generate an improved response that:
           - Addresses missed opportunities
           - Incorporates refinements
           - Maintains accuracy
           - Improves upon originals

        Provide your optimised response.
        """

        model = GenerativeModel("gemini-pro")
        response = await model.generate_content_async(
            [optimisation_prompt],
            generation_config=get_generation_config(0.7, 8192),
            safety_settings=get_safety_settings()
        )
        
        parsed_response = parse_response(response)
        if "error" in parsed_response:
            return {'optimisation_error': parsed_response["error"]}
            
        return {'optimised_response': parsed_response["text"]}
        
    except Exception as e:
        return {'optimisation_error': str(e)}

def display_model_response(response_text: str):
    """Format the response in markdown"""
    st.markdown(response_text)

def main():
    st.set_page_config(page_title="Model Comparison", layout="wide")
    st.title("Model Response Comparison")

    # Initialise with project settings
    project_id = "benjaminwestern-test-genai"
    location = st.selectbox("Region:", ["us-central1", "europe-west4"], index=0)
    
    try:
        initialise_vertex_ai(project_id, location)
        available_models = get_available_models()
        
        # Model selection
        st.sidebar.header("Model Selection")
        
        selected_models = st.sidebar.multiselect(
            "Select exactly 3 models:",
            options=[m['display_name'] for m in available_models],
            max_selections=3
        )
        
        if len(selected_models) != 3:
            st.warning("Please select exactly 3 models.")
            return
            
        # Parameters
        st.sidebar.header("Parameters")
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.sidebar.slider("Max Tokens", 64, 8192, 1024, 64)
        
        parameters = {
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }

        # Prompt input
        prompt = st.text_area("Enter your prompt:", height=150)
        
        if st.button("Generate & Compare") and prompt:
            progress = st.empty()
            
            async def process_all():
                # Get model responses
                progress.text("Generating responses...")
                selected_model_configs = [m for m in available_models if m['display_name'] in selected_models]
                responses = await get_model_responses(prompt, selected_model_configs, parameters)
                
                # Evaluate responses
                progress.text("Evaluating responses...")
                evaluations = await evaluate_responses(prompt, responses)
                
                # Optimise based on evaluations
                progress.text("Generating optimised response...")
                optimisation = await optimise_prompt(prompt, evaluations)
                
                progress.empty()
                return responses, evaluations, optimisation

            responses, evaluations, optimisation = asyncio.run(process_all())
            
            # Display results in tabs
            tabs = st.tabs(selected_models + ["Optimised"])
            
            # Show individual model results
            for i, model_name in enumerate(selected_models):
                with tabs[i]:
                    st.subheader(f"Model: {model_name}")
                    
                    if 'error' in evaluations[i]:
                        st.error(f"Error: {evaluations[i]['error']}")
                        continue
                    
                    # Display original response in markdown
                    st.markdown("### Response:")
                    display_model_response(evaluations[i].get('response', 'No response generated'))
                    
                    # Evaluation results
                    if 'evaluation_error' in evaluations[i]:
                        st.error(f"Evaluation failed: {evaluations[i]['evaluation_error']}")
                    else:
                        eval_data = evaluations[i].get('evaluation', {})
                        if 'error' not in eval_data:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Accuracy", f"{eval_data.get('modelAccuracy', 0)}%")
                                status = eval_data.get('trafficLightStatus', 'red').lower()
                                st.markdown(f"Status: {'🟢' if status == 'green' else '🟡' if status == 'yellow' else '🟥'}")
                            
                            with col2:
                                st.markdown("### Key Findings")
                                st.markdown("**Potential Misses:**")
                                for miss in eval_data.get('potentialMisses', []):
                                    st.markdown(f"- {miss}")
                                    
                                st.markdown("**Areas to Refine:**")
                                for area in eval_data.get('areasToRefine', []):
                                    st.markdown(f"- {area}")
                                    
                                if eval_data.get('additionalInsights'):
                                    st.markdown("**Additional Insights:**")
                                    for insight in eval_data.get('additionalInsights', []):
                                        st.markdown(f"- {insight}")
                        else:
                            st.error(f"Evaluation failed: {eval_data['error']}")
            
            # Show optimised response
            with tabs[-1]:
                st.subheader("Optimised Response")
                if 'optimisation_error' in optimisation:
                    st.error(f"Optimisation failed: {optimisation['optimisation_error']}")
                else:
                    st.markdown("### Response:")
                    display_model_response(optimisation.get('optimised_response', 'No response generated'))
            
            st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Full error details:", exc_info=True)

if __name__ == "__main__":
    main()