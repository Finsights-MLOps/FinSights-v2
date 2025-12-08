# ModelPipeline\finrag_ml_tg1\rag_modules_src\synthesis_pipeline\bedrock_client.py

"""
BedrockClient - AWS Bedrock API wrapper for Claude models.

Design: External service pattern - all dependencies injected.
Responsibility: AWS API calls, response parsing, cost tracking.
Does NOT: Build prompts, format context, manage configuration.
"""

import boto3
import json
from typing import Dict
import logging

from ..utilities.response_cleaner import clean_llm_response

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    Thin wrapper around AWS Bedrock Runtime API.
    
    Follows external service pattern - receives all dependencies
    from caller rather than creating them internally.
    
    Example:
        >>> config = MLConfig()
        >>> model = config.get_default_serving_model()
        >>> client = BedrockClient(
        ...     region=config.region,
        ...     model_id=model['model_id'],
        ...     max_tokens=model['max_tokens'],
        ...     temperature=model['temperature'],
        ...     cost_per_1k_input=model['cost_per_1k_input'],
        ...     cost_per_1k_output=model['cost_per_1k_output']
        ... )
        >>> response = client.invoke(system="You are helpful.", user="Hello!")
    """
    
    def __init__(
        self,
        region: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        cost_per_1k_input: float,
        cost_per_1k_output: float
    ):
        """
        Initialize Bedrock client with explicit dependencies.
        
        All configuration comes from caller - no internal config loading.
        
        Args:
            region: AWS region (e.g., 'us-east-1')
            model_id: Bedrock model identifier 
                     (e.g., 'anthropic.claude-3-5-sonnet-20241022-v2:0')
            max_tokens: Maximum tokens in model response
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            cost_per_1k_input: Cost per 1,000 input tokens in USD
            cost_per_1k_output: Cost per 1,000 output tokens in USD
            
        Example:
            >>> client = BedrockClient(
            ...     region='us-east-1',
            ...     model_id='anthropic.claude-3-5-sonnet-20241022-v2:0',
            ...     max_tokens=8192,
            ...     temperature=0.1,
            ...     cost_per_1k_input=0.003,
            ...     cost_per_1k_output=0.015
            ... )
        """
        # Store configuration
        self.region = region
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        
        # Initialize boto3 client
        self.client = boto3.client('bedrock-runtime', region_name=region)
        
        logger.info(
            f"BedrockClient initialized: model={model_id}, "
            f"region={region}, max_tokens={max_tokens}"
        )
    

    def invoke(self, system: str, user: str) -> Dict:
        """
        Invoke Claude model with system + user prompts.
        
        Uses Claude's Messages API format with separate system prompt.
        Automatically tracks usage and calculates cost.
        
        Args:
            system: System prompt (instructions, role definition)
            user: User prompt (assembled context + query)
            
        Returns:
            Dictionary with structure:
            {
                'content': str,              # Model's response text
                'usage': {
                    'input_tokens': int,     # Tokens in prompt
                    'output_tokens': int     # Tokens in response
                },
                'cost': float,               # Total cost in USD
                'model_id': str,             # Model identifier
                'stop_reason': str           # Why generation stopped
            }
            
        Raises:
            Exception: On AWS API errors (caller should handle)
            
        Example:
            >>> response = client.invoke(
            ...     system="You are a financial analyst.",
            ...     user="What is EBITDA?"
            ... )
            >>> print(response['content'])
            >>> print(f"Cost: ${response['cost']:.4f}")
        """
        # Construct request body (Claude Messages API format)
        # "bedrock-2023-05-31" is the ONLY version, and it's current. 
        # 
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,           # Separate system context
            "messages": [
                {
                    "role": "user",
                    "content": user
                }
            ]
        }
        
        """ // bedrock-2023-05-31 = Messages API format specification, NOT model version
        // AWS maintains backward compatibility - new models work with this format
        Claude 4 Haiku 4.5 (us.anthropic.claude-haiku-4-5-20251001-v1:0) - MODEL
        Claude 4 Sonnet 4.5 (us.anthropic.claude-sonnet-4-5-20250929-v1:0)
        Claude 4 Opus 4.5 (global.anthropic.claude-opus-4-5-20251101-v1:0)
        """
        try:
            # Call AWS Bedrock API
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            # Parse response body
            response_body = json.loads(response['body'].read())
            
            ## updated: on 12/8. wew.
            # Extract response content
            raw_content = response_body['content'][0]['text']
            # Clean response to remove LaTeX/Markdown formatting issues
            content = clean_llm_response(raw_content, log_changes=True)
            print(f"[DEBUG] Raw length: {len(raw_content)}")
            print(f"[DEBUG] Cleaned length: {len(content)}")
            print(f"[DEBUG] First 200 chars: {content[:200]}")

            
            # Extract usage statistics
            usage = response_body['usage']
            input_tokens = usage['input_tokens']
            output_tokens = usage['output_tokens']
            
            # Extract stop reason
            stop_reason = response_body.get('stop_reason', 'unknown')
            
            # Calculate cost
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Log success
            logger.info(
                f"Bedrock invoke success: "
                f"input={input_tokens} tokens, "
                f"output={output_tokens} tokens, "
                f"cost=${cost:.4f}, "
                f"stop_reason={stop_reason}"
            )
            
            # Return structured response
            return {
                'content': content,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                },
                'cost': cost,
                'model_id': self.model_id,
                'stop_reason': stop_reason
            }
            
        except Exception as e:
            # Log error and re-raise for caller to handle
            logger.error(f"Bedrock API error: {e}", exc_info=True)
            raise
    

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate total cost from token usage.
        
        Uses cost rates provided at initialization.
        
        Args:
            input_tokens: Number of tokens in prompt
            output_tokens: Number of tokens in response
            
        Returns:
            Total cost in USD
            
        Example:
            >>> client._calculate_cost(12000, 500)
            0.0435  # (12000/1000 * 0.003) + (500/1000 * 0.015)
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        total_cost = input_cost + output_cost
        
        return total_cost
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Dict:
        """
        Estimate cost before making API call.
        
        Useful for cost-conscious batch processing or user warnings.
        
        Args:
            input_tokens: Expected input token count
            output_tokens: Expected output token count
            
        Returns:
            Dictionary with cost breakdown:
            {
                'input_cost': float,
                'output_cost': float,
                'total_cost': float
            }
            
        Example:
            >>> estimate = client.estimate_cost(10000, 1000)
            >>> print(f"This query will cost ~${estimate['total_cost']:.4f}")
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BedrockClient(model_id='{self.model_id}', "
            f"region='{self.region}', "
            f"max_tokens={self.max_tokens})"
        )


# Utility function for easy client creation from MLConfig
def create_bedrock_client_from_config(
    config,
    model_key: str = None
) -> BedrockClient:
    """
    Factory function to create BedrockClient from MLConfig.
    
    Convenience wrapper that handles config extraction.
    
    Args:
        config: MLConfig instance
        model_key: Optional model key (uses default if None)
        
    Returns:
        Initialized BedrockClient
        
    Example:
        >>> from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
        >>> config = MLConfig()
        >>> client = create_bedrock_client_from_config(config)
    """
    # Get model configuration
    if model_key:
        model = config.get_serving_model(model_key)
    else:
        model = config.get_default_serving_model()
    
    # Create client with extracted values
    return BedrockClient(
        region=config.region,
        model_id=model['model_id'],
        max_tokens=model['max_tokens'],
        temperature=model['temperature'],
        cost_per_1k_input=model['cost_per_1k_input'],
        cost_per_1k_output=model['cost_per_1k_output']
    )