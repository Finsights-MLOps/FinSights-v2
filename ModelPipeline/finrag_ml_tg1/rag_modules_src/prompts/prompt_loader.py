"""
Prompt template loader for FinRAG synthesis pipeline.
Loads and formats YAML-based prompt templates.

Location: ModelPipeline/finrag_ml_tg1/rag_modules_src/prompts/prompt_loader.py
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)



import yaml
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads and formats prompt templates from YAML files.
    
    Design:
        - System prompt: Reusable instructions for LLM behavior
        - Query template: Wrapper for pre-assembled context
        - All assembly logic lives in Python (supply_lines.py)
        - Templates are simple, focused on prompt content
    
    Usage:
        # Initialize
        loader = PromptLoader()
        
        # Load system prompt
        system_prompt = loader.load_system_prompt()
        
        # Format query (context already assembled by build_combined_context)
        user_prompt = loader.format_query_template(
            combined_context=your_assembled_context_string
        )
        
        # Send to LLM
        response = llm_client.invoke(
            system=system_prompt,
            user=user_prompt,
            max_tokens=2048
        )
    
    File Structure:
        prompts/
        ├── system_financial_rag_v1.yaml      # System instructions
        ├── query_template_standard_v1.yaml   # Query wrapper
        └── prompt_loader.py                   # This file
    """
    
    def __init__(
        self, 
        system_prompt_version: str = "v1", 
        query_template_version: str = "v1"
    ):
        """
        Initialize prompt loader.
        
        Args:
            system_prompt_version: System prompt version (e.g., 'v1', 'v2')
            query_template_version: Query template version (e.g., 'v1', 'v2')
        
        Raises:
            FileNotFoundError: If prompt files don't exist at expected paths
        """
        # Use MLConfig service to get ModelPipeline root (centralized path detection)
        from finrag_ml_tg1.loaders.ml_config_loader import MLConfig
        config = MLConfig()
        model_root = config.model_root

        # Prompts directory: rag_modules_src/prompts/
        self.PROMPTS_DIR = model_root / "finrag_ml_tg1" / "rag_modules_src" / "prompts"
        
        # Build file paths
        self.system_prompt_file = self.PROMPTS_DIR / f"system_financial_rag_{system_prompt_version}.yaml"
        self.query_template_file = self.PROMPTS_DIR / f"query_template_standard_{query_template_version}.yaml"
        
        # Validate files exist
        if not self.system_prompt_file.exists():
            raise FileNotFoundError(
                f"System prompt not found: {self.system_prompt_file}\n"
                f"Expected location: rag_modules_src/prompts/"
            )
        
        if not self.query_template_file.exists():
            raise FileNotFoundError(
                f"Query template not found: {self.query_template_file}\n"
                f"Expected location: rag_modules_src/prompts/"
            )
        
        # Load YAML configs
        logger.info(f"Loading system prompt: {self.system_prompt_file.name}")
        with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
            self.system_config = yaml.safe_load(f)
        
        logger.info(f"Loading query template: {self.query_template_file.name}")
        with open(self.query_template_file, 'r', encoding='utf-8') as f:
            self.query_config = yaml.safe_load(f)
        
        logger.info("PromptLoader initialized successfully")
    
    # ════════════════════════════════════════════════════════════════════════
    # Core Methods
    # ════════════════════════════════════════════════════════════════════════
    
    def load_system_prompt(self) -> str:
        """
        Load system prompt text.
        
        Returns:
            System prompt string ready for LLM
        
        Example:
            >>> loader = PromptLoader()
            >>> system = loader.load_system_prompt()
            >>> print(system[:100])
            You are a financial analyst assistant answering questions using corporate SEC 10-K filing data...
        """
        return self.system_config['prompt'].strip()
    
    
    def format_query_template(self, combined_context: str) -> str:
        """
        Format query template with pre-assembled context.
        
        The context should already be fully assembled by supply_lines.build_combined_context(),
        including:
        - KPI Snapshot section
        - Narrative Context section
        - User Question footer
        
        This method is a simple pass-through wrapper that applies the template.
        
        Args:
            combined_context: Complete assembled context string from build_combined_context()
        
        Returns:
            Formatted prompt string ready for LLM
        
        Example:
            >>> # Context already assembled:
            >>> context = build_combined_context(query, rag_components, ...)
            >>> 
            >>> # Just wrap it:
            >>> loader = PromptLoader()
            >>> user_prompt = loader.format_query_template(context)
        """
        template = self.query_config['template']
        
        formatted = template.format(
            combined_context=combined_context.strip()
        )
        
        return formatted.strip()
    
    
    # ════════════════════════════════════════════════════════════════════════
    # Metadata Access
    # ════════════════════════════════════════════════════════════════════════
    
    def get_system_metadata(self) -> Dict:
        """
        Get system prompt metadata.
        
        Returns:
            Dict with keys: target_models, recommended_temperature, 
                          recommended_max_tokens, last_updated
        """
        return self.system_config.get('metadata', {})
    
    
    def get_query_metadata(self) -> Dict:
        """
        Get query template metadata.
        
        Returns:
            Dict with keys: use_cases, context_structure, average_context_size,
                          estimated_response_size, design_notes, last_updated
        """
        return self.query_config.get('metadata', {})
    
    
    def get_recommended_llm_params(self) -> Dict:
        """
        Get recommended LLM parameters from system prompt metadata.
        
        Returns:
            Dict with keys: temperature, max_tokens, target_models
        
        Example:
            >>> loader = PromptLoader()
            >>> params = loader.get_recommended_llm_params()
            >>> print(params)
            {
                'temperature': 0.1,
                'max_tokens': 2048,
                'target_models': ['claude-sonnet-3.5', 'claude-haiku-3.5', 'gpt-4o']
            }
        """
        metadata = self.get_system_metadata()
        
        return {
            'temperature': metadata.get('recommended_temperature', 0.1),
            'max_tokens': metadata.get('recommended_max_tokens', 2048),
            'target_models': metadata.get('target_models', [])
        }


# ════════════════════════════════════════════════════════════════════════
# Standalone Test
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test prompt loader independently.
    
    Run:
        cd ModelPipeline/finrag_ml_tg1/rag_modules_src/prompts/
        python prompt_loader.py
    """
    print("=" * 70)
    print("PROMPT LOADER TEST")
    print("=" * 70)
    
    try:
        # Initialize
        loader = PromptLoader()
        print("\n✓ PromptLoader initialized")
        
        # Load system prompt
        system = loader.load_system_prompt()
        print(f"\n[System Prompt]")
        print(f"  Length: {len(system)} characters")
        print(f"  Preview: {system[:150]}...")
        
        # Load recommended params
        params = loader.get_recommended_llm_params()
        print(f"\n[Recommended LLM Parameters]")
        print(f"  Temperature: {params['temperature']}")
        print(f"  Max Tokens: {params['max_tokens']}")
        print(f"  Target Models: {', '.join(params['target_models'])}")
        
        # Test query template formatting
        test_context = """
══════════════════════════════════════════════════════════════════════
KPI SNAPSHOT - METRIC PIPELINE OUTPUT
══════════════════════════════════════════════════════════════════════

NVDA:
  2020: Total Assets=$17.3B, Net Income=$950.0M

══════════════════════════════════════════════════════════════════════
NARRATIVE CONTEXT - SEC FILINGS
══════════════════════════════════════════════════════════════════════

=== [NVDA] NVIDIA CORP | FY 2020 | Doc: nvda_2020_10k | Item 7: MD&A | Sentences: sent_123 - sent_145 ===

Revenue increased 52% due to datacenter growth.

══════════════════════════════════════════════════════════════════════
USER QUESTION
══════════════════════════════════════════════════════════════════════

What were NVIDIA's 2020 financials?
        """
        
        user_prompt = loader.format_query_template(test_context.strip())
        print(f"\n[Query Template Formatting]")
        print(f"  Input length: {len(test_context)} characters")
        print(f"  Output length: {len(user_prompt)} characters")
        print(f"  Template applied: ✓")
        
        # Metadata
        query_meta = loader.get_query_metadata()
        print(f"\n[Query Template Metadata]")
        print(f"  Use cases: {len(query_meta.get('use_cases', []))}")
        print(f"  Avg context size: {query_meta.get('average_context_size', 'N/A')}")
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()