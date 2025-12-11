#!/usr/bin/env python3
"""
Simple script to query the RAG model directly without frontend.

This script provides a straightforward way to run queries against the FinRAG model
by directly calling the orchestrator function.

Usage:
    # Basic usage
    python query_rag.py "What was NVIDIA's revenue in 2023?"
    
    # Interactive mode
    python query_rag.py --interactive
    
    # With custom options
    python query_rag.py --query "Your question" --model development --no-kpi
    
Examples:
    python query_rag.py "What were Apple's revenues from 2018 to 2020?"
    python query_rag.py --interactive
    python query_rag.py --query "Analyze Microsoft's AI strategy" --export-response
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add ModelPipeline to Python path
MODEL_PIPELINE_ROOT = Path(__file__).parent
if str(MODEL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINE_ROOT))

from finrag_ml_tg1.rag_modules_src.synthesis_pipeline.orchestrator import answer_query


def find_model_root() -> Path:
    """Find ModelPipeline root directory."""
    current = Path(__file__).resolve().parent
    
    # If this script is in ModelPipeline, return it
    if current.name == "ModelPipeline":
        return current
    
    # Otherwise walk up
    for parent in current.parents:
        if parent.name == "ModelPipeline":
            return parent
    
    # Fallback to script's parent directory
    return current


def print_result(result: dict):
    """Print query result in a readable format."""
    print("\n" + "=" * 80)
    print("QUERY RESULT")
    print("=" * 80)
    
    # Check for error
    if result.get("error"):
        print(f"\n❌ ERROR: {result['error']}")
        print(f"   Type: {result.get('error_type', 'Unknown')}")
        print(f"   Stage: {result.get('stage', 'Unknown')}")
        print(f"\n   Query: {result['query']}")
        print("=" * 80)
        return
    
    # Success case
    query = result.get("query", "")
    answer = result.get("answer", "")
    metadata = result.get("metadata", {})
    
    print(f"\n📝 Query: {query}")
    print("\n" + "-" * 80)
    print("💬 Answer:")
    print("-" * 80)
    print(answer)
    print("-" * 80)
    
    # Metadata
    llm = metadata.get("llm", {})
    context_meta = metadata.get("context", {})
    
    print(f"\n📊 Metadata:")
    print(f"   Model: {llm.get('model_id', 'N/A')}")
    print(f"   Tokens: {llm.get('input_tokens', 0):,} in / {llm.get('output_tokens', 0):,} out")
    print(f"   Total Tokens: {llm.get('total_tokens', 0):,}")
    print(f"   Cost: ${llm.get('cost', 0):.4f}")
    print(f"   Context Length: {context_meta.get('context_length', 0):,} chars")
    print(f"   KPI Included: {context_meta.get('kpi_included', False)}")
    print(f"   RAG Included: {context_meta.get('rag_included', False)}")
    
    # Processing time
    processing_time = metadata.get("processing_time_ms")
    if processing_time:
        print(f"   Processing Time: {processing_time:.0f} ms")
    
    # Export files
    exports = result.get("exports", {})
    if exports.get("log_file"):
        print(f"\n📁 Exports:")
        print(f"   Logs: {exports['log_file']}")
        if exports.get("context_file"):
            print(f"   Context: {exports['context_file']}")
        if exports.get("response_file"):
            print(f"   Response: {exports['response_file']}")
    
    print("=" * 80)


def run_interactive_mode(model_root: Path, include_kpi: bool, include_rag: bool, 
                        model_key: Optional[str], export_response: bool):
    """Run in interactive mode where user can enter multiple queries."""
    print("\n" + "=" * 80)
    print("FinRAG Interactive Query Mode")
    print("=" * 80)
    print("Enter your questions (type 'quit' or 'exit' to stop)\n")
    
    while True:
        try:
            query = input("\n💭 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            print("\n⏳ Processing query...")
            
            result = answer_query(
                query=query,
                model_root=model_root,
                include_kpi=include_kpi,
                include_rag=include_rag,
                model_key=model_key,
                export_context=False,  # Don't export context in interactive mode
                export_response=export_response
            )
            
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query the FinRAG model directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python query_rag.py "What was NVIDIA's revenue in 2023?"
  
  # Interactive mode
  python query_rag.py --interactive
  
  # Custom options
  python query_rag.py --query "Your question" --model development
  python query_rag.py --query "Your question" --no-kpi --no-rag
  python query_rag.py --query "Your question" --export-response
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="The question to ask (optional if using --interactive)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (can enter multiple queries)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model key from ml_config.yaml (default: uses default_serving_model)"
    )
    
    parser.add_argument(
        "--no-kpi",
        action="store_true",
        help="Disable KPI data in context"
    )
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG context"
    )
    
    parser.add_argument(
        "--export-response",
        action="store_true",
        help="Export full response to JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.query:
        parser.error("Either provide a query as argument or use --interactive mode")
    
    # Find model root
    try:
        model_root = find_model_root()
        print(f"📍 ModelPipeline root: {model_root}")
    except Exception as e:
        print(f"❌ Error finding ModelPipeline root: {e}")
        sys.exit(1)
    
    # Set options
    include_kpi = not args.no_kpi
    include_rag = not args.no_rag
    
    # Run query
    try:
        if args.interactive:
            run_interactive_mode(
                model_root=model_root,
                include_kpi=include_kpi,
                include_rag=include_rag,
                model_key=args.model,
                export_response=args.export_response
            )
        else:
            print(f"\n⏳ Processing query...")
            
            result = answer_query(
                query=args.query,
                model_root=model_root,
                include_kpi=include_kpi,
                include_rag=include_rag,
                model_key=args.model,
                export_context=False,  # Don't export context by default in simple script
                export_response=args.export_response
            )
            
            print_result(result)
            
            # Exit with error code if query failed
            if result.get("error"):
                sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

