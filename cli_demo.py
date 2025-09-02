"""
CLI Demo for Background Removal Integration

Demonstrates how the background removal module integrates with CLI arguments
as specified in the requirements document.
"""

import argparse
from pathlib import Path
from PIL import Image

from background_remover import BackgroundRemover
from config import REMBG_MODELS, DEFAULT_REMBG_MODEL


def create_cli_parser():
    """
    Create CLI argument parser with background removal options.
    
    This demonstrates the CLI integration points specified in the requirements.
    """
    parser = argparse.ArgumentParser(
        description='Telegram Sticker Animator with Background Removal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Background Removal Models:
{chr(10).join(f'  {k}: {v}' for k, v in REMBG_MODELS.items())}

Example usage:
  python cli_demo.py image.jpg --remove-bg
  python cli_demo.py image.jpg --no-remove-bg
  python cli_demo.py image.jpg --bg-model u2netp
        """
    )
    
    # Input/output arguments
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    
    # Background removal arguments (as specified in requirements)
    bg_group = parser.add_mutually_exclusive_group()
    bg_group.add_argument(
        '--remove-bg', 
        action='store_true', 
        default=True,
        help='Enable background removal (default)'
    )
    bg_group.add_argument(
        '--no-remove-bg', 
        action='store_true',
        help='Disable background removal'
    )
    
    parser.add_argument(
        '--bg-model', 
        choices=list(REMBG_MODELS.keys()),
        default=DEFAULT_REMBG_MODEL,
        help=f'Choose rembg model (default: {DEFAULT_REMBG_MODEL})'
    )
    
    # Other processing arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser


def main():
    """
    Main CLI function demonstrating background removal integration.
    """
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Determine background removal settings
    remove_bg_enabled = not args.no_remove_bg
    bg_model = args.bg_model
    
    if args.verbose:
        print(f"🎨 Background Removal CLI Demo")
        print(f"Input: {args.input}")
        print(f"Background removal enabled: {remove_bg_enabled}")
        if remove_bg_enabled:
            print(f"Model: {bg_model} - {REMBG_MODELS[bg_model]}")
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.input}' not found")
        return 1
    
    try:
        # Initialize background remover with CLI settings
        bg_remover = BackgroundRemover(enabled=remove_bg_enabled, model=bg_model)
        
        # Get model info for reporting
        model_info = bg_remover.get_model_info()
        
        if args.verbose:
            print(f"\n🔧 Background Remover Status:")
            print(f"   Enabled: {model_info['enabled']}")
            print(f"   Available: {model_info['available']}")
            print(f"   Model: {model_info['model']}")
            print(f"   Description: {model_info['model_description']}")
        
        # Load and process image
        print(f"📂 Loading image: {args.input}")
        original_image = Image.open(args.input)
        
        if args.verbose:
            print(f"   Original size: {original_image.size}")
            print(f"   Original mode: {original_image.mode}")
        
        # Apply background removal
        print("🎨 Processing image...")
        processed_image = bg_remover.process(original_image)
        
        if args.verbose:
            print(f"   Processed size: {processed_image.size}")
            print(f"   Processed mode: {processed_image.mode}")
            
            if model_info['available'] and remove_bg_enabled:
                print("   ✅ Background removal applied")
            elif remove_bg_enabled:
                print("   ⚠️  Background removal requested but rembg not available - graceful fallback")
            else:
                print("   ℹ️  Background removal disabled")
        
        # Save result
        output_path = args.output if args.output else f"processed_{input_path.name}"
        processed_image.save(output_path, format='PNG')
        
        print(f"✅ Saved result to: {output_path}")
        print(f"📊 Output format: PNG with alpha channel")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
