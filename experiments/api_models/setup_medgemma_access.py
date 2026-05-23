"""
Setup MedGemma Access for FADA VQA

MedGemma is Google's medical language model optimized for healthcare tasks.
It requires HuggingFace authentication and model access approval.

This script guides you through the setup process.
"""

import sys
from pathlib import Path

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def check_hf_login():
    """Check if user is logged into HuggingFace"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"[OK] Logged in as: {user_info['name']}")
        return True
    except Exception:
        print("[X] Not logged in to HuggingFace")
        return False

def check_model_access():
    """Check if user has access to MedGemma"""
    try:
        from transformers import AutoTokenizer
        model_id = "google/medgemma-4b-it"
        print(f"Checking access to {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"[OK] Access granted to MedGemma!")
        return True
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower():
            print(f"[X] Access denied - Model is gated")
            return False
        else:
            print(f"[X] Error: {e}")
            return False

def main():
    print_header("MedGemma Access Setup for FADA")

    print("MedGemma 4B is a medical language model from Google that requires:")
    print("  1. HuggingFace account")
    print("  2. Authentication token")
    print("  3. Model access approval")
    print("\n")

    # Step 1: Check if logged in
    print_header("Step 1: HuggingFace Authentication")

    if check_hf_login():
        print("\nYou are already authenticated. Proceeding to Step 2...")
    else:
        print("\nTo authenticate:")
        print("\n1. Create HuggingFace account:")
        print("   https://huggingface.co/join")
        print("\n2. Generate access token:")
        print("   https://huggingface.co/settings/tokens")
        print("   - Select 'New token'")
        print("   - Name: 'fada-medgemma'")
        print("   - Type: 'Read'")
        print("   - Copy the token")
        print("\n3. Login via CLI:")
        print("   huggingface-cli login")
        print("   (or in Python: from huggingface_hub import login; login())")
        print("\n")

        response = input("Have you completed authentication? (y/n): ")
        if response.lower() != 'y':
            print("\nSetup incomplete. Run this script again after authentication.")
            sys.exit(0)

        if not check_hf_login():
            print("\n[X] Still not authenticated. Please complete authentication first.")
            sys.exit(1)

    # Step 2: Request model access
    print_header("Step 2: Request MedGemma Access")

    if check_model_access():
        print("\n[OK] SUCCESS! You have access to MedGemma.")
        print("\nYou can now train MedGemma VQA models:")
        print("  python train_medgemma_vqa.py")
    else:
        print("\nTo request access:")
        print("\n1. Visit model page:")
        print("   https://huggingface.co/google/medgemma-4b-it")
        print("\n2. Click 'Agree and access repository'")
        print("\n3. Accept the terms and conditions")
        print("\n4. Wait for approval (usually instant, sometimes takes hours)")
        print("\n5. Run this script again to verify access")
        print("\n")

        print("NOTE: While waiting for MedGemma access, you can use:")
        print("  - BLIP-2 (already working)")
        print("  - LLaVA (if memory permits)")
        print("  - Florence-2 (when compatibility issues resolved)")

    # Step 3: Test inference (if access granted)
    if check_model_access():
        print_header("Step 3: Test Inference")

        print("Testing MedGemma inference...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_id = "google/medgemma-4b-it"
            print(f"\nLoading {model_id}...")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            print(f"[OK] Model loaded successfully")
            print(f"  Parameters: {model.num_parameters() / 1e9:.1f}B")
            print(f"  Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

            # Simple test
            test_prompt = "What is shown in a fetal ultrasound image?"
            inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nTest prompt: {test_prompt}")
            print(f"Response: {response}")

            print("\n[OK] MedGemma is ready for VQA training!")
            print("\nNext steps:")
            print("  1. Create training notebook for MedGemma VQA")
            print("  2. Adapt BLIP-2 training pipeline for MedGemma")
            print("  3. Train on fetal ultrasound dataset")

        except Exception as e:
            print(f"\n[X] Test failed: {e}")
            print("Model access granted but inference failed.")
            print("This might be due to:")
            print("  - Insufficient GPU memory")
            print("  - Missing dependencies")
            print("  - Model compatibility issues")

if __name__ == "__main__":
    main()
