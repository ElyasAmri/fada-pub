"""Quick check of MedGemma access status"""

print("\n" + "="*70)
print("MedGemma Access Status Check")
print("="*70 + "\n")

# Check 1: HuggingFace login
print("1. HuggingFace Authentication:")
try:
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"   [OK] Logged in as: {user_info['name']}")
    hf_authenticated = True
except Exception as e:
    print(f"   [X] Not authenticated")
    print(f"   Error: {str(e)[:100]}")
    hf_authenticated = False

# Check 2: Model access
print("\n2. MedGemma Model Access:")
if hf_authenticated:
    try:
        from transformers import AutoTokenizer
        model_id = "google/medgemma-4b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"   [OK] Access granted to {model_id}")
        model_accessible = True
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "gated" in error_msg.lower():
            print(f"   [X] Access denied - Model is gated")
            print(f"   Visit: https://huggingface.co/google/medgemma-4b-it")
        else:
            print(f"   [X] Error: {error_msg[:100]}")
        model_accessible = False
else:
    print("   [SKIP] Cannot check - not authenticated")
    model_accessible = False

# Summary
print("\n" + "="*70)
print("Summary:")
print("="*70)

if hf_authenticated and model_accessible:
    print("\n[OK] MedGemma is ready to use!")
    print("\nNext steps:")
    print("  - Create MedGemma VQA training notebook")
    print("  - Adapt BLIP-2 pipeline for MedGemma")
    print("  - Train on fetal ultrasound dataset")
elif hf_authenticated and not model_accessible:
    print("\n[PARTIAL] Authenticated but no model access")
    print("\nTo gain access:")
    print("  1. Visit: https://huggingface.co/google/medgemma-4b-it")
    print("  2. Click 'Agree and access repository'")
    print("  3. Accept terms and wait for approval")
else:
    print("\n[NOT READY] Authentication required")
    print("\nSetup instructions:")
    print("  1. Create account: https://huggingface.co/join")
    print("  2. Get token: https://huggingface.co/settings/tokens")
    print("  3. Login: huggingface-cli login")
    print("  4. Run: python setup_medgemma_access.py")

print("\nAlternatives (already working):")
print("  - BLIP-2: Trained and working on 5 categories")
print("  - Use existing models for VQA tasks")
print("="*70 + "\n")
