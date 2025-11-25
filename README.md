# Medical Assistant Chatbot with LoRA

Interactive medical Q&A chatbot using Llama 3.1 8B with LoRA fine-tuning.

## Quick Start (Google Colab)

### Prerequisites
1. Enable GPU: Runtime → Change runtime type → T4 GPU → Save
2. Register ngrok account (free): https://dashboard.ngrok.com/signup
3. Get your authtoken: https://dashboard.ngrok.com/get-started/your-authtoken

### One-Command Setup

```python
# Copy-paste this entire block into a Colab cell

# 1. Install dependencies
!pip install -q streamlit torch transformers peft accelerate pyngrok

# 2. Upload streamlit.py
from google.colab import files
print("Upload streamlit.py:")
uploaded = files.upload()

# 3. Pre-load model (5-10 min first time)
print("\nLoading model...")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER = "Easonwangzk/lora-llama31-med-adapter"
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch_dtype, device_map="auto")
base_model.eval()
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER, torch_dtype=torch_dtype)
model.eval()

globals()['preloaded_model'] = model
globals()['preloaded_tokenizer'] = tokenizer
print("✅ Model loaded!")

# 4. Start Streamlit
import subprocess, time
subprocess.Popen(["streamlit", "run", "streamlit.py", "--server.port", "8501"])
time.sleep(10)

# 5. Create public URL with ngrok
from pyngrok import ngrok

# ⚠️ IMPORTANT: Replace with your authtoken from https://dashboard.ngrok.com
ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE")

ngrok.kill()
public_url = ngrok.connect(8501)

print("\n" + "="*60)
print("🎉 Medical Assistant Ready!")
print("="*60)
print(f"\n📱 Access at:\n{public_url}\n")
print("="*60)
```

## Why Do We Need Ngrok?

### Problem: Can't access Colab's localhost directly

When you run Streamlit in Colab with `streamlit run streamlit.py`, it starts on `localhost:8501` **on Google's server**, not your computer. Your browser cannot access Google's internal network directly.

```
Your Browser → http://localhost:8501 ❌
(This localhost is on Google's server, not accessible from outside)
```

### Solution: Ngrok creates a public tunnel

Ngrok creates a public URL that forwards requests to Colab's localhost:

```
Your Browser → https://1234-abcd.ngrok-free.app (public URL)
       ↓
Ngrok Service (forwards request)
       ↓
Colab's localhost:8501
       ↓
Your Streamlit App ✅
```

## How to Get Ngrok Authtoken

1. **Sign up** (free): https://dashboard.ngrok.com/signup
2. **Login** and go to: https://dashboard.ngrok.com/get-started/your-authtoken
3. **Copy** your authtoken (looks like: `2ab...xyz`)
4. **Paste** it in the code:
   ```python
   ngrok.set_auth_token("2ab...xyz")  # Replace with your token
   ```

## Files

- **streamlit.py** - Main application (ONLY file you need!)
- **requirements.txt** - Dependencies list
- **lora.py** - Original LoRA evaluation script (reference)

## Configuration

Edit these in `streamlit.py`:
```python
BASE_MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER_REPO = "Easonwangzk/lora-llama31-med-adapter"
USE_LORA = True
```

## How Pre-loading Works

The script pre-loads the model into global variables **before** starting Streamlit:

```python
# In Colab cell: Pre-load model once
globals()['preloaded_model'] = model
globals()['preloaded_tokenizer'] = tokenizer

# streamlit.py automatically detects and uses it:
@st.cache_resource
def load_model():
    if 'preloaded_model' in globals():
        return globals()['preloaded_model'], globals()['preloaded_tokenizer']
    # Otherwise load normally...
```

**Benefits:**
- ✅ Model loads once (5-10 min)
- ✅ Streamlit uses pre-loaded model instantly
- ✅ No reloading on every interaction
- ✅ Much faster response times

## Requirements

- GPU: T4 or better (12GB+ VRAM)
- RAM: 12GB+ system memory
- Storage: ~20GB for model weights
- Internet: Required for initial download
- Ngrok account: Free registration required

## Troubleshooting

### Authentication Error?
Make sure you've set your authtoken:
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")  # Replace with your actual token
```

Get token from: https://dashboard.ngrok.com/get-started/your-authtoken

### Out of Memory?
```python
import torch
torch.cuda.empty_cache()
```

### Streamlit Won't Stop?
```python
!pkill -f streamlit
from pyngrok import ngrok
ngrok.kill()
```

### Check GPU
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Port Already in Use?
```python
!pkill -f streamlit  # Stop old streamlit
!streamlit run streamlit.py --server.port 8501 &
```

## Performance

- **First run**: 5-10 minutes (model download ~16GB)
- **Subsequent runs**: ~30 seconds (models cached)
- **Inference**: 2-5 seconds per response (T4 GPU)
- **Token generation**: ~20-50 tokens/sec

## Workflow

1. **One-time setup**: Register ngrok, get authtoken
2. **Every session**:
   - Enable GPU in Colab
   - Copy-paste the setup code
   - Upload streamlit.py
   - Wait for model to load (5-10 min first time)
   - Click the ngrok URL
   - Start chatting!

## Disclaimer

**For demonstration and educational purposes only.**
- ❌ Not for clinical use
- ❌ Not for medical decisions
- ✅ Always consult healthcare professionals

## License

Educational use only. Check individual model licenses:
- Llama 3.1: [Meta's Llama License](https://ai.meta.com/llama/license/)
- LoRA Adapter: See repository for details
