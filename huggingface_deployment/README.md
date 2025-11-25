---
title: Medical Assistant with LoRA
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# Medical Assistant Chatbot

An AI-powered medical question-answering assistant using Llama 3.1 8B with LoRA fine-tuning.

## Features

- 💬 Interactive chat interface with conversation history
- 🎨 Dark/Light theme support
- 🚀 Quick question buttons for common medical queries
- ⚡ GPU-accelerated inference
- 🔒 Privacy-focused (no data collection)

## Models Used

- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct` (8B parameters)
- **LoRA Adapter**: `Easonwangzk/lora-llama31-med-adapter` (Medical domain fine-tuning)

## Try These Questions

- What is diabetes?
- What are the common symptoms of hypertension?
- Explain the difference between Type 1 and Type 2 diabetes.
- What diagnostic tests are used for heart disease?
- What are the risk factors for stroke?

## Quick Start

1. Type your medical question in the chat input
2. Or click one of the quick question buttons
3. Get AI-powered responses based on medical knowledge

## Performance

- **Response Time**: 2-5 seconds (GPU)
- **Context Window**: Maintains last 3 messages for context
- **Max Response**: Up to 512 tokens

## Disclaimer

⚠️ **IMPORTANT**: This system is for **demonstration and educational purposes only**.

- ❌ **NOT for clinical use**
- ❌ **NOT for medical decisions**
- ❌ **NOT a replacement for professional medical advice**

**Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.**

## Technical Details

### Architecture
```
User Input
    ↓
Llama 3.1 8B Base Model
    ↓
LoRA Adapter (Medical Fine-tuning)
    ↓
Generated Response
```

### Generation Parameters
- Temperature: 0.7
- Top-p: 0.95
- Max new tokens: 512
- Do sample: True

### Requirements
- GPU: NVIDIA A10G (24GB) recommended
- Python: 3.9+
- CUDA: 11.8+

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Meta AI for Llama 3.1 base model
- Hugging Face for transformers and PEFT libraries
- Streamlit for the UI framework

## Contact

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ for medical education and research**
