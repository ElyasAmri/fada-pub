# Modular Architecture: Classification → Captioning Evolution

## Design Philosophy
Build a flexible system where components can be swapped/upgraded without rewriting everything.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface Layer                   │
│                 (Gradio → React Later)                   │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│                  Response Generator                      │
│     Phase 1: Template-based                             │
│     Phase 2: LLM Integration                            │
│     Phase 3: End-to-end Captioning                      │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│                  Feature Extractor                       │
│     Shared CNN Backbone (Never Changes)                  │
│     EfficientNet-B0 → 1280-dim features                 │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┬─────────┬──────────┐
        ▼                 ▼         ▼          ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│Classification│ │  Attributes  │ │   Caption    │ │   Future     │
│     Head     │ │     Head     │ │     Head     │ │   VL Head    │
│  (Phase 1)   │ │  (Phase 1)   │ │  (Phase 2)   │ │  (Phase 3)   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Phase 1: Classification-Based Chatbot (Weeks 1-3)

### 1.1 Core Vision Model
```python
class UltrasoundAnalyzer(nn.Module):
    def __init__(self):
        # FROZEN BACKBONE - Never retrain this
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        self.feature_dim = 1280
        
        # Swappable heads
        self.heads = nn.ModuleDict({
            'organ': ClassificationHead(self.feature_dim, 5),
            'abnormality': ClassificationHead(self.feature_dim, 2),
            'quality': ClassificationHead(self.feature_dim, 3),
            'orientation': ClassificationHead(self.feature_dim, 4),
        })
        
        # Feature vector for future captioning
        self.feature_projector = nn.Linear(self.feature_dim, 512)
    
    def forward(self, x, task='all'):
        features = self.backbone(x)  # Shared features
        
        outputs = {}
        if task == 'all':
            for name, head in self.heads.items():
                outputs[name] = head(features)
        else:
            outputs[task] = self.heads[task](features)
        
        # Always output features for future use
        outputs['features'] = self.feature_projector(features)
        return outputs
```

### 1.2 Response Generator V1
```python
class ResponseGenerator:
    def __init__(self, mode='template'):
        self.mode = mode
        self.templates = load_templates()
        
    def generate(self, predictions, user_query=None):
        if self.mode == 'template':
            return self._template_response(predictions)
        elif self.mode == 'llm':
            return self._llm_response(predictions, user_query)
        elif self.mode == 'caption':  # Future
            return self._caption_response(predictions)
    
    def _template_response(self, pred):
        template = self.templates[pred['organ']][pred['abnormality']]
        return template.format(**pred)
```

## Phase 2: Add Captioning Module (Weeks 4-5)

### 2.1 Caption Head Addition
```python
class CaptionHead(nn.Module):
    def __init__(self, feature_dim=512, vocab_size=5000):
        super().__init__()
        # Transformer decoder for captioning
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(512, 8), 
            num_layers=3
        )
        self.vocab_embedding = nn.Embedding(vocab_size, 512)
        self.output_projection = nn.Linear(512, vocab_size)
    
    def forward(self, image_features, text_tokens=None):
        if self.training:
            # Teacher forcing during training
            text_embeds = self.vocab_embedding(text_tokens)
            output = self.transformer(text_embeds, image_features)
            return self.output_projection(output)
        else:
            # Autoregressive generation during inference
            return self.generate(image_features)
```

### 2.2 Dual-Mode Response Generator
```python
class HybridResponseGenerator:
    def __init__(self, vision_model, caption_model=None):
        self.vision_model = vision_model
        self.caption_model = caption_model
        
    def generate_response(self, image, mode='hybrid'):
        # Extract features (shared across all modes)
        features = self.vision_model.backbone(image)
        
        if mode == 'classification':
            # Current approach
            predictions = self.vision_model(image)
            response = self.create_structured_response(predictions)
            
        elif mode == 'caption' and self.caption_model:
            # Future approach
            caption = self.caption_model.generate(features)
            response = caption
            
        elif mode == 'hybrid':
            # Best of both worlds
            predictions = self.vision_model(image)
            if self.caption_model:
                caption = self.caption_model.generate(features)
                response = self.merge_outputs(predictions, caption)
            else:
                response = self.create_structured_response(predictions)
        
        return response
```

## Phase 3: Vision-Language Integration (Future)

### 3.1 Adapter for Pre-trained VLMs
```python
class VLMAdapter(nn.Module):
    """Adapter to connect our backbone to pre-trained VLMs"""
    def __init__(self, backbone, vlm_model='blip'):
        super().__init__()
        self.backbone = backbone  # Our trained EfficientNet
        self.adapter = nn.Linear(1280, 768)  # Match VLM input dim
        self.vlm = load_pretrained_vlm(vlm_model)
        
    def forward(self, image, text=None):
        # Use our features
        features = self.backbone(image)
        adapted_features = self.adapter(features)
        
        # Pass to VLM
        output = self.vlm(adapted_features, text)
        return output
```

## Data Preparation Strategy

### Structured Annotations → Multiple Formats
```python
class FlexibleDataset(Dataset):
    def __init__(self, annotations, mode='classification'):
        self.annotations = annotations
        self.mode = mode
        
    def __getitem__(self, idx):
        image = load_image(self.annotations[idx]['path'])
        
        if self.mode == 'classification':
            # Current: Return class labels
            labels = {
                'organ': self.annotations[idx]['organ'],
                'abnormality': self.annotations[idx]['abnormality']
            }
            return image, labels
            
        elif self.mode == 'captioning':
            # Future: Convert annotations to captions
            caption = self.create_caption(self.annotations[idx])
            return image, caption
            
        elif self.mode == 'multi_task':
            # Both: Return everything
            labels = {...}
            caption = self.create_caption(...)
            return image, labels, caption
    
    def create_caption(self, annotation):
        """Convert structured annotation to natural language"""
        caption = f"{annotation['organ']} ultrasound showing "
        caption += f"{annotation['structures']}. "
        if annotation['abnormality']:
            caption += f"Note: {annotation['findings']}."
        return caption
```

## Training Pipeline

### Progressive Training Strategy
```python
def train_progressive(model, dataset):
    # Stage 1: Classification only (Weeks 1-2)
    dataset.mode = 'classification'
    train_classification_heads(model, dataset, epochs=50)
    
    # Stage 2: Add caption generation (Week 3)
    dataset.mode = 'multi_task'
    model.add_caption_head()
    train_with_captions(model, dataset, epochs=30, freeze_backbone=True)
    
    # Stage 3: Fine-tune everything (Week 4)
    train_end_to_end(model, dataset, epochs=20, freeze_backbone=False)
```

## Web Interface Design

### Modular API
```python
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

app = FastAPI()

class AnalysisMode(BaseModel):
    mode: str = 'classification'  # 'classification', 'caption', 'hybrid'

@app.post("/analyze")
async def analyze_ultrasound(
    file: UploadFile, 
    mode: AnalysisMode,
    user_query: str = None
):
    image = load_image(file)
    
    # Use current best model based on mode
    if mode.mode == 'classification':
        result = classification_pipeline(image)
    elif mode.mode == 'caption' and caption_model_exists():
        result = caption_pipeline(image)
    else:
        result = hybrid_pipeline(image)
    
    # Generate conversational response
    response = generate_chat_response(result, user_query)
    
    return {
        "analysis": result,
        "message": response,
        "confidence": result.get('confidence', 0.0)
    }
```

### Gradio Interface (Supports Both Modes)
```python
import gradio as gr

def analyze_image(image, mode, message, history):
    # Backend handles mode switching
    result = api_call(image, mode=mode, query=message)
    return result['message']

with gr.Blocks() as demo:
    gr.Markdown("# Ultrasound Analysis Chatbot")
    
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            mode = gr.Radio(
                ["Classification", "Caption", "Hybrid"],
                value="Classification",
                label="Analysis Mode"
            )
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask about the image")
            
    msg.submit(analyze_image, [image, mode, msg, chatbot], [chatbot])
```

## Configuration Management

### config.yaml
```yaml
model:
  backbone: efficientnet_b0
  feature_dim: 1280
  projection_dim: 512
  
  heads:
    classification:
      enabled: true
      classes: 
        organ: 5
        abnormality: 2
        quality: 3
    
    caption:
      enabled: false  # Enable in Phase 2
      vocab_size: 5000
      max_length: 100
      
response:
  mode: template  # template, llm, caption
  llm_api: openai  # openai, anthropic, local
  
training:
  phase: 1  # 1: classification, 2: caption, 3: end-to-end
  freeze_backbone: false
  learning_rate: 1e-4
```

## Migration Path

### Phase 1 → Phase 2 Checklist
- [ ] Keep backbone weights frozen
- [ ] Add caption head to existing model
- [ ] Convert annotations to captions
- [ ] Add caption loss to training
- [ ] Update API to support caption mode
- [ ] Update UI to show captions

### No Breaking Changes Required
- Same backbone = same feature extraction
- Classification heads remain functional
- API versioning for compatibility
- Progressive enhancement of UI

## Directory Structure
```
fada-v3/
├── src/
│   ├── models/
│   │   ├── backbone.py        # Shared CNN (never changes)
│   │   ├── heads.py           # All task-specific heads
│   │   ├── unified_model.py   # Combines everything
│   │   └── vlm_adapter.py     # Future VLM integration
│   ├── data/
│   │   ├── dataset.py         # Flexible dataset class
│   │   ├── augmentation.py    
│   │   └── caption_generator.py
│   ├── training/
│   │   ├── classification.py  # Phase 1
│   │   ├── captioning.py      # Phase 2
│   │   └── multi_task.py      # Phase 3
│   └── api/
│       ├── inference.py       # Mode-agnostic inference
│       └── response_gen.py    # Handles all response types
├── configs/
│   ├── phase1_classification.yaml
│   ├── phase2_captioning.yaml
│   └── phase3_hybrid.yaml
└── web/
    ├── gradio_demo.py
    └── static/
```

## Key Design Principles

1. **Shared Backbone**: Train once, use everywhere
2. **Modular Heads**: Add capabilities without retraining
3. **Progressive Training**: Build on what works
4. **API Versioning**: Never break existing functionality
5. **Config-Driven**: Switch modes via configuration

This architecture ensures smooth evolution from classification to full captioning!