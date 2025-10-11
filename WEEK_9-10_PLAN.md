# Week 9-10 Implementation Plan: Optional LoRA Fine-Tuning

**Date**: October 11, 2025  
**Status**: 📋 PLANNING  
**Progress**: Preparing for 9-10/23 weeks (Part 1 completion)

---

## Executive Summary

Week 9-10 will implement **Optional LoRA (Low-Rank Adaptation) Fine-Tuning**, completing **Part 1** of the 23-week implementation plan. This will enable Greta to:

1. **Collect training data** from user interactions
2. **Fine-tune local models** using LoRA for efficiency
3. **Manage fine-tuned adapters** alongside base models
4. **Evaluate fine-tuning quality** with metrics
5. **Deploy fine-tuned models** seamlessly

**Key Decision**: LoRA is marked as "Optional" in the plan. We should discuss whether to:
- **Option A**: Implement full LoRA fine-tuning (2 weeks)
- **Option B**: Implement basic training data collection only (1 week)
- **Option C**: Skip and move to Part 2 (PAI Integration)

---

## Background: What is LoRA?

### LoRA (Low-Rank Adaptation)
- **Purpose**: Fine-tune large language models efficiently
- **Method**: Add small trainable matrices to frozen model weights
- **Benefits**:
  - 10,000x fewer trainable parameters
  - 3x less memory usage
  - Fast training (hours vs days)
  - Multiple adapters for one base model
  - Easy to swap/merge adapters

### Why LoRA for Greta?
1. **Personalization**: Adapt to user's writing style, preferences, domain knowledge
2. **Efficiency**: Fine-tune 7B model on consumer hardware (M1/M2 Mac)
3. **Flexibility**: Multiple adapters for different tasks (coding, writing, analysis)
4. **Integration**: Works with llama.cpp (Week 1) via adapter loading

---

## Implementation Options

### Option A: Full LoRA Implementation (Recommended)

**Duration**: 2 weeks  
**Complexity**: High  
**Value**: High (enables true personalization)

**Components**:
1. Training data collection pipeline
2. LoRA fine-tuning service (using PEFT library)
3. Adapter management system
4. Fine-tuning monitoring and evaluation
5. Adapter deployment to llama.cpp

**Technologies**:
- **PEFT** (Parameter-Efficient Fine-Tuning) by HuggingFace
- **transformers** library
- **datasets** library
- **torch** (PyTorch)
- **wandb** (optional, for monitoring)

**Pros**:
- Complete personalization capability
- Multiple adapters for different tasks
- Efficient training on local hardware
- Completes Part 1 as designed

**Cons**:
- Complex implementation
- Requires GPU (Metal on Mac)
- Training time (hours per adapter)
- Storage for adapters (~100-500MB each)

---

### Option B: Training Data Collection Only

**Duration**: 1 week  
**Complexity**: Medium  
**Value**: Medium (foundation for future fine-tuning)

**Components**:
1. Training data collection pipeline
2. Data quality filtering
3. Dataset management
4. Export to standard formats (JSONL, Parquet)

**Pros**:
- Simpler implementation
- No GPU requirements
- Builds foundation for future fine-tuning
- Can add LoRA later

**Cons**:
- No actual fine-tuning capability
- Incomplete Part 1 implementation
- Deferred value

---

### Option C: Skip to Part 2 (PAI Integration)

**Duration**: 0 weeks  
**Complexity**: N/A  
**Value**: Depends on Part 2 priorities

**Rationale**:
- LoRA is marked "Optional" in plan
- Part 2 (PAI Integration) may be higher priority
- Can return to LoRA later if needed

**Pros**:
- Faster progress to Part 2
- Focus on PAI features
- LoRA can be added later

**Cons**:
- Part 1 incomplete
- No personalization capability
- May need to revisit later

---

## Recommended Approach: Option A (Full LoRA)

### Rationale
1. **Completes Part 1**: Delivers all 10 weeks as designed
2. **High Value**: Personalization is a key differentiator for PAI
3. **Feasible**: M1/M2 Mac can handle LoRA training
4. **Foundation**: Enables advanced PAI features in Part 2

### Implementation Plan (2 Weeks)

#### Week 9: Training Data Collection & LoRA Setup

**Day 1-2: Training Data Collection Pipeline**
- Create `training_data_service.py`
- Collect conversation data from Week 4 service
- Filter quality data (length, coherence, usefulness)
- Format for fine-tuning (instruction-response pairs)
- Store in MongoDB + export to JSONL

**Day 3-4: LoRA Fine-Tuning Service Setup**
- Install PEFT, transformers, torch
- Create `lora_service.py`
- Implement LoRA configuration (rank, alpha, dropout)
- Set up training pipeline (data loading, tokenization)
- Configure for Metal GPU acceleration

**Day 5-7: Basic Fine-Tuning Implementation**
- Implement training loop
- Add checkpointing
- Basic evaluation metrics (loss, perplexity)
- Test with small dataset

#### Week 10: Adapter Management & Deployment

**Day 1-3: Adapter Management**
- Create `adapter_registry.py`
- Adapter storage and versioning
- Adapter metadata (base model, task, metrics)
- Adapter loading/unloading
- Multiple adapter support

**Day 4-5: Integration with llama.cpp**
- Export LoRA adapters to GGUF format
- Integrate with Week 1 llama.cpp service
- Dynamic adapter loading
- Adapter switching API

**Day 6-7: Monitoring, Evaluation & Testing**
- Training monitoring (loss curves, metrics)
- Evaluation suite (perplexity, task-specific metrics)
- A/B testing framework (base vs fine-tuned)
- Comprehensive test suite
- Documentation

---

## Technical Architecture

### Component 1: Training Data Collection

```python
class TrainingDataService:
    """Collect and prepare training data from conversations"""
    
    async def collect_from_conversations(
        self,
        min_quality_score: float = 0.7,
        min_length: int = 50,
        max_samples: int = 10000
    ) -> Dataset
    
    async def filter_quality(self, data: List[Dict]) -> List[Dict]
    
    async def format_for_training(
        self,
        data: List[Dict],
        format: str = "instruction"  # instruction, chat, completion
    ) -> Dataset
    
    async def export_dataset(
        self,
        dataset: Dataset,
        output_path: str,
        format: str = "jsonl"  # jsonl, parquet, csv
    )
```

### Component 2: LoRA Fine-Tuning Service

```python
class LoRAService:
    """LoRA fine-tuning service"""
    
    async def create_lora_config(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: List[str] = ["q_proj", "v_proj"]
    ) -> LoraConfig
    
    async def fine_tune(
        self,
        base_model: str,
        dataset: Dataset,
        config: LoraConfig,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 4
    ) -> TrainingResult
    
    async def evaluate(
        self,
        model: str,
        adapter: str,
        eval_dataset: Dataset
    ) -> EvaluationMetrics
```

### Component 3: Adapter Registry

```python
class AdapterRegistry:
    """Manage LoRA adapters"""
    
    async def register_adapter(
        self,
        adapter_id: str,
        base_model: str,
        task: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any]
    )
    
    async def load_adapter(self, adapter_id: str) -> Adapter
    
    async def list_adapters(
        self,
        base_model: Optional[str] = None,
        task: Optional[str] = None
    ) -> List[AdapterInfo]
    
    async def delete_adapter(self, adapter_id: str)
```

### Component 4: Integration with llama.cpp

```python
# Extend Week 1 LLM Service
class LLMService:
    async def load_adapter(self, adapter_path: str):
        """Load LoRA adapter into llama.cpp"""
        
    async def unload_adapter(self):
        """Unload current adapter"""
        
    async def list_loaded_adapters(self) -> List[str]:
        """List currently loaded adapters"""
```

---

## Integration Points

### Week 1: LLM Service
- Load LoRA adapters into llama.cpp
- Adapter-aware generation
- Adapter switching API

### Week 2: Model Management
- Store adapters alongside models
- Adapter versioning
- Adapter metadata in registry

### Week 4: Conversation Service
- Source of training data
- Conversation quality scoring
- User feedback collection

### Week 5-6: RAG Service
- Domain-specific fine-tuning data
- Document-aware training samples
- Knowledge-grounded responses

### Week 7-8: Advanced AutoGen
- Agent-specific adapters
- Multi-agent fine-tuning
- Specialized agent training

---

## File Structure

```
Greta/
├── backend/
│   ├── services/
│   │   ├── training_data_service.py      # NEW: Training data collection
│   │   ├── lora_service.py                # NEW: LoRA fine-tuning
│   │   ├── adapter_registry.py            # NEW: Adapter management
│   │   └── llm_service.py                 # MODIFIED: Add adapter loading
│   └── models/
│       ├── training_models.py             # NEW: Training data schemas
│       └── adapter_models.py              # NEW: Adapter schemas
├── data/
│   ├── training/                          # NEW: Training datasets
│   ├── adapters/                          # NEW: LoRA adapters
│   └── checkpoints/                       # NEW: Training checkpoints
├── test_week9_10.py                       # NEW: Test suite
└── WEEK_9-10_COMPLETE.md                  # NEW: Completion report
```

---

## Dependencies

### New Python Packages

```bash
# Core fine-tuning
pip install peft==0.7.0              # LoRA implementation
pip install transformers==4.35.0     # HuggingFace transformers
pip install datasets==2.14.0         # Dataset management
pip install accelerate==0.24.0       # Training acceleration

# PyTorch (Metal GPU support for Mac)
pip install torch==2.1.0             # PyTorch
pip install torchvision==0.16.0      # Vision utilities

# Monitoring (optional)
pip install wandb==0.16.0            # Training monitoring
pip install tensorboard==2.15.0      # TensorBoard logging

# Utilities
pip install sentencepiece==0.1.99    # Tokenization
pip install protobuf==4.25.0         # Protocol buffers
```

**Total Size**: ~5-7 GB (PyTorch is large)

---

## Success Criteria

### Week 9 Success Criteria
- ✅ Training data collection pipeline working
- ✅ Quality filtering implemented
- ✅ Dataset export to JSONL/Parquet
- ✅ LoRA service initialized
- ✅ Basic fine-tuning working on test dataset
- ✅ Training metrics collected

### Week 10 Success Criteria
- ✅ Adapter registry implemented
- ✅ Adapter storage and versioning working
- ✅ llama.cpp adapter loading working
- ✅ Adapter switching API functional
- ✅ Evaluation metrics implemented
- ✅ Test suite passing (100% target)
- ✅ Documentation complete

### Part 1 Completion Criteria
- ✅ All 10 weeks implemented (Weeks 1-10)
- ✅ All core LLM functionality complete
- ✅ Ready for Part 2 (PAI Integration)

---

## Risks and Mitigations

### Risk 1: GPU Memory Limitations
**Impact**: High  
**Probability**: Medium  
**Mitigation**:
- Use smaller LoRA rank (r=4 or r=8)
- Reduce batch size
- Use gradient checkpointing
- 4-bit quantization (QLoRA)

### Risk 2: Training Time
**Impact**: Medium  
**Probability**: High  
**Mitigation**:
- Start with small datasets (1000-5000 samples)
- Use efficient LoRA config (low rank)
- Parallel training if multiple GPUs available

### Risk 3: llama.cpp Adapter Compatibility
**Impact**: High  
**Probability**: Medium  
**Mitigation**:
- Test adapter export early
- Use llama.cpp-compatible formats
- Fallback to base model if adapter fails

### Risk 4: Training Data Quality
**Impact**: High  
**Probability**: Medium  
**Mitigation**:
- Implement quality filtering
- Manual review of samples
- Start with curated datasets
- Iterative improvement

---

## Timeline

### Week 9 (Days 1-7)
- Day 1-2: Training data collection
- Day 3-4: LoRA service setup
- Day 5-7: Basic fine-tuning

### Week 10 (Days 8-14)
- Day 8-10: Adapter management
- Day 11-12: llama.cpp integration
- Day 13-14: Testing and documentation

**Total**: 14 days (2 weeks)

---

## Next Steps

### Decision Required
**Question**: Which option should we implement?
- **Option A**: Full LoRA (2 weeks) - Recommended
- **Option B**: Data collection only (1 week)
- **Option C**: Skip to Part 2

### If Option A (Full LoRA) Selected:
1. Install dependencies (PEFT, transformers, torch)
2. Create training_data_service.py
3. Create lora_service.py
4. Create adapter_registry.py
5. Modify llm_service.py for adapter loading
6. Create test suite
7. Test on small dataset
8. Document and complete Part 1

### If Option B (Data Collection) Selected:
1. Create training_data_service.py
2. Implement quality filtering
3. Dataset export functionality
4. Mark Part 1 as "partially complete"
5. Move to Part 2

### If Option C (Skip) Selected:
1. Mark Week 9-10 as "deferred"
2. Begin Part 2 planning
3. Return to LoRA later if needed

---

## Recommendation

**I recommend Option A (Full LoRA Implementation)** for the following reasons:

1. **Completes Part 1**: Delivers all 10 weeks as designed in the 23-week plan
2. **High Value**: Personalization is crucial for a Personal AI system
3. **Feasible**: M1/M2 Mac can handle LoRA training with proper configuration
4. **Foundation**: Enables advanced personalization features in Part 2
5. **Competitive Advantage**: Fine-tuned local models differentiate Greta from cloud-only solutions

**Estimated Effort**: 2 weeks (14 days)  
**Expected Outcome**: Production-ready LoRA fine-tuning system integrated with all previous weeks

---

*Plan created: October 11, 2025*  
*Status: Awaiting decision on implementation option*

