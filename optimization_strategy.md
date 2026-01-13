# ML Model Inference Optimization Strategy

**Project:** Federated Pneumonia Detection System  
**Current Performance:** ~20ms average inference time  
**Goal:** Optimize inference latency without affecting model accuracy  
**Date:** 2026-01-13

---

## Table of Contents
1. [Professional ML Inference Profiling Framework](#professional-ml-inference-profiling-framework)
2. [Phase 1: Establish Baseline Measurement](#phase-1-establish-baseline-measurement)
3. [Phase 2: Stage-by-Stage Profiling](#phase-2-stage-by-stage-profiling)
4. [Phase 3: Bottleneck Identification](#phase-3-bottleneck-identification)
5. [Phase 4: Optimization Strategy Selection](#phase-4-optimization-strategy-selection)
6. [Phase 5: Professional Workflow](#phase-5-professional-workflow)
7. [Data Types for Quantization](#data-types-for-quantization)
8. [Optimization Techniques Comparison](#optimization-techniques-comparison)

---

## Professional ML Inference Profiling Framework

### Overview
The professional approach to ML inference optimization follows a systematic methodology:
1. **Measure** → Establish baseline
2. **Profile** → Identify bottlenecks
3. **Optimize** → Apply targeted fixes
4. **Validate** → Ensure accuracy preserved
5. **Iterate** → Repeat until target achieved

---

## Phase 1: Establish Baseline Measurement

### Step 1.1: End-to-End Timing (Black Box)
Measure TOTAL time without dissecting anything:

**What to measure:**
- Time from: "Raw input received" → "Final output returned"
- Run 1,000+ samples (after 10-50 warmup iterations)
- Track statistics: mean, median, p50, p95, p99, min, max

**Why start here:**
- Establishes baseline for comparison
- Gives target improvement goal (e.g., "need 50% faster")
- Shows real-world performance including all overhead

**Important: Warmup Period**
- First 10-50 inferences are typically slower (GPU initialization, cache loading)
- **Always discard warmup results** from benchmarks
- Only measure after model has "settled"

---

### Step 1.2: Identify Your Pipeline Stages

**Generic ML Inference Pipeline:**
```
Input → [Preprocessing] → [Model Inference] → [Postprocessing] → Output
```

**Examples by Domain:**

| Domain | Preprocessing | Inference | Postprocessing |
|--------|--------------|-----------|----------------|
| **Computer Vision** | Resize/Normalize/ToTensor | CNN Forward Pass | Sigmoid/Softmax/Argmax |
| **NLP/LLMs** | Tokenization | Transformer Forward | Token Sampling/Decoding |
| **Object Detection** | Image Preprocessing | Detector Forward | NMS/Thresholding |
| **Speech** | MFCC/Spectrogram | Acoustic Model | Beam Search/CTC Decode |

**Our Pneumonia Detection:**
```
PIL Image → [Resize/Normalize/ToTensor] → [ResNet Forward Pass] → [Sigmoid/Argmax] → Prediction
```

---

## Phase 2: Stage-by-Stage Profiling (White Box)

### Step 2.1: Manual Instrumentation (Simple but Effective)

**Concept:** Insert timers at each stage boundary

**Generic Approach:**
1. **Timer START** before preprocessing
2. **Timer CHECKPOINT** after preprocessing, before inference
3. **Timer CHECKPOINT** after inference, before postprocessing
4. **Timer END** after postprocessing

**Calculations:**
- Preprocessing time = Checkpoint1 - Start
- Inference time = Checkpoint2 - Checkpoint1
- Postprocessing time = End - Checkpoint2
- Total time = End - Start

**Industry Practice:**
- Run on 100-1,000 samples
- Calculate percentage breakdown
- Example: "Preprocessing: 15%, Inference: 75%, Postprocessing: 10%"

---

### Step 2.2: Automated Profiling Tools (Professional Approach)

Instead of manual timers, use profiling tools that automatically dissect everything:

#### Tool Categories

| Tool Type | Examples | What It Profiles | Best For |
|-----------|----------|------------------|----------|
| **Framework Profilers** | PyTorch Profiler, TF Profiler | Every operation in your model | Deep analysis of model internals |
| **Hardware Profilers** | NVIDIA Nsight, Intel VTune | CPU/GPU utilization, memory bandwidth | Hardware bottlenecks |
| **ONNX/Runtime Profilers** | ONNX Runtime profiler, TensorRT profiler | Optimized model execution | Production inference optimization |
| **General Profilers** | cProfile (Python), perf (Linux) | Entire application including Python overhead | Full-stack analysis |

**Professional workflow uses multiple tools together**

---

### Step 2.3: Statistical Metrics - Why Mean Isn't Enough

| Metric | What It Tells You | Why It Matters |
|--------|------------------|----------------|
| **Mean** | Average performance | Good overall indicator, but hides outliers |
| **Median (p50)** | Typical user experience | More robust to outliers |
| **p95** | 95% of requests faster than this | Production SLA threshold |
| **p99** | 99% of requests faster than this | Worst-case user experience |
| **Min** | Best-case performance | Theoretical limit |
| **Max** | Worst outlier | Indicates instability |

**Example Why This Matters:**
- Mean: 20ms (looks good!)
- p99: 200ms (10x worse - real users experience this!)

In production, those outliers are real user experiences that hurt satisfaction.

---

## Phase 3: Bottleneck Identification

### Step 3.1: Calculate Stage Percentages

After profiling 1,000 samples:

**Example Result:**
```
Total average time: 20ms
- Preprocessing: 3ms (15%)
- Core Inference: 15ms (75%)
- Postprocessing: 2ms (10%)
```

**Decision Rules:**

| Stage Percentage | Priority | Action |
|-----------------|----------|--------|
| **> 50%** | Primary bottleneck | Optimize this FIRST |
| **20-50%** | Secondary bottleneck | Optimize AFTER primary |
| **< 20%** | Low priority | Not worth optimizing yet |

---

### Step 3.2: Detailed Operator-Level Profiling (If Needed)

If inference is the bottleneck, drill deeper:

**For Neural Networks, profilers show:**
- Layer 1 (Conv2d): 2ms
- Layer 2 (BatchNorm): 0.5ms
- Layer 3 (ReLU): 0.1ms
- Layer 4 (Conv2d): 3ms ← **Bottleneck!**
- Layer 5 (MaxPool): 0.8ms
- ...and so on

**This reveals:**
- Which specific layers are slow
- Convolutions vs Attention vs Matrix multiplications
- Memory-bound vs compute-bound operations

---

## Phase 4: Optimization Strategy Selection

### Decision Tree 1: Preprocessing Bottleneck (> 30% of time)

**If preprocessing is slow, your options:**

| Technique | When to Use | Expected Gain | Complexity |
|-----------|-------------|---------------|------------|
| **Move preprocessing to GPU** | Large images, CPU preprocessing | 2-5x faster | Low |
| **Reduce image resolution** | Input larger than model needs | Proportional to size reduction | Low |
| **Optimize data loading** | Reading from disk is slow | 2-10x faster | Medium |
| **Batch preprocessing** | Processing images one-by-one | 2-4x faster | Low |
| **Use optimized libraries** | Using pure Python for transforms | 3-10x faster | Low |
| **Cache preprocessed data** | Same images processed repeatedly | Near-infinite for duplicates | Medium |

**Example for Our Case:**
- Images: 1024×1024 but model uses 224×224
- Preprocessing: resize → normalize → convert
- If this is 30% of time (6ms out of 20ms):
  - Moving to GPU: 6ms → 1-2ms (saves 4-5ms total)
  - Using optimized resize (PIL → OpenCV or GPU): similar gains

---

### Decision Tree 2: Core Inference Bottleneck (> 50% of time)

**First, identify bottleneck type:**

#### Memory-Bound vs Compute-Bound

| Type | Indicators | Optimization Focus |
|------|-----------|-------------------|
| **Memory-bound** | • GPU utilization < 80%<br>• Large tensors moved frequently<br>• Many small operations | Reduce data movement:<br>• Quantization (FP16/INT8)<br>• Operator fusion |
| **Compute-bound** | • GPU utilization > 90%<br>• Heavy operations (large matrix multiplies)<br>• Model very deep/wide | Faster compute:<br>• Better hardware<br>• INT8 quantization<br>• TensorRT |

#### Optimization Techniques by Bottleneck Type

| Bottleneck Type | Optimization Strategy | Expected Gain | Accuracy Impact |
|-----------------|----------------------|---------------|----------------|
| **Memory-bound** | Quantization (FP16/INT8) | 1.5-4x | FP16: <0.1%, INT8: 1-3% |
| **Compute-bound** | Graph optimization, operator fusion | 1.2-3x | 0% |
| **CPU inference (no GPU)** | ONNX Runtime, quantization | 2-10x | Varies |
| **Large model** | Model compression, distillation, pruning | 2-5x | 2-5% |
| **Many operations** | Graph optimization, operator fusion | 1.2-2x | 0% |

---

### Decision Tree 3: Postprocessing Bottleneck (> 30% of time)

**Less common but happens with:**
- Object detection (Non-Maximum Suppression)
- LLMs (beam search, sampling algorithms)
- Instance segmentation

| Technique | When to Use | Expected Gain |
|-----------|-------------|---------------|
| **Vectorize operations** | Using loops instead of NumPy/Tensor ops | 2-10x |
| **Move to GPU** | CPU postprocessing with large tensors | 2-5x |
| **Optimize algorithm** | Suboptimal NMS, beam search implementation | 2-5x |
| **Reduce candidates** | Processing too many outputs | Proportional to reduction |

---

## Phase 5: Professional Workflow (Step-by-Step)

### Week 1: Initial Profiling

**Day 1-2:**
1. Measure end-to-end latency (1,000 samples)
2. Record baseline metrics (mean, p50, p95, p99)
3. Document current performance

**Day 3-4:**
4. Profile with framework profiler (PyTorch Profiler or ONNX Runtime)
5. Generate stage breakdown percentages
6. Create visualization of bottlenecks

**Day 5:**
7. Identify primary bottleneck
8. Research applicable optimization techniques
9. Create optimization plan

---

### Week 1: Decision Point

**Based on profiling results:**

| If This is Slow | Optimize This First | Expected Impact |
|----------------|--------------------|--------------------|
| Preprocessing > 30% | Data pipeline | 20-50% total speedup |
| Inference > 50% | Model execution | 50-80% total speedup |
| Postprocessing > 30% | Output processing | 20-40% total speedup |

---

### Week 2: Apply Optimization

**Day 1-2:**
1. Choose 1-2 techniques from decision tree
2. Implement changes (start with safest options)
3. Document modifications

**Day 3-4:**
4. Re-profile with same methodology
5. Compare new vs baseline metrics
6. Generate before/after comparison

**Day 5:**
7. Measure speedup (old_time / new_time)
8. Validate accuracy unchanged (< 0.5% drop)
9. Document results

---

### Week 2: Validation Checkpoint

**Success Criteria:**
- ✅ Latency improved by target % (e.g., 30%+)
- ✅ Accuracy drop < 0.5%
- ✅ p95/p99 also improved (not just mean)
- ✅ No new errors or edge case failures

**If Successful:**
- Commit changes
- Iterate on next bottleneck
- Continue optimization cycle

**If Unsuccessful:**
- Revert changes
- Try different approach from decision tree
- Consult with team/experts

---

### Week 3+: Iterate

**Iterative Process:**
1. Re-profile optimized system
2. Identify remaining bottlenecks
3. Apply next optimization
4. Validate
5. Repeat until:
   - Target latency achieved, OR
   - Diminishing returns (< 10% improvement per iteration)

---

## Data Types for Quantization

### FP32 (Float32) - The Original

**Characteristics:**
- **Size:** 32 bits per number
- **Range:** ±3.4 × 10³⁸
- **Precision:** ~7 decimal digits
- **Use case:** Default training format, highest precision

**Example:** 0.87654321 stored precisely

---

### FP16 (Float16) - Half Precision

**Characteristics:**
- **Size:** 16 bits per number (2x smaller)
- **Range:** ±65,504
- **Precision:** ~3-4 decimal digits
- **Use case:** Inference optimization on modern GPUs

**Example:** 0.87654321 → stored as 0.8765 (slight rounding)

**Trade-offs:**
- ✅ 2x smaller model size
- ✅ 1.5-2x faster on modern GPUs (Tensor Cores)
- ✅ **Very safe for medical imaging** - minimal accuracy loss
- ⚠️ Smaller range (can cause overflow in extreme cases)

**Why It Works Well:**
- Model weights typically in range -5 to +5
- Probabilities are 0-1
- Don't need 7 decimal places - 3-4 is sufficient

**Real-world Results:**
- Accuracy drop: < 0.1% typically
- Recall drop: < 0.1% typically
- Speed gain: 1.5-2x

---

### INT8 (8-bit Integer) - Aggressive Quantization

**Characteristics:**
- **Size:** 8 bits per number (4x smaller than FP32)
- **Range:** -128 to +127 (signed) or 0 to 255 (unsigned)
- **Precision:** None - integers only!

**How It Works:**
- Original weight: 0.87654321 (FP32)
- Quantization maps to: 223 (INT8)
- Uses "scale factor" to convert back: 223 × scale ≈ 0.876

**Trade-offs:**
- ✅ 4x smaller model size
- ✅ 2-4x faster inference (integer math is cheap)
- ✅ Lower memory bandwidth requirements
- ❌ **Accuracy loss: 1-3%** (significant!)
- ❌ Needs calibration data to find best scale factors

**Why It Loses Accuracy:**
- Many weights round to same integer
- Example: 0.001, 0.002, 0.003 might all become 0
- Fine gradations are lost
- Accumulated error across layers

**Calibration Required for Static INT8:**
- Need ~500-1,000 representative images
- Model runs on these to figure out typical activation ranges
- Determines optimal scale factors per layer
- **Without calibration:** accuracy drops 5-10%
- **With calibration:** accuracy drops 1-3%

---

### Comparison Table

| Format | Bits | Model Size | Speed | Accuracy Impact | Complexity | Medical Imaging Safe? |
|--------|------|------------|-------|----------------|------------|----------------------|
| **FP32** | 32 | 100% (baseline) | 1x | 0% (baseline) | Easy | ✅ Yes (default) |
| **FP16** | 16 | 50% | 1.5-2x | < 0.1% | Easy | ✅ Yes (very safe) |
| **INT8** | 8 | 25% | 2-4x | 1-3% | Hard (calibration) | ⚠️ Needs validation |

---

## Optimization Techniques Comparison

### Full Comparison Matrix

| Technique | What It Does | Speed Gain | Accuracy Impact | Effort | Hardware Needs | Our Recommendation |
|-----------|--------------|------------|----------------|--------|----------------|-------------------|
| **Graph Optimization** | Fuse operations, eliminate redundant nodes | 1.2-1.5x | **0%** | **Low** | Any | ✅ **Do First** |
| **Execution Provider** | Use GPU/TensorRT instead of CPU | 1.5-3x | **0%** | **Low** | GPU | ✅ **Do First** |
| **FP16 Quantization** | Use 16-bit floats | 1.5-2x | < 0.1% | **Low** | Modern GPU | ✅ **Safe Second** |
| **INT8 Quantization** | Use 8-bit integers | 2-4x | 1-3% | **Medium** | CPU/GPU with INT8 | ⚠️ **Validate First** |
| **Unstructured Pruning** | Zero out individual weights | 1.1-1.3x* | 0-2% | **High** | Sparse kernels | ❌ **Not Recommended** |
| **Structured Pruning** | Remove entire filters | 1.3-2x | 2-5% | **Very High** | Any | ❌ **Not Recommended** |
| **Knowledge Distillation** | Train smaller student model | 2-5x | 1-5% | **Very High** | Any | ❌ **Too Much Work** |

*Unstructured pruning needs sparse matrix support - otherwise minimal gain

---

## Recommended Optimization Path for Our Use Case

### Context
- **Domain:** Medical imaging (accuracy is CRITICAL)
- **Current:** 20ms per image
- **Goal:** Faster without harming recall
- **Constraint:** Cannot compromise diagnostic accuracy

---

### Tier 1: Safe & Easy (Do These First) ✅

**Week 1-2:**

1. **ONNX Graph Optimization**
   - Zero accuracy loss
   - 20-30% faster
   - 1-2 days implementation
   - No risk

2. **Execution Provider Selection**
   - Zero accuracy loss
   - 50-100% faster (if GPU available)
   - 1-2 days implementation
   - No risk

**Expected Result:** 20ms → 10-13ms

**Validation:**
- Run 1,000 test samples
- Compare predictions exactly
- Ensure 100% match

---

### Tier 2: Low Risk (If Tier 1 Isn't Enough) ⚠️

**Week 3-4:**

3. **FP16 Quantization**
   - < 0.1% accuracy loss
   - 50-100% faster on modern GPUs
   - 3-5 days including validation
   - Very low risk for medical imaging

**Expected Result:** 10-13ms → 5-8ms

**Validation:**
- Full test set (10,000+ images)
- Measure accuracy, precision, recall, F1, AUROC
- Threshold: < 0.5% drop on any metric
- Special focus on recall (false negatives critical)

---

### Tier 3: Medium Risk (Only if Desperate) ⚠️⚠️

**Week 5-8:**

4. **INT8 Dynamic Quantization**
   - 1-3% accuracy loss (MUST validate!)
   - 2-3x faster
   - 1-2 weeks including calibration and validation
   - Requires extensive medical imaging validation

**Expected Result:** 5-8ms → 2-4ms

**Validation:**
- Calibration on 1,000 representative images
- Full test set evaluation
- Compare confusion matrices
- Clinical review of changed predictions
- Regulatory compliance check

---

### Tier 4: High Risk / High Effort (NOT Recommended) ❌

**Do NOT pursue for medical imaging:**

5. **Structured Pruning**
   - 2-5% accuracy loss unacceptable
   - Requires retraining
   - 4-8 weeks of work
   - Regulatory re-approval needed

6. **Knowledge Distillation**
   - Unpredictable accuracy
   - 4-8 weeks training time
   - New model = new validation cycle

---

## Why NOT Pruning for Our Case?

### Reasons to Avoid Pruning

1. **Medical imaging = accuracy cannot drop 2-5%**
   - False negatives in pneumonia detection are dangerous
   - Regulatory requirements likely strict
   - Patient safety is paramount

2. **Already fast (20ms)**
   - Not a critical bottleneck
   - Pruning makes sense for 500ms+ models
   - Better ROI from safer optimizations

3. **Pruning requires retraining**
   - Model is already trained and validated
   - Retraining = new validation cycle
   - Regulatory re-approval expensive

4. **Better alternatives exist**
   - Graph optimization + FP16 = 3-4x speedup
   - Near-zero accuracy risk
   - Much less effort

---

## Next Steps

### Immediate Actions (This Week)

1. ✅ **Understand concepts** (DONE - this document)
2. ⬜ **Benchmark current PyTorch model**
   - Run on 1,000 test images
   - Record baseline: mean, p50, p95, p99
   - Profile stage breakdown

3. ⬜ **Research export to ONNX**
   - Study PyTorch → ONNX conversion
   - Understand dynamic axes
   - Learn validation techniques

### Short-term (Next 2 Weeks)

4. ⬜ **Export model to ONNX**
5. ⬜ **Enable graph optimizations**
6. ⬜ **Benchmark ONNX model**
7. ⬜ **Validate accuracy maintained**

### Medium-term (Next Month)

8. ⬜ **Apply FP16 quantization** (if needed)
9. ⬜ **Full validation pipeline**
10. ⬜ **A/B test in production**

### Long-term (If Still Needed)

11. ⬜ **Evaluate INT8 quantization**
12. ⬜ **Clinical validation**
13. ⬜ **Regulatory compliance check**

---

## Resources & Tools

### Profiling Tools to Explore

- **PyTorch Profiler** - Built-in, easy to use
- **ONNX Runtime Profiler** - For ONNX optimization
- **NVIDIA Nsight Systems** - GPU profiling (if using NVIDIA)
- **TensorBoard** - Visualization of profiling results

### Learning Resources

- ONNX Runtime Documentation
- PyTorch ONNX Export Guide
- TensorRT Best Practices
- Model Optimization Whitepapers

---

## Conclusion

**Key Takeaways:**

1. **Always profile before optimizing** - measure, don't guess
2. **Start with safe optimizations** - graph optimization, FP16
3. **Validate thoroughly** - accuracy is non-negotiable in medical imaging
4. **Use percentiles, not just mean** - p95/p99 matter for UX
5. **Iterate systematically** - one change at a time

**Expected Final Result:**
- Start: 20ms (PyTorch FP32)
- After Tier 1: 10-13ms (ONNX + Graph Opt + GPU)
- After Tier 2: 5-8ms (+ FP16)
- Total speedup: **2.5-4x faster** with < 0.1% accuracy impact

**This is achievable within 2-4 weeks of focused effort.**

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-13  
**Status:** Learning Phase - No implementation yet
