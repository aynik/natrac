## Differentiable Neural ATRAC1 Codec

**General Principles for the Coding Agent:**
*   **Incremental Development:** Complete and test each step thoroughly before moving to the next.
*   **Version Control:** Commit frequently with clear messages.
*   **Modular Design:** Keep components (NN modules, data loaders, utility functions) well-separated.
*   **Testing is Key:** Follow the integrated testing steps diligently.
*   **Documentation:** Add comments and docstrings as you code.
*   **Experiment Tracking:** Log all experiments, parameters, and results.

---

**Phase 0: Foundations & Data Preparation**

*   [ ] **P0.1: Environment Setup**
    *   [ ] Create virtual environment
    *   [ ] Install PyTorch
    *   [ ] Configure GPU acceleration (CUDA, Dual GPU RTX 3090 24GB)
    *   [ ] Install `pytrac` Python bindings for atracdenc from https://github.com/aynik/pytrac
    *   [ ] Install `numpy`, `scipy`
    *   [ ] Install `librosa` (for audio analysis/metrics)
    *   [ ] Install `matplotlib` (for visualization)
    *   [ ] Choose and set up an experiment tracking tool (e.g., MLflow, Weights & Biases, or simple CSV logging)
*   [ ] **P0.2: Dataset Curation**
    *   [ ] Gather/select a diverse audio dataset (music, speech, sound effects).
    *   [ ] Pre-process all audio to be 44.1 kHz.
    *   [ ] Decide on a primary channel format (e.g., mono first, then stereo). If stereo, ensure pairs are maintained.
    *   [ ] Split dataset into training, validation, and test sets.
*   [ ] **P0.3: Data Extraction Script Development**
    *   [ ] Create a Python script based on `examples/neural_training_example.py`.
    *   [ ] Function to process a single audio file:
        *   [ ] Load WAV file.
        *   [ ] Iterate through it frame by frame (512 samples).
        *   [ ] For each frame:
            *   [ ] Use `atracdenc.FrameProcessor` to get `IntermediateData` (Encoder ground truth).
            *   [ ] Use `atracdenc.FrameDecoder.decode_from_intermediate()` with the `IntermediateData` to get `DecoderIntermediateData` (Decoder ground truth & consistency check PCM).
            *   [ ] Collect all relevant fields from both `IntermediateData` and `DecoderIntermediateData`.
*   [ ] **P0.4: Batch Data Extraction and Storage**
    *   [ ] Run the P0.3 script on the entire training, validation, and test datasets.
    *   [ ] Store the extracted frame-wise data efficiently (e.g., HDF5 files, one per split, or a structured directory of NumPy arrays).
*   [ ] **P0.5: PyTorch/TensorFlow DataLoader Implementation**
    *   [ ] Create a custom `Dataset` class to load the pre-extracted data from P0.4.
    *   [ ] Implement `__len__` and `__getitem__` to return a dictionary of all necessary intermediate tensors for a given frame.
    *   [ ] Implement `DataLoader` instances for training, validation, and testing.
*   [ ] **P0.6: Utility Functions Implementation**
    *   [ ] Implement normalization/denormalization functions for PCM, QMF, and MDCT data (based on statistics from the dataset).
    *   [ ] Implement core loss functions: MSE, L1.
    *   [ ] Implement STFT-based loss (magnitude and/or phase).
    *   [ ] Implement Mel-spectrogram loss.
    *   [ ] Implement audio quality metric functions: SNR, LSD (Log-Spectral Distance).
*   [ ] **P0.7: Data Pipeline Sanity Check (T0)**
    *   [ ] Load a few samples using the DataLoader.
    *   [ ] Verify shapes and data types.
    *   [ ] For a sample, check that `IntermediateData.pcm_input` when processed by `decode_from_intermediate(IntermediateData)` yields a `DecoderIntermediateData.pcm_output` that is audibly similar (it's a lossy C++ round trip).
    *   [ ] Calculate and log statistics (min, max, mean, std) of all key intermediate data fields across a subset of the training data.

---

**Phase 1: Neural Encoder Modules (Mimicking `FrameProcessor`)**

*   **General Sub-steps for each Encoder Module (1.x):**
    *   [ ] **(a) Define NN Architecture:** Choose appropriate layers (CNN, MLP, Transformer, LSTM/GRU).
    *   [ ] **(b) Implement Training Loop:**
        *   [ ] Load specific input/output pairs from the DataLoader for this module.
        *   [ ] Apply normalization.
        *   [ ] Forward pass, calculate loss (e.g., MSE/L1 against C++ output).
        *   [ ] Backward pass and optimizer step.
    *   [ ] **(c) Implement Evaluation:** Calculate metrics (MSE, L1, SNR, accuracy for classification) on the validation set.
    *   [ ] **(d) Implement Visualization:** Plot NN predictions vs. C++ ground truth for a few examples.

*   [ ] **P1.1: NN_PCM_to_QMF Module**
    *   [ ] P1.1.a: Define architecture (Input: PCM frame; Output: QMF bands `qmf_low, qmf_mid, qmf_hi`).
    *   [ ] P1.1.b: Implement training loop (GT: `IntermediateData.pcm_input` -> `IntermediateData.qmf_low/mid/hi`).
    *   [ ] P1.1.c: Implement evaluation.
    *   [ ] P1.1.d: Visualize QMF band outputs.
*   [ ] **P1.2: NN_Transient_Detector (Window Mask Prediction) Module**
    *   [ ] P1.2.a: Define architecture (Input: QMF bands; Output: `window_mask` per channel).
    *   [ ] P1.2.b: Implement training loop (GT: QMF -> `IntermediateData.window_mask`). Loss: Cross-entropy if classification.
    *   [ ] P1.2.c: Implement evaluation (accuracy).
    *   [ ] P1.2.d: Visualize predicted vs. actual window masks.
*   [ ] **P1.3: NN_QMF_to_MDCT Module**
    *   [ ] P1.3.a: Define architecture (Input: QMF bands, `window_mask`; Output: MDCT coefficients).
    *   [ ] P1.3.b: Implement training loop (GT: QMF, `window_mask` -> `IntermediateData.mdct_specs`).
    *   [ ] P1.3.c: Implement evaluation.
    *   [ ] P1.3.d: Visualize MDCT spectra.
*   [ ] **P1.4: NN_Scaler (Scale Factor Indices & Pre-Quant Floats) Module**
    *   [ ] P1.4.a: Define architecture (Input: MDCT coeffs, `window_mask`; Output: SFIs per BFU, pre-quant float values per BFU).
    *   [ ] P1.4.b: Implement training loop (GT: MDCT, `window_mask` -> `IntermediateData.scaled_blocks[ch][bfu].ScaleFactorIndex`, `IntermediateData.scaled_blocks[ch][bfu].Values`). Multi-task loss.
    *   [ ] P1.4.c: Implement evaluation (accuracy for SFIs, MSE for float values).
    *   [ ] P1.4.d: Visualize SFIs and pre-quant values.
*   [ ] **P1.5: NN_Bit_Allocator Module**
    *   [ ] P1.5.a: Define architecture (Input: MDCT features/SFIs; Output: Bits per BFU).
    *   [ ] P1.5.b: Implement training loop (GT: MDCT/SFIs -> `IntermediateData.bits_per_bfu`). Loss: Cross-entropy or MSE then discretize.
    *   [ ] P1.5.c: Implement evaluation (accuracy/MSE).
    *   [ ] P1.5.d: Visualize bit allocation patterns.
*   [ ] **P1.6: NN_Quantizer (Differentiable) Module**
    *   [ ] P1.6.a: Implement/choose a differentiable quantization method (e.g., Straight-Through Estimator, STE).
    *   [ ] P1.6.b: Define architecture (Input: Pre-quant floats from P1.4's NN, bits per BFU from P1.5's NN; Output: Quantized integer values).
    *   [ ] P1.6.c: Implement training loop (GT: Pre-quant floats, bits_per_bfu -> `IntermediateData.quantized_values`). Loss: MSE on the *reconstructed float values* after NN dequantization, or directly on quantized integers if using STE carefully.
    *   [ ] P1.6.d: Implement evaluation.
    *   [ ] P1.6.e: Visualize quantized values and quantization error (if predicted as auxiliary).

---

**Phase 2: Neural Decoder Modules (Mimicking `FrameDecoder`)**

*   **General Sub-steps for each Decoder Module (2.x):** (Similar to Phase 1)
    *   [ ] **(a) Define NN Architecture.**
    *   [ ] **(b) Implement Training Loop.**
    *   [ ] **(c) Implement Evaluation.**
    *   [ ] **(d) Implement Visualization.**

*   [ ] **P2.1: NN_Dequantizer Module**
    *   [ ] P2.1.a: Define architecture (Input: `IntermediateData.quantized_values`, `IntermediateData.scaled_blocks[ch][bfu].ScaleFactorIndex`, `IntermediateData.bits_per_bfu`; Output: Dequantized float MDCT coefficients).
    *   [ ] P2.1.b: Implement training loop (GT: Inputs -> `DecoderIntermediateData.mdct_specs` from `decode_from_intermediate(IntermediateData)`).
    *   [ ] P2.1.c: Implement evaluation.
    *   [ ] P2.1.d: Visualize dequantized MDCT.
*   [ ] **P2.2: NN_IMDCT Module**
    *   [ ] P2.2.a: Define architecture (Input: Dequantized MDCT from P2.1's NN, `IntermediateData.window_mask`; Output: QMF synthesis bands).
    *   [ ] P2.2.b: Implement training loop (GT: MDCT, `window_mask` -> `DecoderIntermediateData.qmf_low/mid/hi`).
    *   [ ] P2.2.c: Implement evaluation.
    *   [ ] P2.2.d: Visualize QMF synthesis bands.
*   [ ] **P2.3: NN_IQMF (QMF Synthesis) Module**
    *   [ ] P2.3.a: Define architecture (Input: QMF synthesis bands from P2.2's NN; Output: Reconstructed PCM frame).
    *   [ ] P2.3.b: Implement training loop (GT: QMF synth bands -> `DecoderIntermediateData.pcm_output`).
    *   [ ] P2.3.c: Implement evaluation (PCM reconstruction SNR, LSD).
    *   [ ] P2.3.d: Listen to reconstructed PCM snippets. Visualize waveforms.

---

**Phase 3: Integrated Neural Codec & End-to-End Fine-tuning**

*   [ ] **P3.1: Assemble NN Encoder Chain**
    *   [ ] Create a master `NNEncoder` class that sequentially calls modules P1.1 -> P1.2 -> P1.3 -> P1.4 -> P1.5 -> P1.6.
    *   [ ] Load pre-trained weights for each module.
    *   [ ] Test chain: Feed PCM, compare output (quantized values, SFIs, etc.) with C++ `IntermediateData`. Calculate element-wise differences/accuracy.
*   [ ] **P3.2: Assemble NN Decoder Chain**
    *   [ ] Create a master `NNDecoder` class that sequentially calls modules P2.1 -> P2.2 -> P2.3.
    *   [ ] Load pre-trained weights for each module.
    *   [ ] Test chain: Feed quantized part of C++ `IntermediateData`, compare output PCM with C++ `DecoderIntermediateData.pcm_output`.
*   [ ] **P3.3: Full Neural Codec Assembly**
    *   [ ] Create a master `NNFullCodec` class: `NNEncoder` -> `NNDecoder`.
    *   [ ] Test: Feed PCM -> `NNFullCodec` -> Reconstructed PCM.
    *   [ ] Evaluate initial reconstruction quality (SNR, LSD). Listen to samples.
*   [ ] **P3.4: End-to-End Fine-tuning**
    *   [ ] Implement an end-to-end training loop for `NNFullCodec`.
    *   [ ] Primary Loss: PCM reconstruction loss (e.g., L1/MSE in time domain, or STFT/Mel-spectrogram loss between original PCM and NN reconstructed PCM).
    *   [ ] (Optional) Define and add auxiliary losses at intermediate stages if needed for stability or quality.
    *   [ ] Fine-tune the entire model.
    *   [ ] Evaluate regularly on validation set (objective metrics and listening tests).

---

**Phase 4: ATRAC1 Bitstream Compatibility & File I/O**

*   [ ] **P4.1: Neural Output to C++ Parameter Conversion**
    *   [ ] Implement a Python function `neural_outputs_to_atrac_params(nn_encoder_output)`:
        *   [ ] Takes the dictionary of tensors from the `NNEncoder`.
        *   [ ] Converts `window_mask` to the correct integer format.
        *   [ ] Converts `ScaleFactorIndices` to the correct integer format.
        *   [ ] Converts `bits_per_bfu` to the correct integer format.
        *   [ ] Converts `quantized_values` (integers) to a list of lists format.
        *   [ ] Bundles these into a structure suitable for writing an ATRAC1 frame.
*   [ ] **P4.2: Frame Encoding to AEA Bitstream (Strategy Decision)**
    *   [ ] **Decision:** Choose Option A (Simplified C++ Writer) or Option B (Python Bitstream Writer).
        *   *Recommendation: Start with Option A for simplicity if feasible, as it avoids re-implementing complex bit-packing.*
*   [ ] **P4.2.A: (If Option A Chosen for Encoder) Modify C++ Bindings for Controlled Bitstream Writing**
    *   [ ] Design and implement a C++ function (exposed via Pybind11) like `write_precomputed_atrac_frame(output_stream, window_mask_ch0, window_mask_ch1, ..., list_of_sfi_ch0, ..., list_of_wordlen_ch0, ..., list_of_quantval_ch0, ..., bfu_amount_idx_ch0, ...)` that bypasses the C++ internal decision-making (scaling, bit allocation, quantization) and directly writes the bitstream using provided parameters. This function would essentially be a stripped-down version of `TAtrac1BitStreamWriter::WriteBitStream`.
    *   [ ] Re-build and test Python bindings.
*   [ ] **P4.2.B: (If Option B Chosen for Encoder) Implement Python Bitstream Writer**
    *   [ ] Carefully re-implement the logic of `TAtrac1BitStreamWriter::WriteBitStream` in Python to pack the parameters from P4.1 into a byte array representing one ATRAC1 frame.
*   [ ] **P4.2.C: Implement AEA File Encoder**
    *   [ ] Python script to take an input WAV file.
    *   [ ] Read WAV, process frame-by-frame with `NNEncoder`.
    *   [ ] Convert NN outputs to ATRAC params (P4.1).
    *   [ ] Use the chosen method (P4.2.A or P4.2.B) to generate the byte data for each ATRAC1 frame.
    *   [ ] Write AEA header (mimicking `TAeaOutput::CreateMeta`).
    *   [ ] Write the sequence of frame byte data to an output `.aea` file.
*   [ ] **P4.3: AEA Frame Parsing to Neural Decoder Input (Strategy Decision)**
    *   [ ] **Decision:** Choose Option A (Use C++ Parser) or Option B (Python Bitstream Parser).
        *   *Recommendation: Option A is likely easier.*
*   [ ] **P4.3.A: (If Option A Chosen for Decoder) Ensure C++ Parser Output is Suitable**
    *   [ ] Verify that `atracdenc.FrameDecoder.decode_frame()` provides all necessary parsed elements (quantized values, SFIs, word lengths, window masks) in its `DecoderIntermediateData` to feed into the `NNDecoder` (specifically, `NN_Dequantizer`).
*   [ ] **P4.3.B: (If Option B Chosen for Decoder) Implement Python Bitstream Parser**
    *   [ ] Re-implement the bitstream parsing logic from `TAtrac1Dequantiser::Dequant` in Python.
*   [ ] **P4.3.C: Implement AEA File Decoder**
    *   [ ] Python script to take an input `.aea` file.
    *   [ ] Read AEA header.
    *   [ ] Read AEA frame-by-frame.
    *   [ ] Parse each frame's byte data into parameters suitable for `NNDecoder` (using P4.3.A or P4.3.B).
    *   [ ] Process parameters through `NNDecoder` to get PCM.
    *   [ ] Write output PCM frames to a WAV file.
*   [ ] **P4.4: File I/O and Compatibility Testing (T4)**
    *   [ ] Test: `MyInput.wav -> NN_File_Encoder -> MyOutput.aea`.
    *   [ ] Then: `MyOutput.aea -> C++_atracdenc.decode_file -> DecodedFromNN.wav`.
    *   [ ] Evaluate `DecodedFromNN.wav` (play, compare to `MyInput.wav`).
    *   [ ] Test: `OriginalCpp.wav -> C++_atracdenc.encode_file -> OriginalCpp.aea`.
    *   [ ] Then: `OriginalCpp.aea -> NN_File_Decoder -> DecodedFromCpp.wav`.
    *   [ ] Evaluate `DecodedFromCpp.wav` (play, compare to `OriginalCpp.wav`).
    *   [ ] Check if `MyOutput.aea` is a valid AEA file (e.g., does C++ decoder handle it without crashing, even if quality is TBD).

---

**Phase 5: Optimization & Improvement**

*   [ ] **P5.1: Bit Allocation Refinement (`NN_Bit_Allocator`)**
    *   [ ] Freeze weights of all other NN modules (from Phase 3).
    *   [ ] Design a new training loop for `NN_Bit_Allocator`:
        *   Input: Features from `NN_Scaler` (or MDCT).
        *   Action: `NN_Bit_Allocator` predicts bit allocation.
        *   Consequence: Pass this allocation + other NN outputs through the *rest of the NN encoder and the full NN decoder*.
        *   Reward/Loss: End-to-end PCM reconstruction loss, or a psychoacoustic quality metric (if differentiable or using RL).
    *   [ ] Train `NN_Bit_Allocator`. The goal is for it to learn allocations that improve final quality for a given (implicit or explicit) total bit budget.
    *   [ ] Experiment with different input features to `NN_Bit_Allocator` (e.g., perceptual entropy, raw MDCT BFU energies).
*   [ ] **P5.2: Quantization Strategy Refinement (`NN_Quantizer`)**
    *   [ ] Research and implement alternative/improved differentiable quantization techniques.
    *   [ ] Fine-tune `NN_Quantizer` jointly with `NN_Bit_Allocator` (using the P5.1 setup), as their performance is coupled.
*   [ ] **P5.3: End-to-End Re-training with Optimized Components**
    *   [ ] Unfreeze all modules.
    *   [ ] Re-run end-to-end fine-tuning (as in P3.4) with the improved `NN_Bit_Allocator` and `NN_Quantizer`.
*   [ ] **P5.4: (Optional) Model Compression**
    *   [ ] If model size/speed is a concern, apply techniques like weight pruning, knowledge distillation (e.g., distill to a smaller student model), or neural network quantization (quantizing the NN weights themselves).
*   [ ] **P5.5: Final Evaluation and Comparison**
    *   [ ] Conduct comprehensive objective (SNR, LSD, STOI, PESQ if available) and subjective (A/B listening tests) evaluations.
    *   [ ] Compare:
        1.  Original C++ `atracdenc`.
        2.  Baseline NN Codec (from end of Phase 3/4).
        3.  Optimized NN Codec (from P5.3).
    *   [ ] Analyze performance on different types of audio content.
    *   [ ] Document findings and potential areas for further improvement.
