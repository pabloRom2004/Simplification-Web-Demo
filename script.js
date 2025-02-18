import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.min.js';

class TextSimplifier {
    constructor() {
        // Model paths
        this.models = {
            elementary: 'pabRomero/BART-Firefox-Simplification-Elementary-ONNX',
            secondary: 'pabRomero/BART-Firefox-Simplification-Secondary-ONNX',
            advanced: 'pabRomero/BART-Firefox-Simplification-Advanced-ONNX'
        };

        // Initially, no model is downloaded
        this.downloadedModels = {
            elementary: false,
            secondary: false,
            advanced: false
        };

        // Each model’s pipeline instance (only one quantization chosen at startup)
        this.modelInstances = {
            elementary: null,
            secondary: null,
            advanced: null
        };

        // Dummy model sizes based on quant
        this.modelSizes = {
            elementary: { fp32: '540 MB', fp16: '250 MB', int8: '150 MB', bnb4: '212 MB' },
            secondary: { fp32: '540 MB', fp16: '200 MB', int8: '150 MB', bnb4: '212 MB' },
            advanced: { fp32: '540 MB', fp16: '350 MB', int8: '150 MB', bnb4: '212 MB' }
        };

        this.sentences = [];
        this.isProcessing = false;
        this.currentQuantization = null; // set on splash selection

        this.initSplashScreen();
    }

    // Show splash screen for quant selection
    initSplashScreen() {
        const splashScreen = document.getElementById('splashScreen');
        const quantButtons = document.querySelectorAll('.quant-button');

        quantButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const quant = e.target.getAttribute('data-quant');
                this.currentQuantization = quant;
                // Hide splash, show main UI
                splashScreen.classList.add('hidden');
                document.getElementById('mainUI').classList.remove('hidden');
                // Show the user which quant is used, mention WebGPU
                document.getElementById('currentQuantLabel').textContent =
                    `Current Quantization: ${quant} (using WebGPU). To change, reload the page.`;
                // Set label text and size for each model
                this.setModelDownloadInfo();
                // Continue init
                this.init();
            });
        });
    }

    async init() {
        try {
            await this.loadSentences();
            this.setupEventListeners();
            // We DO NOT auto-download any model; user must manually download.

            this.hideLoading();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize the application. Please refresh the page.');
        }
    }

    setupEventListeners() {
        // Buttons for random text & simplifying
        document.getElementById('randomButton')
            .addEventListener('click', () => this.setRandomSentence());
        document.getElementById('simplifyButton')
            .addEventListener('click', () => this.handleTextChange());

        // Download buttons for each model
        document.getElementById('elementary-download')
            .addEventListener('click', () => this.handleModelDownload('elementary'));
        document.getElementById('secondary-download')
            .addEventListener('click', () => this.handleModelDownload('secondary'));
        document.getElementById('advanced-download')
            .addEventListener('click', () => this.handleModelDownload('advanced'));
    }

    setModelDownloadInfo() {
        // For each model, set the quant label and show the size
        ['elementary', 'secondary', 'advanced'].forEach(model => {
            const quantLabel = document.getElementById(`${model}-quant-label`);
            const sizeLabel = document.getElementById(`${model}-size`);
            if (quantLabel) quantLabel.textContent = this.currentQuantization;
            if (sizeLabel) {
                sizeLabel.textContent = `(${this.modelSizes[model][this.currentQuantization]})`;
            }
        });
    }

    async loadSentences() {
        try {
            const response = await fetch('asset_test_sentences.txt');
            if (!response.ok) throw new Error('Failed to load sentences file');
            const text = await response.text();
            this.sentences = text.split('\n').filter(s => s.trim());
        } catch (error) {
            console.error('Error loading sentences:', error);
            this.showError('Failed to load example sentences.');
        }
    }

    updateModelStatus(model, status) {
        const statusElement = document.getElementById(`${model}-status`);
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `status ${status.toLowerCase()}`;
        }
    }

    async handleModelDownload(model) {
        const downloadBtn = document.getElementById(`${model}-download`);
        const progressBar = document.getElementById(`${model}-progress`);
        const progressElement = progressBar.querySelector('.progress');

        // Disable the button & show progress bar
        downloadBtn.disabled = true;
        progressBar.classList.remove('hidden');
        // Make the bar "indeterminate" style, so we just show it animating
        progressElement.style.width = '100%';
        progressElement.classList.add('indeterminate'); // e.g., a CSS class that animates

        this.updateModelStatus(model, 'Downloading');

        try {
            // Actually load the pipeline. 
            // If the model is cached locally, this should return quickly.
            // If not cached, it will fetch from HF Hub (no partial progress callback).
            await this.loadModel(model);

            this.downloadedModels[model] = true;
            downloadBtn.textContent = 'Downloaded';
            this.updateModelStatus(model, 'Ready');
        } catch (error) {
            console.error(`Failed to download ${model} model:`, error);
            this.updateModelStatus(model, 'Failed');
            downloadBtn.disabled = false;
        } finally {
            // Hide progress bar & remove 'indeterminate' style
            progressBar.classList.add('hidden');
            progressElement.classList.remove('indeterminate');
            progressElement.style.width = '0%';
        }
    }

    async loadModel(model) {
        try {
            // This triggers the actual retrieval from 
            // Hugging Face Hub if not cached locally.
            const pipelineInstance = await pipeline(
                'text2text-generation',
                this.models[model],
                {
                    dtype: this.currentQuantization,
                }
            );
            this.modelInstances[model] = pipelineInstance;
        } catch (error) {
            throw new Error(`Failed to load ${model} model with quantization ${this.currentQuantization}`);
        }
    }


    // Process models in sequence to avoid session conflicts
    async handleTextChange() {
        const text = document.getElementById('inputText').value.trim();
        if (!text || this.isProcessing) return;

        this.isProcessing = true;
        this.showLoading('Simplifying text...');
        this.clearOutputs();
        document.getElementById('original-text').textContent = text;

        const modelOrder = ['elementary', 'secondary', 'advanced'];
        const results = [];

        try {
            // Run each model in series
            for (const model of modelOrder) {
                if (!this.downloadedModels[model]) {
                    results.push(`Model not downloaded yet. Please download ${model} (${this.currentQuantization}).`);
                } else {
                    // Inference
                    const output = await this.simplifyText(model, text);
                    results.push(output);
                }
            }
            this.updateOutputs(results);
        } catch (error) {
            console.error('Simplification error:', error);
            this.showError('Failed to simplify text. Please try again.');
        } finally {
            this.hideLoading();
            this.isProcessing = false;
        }
    }

    async simplifyText(model, text) {
        try {
            const pipelineInstance = this.modelInstances[model];
            if (!pipelineInstance) {
                throw new Error(`Model instance not loaded for ${model}`);
            }

            // Choose correct decoder_start_token_id:
            // - If your config sets "decoder_start_token_id": 2, use 2 here
            // - If your fine-tuned model uses 0, keep it at 0
            // (In standard BART, it's 2)
            const chosenDecoderStart = 0; // or 2

            // Combined config for both input & output constraints + generation style:
            const generationConfig = {
                // 1) Truncate the *input* to 80 tokens
                truncation: true,
                max_length: 80,

                // 2) Generate up to 80 new tokens
                max_new_tokens: 80,

                // 3) Beam search / no-sampling BART settings
                do_sample: false,
                decoder_start_token_id: 2,
                forced_eos_token_id: 2,        // BART typically uses eos=2
                // forced_bos_token_id: null,
                // no_repeat_ngram_size: 3,       // often used in summarization tasks
                num_beams: 4,
                // min_length: 12,
                // length_penalty: 1.0,
                // early_stopping: true
            };

            const result = await pipelineInstance(text, generationConfig);
            return result[0].generated_text;
        } catch (error) {
            throw new Error(`Failed to simplify text with ${model} model`);
        }
    }

    setRandomSentence() {
        if (this.sentences.length === 0) {
            this.showError('No example sentences available.');
            return;
        }
        const randomIndex = Math.floor(Math.random() * this.sentences.length);
        document.getElementById('inputText').value = this.sentences[randomIndex];
        this.handleTextChange();
    }

    updateOutputs(results) {
        const modelOrder = ['elementary', 'secondary', 'advanced'];
        modelOrder.forEach((model, idx) => {
            const output = document.getElementById(`${model}-output`);
            const text = results[idx];
            output.textContent = text;
            if (text.includes('Model not downloaded yet')) {
                output.classList.add('not-downloaded-output');
            } else {
                output.classList.remove('not-downloaded-output');
            }
        });
    }

    clearOutputs() {
        document.getElementById('original-text').textContent = '';
        document.getElementById('elementary-output').textContent = '';
        document.getElementById('secondary-output').textContent = '';
        document.getElementById('advanced-output').textContent = '';
        this.hideError();
    }

    showLoading(message = 'Processing...') {
        const loadingIndicator = document.getElementById('loadingIndicator');
        loadingIndicator.querySelector('p').textContent = message;
        loadingIndicator.classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loadingIndicator').classList.add('hidden');
    }

    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        errorElement.textContent = message;
        errorElement.classList.remove('hidden');
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        document.getElementById('errorMessage').classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new TextSimplifier();
});
