class TextSimplifier {
    constructor() {
        this.models = {
            elementary: 'pabRomero/BART-Firefox-Simplification-Elementary-ONNX',
            secondary: 'pabRomero/BART-Firefox-Simplification-Secondary-ONNX',
            advanced: 'pabRomero/BART-Firefox-Simplification-Advanced-ONNX'
        };
        
        this.modelInstances = {};
        this.sentences = [];
        this.isProcessing = false;
        this.currentQuantization = 'int8';
        
        this.init();
    }

    async init() {
        try {
            await this.loadSentences();
            this.setupEventListeners();
            await this.loadModels();
            this.hideLoading();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize the application. Please refresh the page.');
        }
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
        statusElement.textContent = status;
        statusElement.className = `status ${status.toLowerCase()}`;
    }

    updateQuantizationLabels() {
        const models = ['elementary', 'secondary', 'advanced'];
        models.forEach(model => {
            const label = document.getElementById(`${model}-label`);
            label.textContent = `${model.charAt(0).toUpperCase() + model.slice(1)} (${this.currentQuantization}):`;
        });
    }

    async loadModels() {
        this.showLoading('Loading models...');
        this.currentQuantization = document.getElementById('quantization').value;
        
        try {
            const modelPromises = Object.entries(this.models).map(async ([key, modelPath]) => {
                this.updateModelStatus(key, 'Loading');
                this.modelInstances[key] = await pipeline(
                    'text2text-generation',
                    modelPath,
                    { dtype: this.currentQuantization }
                );
                this.updateModelStatus(key, 'Ready');
            });

            await Promise.all(modelPromises);
            this.updateQuantizationLabels();
        } catch (error) {
            console.error('Model loading error:', error);
            this.showError('Failed to load one or more models. Please check your connection and refresh.');
            throw error;
        }
    }

    setupEventListeners() {
        const inputText = document.getElementById('inputText');
        const randomButton = document.getElementById('randomButton');
        const simplifyButton = document.getElementById('simplifyButton');
        const quantization = document.getElementById('quantization');

        simplifyButton.addEventListener('click', () => this.handleTextChange());
        randomButton.addEventListener('click', () => this.setRandomSentence());
        quantization.addEventListener('change', () => this.handleQuantizationChange());
    }

    async handleTextChange() {
        const text = document.getElementById('inputText').value.trim();
        if (!text || this.isProcessing) return;

        this.isProcessing = true;
        this.showLoading('Simplifying text...');
        this.clearOutputs();

        // Update original text
        document.getElementById('original-text').textContent = text;

        try {
            const results = await Promise.all([
                this.simplifyText('elementary', text),
                this.simplifyText('secondary', text),
                this.simplifyText('advanced', text)
            ]);

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
            const result = await this.modelInstances[model](text, {
                max_new_tokens: 1000,
                temperature: 0
            });
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
        const inputText = document.getElementById('inputText');
        inputText.value = this.sentences[randomIndex];
        this.handleTextChange();
    }

    async handleQuantizationChange() {
        if (this.isProcessing) return;
        
        try {
            this.clearOutputs();
            await this.loadModels();
            const inputText = document.getElementById('inputText');
            if (inputText.value.trim()) {
                this.handleTextChange();
            }
        } catch (error) {
            console.error('Quantization change error:', error);
            this.showError('Failed to change quantization. Please refresh the page.');
        }
    }

    updateOutputs(results) {
        const [elementary, secondary, advanced] = results;
        
        document.getElementById('elementary-output').textContent = elementary;
        document.getElementById('secondary-output').textContent = secondary;
        document.getElementById('advanced-output').textContent = advanced;
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
        setTimeout(() => this.hideError(), 5000); // Hide error after 5 seconds
    }

    hideError() {
        document.getElementById('errorMessage').classList.add('hidden');
    }

    cleanup() {
        this.isProcessing = false;
        this.hideLoading();
        this.hideError();
        this.clearOutputs();
    }
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const simplifier = new TextSimplifier();
});