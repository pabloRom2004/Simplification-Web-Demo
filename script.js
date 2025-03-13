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

        // Each model's pipeline instance (only one quantization chosen at startup)
        this.modelInstances = {
            elementary: null,
            secondary: null,
            advanced: null
        };

        // Dummy model sizes based on quant
        this.modelSizes = {
            elementary: { fp32: '540 MB', fp16: '250 MB', int8: '136 MB', bnb4: '212 MB' },
            secondary: { fp32: '540 MB', fp16: '200 MB', int8: '136 MB', bnb4: '212 MB' },
            advanced: { fp32: '540 MB', fp16: '350 MB', int8: '136 MB', bnb4: '212 MB' }
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
                // Show the user which quant is used
                document.getElementById('currentQuantLabel').textContent =
                    `Current Quantization: ${quant}. To change, return to selection screen.`;
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
        // Button to return to selection screen
        document.getElementById('returnToSelectButton')
            .addEventListener('click', () => window.location.reload());
            
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
        document.getElementById('highlightCheckbox').addEventListener('change', (e) => {
            const resultsContainer = document.querySelector('.results-container');
            if (e.target.checked) {
                resultsContainer.classList.remove('no-highlight');
            } else {
                resultsContainer.classList.add('no-highlight');
            }
        });
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
        // Make the bar "indeterminate" style with animation
        progressElement.style.width = '100%';
        progressElement.classList.add('indeterminate');

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

    // Syllable counting function
    sylco(word) {
        word = word.toLowerCase();
        
        // exception_add are words that need extra syllables
        // exception_del are words that need less syllables
        const exception_add = ['serious', 'crucial'];
        const exception_del = ['fortunately', 'unfortunately'];
        
        const co_one = ['cool', 'coach', 'coat', 'coal', 'count', 'coin', 'coarse', 'coup', 'coif', 'cook', 'coign', 'coiffe', 'coof', 'court'];
        const co_two = ['coapt', 'coed', 'coinci'];
        const pre_one = ['preach'];
        
        let syls = 0;  // added syllable number
        let disc = 0;  // discarded syllable number
        
        // 1) if letters < 3 : return 1
        if (word.length <= 3) {
            return 1;
        }
            
        // 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end
        if (word.endsWith("es") || word.endsWith("ed")) {
            const doubleAndtripple_1 = (word.match(/[eaoui][eaoui]/g) || []).length;
            if (doubleAndtripple_1 > 1 || (word.match(/[eaoui][^eaoui]/g) || []).length > 1) {
                if (word.endsWith("ted") || word.endsWith("tes") || word.endsWith("ses") || word.endsWith("ied") || word.endsWith("ies")) {
                    // pass
                } else {
                    disc += 1;
                }
            }
        }
                    
        // 3) discard trailing "e", except where ending is "le"
        const le_except = ['whole', 'mobile', 'pole', 'male', 'female', 'hale', 'pale', 'tale', 'sale', 'aisle', 'whale', 'while'];
        if (word.endsWith("e")) {
            if (word.endsWith("le") && !le_except.includes(word)) {
                // pass
            } else {
                disc += 1;
            }
        }
                
        // 4) check if consecutive vowels exists, triplets or pairs, count them as one
        const doubleAndtripple = (word.match(/[eaoui][eaoui]/g) || []).length;
        const tripple = (word.match(/[eaoui][eaoui][eaoui]/g) || []).length;
        disc += doubleAndtripple + tripple;
        
        // 5) count remaining vowels in word
        const numVowels = (word.match(/[eaoui]/g) || []).length;
        
        // 6) add one if starts with "mc"
        if (word.startsWith("mc")) {
            syls += 1;
        }
            
        // 7) add one if ends with "y" but is not surrounded by vowel
        if (word.endsWith("y") && !"aeoui".includes(word.charAt(word.length - 2))) {
            syls += 1;
        }
            
        // 8) add one if "y" is surrounded by non-vowels and is not in the last word
        for (let i = 0; i < word.length; i++) {
            if (word[i] === "y") {
                if (i !== 0 && i !== word.length - 1) {
                    if (!"aeoui".includes(word[i-1]) && !"aeoui".includes(word[i+1])) {
                        syls += 1;
                    }
                }
            }
        }
                
        // 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one
        if (word.startsWith("tri") && "aeoui".includes(word[3])) {
            syls += 1;
        }
        if (word.startsWith("bi") && "aeoui".includes(word[2])) {
            syls += 1;
        }
            
        // 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
        if (word.endsWith("ian")) {
            if (word.endsWith("cian") || word.endsWith("tian")) {
                // pass
            } else {
                syls += 1;
            }
        }
                
        // 11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary
        if (word.startsWith("co") && "eaoui".includes(word[2])) {
            if (co_two.some(prefix => word.startsWith(prefix))) {
                syls += 1;
            } else if (co_one.some(prefix => word.startsWith(prefix))) {
                // pass
            } else {
                syls += 1;
            }
        }
                
        // 12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary
        if (word.startsWith("pre") && "eaoui".includes(word[3])) {
            if (pre_one.some(prefix => word.startsWith(prefix))) {
                // pass
            } else {
                syls += 1;
            }
        }
                
        // 13) check for "-n't" and cross match with dictionary to add syllable
        const negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"];
        if (word.endsWith("n't") && negative.includes(word)) {
            syls += 1;
        }
                
        // 14) Handling the exceptional words
        if (exception_del.includes(word)) {
            disc += 1;
        }
        if (exception_add.includes(word)) {
            syls += 1;
        }
            
        return numVowels - disc + syls;
    }

    // Calculate reading metrics (Flesch Reading Ease)
    calculateReadingMetrics(text) {
        try {
            // Simple sentence tokenizer
            const sentences = splitSentences(text);
            
            // Simple word tokenizer - find words with alphanumeric characters
            const words = text.match(/\b\w+\b/g) || [];
            
            if (sentences.length === 0 || words.length === 0) {
                return 0;
            }
            
            const totalSyllables = words.reduce((sum, word) => sum + this.sylco(word), 0);
            
            const score = 206.835 - 1.015 * (words.length / sentences.length) - 84.6 * (totalSyllables / words.length);
            return Math.max(0, Math.min(100, score));
        } catch (e) {
            console.error("Error processing text:", e);
            return 0;
        }
    }

    // Translate score to grade level
    scoreToGradeLevel(score) {
        if (score >= 90) return "5th Grade";
        if (score >= 80) return "6th Grade";
        if (score >= 70) return "7th Grade";
        if (score >= 60) return "8th-9th Grade";
        if (score >= 50) return "10th-12th Grade";
        if (score >= 30) return "College";
        return "College Graduate";
    }

    // Process models in sequence to avoid session conflicts
    async handleTextChange() {
        const text = document.getElementById('inputText').value.trim();
        if (!text || this.isProcessing) return;
    
        this.isProcessing = true;
        this.showLoading('Simplifying text...');
        this.clearOutputs();
        
        // Calculate reading score for original text
        const originalScore = this.calculateReadingMetrics(text);
        const originalGrade = this.scoreToGradeLevel(originalScore);
        
        // Display original text
        document.getElementById('original-text').textContent = text;
        
        // Add reading score display for original text
        const originalScoreElement = document.createElement('div');
        originalScoreElement.className = 'reading-score';
        originalScoreElement.textContent = `Reading Level: ${originalScore.toFixed(1)} (${originalGrade})`;
        document.getElementById('original-text').after(originalScoreElement);
    
        // Split text into sentences for processing
        const inputSentences = splitSentences(text);
        
        // Show/hide highlight toggle based on sentence count
        const highlightToggle = document.getElementById('highlightToggle');
        if (inputSentences.length > 1) {
            highlightToggle.classList.remove('hidden');
            
            // Visual representation of input sentences if multiple
            let originalHtml = '';
            inputSentences.forEach((sentence, idx) => {
                const colorClass = `highlighted-sentence-${idx % 4}`;
                originalHtml += `<span class="highlighted-sentence ${colorClass}" data-sentence-idx="${idx}">${sentence}</span> `;
            });
            document.getElementById('original-text').innerHTML = originalHtml.trim();
        } else {
            highlightToggle.classList.add('hidden');
            // For single sentence, just use plain text (no highlighting)
            document.getElementById('original-text').textContent = text;
        }
    
        const modelOrder = ['elementary', 'secondary', 'advanced'];
        const modelResults = {}; // Store results by model
        const modelOutputTexts = {}; // Store plain text for scoring
    
        try {
            // Process each model
            for (const model of modelOrder) {
                if (!this.downloadedModels[model]) {
                    // Handle non-downloaded models
                    const notDownloadedMsg = `Model not downloaded yet. Please download ${model} (${this.currentQuantization}).`;
                    modelResults[model] = notDownloadedMsg;
                    modelOutputTexts[model] = notDownloadedMsg;
                } else {
                    // Process each input sentence separately but preserve coloring
                    let modelOutputHtml = '';
                    let fullOutputText = '';
                    
                    for (let i = 0; i < inputSentences.length; i++) {
                        const inputSentence = inputSentences[i];
                        const simplified = await this.simplifyText(model, inputSentence);
                        
                        // Add to HTML with appropriate color class (if multiple sentences)
                        if (inputSentences.length > 1) {
                            const colorClass = `highlighted-sentence-${i % 4}`;
                            modelOutputHtml += `<span class="highlighted-sentence ${colorClass}" data-sentence-idx="${i}">${simplified}</span> `;
                        }
                        
                        // Add to plain text for reading score calculation
                        fullOutputText += simplified + ' ';
                    }
                    
                    modelResults[model] = modelOutputHtml.trim();
                    modelOutputTexts[model] = fullOutputText.trim();
                }
            }
            
            // Update outputs with HTML and calculate scores
            this.updateOutputsWithInputBasedColoring(modelResults, modelOutputTexts, inputSentences.length);
        } catch (error) {
            console.error('Simplification error:', error);
            this.showError('Failed to simplify text. Please try again.');
        } finally {
            this.hideLoading();
            this.isProcessing = false;
        }
    }
    
    // New method to handle the updated coloring approach
    updateOutputsWithInputBasedColoring(modelResults, modelOutputTexts, sentenceCount) {
        const modelOrder = ['elementary', 'secondary', 'advanced'];
        
        modelOrder.forEach((model) => {
            const output = document.getElementById(`${model}-output`);
            const result = modelResults[model];
            
            if (result.includes('Model not downloaded yet')) {
                output.textContent = result;
                output.classList.add('not-downloaded-output');
            } else {
                // Only use HTML with highlighting if there were multiple input sentences
                if (sentenceCount > 1) {
                    output.innerHTML = result;
                } else {
                    // For single sentence, use plain text without highlighting
                    output.textContent = modelOutputTexts[model];
                }
                
                // Calculate and show reading score based on plain text
                const score = this.calculateReadingMetrics(modelOutputTexts[model]);
                const grade = this.scoreToGradeLevel(score);
                
                const scoreElement = document.createElement('div');
                scoreElement.className = 'reading-score';
                scoreElement.textContent = `Reading Level: ${score.toFixed(1)} (${grade})`;
                output.after(scoreElement);
            }
        });
    }

    // Update outputs with reading scores
    updateOutputsWithScores(results, scores) {
        const modelOrder = ['elementary', 'secondary', 'advanced'];
        
        modelOrder.forEach((model, idx) => {
            const output = document.getElementById(`${model}-output`);
            const text = results[idx];
            const score = scores[idx];
            
            if (text.includes('Model not downloaded yet')) {
                output.textContent = text;
                output.classList.add('not-downloaded-output');
            } else {
                // Create visual representation if multiple sentences
                const sentences = splitSentences(text);
                if (sentences.length > 1) {
                    let html = '';
                    sentences.forEach((sentence, idx) => {
                        const colorClass = `highlighted-sentence-${idx % 4}`; // Cycle through 0-3
                        html += `<span class="highlighted-sentence ${colorClass}" data-sentence-idx="${idx}">${sentence}</span> `;
                    });
                    output.innerHTML = html.trim();
                }
                
                // Add reading score
                const grade = this.scoreToGradeLevel(score);
                const scoreElement = document.createElement('div');
                scoreElement.className = 'reading-score';
                scoreElement.textContent = `Reading Level: ${score.toFixed(1)} (${grade})`;
                output.after(scoreElement);
            }
        });
    }

    async simplifyText(model, text) {
        try {
            const pipelineInstance = this.modelInstances[model];
            if (!pipelineInstance) {
                throw new Error(`Model instance not loaded for ${model}`);
            }

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
                num_beams: 4,
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
        
        // 25% chance to combine multiple sentences
        const shouldCombine = Math.random() < 0.25;
        
        if (shouldCombine && this.sentences.length >= 2) {
            // Choose between 2-5 sentences
            const sentenceCount = Math.min(
                Math.floor(Math.random() * 4) + 2, // 2-5
                this.sentences.length // Don't exceed available sentences
            );
            
            // Get unique random sentences
            const selectedIndices = [];
            const combinedSentences = [];
            
            while (selectedIndices.length < sentenceCount) {
                const randomIndex = Math.floor(Math.random() * this.sentences.length);
                if (!selectedIndices.includes(randomIndex)) {
                    selectedIndices.push(randomIndex);
                    combinedSentences.push(this.sentences[randomIndex]);
                }
            }
            
            document.getElementById('inputText').value = combinedSentences.join(' ');
        } else {
            // Just one random sentence as before
            const randomIndex = Math.floor(Math.random() * this.sentences.length);
            document.getElementById('inputText').value = this.sentences[randomIndex];
        }
        
        this.handleTextChange();
    }

    clearOutputs() {
        document.getElementById('original-text').textContent = '';
        document.getElementById('elementary-output').textContent = '';
        document.getElementById('secondary-output').textContent = '';
        document.getElementById('advanced-output').textContent = '';
        // In clearOutputs method, you can add:
        document.getElementById('highlightToggle').classList.add('hidden');
        // Remove any reading score elements
        document.querySelectorAll('.reading-score').forEach(el => el.remove());
        
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

// Sentence splitting function
function splitSentences(text) {
    // More sophisticated sentence splitting that handles common abbreviations
    const abbrevs = ["Mr.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.", "e.g.", "i.e.", "etc.", "vs.", "U.S.", "U.K.", "A.M.", "P.M."];
    
    // Replace common abbreviations with temporary markers
    let processed = text;
    const replacements = [];
    
    abbrevs.forEach((abbrev, index) => {
        const marker = `__ABBREV${index}__`;
        const regex = new RegExp(abbrev.replace(/\./g, "\\."), "g");
        processed = processed.replace(regex, marker);
        replacements.push({ marker, original: abbrev });
    });
    
    // Split on sentence-ending punctuation followed by space and capital letter
    const sentences = processed.split(/([.!?])\s+(?=[A-Z])/);
    
    // Recombine sentence-ending punctuation with the sentence
    const result = [];
    for (let i = 0; i < sentences.length; i += 2) {
        if (i + 1 < sentences.length) {
            result.push(sentences[i] + sentences[i + 1]);
        } else {
            result.push(sentences[i]);
        }
    }
    
    // Restore abbreviations
    return result.map(sentence => {
        let restored = sentence;
        replacements.forEach(({ marker, original }) => {
            restored = restored.replace(new RegExp(marker, "g"), original);
        });
        return restored.trim();
    }).filter(s => s);
}

document.addEventListener('DOMContentLoaded', () => {
    new TextSimplifier();
});